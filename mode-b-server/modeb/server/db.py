"""事件落库与统计 —— SQLite。

为什么是 SQLite：这是选型验证阶段的原型，甲方规模未定。真实车队（>200 台车、
事件按亿计）应换成 PostgreSQL + TimescaleDB 或 ClickHouse，
但**表结构与查询口径可以原样搬过去**，所以这里的 schema 是按生产口径设计的，
不是临时凑的：事件表按时间分区友好、统计查询全部走索引、证据文件与元数据分离存放。

「汇总」是模式B 相对另外两条路线的核心优势，因此统计查询是一等公民：
车队总览、违规排行、驾驶员安全评分、时段趋势、复核工作流，全部在这里实现。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

_SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    vehicle_id    TEXT NOT NULL,
    violation     TEXT NOT NULL,
    severity      TEXT NOT NULL,
    decision      TEXT NOT NULL DEFAULT 'confirmed',   -- confirmed | undecidable
    role          TEXT,
    seat          TEXT,
    confidence    REAL,
    duration_s    REAL,
    message       TEXT,
    mode          TEXT,
    raw_json      TEXT NOT NULL,
    evidence_path TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending',   -- pending|confirmed|dismissed|appealed
    review_note   TEXT,
    reviewed_at   REAL
);
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id   TEXT PRIMARY KEY,
    plate        TEXT,
    fleet        TEXT,
    driver_name  TEXT,
    driver_id    TEXT,
    source_kind  TEXT,
    status       TEXT DEFAULT 'offline',
    registered_at REAL,
    last_seen    REAL,
    meta_json    TEXT
);
"""

# 索引单独一段：必须在 `_migrate()` 给老库补完列之后再建，
# 否则 `idx_events_decision` 会在还没有 decision 列的老库上直接报错。
_SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_events_ts        ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_vehicle   ON events(vehicle_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_violation ON events(violation, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_review    ON events(review_status, ts DESC);
-- 「判不了」的记录必须能被单独查出来：它既不能混进违规统计，也绝不能被当成合规而丢掉
CREATE INDEX IF NOT EXISTS idx_events_decision  ON events(decision, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vehicles_seen    ON vehicles(last_seen DESC);
"""

# 违规扣分表 —— 驾驶员安全评分的依据。分值按「是否直接致命」排序，
# 不是按检测难度排序：未系安全带和疲劳是事故致死的主因，扣分最重。
PENALTY: dict[str, int] = {
    "driver.fatigue": 12,
    "driver.no_seatbelt": 10,
    "driver.phone_use": 10,
    "vehicle.speeding": 10,
    "driver.identity_mismatch": 10,
    "driver.distraction": 6,
    "driver.smoking": 4,
    "driver.hands_off_wheel": 4,
    "passenger.no_seatbelt": 3,
    "passenger.overload": 6,
    "passenger.child_front_seat": 5,
    "vehicle.harsh_driving": 3,
    "driver.absent": 1,
    "system.camera_blocked": 8,   # 遮挡摄像头是主观规避行为，扣分不能轻
}
DEFAULT_PENALTY = 3


class EventStore:
    """线程安全的 SQLite 封装（FastAPI 里多个请求线程会并发访问）。"""

    def __init__(self, path: str | Path, evidence_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = Path(evidence_dir) if evidence_dir else self.path.parent / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_TABLES)
        self._migrate()                                  # 老库补列，必须在建索引之前
        self._conn.executescript(_SCHEMA_INDEXES)
        self._conn.execute("PRAGMA journal_mode=WAL")   # 写入与看板查询并发
        self._conn.commit()

    def _migrate(self) -> None:
        """就地升级老库 —— 演示环境里 runs/modeb.db 常常是上一版留下的。"""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(events)")}
        if "decision" not in cols:
            self._conn.execute(
                "ALTER TABLE events ADD COLUMN decision TEXT NOT NULL DEFAULT 'confirmed'")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- 车辆 ---------------------------------------------------------------
    def register_vehicle(self, vehicle_id: str, **fields: Any) -> None:
        now = time.time()
        meta = json.dumps(fields.pop("meta", {}) or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """INSERT INTO vehicles(vehicle_id, plate, fleet, driver_name, driver_id,
                                        source_kind, status, registered_at, last_seen, meta_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(vehicle_id) DO UPDATE SET
                       plate=COALESCE(excluded.plate, vehicles.plate),
                       fleet=COALESCE(excluded.fleet, vehicles.fleet),
                       driver_name=COALESCE(excluded.driver_name, vehicles.driver_name),
                       driver_id=COALESCE(excluded.driver_id, vehicles.driver_id),
                       source_kind=COALESCE(excluded.source_kind, vehicles.source_kind),
                       status=excluded.status, last_seen=excluded.last_seen,
                       meta_json=excluded.meta_json""",
                (vehicle_id, fields.get("plate"), fields.get("fleet"), fields.get("driver_name"),
                 fields.get("driver_id"), fields.get("source_kind"), fields.get("status", "online"),
                 now, now, meta))
            self._conn.commit()

    def touch_vehicle(self, vehicle_id: str, status: str = "online") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE vehicles SET last_seen=?, status=? WHERE vehicle_id=?",
                (time.time(), status, vehicle_id))
            self._conn.commit()

    def list_vehicles(self, offline_after_s: float = 20.0) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM vehicles ORDER BY last_seen DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["meta"] = json.loads(d.pop("meta_json") or "{}")
            idle = now - (d.get("last_seen") or 0)
            d["idle_s"] = round(idle, 1)
            d["online"] = idle <= offline_after_s
            if not d["online"]:
                d["status"] = "offline"
            out.append(d)
        return out

    # -- 事件 ---------------------------------------------------------------
    def insert_event(self, event_dict: dict[str, Any]) -> str:
        """写入一条事件。证据图另存文件，DB 只留路径，避免行体积失控。"""
        ev = dict(event_dict)
        ev_id = ev.get("event_id") or f"e{int(time.time() * 1000)}"
        evidence = dict(ev.get("evidence") or {})
        path = None
        b64 = evidence.get("frame_b64")
        if b64:
            path = self._save_evidence(ev_id, b64)
            evidence["frame_b64"] = None
            evidence["frame_uri"] = f"/api/v1/events/{ev_id}/evidence"
            ev["evidence"] = evidence

        subject = ev.get("subject") or {}
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO events(event_id, ts, vehicle_id, violation, severity,
                       decision, role, seat, confidence, duration_s, message, mode, raw_json,
                       evidence_path, review_status)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                          COALESCE((SELECT review_status FROM events WHERE event_id=?), 'pending'))""",
                (ev_id, float(ev.get("ts", time.time())), ev.get("vehicle_id", ""),
                 ev.get("violation", ""), ev.get("severity", "warn"),
                 ev.get("decision", "confirmed"), subject.get("role"),
                 subject.get("seat"), float(ev.get("confidence") or 0.0),
                 float(ev.get("duration_s") or 0.0), ev.get("message", ""), ev.get("mode", ""),
                 json.dumps(ev, ensure_ascii=False), str(path) if path else None, ev_id))
            self._conn.commit()
        return ev_id

    def _save_evidence(self, event_id: str, data_uri: str) -> Path | None:
        import base64
        try:
            b64 = data_uri.split(",", 1)[-1]
            path = self.evidence_dir / f"{event_id}.jpg"
            path.write_bytes(base64.b64decode(b64))
            return path
        except Exception:  # noqa: BLE001
            return None

    def evidence_path(self, event_id: str) -> Path | None:
        with self._lock:
            row = self._conn.execute("SELECT evidence_path FROM events WHERE event_id=?",
                                     (event_id,)).fetchone()
        if not row or not row["evidence_path"]:
            return None
        p = Path(row["evidence_path"])
        return p if p.exists() else None

    def query_events(self, *, vehicle_id: str | None = None, violation: str | None = None,
                     severity: str | None = None, review_status: str | None = None,
                     decision: str | None = None,
                     since: float | None = None, until: float | None = None,
                     limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        sql = ["SELECT * FROM events WHERE 1=1"]
        args: list[Any] = []
        for col, val in (("vehicle_id", vehicle_id), ("violation", violation),
                         ("severity", severity), ("review_status", review_status),
                         ("decision", decision)):
            if val:
                sql.append(f"AND {col}=?")
                args.append(val)
        if since is not None:
            sql.append("AND ts>=?")
            args.append(since)
        if until is not None:
            sql.append("AND ts<=?")
            args.append(until)
        sql.append("ORDER BY ts DESC LIMIT ? OFFSET ?")
        args += [int(limit), int(offset)]
        with self._lock:
            rows = self._conn.execute(" ".join(sql), args).fetchall()
        return [_row_to_event(r) for r in rows]

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        return _row_to_event(row) if row else None

    def review_event(self, event_id: str, status: str, note: str = "") -> bool:
        if status not in ("pending", "confirmed", "dismissed", "appealed"):
            raise ValueError(f"非法复核状态: {status}")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE events SET review_status=?, review_note=?, reviewed_at=? WHERE event_id=?",
                (status, note, time.time(), event_id))
            self._conn.commit()
        return cur.rowcount > 0

    # -- 汇总统计（模式B 的核心价值） ---------------------------------------
    def overview(self, window_s: float = 86400.0) -> dict[str, Any]:
        """总览。`events` / `critical` **只统计确认违规**，「判不了」单列为 `undecidable`。

        把两者混在一起会同时犯两个错：误报率虚高，以及更糟的——
        漏检被当成合规。这两个数字必须分开看。
        """
        since = time.time() - window_s
        C = "decision='confirmed'"
        with self._lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) c FROM events WHERE ts>=? AND {C}", (since,)).fetchone()["c"]
            crit = self._conn.execute(
                f"SELECT COUNT(*) c FROM events WHERE ts>=? AND {C} AND severity='critical'",
                (since,)).fetchone()["c"]
            undecidable = self._conn.execute(
                "SELECT COUNT(*) c FROM events WHERE ts>=? AND decision='undecidable'",
                (since,)).fetchone()["c"]
            und_vehicles = self._conn.execute(
                "SELECT COUNT(DISTINCT vehicle_id) c FROM events WHERE ts>=? AND decision='undecidable'",
                (since,)).fetchone()["c"]
            pending = self._conn.execute(
                f"SELECT COUNT(*) c FROM events WHERE ts>=? AND {C} AND review_status='pending'",
                (since,)).fetchone()["c"]
            vehicles = self._conn.execute("SELECT COUNT(*) c FROM vehicles").fetchone()["c"]
            involved = self._conn.execute(
                f"SELECT COUNT(DISTINCT vehicle_id) c FROM events WHERE ts>=? AND {C}",
                (since,)).fetchone()["c"]
        online = sum(1 for v in self.list_vehicles() if v["online"])
        return {"window_s": window_s, "events": total, "critical": crit, "pending_review": pending,
                "undecidable": undecidable, "vehicles_unchecked": und_vehicles,
                "vehicles": vehicles, "vehicles_online": online, "vehicles_with_events": involved}

    def data_quality(self, window_s: float = 86400.0, limit: int = 20) -> dict[str, Any]:
        """检查完成度 —— 「这车没问题」和「这车没看清」必须能分开回答。

        这是车队管理系统的基本要求：发车前放行一台「没看清」的车，
        和放行一台「确认合规」的车，责任完全不同。
        """
        since = time.time() - window_s
        with self._lock:
            by_v = self._conn.execute(
                """SELECT violation, COUNT(*) n FROM events
                   WHERE ts>=? AND decision='undecidable'
                   GROUP BY violation ORDER BY n DESC""", (since,)).fetchall()
            by_veh = self._conn.execute(
                """SELECT e.vehicle_id, v.plate, v.driver_name, COUNT(*) n
                   FROM events e LEFT JOIN vehicles v ON v.vehicle_id=e.vehicle_id
                   WHERE e.ts>=? AND e.decision='undecidable'
                   GROUP BY e.vehicle_id ORDER BY n DESC LIMIT ?""", (since, limit)).fetchall()
            reasons = self._conn.execute(
                """SELECT raw_json FROM events WHERE ts>=? AND decision='undecidable'
                   ORDER BY ts DESC LIMIT 300""", (since,)).fetchall()
        reason_count: dict[str, int] = {}
        for r in reasons:
            try:
                why = (json.loads(r["raw_json"]).get("raw_signals") or {}).get("undecidable_reason")
            except (ValueError, TypeError):
                why = None
            if why:
                reason_count[why] = reason_count.get(why, 0) + 1
        return {"by_violation": [dict(r) for r in by_v],
                "by_vehicle": [dict(r) for r in by_veh],
                "by_reason": sorted(({"reason": k, "n": v} for k, v in reason_count.items()),
                                    key=lambda x: x["n"], reverse=True)}

    def violation_ranking(self, window_s: float = 86400.0, limit: int = 20) -> list[dict[str, Any]]:
        since = time.time() - window_s
        with self._lock:
            rows = self._conn.execute(
                """SELECT violation, COUNT(*) n,
                          SUM(CASE WHEN severity='critical' THEN 1 ELSE 0 END) n_critical
                   FROM events WHERE ts>=? AND decision='confirmed'
                   GROUP BY violation ORDER BY n DESC LIMIT ?""",
                (since, limit)).fetchall()
        return [dict(r) for r in rows]

    def vehicle_ranking(self, window_s: float = 86400.0, limit: int = 20) -> list[dict[str, Any]]:
        """违规排行 —— 按加权扣分排序，而不是按事件条数。

        按条数排会把「乘客没系安全带 20 次」排在「疲劳驾驶 2 次」前面，
        而后者才是真正会出人命的。管理者要看的是风险，不是计数。
        """
        since = time.time() - window_s
        with self._lock:
            rows = self._conn.execute(
                """SELECT e.vehicle_id, e.violation, COUNT(*) n, v.plate, v.driver_name, v.driver_id
                   FROM events e LEFT JOIN vehicles v ON v.vehicle_id = e.vehicle_id
                   WHERE e.ts>=? AND e.decision='confirmed'
                   GROUP BY e.vehicle_id, e.violation""", (since,)).fetchall()
        agg: dict[str, dict[str, Any]] = {}
        for r in rows:
            d = agg.setdefault(r["vehicle_id"], {
                "vehicle_id": r["vehicle_id"], "plate": r["plate"], "driver_name": r["driver_name"],
                "driver_id": r["driver_id"], "events": 0, "penalty": 0, "by_violation": {}})
            d["events"] += r["n"]
            d["penalty"] += PENALTY.get(r["violation"], DEFAULT_PENALTY) * r["n"]
            d["by_violation"][r["violation"]] = r["n"]
        out = sorted(agg.values(), key=lambda x: x["penalty"], reverse=True)[:limit]
        for d in out:
            d["score"] = max(0, 100 - d["penalty"])       # 安全评分：100 分制，扣完为止
            d["grade"] = _grade(d["score"])
        return out

    def driver_scores(self, window_s: float = 7 * 86400.0, limit: int = 50) -> list[dict[str, Any]]:
        """驾驶员安全评分 —— 按人聚合（一人可能开多台车）。"""
        since = time.time() - window_s
        with self._lock:
            rows = self._conn.execute(
                """SELECT COALESCE(v.driver_id, e.vehicle_id) did,
                          COALESCE(v.driver_name, e.vehicle_id) dname,
                          e.violation, COUNT(*) n, COUNT(DISTINCT e.vehicle_id) nveh
                   FROM events e LEFT JOIN vehicles v ON v.vehicle_id = e.vehicle_id
                   WHERE e.ts>=? AND e.decision='confirmed'
                   GROUP BY did, e.violation""", (since,)).fetchall()
        agg: dict[str, dict[str, Any]] = {}
        for r in rows:
            d = agg.setdefault(r["did"], {"driver_id": r["did"], "driver_name": r["dname"],
                                          "events": 0, "penalty": 0, "vehicles": 0,
                                          "by_violation": {}})
            d["events"] += r["n"]
            d["vehicles"] = max(d["vehicles"], r["nveh"])
            d["penalty"] += PENALTY.get(r["violation"], DEFAULT_PENALTY) * r["n"]
            d["by_violation"][r["violation"]] = r["n"]
        out = []
        for d in agg.values():
            d["score"] = max(0, 100 - d["penalty"])
            d["grade"] = _grade(d["score"])
            out.append(d)
        return sorted(out, key=lambda x: x["score"])[:limit]

    def timeline(self, window_s: float = 86400.0, buckets: int = 24) -> list[dict[str, Any]]:
        """时段趋势 —— 疲劳驾驶在凌晨聚集是行业常识，看板要能把它显出来。"""
        now = time.time()
        since = now - window_s
        step = window_s / buckets
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, severity FROM events WHERE ts>=? AND decision='confirmed'",
                (since,)).fetchall()
        out = [{"t": round(since + i * step), "total": 0, "critical": 0} for i in range(buckets)]
        for r in rows:
            i = min(buckets - 1, max(0, int((r["ts"] - since) / step)))
            out[i]["total"] += 1
            if r["severity"] == "critical":
                out[i]["critical"] += 1
        return out

    def bulk_insert(self, events: Iterable[dict[str, Any]]) -> int:
        n = 0
        for e in events:
            self.insert_event(e)
            n += 1
        return n


def _grade(score: int) -> str:
    if score >= 90:
        return "优秀"
    if score >= 75:
        return "良好"
    if score >= 60:
        return "需关注"
    return "高风险"


# 后台自己加的字段 —— 它们不属于 SafetyEvent 契约，
# 任何要把查询结果还原成 SafetyEvent 的地方都必须先剥掉
BACKEND_ONLY_FIELDS = ("review_status", "review_note", "reviewed_at")


def strip_backend_fields(row: dict[str, Any]) -> dict[str, Any]:
    """把 `/api/v1/events` 的返回值还原成纯 SafetyEvent 字典。

    `SafetyEvent.from_dict()` 对未知字段是**报错**而不是忽略，
    所以多带一个 `review_status` 就会让下游反序列化失败。
    这条已作为契约层修改建议提给汇总方（见 README「对契约层的修改建议」）。
    """
    d = {k: v for k, v in row.items() if k not in BACKEND_ONLY_FIELDS}
    d.pop("schema_version", None)
    return d


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    d = json.loads(row["raw_json"])
    d["event_id"] = row["event_id"]
    d.setdefault("decision", row["decision"])
    d["review_status"] = row["review_status"]
    d["review_note"] = row["review_note"]
    d["reviewed_at"] = row["reviewed_at"]
    if row["evidence_path"]:
        d.setdefault("evidence", {})["frame_uri"] = f"/api/v1/events/{row['event_id']}/evidence"
    return d
