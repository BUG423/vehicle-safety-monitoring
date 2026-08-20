"""极简后台接收端 —— 只为演示模式C 的上报与断网补传。

**这不是模式B。** 模式B 那条路线做的是完整的车队级看板、复核工作流与统计分析；
这里只需要一个能收事件、能被人为「拔网线」的端点，用来证明：

  1. 车载设备产生的 `SafetyEvent` 能被后台原样收下（三条路线事件格式一致）；
  2. 后台不可达时事件落盘不丢；
  3. 后台恢复后自动补传，且顺序与内容完好。

    python3 mode-c-edge/backend/receiver.py --port 18080

接口：
    POST /api/events      收一条事件（JSON），落 JSONL
    GET  /api/stats       统计
    GET  /api/events      最近 N 条（?limit=20）
    POST /api/control     {"online": false} 模拟后台不可达（返回 503）
    GET  /health          健康检查
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "schema"))


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.online = True
        self.first_ts: float | None = None

    def add(self, ev: dict) -> None:
        with self.lock:
            self.events.append(ev)
            if self.first_ts is None:
                self.first_ts = time.time()
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def stats(self) -> dict:
        with self.lock:
            by_v = Counter(e.get("violation", "?") for e in self.events)
            by_s = Counter(e.get("severity", "?") for e in self.events)
            return {"total": len(self.events), "online": self.online,
                    "by_violation": dict(by_v), "by_severity": dict(by_s),
                    "store": str(self.path)}


class Handler(BaseHTTPRequestHandler):
    store: Store = None            # type: ignore[assignment]
    quiet = False

    def log_message(self, fmt, *args):  # noqa: A003 - 压掉默认访问日志
        if not self.quiet:
            pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            return self._json(200, {"ok": True, "online": self.store.online})
        if self.path.startswith("/api/stats"):
            return self._json(200, self.store.stats())
        if self.path.startswith("/api/events"):
            limit = 20
            if "?" in self.path and "limit=" in self.path:
                try:
                    limit = int(self.path.split("limit=")[1].split("&")[0])
                except ValueError:
                    pass
            with self.store.lock:
                return self._json(200, {"events": self.store.events[-limit:]})
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        if self.path.startswith("/api/control"):
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self._json(400, {"error": "bad json"})
            if "online" in body:
                self.store.online = bool(body["online"])
                state = "可达" if self.store.online else "不可达（模拟断网）"
                print(f"[后台] 状态切换 → {state}")
            return self._json(200, {"online": self.store.online})

        if self.path.startswith("/api/events"):
            if not self.store.online:
                return self._json(503, {"error": "backend offline (simulated)"})
            try:
                ev = json.loads(raw)
            except ValueError:
                return self._json(400, {"error": "bad json"})
            self.store.add(ev)
            b64 = (ev.get("evidence") or {}).get("frame_b64")
            print(f"[后台] 收到 #{len(self.store.events):<3d} {ev.get('violation'):<26s} "
                  f"{ev.get('severity'):<8s} 车辆={ev.get('vehicle_id')} "
                  f"证据图={'有' if b64 else '无'}")
            return self._json(200, {"ok": True, "n": len(self.store.events)})
        return self._json(404, {"error": "not found"})


def serve(port: int, store_path: Path, *, block: bool = True):
    Handler.store = Store(store_path)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[后台] 监听 http://127.0.0.1:{port}  事件落库 {store_path}")
    if block:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[后台] 退出")
        finally:
            srv.server_close()
        return srv
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="backend-http")
    t.start()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description="模式C 演示用的极简后台接收端")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--store", default=str(Path(__file__).resolve().parent / "received_events.jsonl"))
    args = ap.parse_args()
    serve(args.port, Path(args.store))


if __name__ == "__main__":
    main()
