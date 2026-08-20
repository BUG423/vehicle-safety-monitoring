"""单车流水线 —— 一辆车一条时间线。

    感知结果 → 逐帧判定(rules) → 防误报确认(common.ViolationConfirmer)
             → SafetyEvent → AlertDispatcher（车内 + 后台）

一辆车一个实例，因为确认器、PERCLOS、跟踪器都是**有状态**的，不能跨车共享。
GPU 推理是跨车共享的，由 scheduler 统一批处理 —— 这正是模式B 的成本优势所在。
"""
from __future__ import annotations

import base64
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import (AlertDispatcher, DetectionMode, Evidence, SafetyEvent,  # noqa: E402
                    Subject, VehicleContext, ViolationConfirmer)

from ..config import Config  # noqa: E402
from ..perception.base import IoUTracker, PerceptionResult, assign_seats  # noqa: E402
from ..sources.base import Frame  # noqa: E402
from .rules import RawHit, ViolationRuleEngine  # noqa: E402


@dataclass
class PipelineStats:
    frames: int = 0
    events: int = 0
    last_ts: float = 0.0
    infer_ms_avg: float = 0.0
    e2e_ms_avg: float = 0.0
    persons_last: int = 0
    _e2e: list[float] = field(default_factory=list)

    def observe(self, infer_ms: float, e2e_ms: float, persons: int) -> None:
        self.frames += 1
        self.last_ts = time.time()
        self.persons_last = persons
        a = 0.1
        self.infer_ms_avg = infer_ms if self.frames == 1 else (1 - a) * self.infer_ms_avg + a * infer_ms
        self.e2e_ms_avg = e2e_ms if self.frames == 1 else (1 - a) * self.e2e_ms_avg + a * e2e_ms
        self._e2e.append(e2e_ms)
        if len(self._e2e) > 2000:
            del self._e2e[:1000]

    def percentile(self, q: float) -> float:
        if not self._e2e:
            return 0.0
        return float(np.percentile(self._e2e, q))

    def to_dict(self) -> dict[str, Any]:
        return {"frames": self.frames, "events": self.events,
                "infer_ms_avg": round(self.infer_ms_avg, 1),
                "e2e_ms_avg": round(self.e2e_ms_avg, 1),
                "e2e_ms_p95": round(self.percentile(95), 1),
                "persons_last": self.persons_last,
                "last_ts": round(self.last_ts, 3)}


class VehiclePipeline:
    """一辆车的完整判定链路。"""

    def __init__(self, vehicle_id: str, cfg: Config, *,
                 dispatcher: AlertDispatcher | None = None,
                 face_module: Any = None,
                 on_event: Callable[[SafetyEvent], None] | None = None,
                 evidence_max_side: int = 320) -> None:
        self.vehicle_id = vehicle_id
        self.cfg = cfg
        self.tracker = IoUTracker()
        self.rules = ViolationRuleEngine(cfg.rules, face_module=face_module,
                                         kp_thr=cfg.perception.keypoint_score_thr)
        self.confirmer = ViolationConfirmer()
        self.dispatcher = dispatcher
        self.on_event = on_event
        self.evidence_max_side = evidence_max_side
        self.stats = PipelineStats()
        self.context: VehicleContext | None = None
        self.last_hits: list[RawHit] = []
        self.active: dict[str, dict[str, Any]] = {}   # 当前处于违规态的项，供看板显示实时状态

    def set_context(self, ctx: VehicleContext | None) -> None:
        self.context = ctx

    # -- 主入口 -------------------------------------------------------------
    def process(self, frame: Frame, res: PerceptionResult) -> list[SafetyEvent]:
        assign_seats(res.persons, res.width, res.height, self.cfg.rules.driver_side)
        self.tracker.update(res.persons, res.ts)

        ctx = self.context
        if frame.meta:
            ctx = _ctx_from_meta(self.vehicle_id, frame.meta) or ctx

        hits = self.rules.evaluate(res, frame.image, ctx)
        self.last_hits = hits

        events: list[SafetyEvent] = []
        now = res.ts
        for hit in hits:
            c = self.confirmer.update(hit.violation, hit.hit, confidence=hit.confidence,
                                      key=f"{hit.key}", now=now)
            slot = f"{hit.violation.value}@{hit.key}"
            if c.active:
                self.active[slot] = {"violation": hit.violation.value, "key": hit.key,
                                     "label": hit.violation.label_zh,
                                     "duration_s": round(c.duration_s, 1),
                                     "severity": c.severity.value}
            else:
                self.active.pop(slot, None)
            if c.should_alert:
                events.append(self._build_event(frame, res, hit, c))

        for ev in events:
            self.stats.events += 1
            if self.dispatcher is not None:
                results = self.dispatcher.dispatch(ev)
                ev.raw_signals["alert_channels"] = results
            if self.on_event is not None:
                self.on_event(ev)

        e2e_ms = (time.time() - frame.ts) * 1000.0
        self.stats.observe(res.infer_ms, e2e_ms, len(res.persons))
        return events

    # -- 事件构造 -----------------------------------------------------------
    def _build_event(self, frame: Frame, res: PerceptionResult, hit: RawHit, c: Any) -> SafetyEvent:
        signals = dict(hit.signals)
        signals.update({
            "backend": res.backend,
            "infer_ms": res.infer_ms,
            "frame_seq": frame.seq,
            "hit_ratio": c.confidence,
        })
        # bench/score.py 用 clip_t 对齐时间轴；车端 meta 里带了就透传
        if "clip_t" in frame.meta:
            signals["clip_t"] = round(float(frame.meta["clip_t"]), 2)
        if self.context is not None and self.context.speed_kmh is not None:
            signals["speed_kmh"] = self.context.speed_kmh

        return SafetyEvent(
            violation=hit.violation,
            vehicle_id=self.vehicle_id,
            mode=DetectionMode.SERVER,
            ts=res.ts,
            severity=c.severity,
            subject=Subject(role=hit.role, seat=hit.seat, track_id=hit.track_id),
            confidence=round(float(c.confidence), 4),
            duration_s=round(float(c.duration_s), 2),
            evidence=Evidence(
                frame_b64=self._thumbnail(frame.image, hit.bbox),
                bbox=hit.bbox,
                captured_at=frame.ts,
            ),
            raw_signals=signals,
        )

    def _thumbnail(self, img: np.ndarray, bbox: list[float] | None) -> str | None:
        """事件证据缩略图。

        画上违规框再压成小 JPEG —— 后台复核时人眼要能一眼看出「系统在指什么」，
        没有可视化的证据帧，申诉流程根本走不下去。
        """
        try:
            vis = img.copy()
            h, w = vis.shape[:2]
            if bbox:
                x1, y1, x2, y2 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(bbox)]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            s = self.evidence_max_side / max(h, w)
            if s < 1.0:
                vis = cv2.resize(vis, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                return None
            return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception:  # noqa: BLE001 - 证据生成失败不能影响告警
            return None


def _ctx_from_meta(vehicle_id: str, meta: dict[str, Any]) -> VehicleContext | None:
    """车端随帧上传的 OBD/GPS 信号 → VehicleContext。"""
    keys = ("speed_kmh", "speed_limit_kmh", "gear", "engine_on", "gps")
    if not any(k in meta for k in keys):
        return None
    gps = meta.get("gps")
    return VehicleContext(
        vehicle_id=vehicle_id,
        speed_kmh=meta.get("speed_kmh"),
        speed_limit_kmh=meta.get("speed_limit_kmh"),
        gear=meta.get("gear"),
        engine_on=meta.get("engine_on"),
        gps=tuple(gps) if isinstance(gps, (list, tuple)) and len(gps) == 2 else None,
        seatbelt_switch=meta.get("seatbelt_switch") or {},
    )
