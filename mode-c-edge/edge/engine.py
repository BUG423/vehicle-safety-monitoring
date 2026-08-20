"""嵌入式端主引擎：采集 → 感知 → 判定 → 时序确认 → 事件 → 双通道告警。

嵌入式约束在这里体现得最集中：

* **限帧率**：主循环按 `target_fps` 主动 sleep。不限帧的话 CPU 会跑满，
  设备温度上去后降频，反而比限帧还慢，而且车内无风扇散热。
* **内存受控**：不缓存视频，证据只留一张压到 240px 的 JPEG；
  滑窗全部是 `deque` 且按时间裁剪，长度不随运行时长增长。
* **单调时钟**：确认逻辑用「逻辑时间」而非墙钟。离线跑视频时逻辑时间 = 视频内时间，
  这样 3 秒滑窗就是视频里的 3 秒，跟跑得多快无关；实时模式下逻辑时间 = 墙钟。
* **不能被告警拖垮**：后台上报走 `BackendChannel` 的异步队列，网络卡住不阻塞主循环。
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from common import (AlertDispatcher, BackendChannel, DetectionMode, Evidence, InCabinChannel,
                    SafetyEvent, Severity, Subject, ViolationConfirmer)

from .alerting import HttpBackendSender, JsonlSink, LocalVoiceSink
from .analyzers import RuleEngine
from .obd import SimulatedObd
from .perception import Perception, PerceptionBackend


@dataclass
class RunStats:
    frames: int = 0
    events: int = 0
    perception_ms: list[float] = field(default_factory=list)
    loop_ms: list[float] = field(default_factory=list)
    local_alert_ms: list[float] = field(default_factory=list)   # 采集 → 车内播报
    wall_start: float = 0.0
    wall_end: float = 0.0
    rss_start_mb: float = 0.0
    rss_peak_mb: float = 0.0

    @staticmethod
    def _pct(xs: list[float], q: float) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        return s[min(len(s) - 1, int(q * (len(s) - 1)))]

    def summary(self) -> dict[str, Any]:
        wall = max(self.wall_end - self.wall_start, 1e-9)
        return {
            "帧数": self.frames,
            "事件数": self.events,
            "墙钟耗时_s": round(wall, 2),
            "处理吞吐_fps": round(self.frames / wall, 2),
            "单帧感知_ms_均值": round(sum(self.perception_ms) / len(self.perception_ms), 2)
            if self.perception_ms else None,
            "单帧感知_ms_p95": round(self._pct(self.perception_ms, 0.95), 2)
            if self.perception_ms else None,
            "单帧主循环_ms_均值": round(sum(self.loop_ms) / len(self.loop_ms), 2)
            if self.loop_ms else None,
            "本地告警延迟_ms_均值": round(sum(self.local_alert_ms) / len(self.local_alert_ms), 2)
            if self.local_alert_ms else None,
            "本地告警延迟_ms_最大": round(max(self.local_alert_ms), 2) if self.local_alert_ms else None,
            "常驻内存_MB_起始": round(self.rss_start_mb, 1),
            "常驻内存_MB_峰值": round(self.rss_peak_mb, 1),
        }


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        try:
            for line in open("/proc/self/status", encoding="utf-8"):
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
        except Exception:  # noqa: BLE001
            pass
    return 0.0


class EdgeEngine:
    """一台车载设备上的一个检测进程。"""

    def __init__(self, cfg, backend: PerceptionBackend, *,
                 obd: SimulatedObd | None = None,
                 events_path: Path | None = None,
                 sender: Callable[[SafetyEvent], bool] | None = None,
                 save_evidence: bool = True,
                 on_event: Callable[[SafetyEvent], None] | None = None) -> None:
        cfg.ensure_dirs()
        self.cfg = cfg
        self.backend = backend
        self.obd = obd
        self.rules = RuleEngine(cfg)
        self.confirmer = ViolationConfirmer()
        self.save_evidence = save_evidence
        self.on_event = on_event
        self.stats = RunStats()

        self.voice = LocalVoiceSink(cfg.audio_dir)
        self.sender = sender if sender is not None else HttpBackendSender(
            cfg.backend_url, cfg.backend_timeout_s, drop_b64=False)
        self.in_cabin = InCabinChannel(sink=self.voice, min_severity=Severity.WARN)
        self.backend_ch = BackendChannel(self.sender, spool_dir=cfg.spool_dir,
                                         retry_interval_s=cfg.retry_interval_s)
        self.dispatcher = AlertDispatcher([self.in_cabin, self.backend_ch])
        self.jsonl = JsonlSink(events_path or cfg.event_log)
        self.dispatch_results: list[dict[str, bool]] = []
        self._t_frame_perf = 0.0

    # ------------------------------------------------------------------
    def _evidence(self, frame: np.ndarray, obs_bbox) -> Evidence:
        h, w = frame.shape[:2]
        scale = min(1.0, self.cfg.evidence_max_side / max(h, w))
        thumb = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        ok, buf = cv2.imencode(".jpg", thumb,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.evidence_jpeg_quality])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        uri = None
        if ok and self.save_evidence:
            p = Path(self.cfg.evidence_dir) / f"ev_{int(time.time() * 1000)}.jpg"
            p.write_bytes(buf.tobytes())
            uri = str(p)
        bbox = None
        if obs_bbox is not None:
            x1, y1, x2, y2 = obs_bbox
            bbox = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
        return Evidence(frame_uri=uri, frame_b64=b64, bbox=bbox, captured_at=time.time())

    # ------------------------------------------------------------------
    def _emit(self, hit, conf, p: Perception, frame: np.ndarray, clip_t: float) -> SafetyEvent:
        cfg = self.cfg
        obs = p.seats.get(hit.seat) if hit.seat else None
        signals = dict(hit.signals)
        signals.update({
            "clip_t": round(clip_t, 3),                 # bench/score.py 用它对齐时间轴
            "backend": p.backend,
            "frame_idx": p.frame_idx,
            "perception_ms": round(p.total_latency_ms, 2),
            "detection_site": "edge_device",            # 与模式B 的 cloud 区分
        })
        ev = SafetyEvent(
            violation=hit.violation, vehicle_id=cfg.vehicle_id, mode=DetectionMode.EDGE,
            severity=conf.severity, confidence=round(conf.confidence, 3),
            duration_s=round(conf.duration_s, 2),
            subject=Subject(role=hit.role, seat=hit.seat),
            evidence=self._evidence(frame, obs.head_bbox if obs else None),
            raw_signals=signals,
        )
        # 本地告警延迟：这一帧被采集 → 车内播报指令发出。全程不经过网络。
        res = self.dispatcher.dispatch(ev)
        self.dispatch_results.append(res)
        # 采集这一帧 → 车内播报指令实际发出（含 TTS/提示音合成），全程不经过网络
        self.stats.local_alert_ms.append((time.perf_counter() - self._t_frame_perf) * 1000)
        self.jsonl.write(ev)
        self.stats.events += 1
        if self.on_event:
            self.on_event(ev)
        if not res.get("in_cabin", False):
            print(f"  [警告] 车内播报失败：{ev.violation.value}")
        return ev

    # ------------------------------------------------------------------
    def run(self, source, *, max_frames: int | None = None, realtime: bool = False,
            verbose: bool = True,
            frame_hook: Callable[[float], None] | None = None) -> RunStats:
        """`frame_hook(clip_t)` 每帧回调一次，用于演示「跑到第 N 秒时拔网线」。"""
        cfg = self.cfg
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"打不开视频源: {source}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.target_fps
        stride = max(1, int(round(src_fps / cfg.target_fps)))   # 抽帧到目标帧率
        t_wall0 = time.time()
        self.stats.wall_start = t_wall0
        self.stats.rss_start_mb = self.stats.rss_peak_mb = _rss_mb()

        raw_idx = 0
        idx = 0
        period = 1.0 / cfg.target_fps
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if raw_idx % stride:
                    raw_idx += 1
                    continue
                clip_t = raw_idx / src_fps
                raw_idx += 1
                loop_t0 = time.perf_counter()
                self._t_frame_perf = loop_t0
                if frame_hook is not None:
                    frame_hook(clip_t)

                if frame.shape[1] != cfg.frame_width or frame.shape[0] != cfg.frame_height:
                    frame = cv2.resize(frame, (cfg.frame_width, cfg.frame_height))

                # 逻辑时钟：离线用视频内时间，实时用墙钟
                now = time.time() if realtime else t_wall0 + clip_t

                p = self.backend.process(frame, idx, ts=now)
                self.stats.perception_ms.append(p.total_latency_ms)

                ctx = self.obd.read(clip_t, now) if self.obd else None
                for hit in self.rules.evaluate(p, ctx):
                    c = self.confirmer.update(hit.violation, hit.hit,
                                              confidence=hit.confidence, key=hit.key, now=now)
                    if c.should_alert:
                        ev = self._emit(hit, c, p, frame, clip_t)
                        if verbose:
                            print(f"[t={clip_t:6.2f}s] 事件 {ev.violation.value:<26s} "
                                  f"{ev.severity.value:<8s} conf={ev.confidence:.2f} "
                                  f"dur={ev.duration_s:.1f}s")

                self.stats.frames = idx + 1
                self.stats.loop_ms.append((time.perf_counter() - loop_t0) * 1000)
                if idx % 25 == 0:
                    self.stats.rss_peak_mb = max(self.stats.rss_peak_mb, _rss_mb())
                idx += 1
                if max_frames and idx >= max_frames:
                    break
                if realtime:
                    slack = period - (time.perf_counter() - loop_t0)
                    if slack > 0:
                        time.sleep(slack)     # 主动让出 CPU，控温控功耗
        finally:
            cap.release()
            self.stats.wall_end = time.time()
            self.stats.rss_peak_mb = max(self.stats.rss_peak_mb, _rss_mb())
        return self.stats

    # ------------------------------------------------------------------
    def flush(self, timeout_s: float = 6.0) -> int:
        """等待后台队列与落盘补传清空，返回仍未送达的条数。"""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.backend_ch.pending == 0:
                return 0
            time.sleep(0.2)
        return self.backend_ch.pending

    def close(self) -> None:
        self.dispatcher.close()
        self.jsonl.close()
        self.backend.close()
