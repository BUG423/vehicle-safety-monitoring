"""多路并发调度器 —— 模式B 的核心工程问题。

一台 A100 要同时看几十路车。朴素做法「一路一线程、每帧单独前向」会把 GPU 打成
一堆小 kernel，利用率极低。这里做三件事：

1. **跨车攒批**：从所有活跃视频源各取一帧，凑够 `max_batch` 或等满 `batch_wait_ms`
   就发一次前向。批大小直接决定单卡能带多少车。
2. **有界积压 + 丢旧帧**：源侧队列满时丢最旧的帧。实时监管里处理 3 秒前的画面没有意义，
   宁可降帧率也要保时效——这是和离线视频分析最根本的区别。
3. **公平轮转**：按 round-robin 从各源取帧，避免高帧率的车饿死低帧率的车。

后处理（规则 + 确认 + 告警）放在独立线程池里，避免 CPU 端的 JPEG 编码、
安全带 ROI 采样阻塞 GPU 批处理循环。
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Config
from ..perception.base import Detector
from ..sources.base import Frame, FrameSource
from .pipeline import VehiclePipeline


@dataclass
class SchedulerStats:
    batches: int = 0
    frames: int = 0
    dropped: int = 0
    batch_size_avg: float = 0.0
    gpu_ms_avg: float = 0.0
    post_ms_avg: float = 0.0
    started_at: float = field(default_factory=time.time)

    def observe(self, n: int, gpu_ms: float) -> None:
        self.batches += 1
        self.frames += n
        a = 0.05
        self.batch_size_avg = n if self.batches == 1 else (1 - a) * self.batch_size_avg + a * n
        self.gpu_ms_avg = gpu_ms if self.batches == 1 else (1 - a) * self.gpu_ms_avg + a * gpu_ms

    @property
    def uptime_s(self) -> float:
        return time.time() - self.started_at

    @property
    def throughput_fps(self) -> float:
        return self.frames / max(self.uptime_s, 1e-6)

    def to_dict(self) -> dict[str, Any]:
        return {"batches": self.batches, "frames": self.frames, "dropped": self.dropped,
                "batch_size_avg": round(self.batch_size_avg, 2),
                "gpu_ms_per_frame_avg": round(self.gpu_ms_avg, 2),
                "post_ms_avg": round(self.post_ms_avg, 2),
                "throughput_fps": round(self.throughput_fps, 2),
                "uptime_s": round(self.uptime_s, 1)}


class InferenceScheduler:
    """把 N 路视频源喂给一个共享的 GPU 检测器。"""

    def __init__(self, detector: Detector, cfg: Config, *,
                 pipeline_factory: Callable[[str], VehiclePipeline],
                 post_workers: int = 2) -> None:
        self.detector = detector
        self.cfg = cfg
        self.pipeline_factory = pipeline_factory
        self.stats = SchedulerStats()

        self._sources: dict[str, FrameSource] = {}
        self._pipelines: dict[str, VehiclePipeline] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._rr = 0

        self._post_q: queue.Queue = queue.Queue(maxsize=256)
        self._threads = [threading.Thread(target=self._gpu_loop, name="modeb-gpu", daemon=True)]
        for i in range(post_workers):
            self._threads.append(threading.Thread(target=self._post_loop, name=f"modeb-post{i}",
                                                  daemon=True))

    # -- 生命周期 -----------------------------------------------------------
    def start(self) -> None:
        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout)
        with self._lock:
            for s in self._sources.values():
                s.close()

    # -- 源管理 -------------------------------------------------------------
    def add_source(self, source: FrameSource) -> VehiclePipeline:
        with self._lock:
            old = self._sources.get(source.vehicle_id)
            if old is not None and old is not source:
                old.close()
            self._sources[source.vehicle_id] = source
            pipe = self._pipelines.get(source.vehicle_id)
            if pipe is None:
                pipe = self.pipeline_factory(source.vehicle_id)
                self._pipelines[source.vehicle_id] = pipe
            return pipe

    def remove_source(self, vehicle_id: str) -> None:
        with self._lock:
            s = self._sources.pop(vehicle_id, None)
        if s is not None:
            s.close()

    def get_source(self, vehicle_id: str) -> FrameSource | None:
        with self._lock:
            return self._sources.get(vehicle_id)

    def pipeline(self, vehicle_id: str) -> VehiclePipeline | None:
        with self._lock:
            return self._pipelines.get(vehicle_id)

    @property
    def n_sources(self) -> int:
        with self._lock:
            return len(self._sources)

    def describe(self) -> dict[str, Any]:
        with self._lock:
            srcs = [s.describe() for s in self._sources.values()]
        return {"detector": self.detector.describe(), "sources": srcs,
                "scheduler": self.stats.to_dict()}

    # -- GPU 主循环 ---------------------------------------------------------
    def _collect_batch(self) -> list[Frame]:
        """公平轮转地从各源取帧，凑批或超时即返回。"""
        deadline = time.time() + self.cfg.perception.batch_wait_ms / 1000.0
        batch: list[Frame] = []
        seen_empty_rounds = 0
        while len(batch) < self.cfg.perception.max_batch and time.time() < deadline:
            with self._lock:
                sources = list(self._sources.values())
                dead = [s.vehicle_id for s in sources if not s.is_open]
                for vid in dead:
                    self._sources.pop(vid, None)
                sources = [s for s in sources if s.is_open]
            if not sources:
                time.sleep(0.01)
                return batch

            got = 0
            n = len(sources)
            for i in range(n):
                src = sources[(self._rr + i) % n]
                try:
                    f = src.read()
                except Exception:  # noqa: BLE001 - 单个源出错不能拖垮调度
                    src.stats["errors"] = src.stats.get("errors", 0) + 1
                    continue
                if f is not None:
                    batch.append(f)
                    got += 1
                    if len(batch) >= self.cfg.perception.max_batch:
                        break
            self._rr = (self._rr + 1) % max(n, 1)
            if got == 0:
                seen_empty_rounds += 1
                if seen_empty_rounds >= 2 and batch:
                    break
                time.sleep(0.002)
        return batch

    def _gpu_loop(self) -> None:
        while not self._stop.is_set():
            batch = self._collect_batch()
            if not batch:
                continue
            t0 = time.perf_counter()
            try:
                results = self.detector.infer_batch([f.image for f in batch],
                                                    vehicle_ids=[f.vehicle_id for f in batch])
            except Exception as exc:  # noqa: BLE001
                print(f"[scheduler] 推理失败，跳过该批 ({len(batch)} 帧): {exc}")
                continue
            gpu_ms = (time.perf_counter() - t0) * 1000.0 / len(batch)
            self.stats.observe(len(batch), gpu_ms)

            for f, r in zip(batch, results):
                try:
                    self._post_q.put_nowait((f, r))
                except queue.Full:
                    self.stats.dropped += 1   # 后处理跟不上，丢帧而不是无限积压

    def _post_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame, res = self._post_q.get(timeout=0.5)
            except queue.Empty:
                continue
            pipe = self.pipeline(frame.vehicle_id)
            if pipe is None:
                continue
            t0 = time.perf_counter()
            try:
                pipe.process(frame, res)
            except Exception as exc:  # noqa: BLE001
                print(f"[scheduler] 后处理异常 vehicle={frame.vehicle_id}: {exc}")
            dt = (time.perf_counter() - t0) * 1000.0
            a = 0.05
            self.stats.post_ms_avg = (dt if self.stats.post_ms_avg == 0
                                      else (1 - a) * self.stats.post_ms_avg + a * dt)
