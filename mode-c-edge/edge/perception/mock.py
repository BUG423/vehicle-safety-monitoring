"""脚本化信号发生器 —— 不读画面，只用来验证「确认层 + 告警链路」。

用途：在没有摄像头、没有视频、没有模型的机器上（比如刚刷完机的设备做自检）
也能把 `确认 → 事件 → 车内播报 → 后台上报 → 断网补传` 整条链路跑一遍。

它**故意注入噪声**：正常状态下也有 8% 的随机误命中（约等于眨眼、抬手的频率），
用来证明 `ViolationConfirmer` 真的把逐帧噪声挡住了，而不是靠输入本来就干净。
"""
from __future__ import annotations

import random
from typing import Any

from .base import DRIVER, FRONT_PASSENGER, Perception, PerceptionBackend, SeatObs, now_ts

# (违规信号名, 起始秒, 结束秒)
DEFAULT_SCRIPT = [
    ("driver_belt_off", 1.0, 12.0),
    ("passenger_belt_off", 3.0, 25.0),
    ("driver_eyes_closed", 15.0, 28.0),
    ("driver_phone", 32.0, 40.0),
    ("driver_yaw", 45.0, 52.0),
]
NOISE_P = 0.08          # 逐帧误命中率，模拟眨眼/抬手
DURATION_S = 60.0


class MockBackend(PerceptionBackend):
    name = "mock"

    def __init__(self, cfg, *, seed: int = 20260820, script=None, noise_p: float = NOISE_P) -> None:
        self.cfg = cfg
        self.script = script or DEFAULT_SCRIPT
        self.noise_p = noise_p
        self._rng = random.Random(seed)
        self._t0: float | None = None

    def _on(self, name: str, t: float) -> bool:
        return any(n == name and a <= t < b for n, a, b in self.script)

    def process(self, frame, frame_idx: int, ts: float | None = None) -> Perception:
        ts = now_ts(ts)
        if self._t0 is None:
            self._t0 = ts
        t = ts - self._t0
        r = self._rng.random

        def noisy(on: bool) -> bool:
            return (not on) if r() < self.noise_p else on

        d = SeatObs(seat=DRIVER, present=True, score=0.95, head_bbox=(60, 40, 150, 150))
        d.eye_open = 0.10 if noisy(self._on("driver_eyes_closed", t)) else 0.38
        d.mouth_open = 0.20
        d.yaw_deg = 45.0 if noisy(self._on("driver_yaw", t)) else 3.0
        d.pitch_deg = 2.0
        d.belt_score = 0.05 if noisy(self._on("driver_belt_off", t)) else 0.75
        d.phone_score = 0.80 if noisy(self._on("driver_phone", t)) else 0.02

        p = SeatObs(seat=FRONT_PASSENGER, present=True, score=0.9, head_bbox=(200, 40, 285, 150))
        p.eye_open, p.mouth_open, p.yaw_deg, p.pitch_deg = 0.38, 0.2, 0.0, 0.0
        p.belt_score = 0.05 if noisy(self._on("passenger_belt_off", t)) else 0.75
        p.phone_score = 0.0

        return Perception(ts=ts, frame_idx=frame_idx, frame_shape=(240, 320), backend=self.name,
                          seats={DRIVER: d, FRONT_PASSENGER: p}, objects=[],
                          sharpness=100.0, latency_ms={"mock": 0.01})

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "real_model": False, "reads_pixels": False,
                "noise_p": self.noise_p,
                "purpose": "不读画面的脚本化信号，仅验证确认层与告警链路",
                "script": self.script}
