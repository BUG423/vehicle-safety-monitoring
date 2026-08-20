"""感知层契约 —— 让感知后端可插拔。

模式C 的感知后端必须在**几 TOPS 算力、1~2 个 CPU 大核**的预算内跑完，
所以这里定义的输出刻意做得很「薄」：只给下游需要的标量与框，
不传整帧、不传特征图，避免在内存受限的设备上产生额外拷贝。

各后端只负责「看见什么」，不负责「算不算违规」——
违规判定在 `edge/analyzers.py`，时序确认在 `common/schema/confirmation.py`。
这样换后端（换模型、换芯片）不影响告警行为，也让三条路线可以逐条 diff 对比。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

DRIVER = "driver"
FRONT_PASSENGER = "front_passenger"


@dataclass
class SeatObs:
    """一个座位上的一次观测。

    `eye_open` / `belt_score` 等量纲由各后端自己定义，**不要求跨后端可比**；
    下游一律用「相对个体基线」的方式使用它们（见 `analyzers.AdaptiveBaseline`），
    这样换模型时不需要重新标定所有阈值。
    """

    seat: str
    present: bool = False
    score: float = 0.0                      # 座位有人的置信度
    head_bbox: tuple[int, int, int, int] | None = None   # 像素坐标 (x1,y1,x2,y2)
    eye_open: float | None = None           # 眼睛开合度，越大越睁开
    mouth_open: float | None = None         # 嘴部张开度，用于哈欠
    yaw_deg: float | None = None            # 头部姿态：左右转头
    pitch_deg: float | None = None          # 抬头/低头
    roll_deg: float | None = None
    belt_score: float | None = None         # 视觉安全带证据，越大越像系着
    phone_score: float | None = None        # 手机/屏幕光证据
    smoke_score: float | None = None        # 抽烟证据（当前无真实模型，见 README）


@dataclass
class ObjectObs:
    """通用目标检测框（COCO 类别）。"""

    label: str
    score: float
    bbox: tuple[int, int, int, int]


@dataclass
class Perception:
    """一帧的感知结果。"""

    ts: float                                # 采集时刻（logical clock）
    frame_idx: int
    frame_shape: tuple[int, int]             # (h, w)
    backend: str
    seats: dict[str, SeatObs] = field(default_factory=dict)
    objects: list[ObjectObs] = field(default_factory=list)
    sharpness: float = 0.0                   # 拉普拉斯方差 → 镜头遮挡/失焦判定
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return sum(self.latency_ms.values())


class PerceptionBackend:
    """感知后端基类。"""

    name = "base"

    def process(self, frame, frame_idx: int, ts: float | None = None) -> Perception:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """自述能力边界 —— 会被写进 README / 运行日志，避免把没跑通的当成跑通了。"""
        return {"name": self.name, "real_model": False, "capabilities": {}}

    def close(self) -> None:
        pass


def now_ts(ts: float | None) -> float:
    return time.time() if ts is None else ts
