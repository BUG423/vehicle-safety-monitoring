"""感知层数据结构与检测器接口。

这一层刻意只描述「画面里有什么」，不描述「这算不算违规」：
  - 检测器 Detector 输出 PerceptionResult（人、关键点、物体）
  - 规则层 engine.rules 才把它翻译成 ViolationType

这样真模型和 Mock 可以完全互换，规则层一行都不用改。
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# COCO 17 关键点顺序（torchvision Keypoint R-CNN 与绝大多数姿态模型一致）
COCO_KP = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP_INDEX = {name: i for i, name in enumerate(COCO_KP)}


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def normalized(self, w: int, h: int) -> list[float]:
        return [round(self.x1 / w, 4), round(self.y1 / h, 4),
                round(self.x2 / w, 4), round(self.y2 / h, 4)]

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clip(self, w: int, h: int) -> "BBox":
        return BBox(max(0.0, self.x1), max(0.0, self.y1), min(float(w), self.x2), min(float(h), self.y2))

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class ObjectDet:
    """通用目标检测框（手机、瓶子……）。"""

    label: str
    score: float
    box: BBox


@dataclass
class PersonObs:
    """画面中的一个人。"""

    box: BBox
    score: float
    keypoints: np.ndarray | None = None   # (17, 3) -> x, y, score
    track_id: int = -1
    seat: str = "unknown"                 # driver | front_passenger | rear_* | unknown

    def kp(self, name: str, thr: float = 0.0) -> tuple[float, float] | None:
        """取一个具名关键点，分数不够返回 None。"""
        if self.keypoints is None:
            return None
        i = KP_INDEX[name]
        x, y, s = self.keypoints[i]
        if s < thr:
            return None
        return float(x), float(y)


@dataclass
class PerceptionResult:
    """一帧的感知结果。"""

    ts: float
    width: int
    height: int
    persons: list[PersonObs] = field(default_factory=list)
    objects: list[ObjectDet] = field(default_factory=list)
    backend: str = "unknown"
    infer_ms: float = 0.0
    frame_stats: dict[str, float] = field(default_factory=dict)  # 亮度/清晰度等全图指标
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def occupancy(self) -> int:
        return len(self.persons)

    def driver(self) -> PersonObs | None:
        for p in self.persons:
            if p.seat == "driver":
                return p
        return None


class Detector(abc.ABC):
    """检测器接口 —— 真模型与 Mock 都实现它。

    约定：
      - `infer_batch` 是主入口，单帧只是 batch=1 的特例（GPU 上批处理收益明显）
      - 实现必须自己保证线程安全，或声明 `thread_safe = False` 由调度器串行化
    """

    name = "base"
    thread_safe = False

    @abc.abstractmethod
    def infer_batch(self, images: list[np.ndarray], *, vehicle_ids: list[str] | None = None) -> list[PerceptionResult]:
        ...

    def infer(self, image: np.ndarray, *, vehicle_id: str = "") -> PerceptionResult:
        return self.infer_batch([image], vehicle_ids=[vehicle_id])[0]

    def warmup(self, n: int = 2, size: tuple[int, int] = (480, 640)) -> None:
        dummy = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        for _ in range(n):
            try:
                self.infer_batch([dummy])
            except Exception:  # noqa: BLE001
                break

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "thread_safe": self.thread_safe}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 座位归属 & 简易 IoU 跟踪
# ---------------------------------------------------------------------------
def assign_seats(persons: list[PersonObs], width: int, height: int, driver_side: str = "left") -> None:
    """按画面位置把人分配到座位。

    舱内 DMS/OMS 摄像头装在后视镜或 A 柱上，画面构图是固定的，
    因此「谁是司机」用几何位置判断就够了，不需要再训一个座位分类模型。
    前排 = 画面上半部分（离镜头近、框更大），驾驶位 = 指定的那一侧。
    """
    if not persons:
        return
    for p in persons:
        front = p.box.area >= 0.06 * width * height or p.box.cy < height * 0.75
        left = p.box.cx < width * 0.5
        is_driver_side = (left if driver_side == "left" else not left)
        if front and is_driver_side:
            p.seat = "driver"
        elif front:
            p.seat = "front_passenger"
        else:
            p.seat = "rear_left" if left else "rear_right"

    # 驾驶位只能有一个：多个候选时取框最大的（离镜头最近）
    drivers = [p for p in persons if p.seat == "driver"]
    if len(drivers) > 1:
        drivers.sort(key=lambda p: p.box.area, reverse=True)
        for p in drivers[1:]:
            p.seat = "rear_left" if driver_side == "left" else "rear_right"


class IoUTracker:
    """最小可用的 IoU 跟踪器。

    舱内场景里人几乎不动、也不会互相遮挡，所以不需要 ByteTrack/DeepSORT 那种复杂度；
    跟踪的唯一目的是让 track_id 稳定，好让确认器按人累计时序。
    """

    def __init__(self, iou_thr: float = 0.3, max_age: float = 2.0) -> None:
        self.iou_thr = iou_thr
        self.max_age = max_age
        self._tracks: dict[int, tuple[BBox, float]] = {}
        self._next_id = 1

    def update(self, persons: list[PersonObs], now: float | None = None) -> None:
        now = time.time() if now is None else now
        for tid in [t for t, (_, ts) in self._tracks.items() if now - ts > self.max_age]:
            del self._tracks[tid]

        used: set[int] = set()
        for p in sorted(persons, key=lambda x: x.box.area, reverse=True):
            best_id, best_iou = -1, 0.0
            for tid, (box, _) in self._tracks.items():
                if tid in used:
                    continue
                v = p.box.iou(box)
                if v > best_iou:
                    best_id, best_iou = tid, v
            if best_iou >= self.iou_thr:
                p.track_id = best_id
            else:
                p.track_id = self._next_id
                self._next_id += 1
            used.add(p.track_id)
            self._tracks[p.track_id] = (p.box, now)
