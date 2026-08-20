"""经典 CV 规则感知后端 —— 无学习成分，为最低算力档位与链路评测准备。

它的定位（务必与真模型后端区分开，不要混着报数字）：

1. **超低端硬件的可行方案**。ESP32-S3 / MCU 这一档跑不动 CNN，
   但车内 DMS 相机是**固定安装**的，座位位置固定，可以按车型标定 ROI，
   再在 ROI 内用亮度、边缘、霍夫线这类几十 KB 内存就能跑的算子出信号。
   这是本条路线「往下探到多低成本」的下界。
2. **告警链路的评测夹具**。`bench/` 的合成卡通素材上，真实人脸/关键点模型
   检不到有效目标（实测：关键点输出恒定，无判别力），而本后端读的是真实像素、
   不读 ground truth，因此能在这段素材上真正跑通「感知 → 确认 → 事件 → 告警」。

**它不是推荐的量产感知方案。** 对真实座舱画面（光照变化、乘员体型差异、
深色衣物上的深色安全带）这些阈值会失效，量产必须用 `onnx_dms` 那一档。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .base import DRIVER, FRONT_PASSENGER, Perception, PerceptionBackend, SeatObs, now_ts


@dataclass
class SeatCalibration:
    """一个座位的标定框，全部为画面归一化坐标 (x1, y1, x2, y2)。

    实车安装时用一次性标定工具生成：让乘员正常就座，框出头部、躯干、手部活动区。
    """

    head: tuple[float, float, float, float]
    torso: tuple[float, float, float, float]
    hand: tuple[float, float, float, float]


# 默认标定档：对应 bench/make_clip.py 生成的合成座舱（司机在左、乘客在右）。
# 换车型必须重新标定，这正是本后端的主要局限。
BENCH_CALIBRATION: dict[str, SeatCalibration] = {
    DRIVER: SeatCalibration(head=(0.216, 0.392, 0.378, 0.650),
                            torso=(0.150, 0.620, 0.450, 0.930),
                            hand=(0.385, 0.470, 0.510, 0.700)),
    FRONT_PASSENGER: SeatCalibration(head=(0.622, 0.392, 0.784, 0.650),
                                     torso=(0.556, 0.620, 0.856, 0.930),
                                     hand=(0.791, 0.470, 0.916, 0.700)),
}


def _crop(img: np.ndarray, b) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = b
    return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def _abs_box(shape, b) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = b
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


class RuleCvBackend(PerceptionBackend):
    """标定 ROI + 亮度/霍夫线的规则后端。"""

    name = "rule_cv"

    def __init__(self, cfg, *, calibration: dict[str, SeatCalibration] | None = None) -> None:
        self.cfg = cfg
        self.cal = calibration or BENCH_CALIBRATION
        cv2.setNumThreads(cfg.num_threads)

    # ---- 眼睛：巩膜（眼白）像素占比 ----
    @staticmethod
    def _eye_open(img: np.ndarray, head) -> float:
        r = _crop(img, head)
        if r.size == 0:
            return 0.0
        h = r.shape[0]
        band = r[int(0.30 * h):int(0.52 * h)]
        if band.size == 0:
            return 0.0
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        return float((g > 190).mean())

    # ---- 嘴：下半张脸的暗区高度占比（张嘴时变大）----
    @staticmethod
    def _mouth_open(img: np.ndarray, head) -> float:
        r = _crop(img, head)
        if r.size == 0:
            return 0.0
        h = r.shape[0]
        band = r[int(0.62 * h):int(0.88 * h)]
        if band.size == 0:
            return 0.0
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        return float((g < max(60, int(np.percentile(g, 12)))).mean())

    # ---- 安全带：躯干 ROI 内的斜跨亮带 ----
    def _belt(self, img: np.ndarray, torso) -> float:
        r = _crop(img, torso)
        if r.size == 0 or min(r.shape[:2]) < 12:
            return 0.0
        g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        thr = max(120, int(np.percentile(g, 90)))
        mask = ((g > thr).astype(np.uint8)) * 255
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=22,
                                minLineLength=int(self.cfg.belt_line_min_len * max(r.shape)),
                                maxLineGap=12)
        lo, hi = self.cfg.belt_angle_range
        best = 0.0
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                dx, dy = float(x2 - x1), float(y2 - y1)
                a = abs(np.degrees(np.arctan2(dy, dx)))
                a = min(a, 180.0 - a)
                if lo <= a <= hi:
                    best = max(best, float(np.hypot(dx, dy)) / max(r.shape))
        return min(best, 1.0)

    # ---- 手机：手部 ROI 内的高亮小块（屏幕光）----
    @staticmethod
    def _screen_glow(img: np.ndarray, hand) -> float:
        """归一化到 0~1 的置信度：手部 ROI 里高亮像素占比达到 4% 即记满分。

        归一化这一步不能省 —— 各后端的原始量纲完全不同（本后端是像素占比，
        onnx 后端是检测框置信度），不归一化就无法共用同一个阈值。
        """
        r = _crop(img, hand)
        if r.size == 0:
            return 0.0
        v = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)[:, :, 2]
        return float(min(1.0, (v > 200).mean() / 0.04))

    def process(self, frame: np.ndarray, frame_idx: int, ts: float | None = None) -> Perception:
        ts = now_ts(ts)
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        seats: dict[str, SeatObs] = {}
        for seat, cal in self.cal.items():
            head = _crop(frame, cal.head)
            # 座位有人：头部 ROI 的像素分布明显偏离「空座」的均一背景
            present = bool(head.size and float(cv2.cvtColor(head, cv2.COLOR_BGR2GRAY).std()) > 8.0)
            seats[seat] = SeatObs(
                seat=seat, present=present, score=1.0 if present else 0.0,
                head_bbox=_abs_box(frame.shape, cal.head),
                eye_open=self._eye_open(frame, cal.head) if present else None,
                mouth_open=self._mouth_open(frame, cal.head) if present else None,
                belt_score=self._belt(frame, cal.torso) if present else None,
                phone_score=self._screen_glow(frame, cal.hand) if present else None,
            )
        lat = {"rule_cv": (time.perf_counter() - t0) * 1000}
        sharp = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        return Perception(ts=ts, frame_idx=frame_idx, frame_shape=(h, w), backend=self.name,
                          seats=seats, objects=[], sharpness=sharp, latency_ms=lat)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "real_model": False,
            "reads_pixels": True,
            "capabilities": {
                "眼睛开合": "真实读像素（眼区巩膜占比），仅在标定 ROI 下成立",
                "安全带": "真实读像素（躯干 ROI 霍夫斜带线）",
                "手机": "真实读像素（手部 ROI 屏幕高亮块）",
                "头部姿态": "未实现 —— 本后端不产出 yaw/pitch，因此分心不做判定",
                "抽烟": "未实现",
            },
            "limitation": "依赖按车型标定的固定 ROI，对真实座舱的光照/体型变化不鲁棒，非量产方案",
            "seats": list(self.cal),
        }
