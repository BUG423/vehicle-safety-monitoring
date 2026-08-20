"""人脸精细模块 —— MediaPipe Face Landmarker（478 点 + blendshapes）。

它解决的是 COCO 关键点解决不了的问题：**眼睑轮廓**。
有了眼睑轮廓才谈得上标准 EAR（Eye Aspect Ratio）与 PERCLOS，
疲劳判定才从「代理指标」升级成「行业通用指标」。

    EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2·‖p1-p4‖)
    Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks" (2016)

同时 Face Landmarker 直接输出：
  - `eyeBlinkLeft/Right` blendshape：模型自己给出的闭眼概率，与 EAR 互为交叉验证
  - 4x4 facial transformation matrix：真实的头部旋转矩阵，比 5 点 solvePnP 稳得多

本模块是**可选增强**：拿不到（没装 mediapipe / 缺图形库 / 头太小）时返回 None，
上层自动回落到 analyzers 里的代理实现，不会崩。
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..compat import ensure_mediapipe_runtime

# MediaPipe Face Mesh 的眼睑 6 点（对应 EAR 论文里的 p1..p6）
_LEFT_EYE = [33, 160, 158, 133, 153, 144]     # 画面左侧的眼（受试者右眼）
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]


@dataclass
class FaceMetrics:
    """一张脸的细粒度指标。"""

    ear_left: float
    ear_right: float
    ear: float                  # 双眼均值
    blink_left: float           # blendshape 闭眼概率 0~1
    blink_right: float
    yaw: float                  # 度，来自 4x4 变换矩阵
    pitch: float
    roll: float
    mouth_open: float           # jawOpen blendshape，用于打哈欠
    source: str = "mediapipe_face_landmarker"

    @property
    def eyes_closed(self) -> bool:
        """EAR 与 blendshape 双证据 —— 任一强信号成立即判闭眼。"""
        return self.ear < 0.19 or max(self.blink_left, self.blink_right) > 0.55

    def to_dict(self) -> dict[str, float | str]:
        return {"ear": round(self.ear, 4), "ear_l": round(self.ear_left, 4),
                "ear_r": round(self.ear_right, 4),
                "blink_l": round(self.blink_left, 3), "blink_r": round(self.blink_right, 3),
                "yaw": round(self.yaw, 1), "pitch": round(self.pitch, 1),
                "roll": round(self.roll, 1), "mouth_open": round(self.mouth_open, 3),
                "source": self.source}


class MediaPipeFaceModule:
    """封装 FaceLandmarker，线程安全（内部串行化）。"""

    available_reason: str | None = None

    def __init__(self, model_path: str, *, num_faces: int = 2) -> None:
        if not ensure_mediapipe_runtime():
            raise RuntimeError("MediaPipe 运行时不可用（缺 libGLESv2）")
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._lock = threading.Lock()
        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=model_path),
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=num_faces,
                running_mode=vision.RunningMode.IMAGE,
            )
        )
        self.model_path = model_path

    def analyze(self, bgr_roi: np.ndarray) -> FaceMetrics | None:
        """输入一块**头部 ROI**（BGR），返回其中最大那张脸的指标。"""
        if bgr_roi is None or bgr_roi.size == 0 or min(bgr_roi.shape[:2]) < 32:
            return None
        import cv2
        rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        with self._lock:
            try:
                res = self._landmarker.detect(image)
            except Exception:  # noqa: BLE001
                return None
        if not res.face_landmarks:
            return None

        h, w = bgr_roi.shape[:2]
        lms = res.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float64)
        ear_l = _ear(pts, _LEFT_EYE)
        ear_r = _ear(pts, _RIGHT_EYE)

        blink_l = blink_r = mouth = 0.0
        if res.face_blendshapes:
            for c in res.face_blendshapes[0]:
                if c.category_name == "eyeBlinkLeft":
                    blink_l = float(c.score)
                elif c.category_name == "eyeBlinkRight":
                    blink_r = float(c.score)
                elif c.category_name == "jawOpen":
                    mouth = float(c.score)

        yaw = pitch = roll = 0.0
        if res.facial_transformation_matrixes:
            yaw, pitch, roll = _euler_from_matrix(np.array(res.facial_transformation_matrixes[0]))

        return FaceMetrics(ear_left=ear_l, ear_right=ear_r, ear=(ear_l + ear_r) / 2.0,
                           blink_left=blink_l, blink_right=blink_r,
                           yaw=yaw, pitch=pitch, roll=roll, mouth_open=mouth)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # noqa: BLE001
            pass

    def describe(self) -> dict[str, Any]:
        return {"name": "mediapipe_face_landmarker", "model": self.model_path,
                "landmarks": 478, "outputs": ["EAR", "blendshapes", "head_pose_matrix"]}


def _ear(pts: np.ndarray, idx: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horiz = np.linalg.norm(p1 - p4)
    if horiz < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horiz))


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    R = m[:3, :3]
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
        roll = 0.0
    return round(yaw, 2), round(-pitch, 2), round(roll, 2)


def try_build(model_path: str, num_faces: int = 2) -> tuple["MediaPipeFaceModule | None", str | None]:
    """构建失败时返回 (None, 原因) —— 原因要留痕，演示时才说得清是真跑通还是降级了。"""
    try:
        return MediaPipeFaceModule(model_path, num_faces=num_faces), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
