"""真模型后端 —— Ultralytics YOLO11。

相对 Keypoint R-CNN 的取舍：

  YOLO11-pose      单阶段，一次前向出人体框 + COCO 17 关键点。
                   n/s 两档在 A100 上是 Keypoint R-CNN 的数倍吞吐，
                   模式B 要算「单卡能撑多少路车」，吞吐直接等于成本，所以它是默认选择。
  YOLO11n (detect) COCO 80 类，用来拿 `cell phone`。

诚实边界（与 torchvision 后端相同，因为受限于 COCO 类表而非模型本身）：
  - COCO **没有** 安全带、香烟类别 → 这两项无法直接检测
  - COCO 关键点无眼睑轮廓 → 疲劳判定需要外挂 MediaPipe 人脸模块（face_module.py）

权重放在 `mode-b-server/models/`，由 `tools/fetch_models.py` 下载，不入版本库。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PerceptionConfig
from .base import BBox, Detector, ObjectDet, PerceptionResult, PersonObs
from .torchvision_detector import frame_quality

_INTEREST = {"cell phone", "bottle", "cup"}
_DEFAULT_POSE = "models/yolo11n-pose.pt"
_DEFAULT_DET = "models/yolo11n.pt"


class YoloDetector(Detector):
    """YOLO11-pose（人体 + 关键点）+ YOLO11（COCO 物体）。"""

    name = "yolo"
    thread_safe = False

    def __init__(self, cfg: PerceptionConfig | None = None, *,
                 pose_weights: str | None = None, det_weights: str | None = None) -> None:
        from ultralytics import YOLO
        import torch

        self.cfg = cfg or PerceptionConfig()
        self.torch = torch
        self.device = self.cfg.device if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()

        root = Path(__file__).resolve().parents[2]
        pose_path = _resolve(pose_weights or _DEFAULT_POSE, root)
        t0 = time.time()
        self.pose = YOLO(str(pose_path))
        self.pose.to(self.device)

        self.objdet = None
        if self.cfg.enable_object_detector:
            det_path = _resolve(det_weights or _DEFAULT_DET, root)
            self.objdet = YOLO(str(det_path))
            self.objdet.to(self.device)
        self.load_s = round(time.time() - t0, 2)
        self.pose_weights = str(pose_path)
        self.det_weights = str(det_path) if self.objdet is not None else None

    def infer_batch(self, images: list[np.ndarray], *, vehicle_ids: list[str] | None = None
                    ) -> list[PerceptionResult]:
        if not images:
            return []
        now = time.time()
        imgsz = _round32(self.cfg.infer_short_side)

        t0 = time.perf_counter()
        with self._lock:
            pose_out = self.pose.predict(images, imgsz=imgsz, device=self.device,
                                         conf=self.cfg.person_score_thr, verbose=False)
            obj_out = (self.objdet.predict(images, imgsz=imgsz, device=self.device,
                                           conf=self.cfg.object_score_thr, verbose=False)
                       if self.objdet is not None else [None] * len(images))
        infer_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(images))

        results = []
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            res = PerceptionResult(ts=now, width=w, height=h, backend=self.name,
                                   infer_ms=round(infer_ms, 2))
            res.persons = self._parse_pose(pose_out[i])
            if obj_out[i] is not None:
                res.objects = self._parse_objects(obj_out[i])
            res.frame_stats = frame_quality(img)
            results.append(res)
        return results

    def _parse_pose(self, r: Any) -> list[PersonObs]:
        persons: list[PersonObs] = []
        if r.boxes is None or len(r.boxes) == 0:
            return persons
        boxes = r.boxes.xyxy.float().cpu().numpy()
        scores = r.boxes.conf.float().cpu().numpy()
        kxy = r.keypoints.xy.float().cpu().numpy() if r.keypoints is not None else None
        kconf = (r.keypoints.conf.float().cpu().numpy()
                 if (r.keypoints is not None and r.keypoints.conf is not None) else None)
        for i, (b, s) in enumerate(zip(boxes, scores)):
            kp = None
            if kxy is not None and i < len(kxy):
                kp = np.zeros((17, 3), dtype=np.float32)
                kp[:, :2] = kxy[i]
                # YOLO 输出 0~1 的关键点置信度；统一乘 10 对齐 Keypoint R-CNN 的 logit 量级，
                # 让上层 kp_thr 阈值对两个后端通用
                kp[:, 2] = (kconf[i] * 10.0) if kconf is not None else 10.0
            persons.append(PersonObs(box=BBox(*b.tolist()), score=float(s), keypoints=kp))
        return persons

    def _parse_objects(self, r: Any) -> list[ObjectDet]:
        objs: list[ObjectDet] = []
        if r.boxes is None or len(r.boxes) == 0:
            return objs
        names = r.names
        for b, s, c in zip(r.boxes.xyxy.float().cpu().numpy(),
                           r.boxes.conf.float().cpu().numpy(),
                           r.boxes.cls.int().cpu().numpy()):
            label = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
            if label not in _INTEREST:
                continue
            objs.append(ObjectDet(label=label, score=float(s), box=BBox(*b.tolist())))
        return objs

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "device": self.device,
                "pose_model": Path(self.pose_weights).name,
                "object_model": Path(self.det_weights).name if self.det_weights else None,
                "imgsz": _round32(self.cfg.infer_short_side),
                "load_s": self.load_s, "thread_safe": self.thread_safe}


def _resolve(p: str, root: Path) -> Path:
    path = Path(p)
    if path.is_absolute() and path.exists():
        return path
    for cand in (Path.cwd() / p, root / p, root / "models" / Path(p).name):
        if cand.exists():
            return cand
    # 交给 ultralytics 自己去 GitHub releases 拉
    return Path(Path(p).name)


def _round32(v: int) -> int:
    return max(160, int(round(v / 32.0)) * 32)
