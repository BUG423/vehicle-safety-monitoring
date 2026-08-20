"""真模型后端 —— torchvision 预训练检测器。

选型理由（详见 DESIGN.md 第 3 节）：

  人体 + 关键点  Keypoint R-CNN (ResNet50-FPN, COCO)
      一次前向同时给出人体框和 17 个关键点，省掉「检测 → 裁剪 → 姿态」两段式的
      调度复杂度；关键点是后续所有判定（安全带 ROI、眼部 ROI、手部位置、头姿）的
      共同输入。torchvision 自带权重，不需要额外下载渠道。

  通用物体      Faster R-CNN (MobileNetV3-Large-FPN, COCO)
      只为了拿 COCO 的 `cell phone` 类。选 MobileNet 版是因为它比 ResNet50 版快
      3~4 倍，而「手机」这种中等尺寸物体在舱内特写里并不难检。

诚实声明：
  - COCO **没有香烟类别**，也没有安全带类别。这两项无法由本后端直接检测，
    见 analyzers.py 里的启发式实现与其局限说明。
  - COCO 关键点只有 5 个面部点（鼻、双眼中心、双耳），**没有眼睑轮廓**，
    因此无法计算标准 EAR（Eye Aspect Ratio）。本后端用眼部 ROI 的形态学代理，
    见 analyzers.FatigueAnalyzer。生产环境应换成 68/478 点人脸对齐模型。
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from ..config import PerceptionConfig
from .base import BBox, Detector, ObjectDet, PerceptionResult, PersonObs

# COCO 91 类中我们真正关心的几类（索引来自 torchvision COCO_INSTANCE_CATEGORY_NAMES）
_COCO_NAMES: dict[int, str] = {
    1: "person", 44: "bottle", 47: "cup", 77: "cell phone", 84: "book",
}
_INTEREST = {"cell phone", "bottle", "cup"}


class TorchvisionDetector(Detector):
    """Keypoint R-CNN + （可选）Faster R-CNN 的组合后端。"""

    name = "torchvision"
    thread_safe = False   # 单个 CUDA 模型实例串行使用，并发由调度器的批处理承担

    def __init__(self, cfg: PerceptionConfig | None = None) -> None:
        import torch  # 延迟导入：mock 后端不需要 torch
        from torchvision.models.detection import (
            KeypointRCNN_ResNet50_FPN_Weights,
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            keypointrcnn_resnet50_fpn,
            fasterrcnn_mobilenet_v3_large_fpn,
        )

        self.cfg = cfg or PerceptionConfig()
        self.torch = torch
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()

        t0 = time.time()
        self.pose = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
            box_score_thresh=self.cfg.person_score_thr,
        ).eval().to(self.device)

        self.objdet = None
        if self.cfg.enable_object_detector:
            self.objdet = fasterrcnn_mobilenet_v3_large_fpn(
                weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                box_score_thresh=self.cfg.object_score_thr,
            ).eval().to(self.device)
        self.load_s = round(time.time() - t0, 2)

        self.use_half = self.device.type == "cuda"
        if self.use_half:
            # 检测头对 fp16 敏感，这里只用 autocast 而不整体 .half()
            self._autocast = lambda: torch.autocast("cuda", dtype=torch.float16)
        else:
            self._autocast = _nullcontext

    # -- 主入口 -------------------------------------------------------------
    def infer_batch(self, images: list[np.ndarray], *, vehicle_ids: list[str] | None = None) -> list[PerceptionResult]:
        if not images:
            return []
        torch = self.torch
        now = time.time()

        scaled, scales, sizes = [], [], []
        for img in images:
            h, w = img.shape[:2]
            sizes.append((w, h))
            small, s = _scale_to_short_side(img, self.cfg.infer_short_side)
            scales.append(s)
            # BGR -> RGB -> CHW float[0,1]
            t = torch.from_numpy(np.ascontiguousarray(small[:, :, ::-1])).permute(2, 0, 1)
            scaled.append(t.to(self.device, non_blocking=True).float().div_(255.0))

        t0 = time.perf_counter()
        with self._lock, torch.inference_mode():
            with self._autocast():
                pose_out = self.pose(scaled)
                obj_out = self.objdet(scaled) if self.objdet is not None else [None] * len(scaled)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        infer_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(images))

        results = []
        for i, img in enumerate(images):
            w, h = sizes[i]
            res = PerceptionResult(ts=now, width=w, height=h, backend=self.name,
                                   infer_ms=round(infer_ms, 2))
            res.persons = self._parse_pose(pose_out[i], 1.0 / scales[i])
            if obj_out[i] is not None:
                res.objects = self._parse_objects(obj_out[i], 1.0 / scales[i])
            res.frame_stats = frame_quality(img)
            results.append(res)
        return results

    # -- 输出解析 -----------------------------------------------------------
    def _parse_pose(self, out: dict[str, Any], inv_scale: float) -> list[PersonObs]:
        boxes = out["boxes"].float().cpu().numpy() * inv_scale
        scores = out["scores"].float().cpu().numpy()
        kps = out["keypoints"].float().cpu().numpy()
        kp_scores = out["keypoints_scores"].float().cpu().numpy()

        persons: list[PersonObs] = []
        for b, s, k, ks in zip(boxes, scores, kps, kp_scores):
            if s < self.cfg.person_score_thr:
                continue
            kp = np.zeros((17, 3), dtype=np.float32)
            kp[:, 0] = k[:, 0] * inv_scale
            kp[:, 1] = k[:, 1] * inv_scale
            kp[:, 2] = ks
            persons.append(PersonObs(box=BBox(*b.tolist()), score=float(s), keypoints=kp))
        return persons

    def _parse_objects(self, out: dict[str, Any], inv_scale: float) -> list[ObjectDet]:
        boxes = out["boxes"].float().cpu().numpy() * inv_scale
        scores = out["scores"].float().cpu().numpy()
        labels = out["labels"].cpu().numpy()
        objs = []
        for b, s, lb in zip(boxes, scores, labels):
            name = _COCO_NAMES.get(int(lb))
            if name is None or name not in _INTEREST or s < self.cfg.object_score_thr:
                continue
            objs.append(ObjectDet(label=name, score=float(s), box=BBox(*b.tolist())))
        return objs

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": str(self.device),
            "pose_model": "keypointrcnn_resnet50_fpn (COCO)",
            "object_model": "fasterrcnn_mobilenet_v3_large_fpn (COCO)" if self.objdet else None,
            "infer_short_side": self.cfg.infer_short_side,
            "load_s": self.load_s,
            "thread_safe": self.thread_safe,
        }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _scale_to_short_side(img: np.ndarray, short_side: int) -> tuple[np.ndarray, float]:
    import cv2
    h, w = img.shape[:2]
    s = short_side / min(h, w)
    if abs(s - 1.0) < 0.02:
        return img, 1.0
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=interp), s


def frame_quality(img: np.ndarray) -> dict[str, float]:
    """全图质量指标 —— 用来判断摄像头是否被遮挡/脏污/夜间失效。

    这是模式B 少数几个「不依赖深度模型」的真实检测项，也是实际部署中最常触发的一项：
    司机为了躲监控用胶带贴摄像头，比疲劳驾驶常见得多。
    """
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
    return {
        "mean": float(small.mean()),
        "std": float(small.std()),
        "lap_var": float(cv2.Laplacian(small, cv2.CV_64F).var()),
    }
