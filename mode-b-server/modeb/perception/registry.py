"""检测器工厂 —— 真模型与降级实现的唯一切换点。

`backend="auto"` 的语义：**按 yolo → torchvision → mock 的顺序试，失败就降级并留痕**。
降级必须留痕（`fallback_log()`），否则「到底是真跑通了还是悄悄降级了」在演示时说不清楚，
而这正是本项目明确要求不能含糊的地方。
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from ..config import PerceptionConfig
from .base import Detector
from .mock_detector import CartoonCockpitDetector, ScriptedMockDetector

_FALLBACK_LOG: list[dict[str, str]] = []
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def build_detector(cfg: PerceptionConfig | None = None) -> Detector:
    cfg = cfg or PerceptionConfig()
    backend = (cfg.backend or "auto").lower()

    if backend in ("mock", "scripted"):
        return ScriptedMockDetector()
    if backend in ("cartoon", "synthetic"):
        return CartoonCockpitDetector()

    order = {"auto": ["yolo", "torchvision"], "yolo": ["yolo"],
             "torchvision": ["torchvision"], "real": ["yolo", "torchvision"]}.get(backend)
    if order is None:
        raise ValueError(f"未知的感知后端: {cfg.backend}（可选: {list(available_backends())}）")

    for name in order:
        try:
            det = _build_real(name, cfg)
            det.warmup(2, (cfg.infer_short_side, int(cfg.infer_short_side * 4 / 3)))
            return det
        except Exception as exc:  # noqa: BLE001
            _FALLBACK_LOG.append({"backend": name, "reason": f"{type(exc).__name__}: {exc}",
                                  "trace": traceback.format_exc(limit=3)})
            print(f"[registry] 后端 {name} 不可用: {type(exc).__name__}: {exc}")

    if backend == "auto":
        print("[registry] 所有真模型后端均不可用，降级到脚本 Mock。原因见 fallback_log()")
        return ScriptedMockDetector()
    raise RuntimeError(f"后端 {backend} 构建失败，详见 fallback_log(): {_FALLBACK_LOG[-1]['reason']}")


def _build_real(name: str, cfg: PerceptionConfig) -> Detector:
    if name == "yolo":
        from .yolo_detector import YoloDetector
        return YoloDetector(cfg)
    if name == "torchvision":
        from .torchvision_detector import TorchvisionDetector
        return TorchvisionDetector(cfg)
    raise ValueError(name)


def build_face_module(model_path: str | None = None, num_faces: int = 2):
    """构建 MediaPipe 人脸模块。返回 (module | None, 失败原因 | None)。

    它是**可选增强**：有它疲劳判定用标准 EAR/PERCLOS，没它回落到眼部 ROI 代理。
    两种情况都能跑，但事件的 `raw_signals.source` 会如实标明用的是哪一种。
    """
    from .face_module import try_build
    path = Path(model_path) if model_path else (_MODELS_DIR / "face_landmarker.task")
    if not path.exists():
        return None, f"未找到人脸模型 {path}，请先运行 tools/fetch_models.py"
    return try_build(str(path), num_faces)


def fallback_log() -> list[dict[str, str]]:
    return list(_FALLBACK_LOG)


def available_backends() -> dict[str, str]:
    return {
        "yolo": "真模型：YOLO11-pose（人体+17关键点）+ YOLO11（COCO 手机），吞吐最高",
        "torchvision": "真模型：Keypoint R-CNN + Faster R-CNN MobileNetV3，精度参考基线",
        "cartoon": "经典 CV：解析合成卡通驾驶舱，仅对 bench/ 合成素材有效",
        "mock": "脚本化：不读画面，仅用于冒烟测试与车队压测",
        "auto": "yolo → torchvision → mock，逐级降级并记录原因",
    }
