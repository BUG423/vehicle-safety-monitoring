"""检测器工厂 —— 真模型与降级实现的唯一切换点。

`backend="auto"` 的语义：**先试真模型，失败就降级并把原因记下来**。
降级必须留痕（`last_fallback_reason`），否则「跑通了还是降级了」在演示时说不清楚，
这正是本项目要求不能含糊的地方。
"""
from __future__ import annotations

import traceback
from typing import Any

from ..config import PerceptionConfig
from .base import Detector
from .mock_detector import CartoonCockpitDetector, ScriptedMockDetector

_LAST_FALLBACK: dict[str, str] = {}


def build_detector(cfg: PerceptionConfig | None = None) -> Detector:
    cfg = cfg or PerceptionConfig()
    backend = (cfg.backend or "auto").lower()

    if backend in ("mock", "scripted"):
        return ScriptedMockDetector()
    if backend in ("cartoon", "synthetic"):
        return CartoonCockpitDetector()

    if backend in ("auto", "torchvision", "real"):
        try:
            from .torchvision_detector import TorchvisionDetector
            det = TorchvisionDetector(cfg)
            det.warmup(2, (cfg.infer_short_side, int(cfg.infer_short_side * 4 / 3)))
            _LAST_FALLBACK.pop("reason", None)
            return det
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            _LAST_FALLBACK["reason"] = reason
            _LAST_FALLBACK["trace"] = traceback.format_exc(limit=3)
            if backend != "auto":
                raise
            print(f"[registry] 真模型后端不可用，降级到脚本 Mock；原因: {reason}")
            return ScriptedMockDetector()

    raise ValueError(f"未知的感知后端: {cfg.backend}")


def last_fallback_reason() -> str | None:
    return _LAST_FALLBACK.get("reason")


def available_backends() -> dict[str, str]:
    return {
        "torchvision": "真模型：Keypoint R-CNN + Faster R-CNN(MobileNetV3)，需要 GPU 与预训练权重",
        "cartoon": "经典 CV：解析合成卡通驾驶舱，仅对合成素材有效",
        "mock": "脚本化：不读画面，用于冒烟测试与车队压测",
        "auto": "优先真模型，失败自动降级到 mock 并记录原因",
    }
