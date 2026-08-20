"""可插拔感知后端。

  onnx_dms   真模型：YuNet 人脸检测 + insightface 106 点关键点 + NanoDet 目标检测
  rule_cv    经典 CV 规则：标定座位 ROI + 亮度/霍夫线，无学习成分，用于超低端硬件与链路评测
  mock       脚本化信号发生器：不读画面，用于在没有任何输入时验证确认层与告警链路
"""
from __future__ import annotations

from .base import DRIVER, FRONT_PASSENGER, ObjectObs, Perception, PerceptionBackend, SeatObs

__all__ = ["DRIVER", "FRONT_PASSENGER", "ObjectObs", "Perception", "PerceptionBackend",
           "SeatObs", "build_backend"]


def build_backend(kind: str, cfg, **kw) -> PerceptionBackend:
    """按名字构造后端；真模型缺失时**明确报错**，不静默降级成 Mock。"""
    kind = kind.lower()
    if kind == "onnx":
        from .onnx_dms import OnnxDmsBackend
        return OnnxDmsBackend(cfg, **kw)
    if kind == "rule":
        from .rule_cv import RuleCvBackend
        return RuleCvBackend(cfg, **kw)
    if kind == "mock":
        from .mock import MockBackend
        return MockBackend(cfg, **kw)
    raise ValueError(f"未知感知后端: {kind}（可选 onnx / rule / mock）")
