"""感知层：检测器接口、真模型后端、降级实现、行为分析器。"""
from .base import (BBox, COCO_KP, Detector, IoUTracker, KP_INDEX, ObjectDet,
                   PerceptionResult, PersonObs, assign_seats)
from .mock_detector import CartoonCockpitDetector, ScriptedMockDetector
from .registry import (available_backends, build_detector, build_face_module, fallback_log)

__all__ = [
    "BBox", "COCO_KP", "Detector", "IoUTracker", "KP_INDEX", "ObjectDet",
    "PerceptionResult", "PersonObs", "assign_seats",
    "CartoonCockpitDetector", "ScriptedMockDetector",
    "build_detector", "build_face_module", "available_backends", "fallback_log",
]
