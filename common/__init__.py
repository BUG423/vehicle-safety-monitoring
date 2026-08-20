"""三种技术模式共享的契约层：事件模型、确认逻辑、告警通道。"""
from .schema.violation_types import ViolationType, Severity, SubjectRole, DetectionMode
from .schema.safety_event import SafetyEvent, Subject, Evidence, VehicleContext, SCHEMA_VERSION
from .schema.confirmation import ViolationConfirmer, ConfirmRule, Confirmation
from .alerting.channels import AlertDispatcher, InCabinChannel, BackendChannel, CabinPrompt

__all__ = [
    "ViolationType", "Severity", "SubjectRole", "DetectionMode",
    "SafetyEvent", "Subject", "Evidence", "VehicleContext", "SCHEMA_VERSION",
    "ViolationConfirmer", "ConfirmRule", "Confirmation",
    "AlertDispatcher", "InCabinChannel", "BackendChannel", "CabinPrompt",
]
