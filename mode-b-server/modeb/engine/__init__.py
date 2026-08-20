"""事件引擎：逐帧判定 → 防误报确认 → SafetyEvent → 告警分发 → 多路调度。"""
from .pipeline import PipelineStats, VehiclePipeline
from .rules import RawHit, ViolationRuleEngine
from .scheduler import InferenceScheduler, SchedulerStats

__all__ = ["VehiclePipeline", "PipelineStats", "RawHit", "ViolationRuleEngine",
           "InferenceScheduler", "SchedulerStats"]
