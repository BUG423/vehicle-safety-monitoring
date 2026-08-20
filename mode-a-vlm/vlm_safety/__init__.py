"""模式A：VLM（视觉语言大模型）直接理解车内图片的安全检查实现。

对外暴露三个入口：
  - ``get_provider``  按配置拿到一个可插拔的 VLM 后端
  - ``SafetyPipeline`` 端到端流水线：图片 -> VLM -> 结构化观测 -> SafetyEvent -> 告警
  - ``build_app``      FastAPI 服务（含手机可访问的 H5 演示页）

所有产出事件均为 ``common.schema.safety_event.SafetyEvent``，``mode=DetectionMode.VLM``。
"""
from .config import Settings, load_settings
from .providers import get_provider, list_providers
from .pipeline import SafetyPipeline, PipelineResult

__all__ = [
    "Settings", "load_settings",
    "get_provider", "list_providers",
    "SafetyPipeline", "PipelineResult",
]
