"""VLM 后端注册表 —— 通过 ``VSM_VLM_PROVIDER`` 环境变量切换。"""
from __future__ import annotations

from ..config import Settings, load_settings
from .base import VLMProvider, VLMResponse

_REGISTRY: dict[str, str] = {
    "auto": "按已配置的凭据自动选择，全都没有则退回 mock",
    "mock": "模拟后端（无需 key，规则化返回，仅供演示与回归）",
    "siliconflow": "硅基流动 Qwen3-VL / GLM-4.5V（SILICONFLOW_API_KEY）",
    "anthropic": "Anthropic Claude（ANTHROPIC_API_KEY）",
    "openai": "OpenAI 兼容接口，含自建 vLLM 网关（OPENAI_API_KEY / OPENAI_BASE_URL）",
    "dashscope": "阿里云百炼 Qwen-VL（DASHSCOPE_API_KEY）",
    "local": "本地 transformers 权重（VSM_LOCAL_MODEL_PATH）",
}


def list_providers() -> dict[str, str]:
    return dict(_REGISTRY)


def get_provider(settings: Settings | None = None, **kwargs) -> VLMProvider:
    """按配置构造 provider。未知名字直接报错，不静默回退到 mock。

    静默回退是演示类项目里最容易产生「假跑通」的地方：明明 key 错了，
    界面上却照样出结果。这里宁可报错。
    """
    settings = settings or load_settings()
    name = (kwargs.pop("provider", None) or settings.provider or "auto").lower()
    if name == "auto":
        name = settings.resolve_provider()
    if name == "mock":
        from .mock import MockProvider
        return MockProvider(settings, **kwargs)
    if name == "anthropic":
        from .cloud import AnthropicProvider
        return AnthropicProvider(settings)
    if name in ("openai", "openai_compat"):
        from .cloud import OpenAICompatProvider
        return OpenAICompatProvider(settings)
    if name == "dashscope":
        from .cloud import DashScopeProvider
        return DashScopeProvider(settings)
    if name in ("siliconflow", "sf"):
        from .cloud import SiliconFlowProvider
        return SiliconFlowProvider(settings)
    if name in ("local", "local_hf", "hf"):
        from .local_hf import LocalHFProvider
        return LocalHFProvider(settings)
    raise ValueError(f"未知 VLM 后端 '{name}'，可选：{', '.join(_REGISTRY)}")


__all__ = ["get_provider", "list_providers", "VLMProvider", "VLMResponse"]
