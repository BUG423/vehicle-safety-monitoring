"""VLM Provider 抽象。

所有后端（云端 API / 本地权重 / Mock）实现同一个 ``analyze`` 接口，
返回同一个 ``VLMResponse``。切换后端不影响下游任何一行代码 —— 这是模式A
能在「没有 key 的演示机」和「甲方生产环境」之间平移的前提。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from ..config import Settings
from ..imaging import PreparedImage


@dataclass
class VLMResponse:
    """一次 VLM 调用的原始结果 —— 只到「文本」为止，解析交给 parser。"""

    text: str                              # 模型原始回答（未解析）
    provider: str
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: 是否为模拟输出。Mock provider 恒为 True，任何真实模型恒为 False。
    #: 这个字段会一路透传到 API 响应和演示页，用于诚实标注。
    simulated: bool = False
    error: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "simulated": self.simulated,
            "error": self.error,
            "text": self.text,
        }


class VLMProvider(ABC):
    """可插拔的 VLM 后端。"""

    name = "base"
    #: 真实模型置 False；Mock 置 True。
    simulated = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.model or self.default_model
        #: 由调用方按本次请求的帧数临时抬高输出预算（见 max_tokens）
        self.max_tokens_override: int | None = None

    @property
    def max_tokens(self) -> int:
        """本次调用允许的输出 token 上限。

        多帧请求的输出长度随帧数、乘员数线性增长。踩过的坑：固定 1200 时
        2 帧 × 2 名乘员就会 ``finish_reason=length`` 被截断，JSON 不完整、
        解析全部失败——日志上看起来像"模型不会输出 JSON"，实际是预算给少了。
        """
        return self.max_tokens_override or self.settings.max_tokens

    @property
    def default_model(self) -> str:
        return ""

    @abstractmethod
    def _invoke(self, images: Sequence[PreparedImage], system: str, user: str) -> VLMResponse:
        """子类实现：把图 + 提示词送给模型，拿回文本。"""

    def analyze(self, images: Sequence[PreparedImage], system: str, user: str) -> VLMResponse:
        """带计时与异常兜底的统一调用入口。

        任何后端异常都被转成 ``error`` 字段返回，绝不向上抛 ——
        车载安全系统里，感知模块挂掉不能导致整个服务 500。
        """
        t0 = time.perf_counter()
        try:
            resp = self._invoke(images, system, user)
        except Exception as exc:  # noqa: BLE001
            resp = VLMResponse(text="", provider=self.name, model=self.model,
                               simulated=self.simulated, error=f"{type(exc).__name__}: {exc}")
        if not resp.latency_ms:
            resp.latency_ms = (time.perf_counter() - t0) * 1000
        resp.simulated = resp.simulated or self.simulated
        return resp

    def health(self) -> dict:
        """给 /health 接口用的自检信息。"""
        return {"provider": self.name, "model": self.model, "simulated": self.simulated,
                "ready": True, "detail": ""}

    def close(self) -> None:
        """释放资源（本地模型卸载显存等）。默认无操作。"""
