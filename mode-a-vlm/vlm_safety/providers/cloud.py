"""云端 VLM 后端：Anthropic Claude / OpenAI 兼容接口 / 阿里 DashScope(Qwen-VL)。

三家都用裸 HTTP 调用（只依赖 requests），不引入各家 SDK：
车载/边缘侧镜像越薄越好，而且 SDK 版本漂移是长期维护里最常见的坑。

**当前运行环境没有任何云端 API key，因此这三个后端未经真实调用验证。**
代码按各家公开的接口规范编写，缺 key 时 ``health()`` 会明确报告 not ready。
"""
from __future__ import annotations

import json
from typing import Sequence

import requests

from ..imaging import PreparedImage
from .base import VLMProvider, VLMResponse


class _HttpProvider(VLMProvider):
    """共用的 HTTP 调用骨架。"""

    api_key_attr = ""
    key_env_name = ""

    @property
    def api_key(self) -> str:
        return getattr(self.settings, self.api_key_attr, "")

    def health(self) -> dict:
        d = super().health()
        if not self.api_key:
            d["ready"] = False
            d["detail"] = f"缺少环境变量 {self.key_env_name}，该后端不可用"
        else:
            d["detail"] = "已配置密钥（未做联网探活，避免产生费用）"
        return d

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        resp = requests.post(url, headers=headers, json=body, timeout=self.settings.timeout_s)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        return resp.json()

    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(f"未配置 {self.key_env_name}，无法调用 {self.name}")


class AnthropicProvider(_HttpProvider):
    """Anthropic Messages API。多图直接放进同一个 user turn 的 content 数组。"""

    name = "anthropic"
    api_key_attr = "anthropic_api_key"
    key_env_name = "ANTHROPIC_API_KEY"

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-5"

    def _invoke(self, images: Sequence[PreparedImage], system: str, user: str) -> VLMResponse:
        self._require_key()
        content: list[dict] = []
        for i, im in enumerate(images):
            if len(images) > 1:
                content.append({"type": "text", "text": f"第 {i + 1} 帧："})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": im.b64},
            })
        content.append({"type": "text", "text": user})

        body = {
            "model": self.model,
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        data = self._post(
            f"{self.settings.anthropic_base_url.rstrip('/')}/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
             "content-type": "application/json"},
            body,
        )
        text = "".join(blk.get("text", "") for blk in data.get("content", [])
                       if blk.get("type") == "text")
        usage = data.get("usage", {})
        return VLMResponse(text=text, provider=self.name, model=self.model,
                           prompt_tokens=usage.get("input_tokens"),
                           completion_tokens=usage.get("output_tokens"),
                           raw={"stop_reason": data.get("stop_reason")})


class OpenAICompatProvider(_HttpProvider):
    """OpenAI 兼容的 /chat/completions。

    这条路覆盖面最广：OpenAI 官方、vLLM/SGLang 自建服务、以及绝大多数国产模型
    的兼容网关都认这个协议 —— 甲方私有化部署时通常只需要改 ``OPENAI_BASE_URL``。
    """

    name = "openai"
    api_key_attr = "openai_api_key"
    key_env_name = "OPENAI_API_KEY"

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"

    @property
    def base_url(self) -> str:
        return self.settings.openai_base_url.rstrip("/")

    def _build_messages(self, images, system, user) -> list[dict]:
        content: list[dict] = []
        for i, im in enumerate(images):
            if len(images) > 1:
                content.append({"type": "text", "text": f"第 {i + 1} 帧："})
            content.append({"type": "image_url", "image_url": {"url": im.data_uri}})
        content.append({"type": "text", "text": user})
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

    def _invoke(self, images, system, user) -> VLMResponse:
        self._require_key()
        body = {
            "model": self.model,
            "messages": self._build_messages(images, system, user),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            # 要求 JSON 输出。部分兼容网关不支持该字段，失败时会在 error 里体现，
            # 由调用方去掉 response_format 重试（见 pipeline 的降级逻辑）。
            "response_format": {"type": "json_object"},
        }
        data = self._post(f"{self.base_url}/chat/completions",
                          {"Authorization": f"Bearer {self.api_key}",
                           "Content-Type": "application/json"}, body)
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage", {})
        return VLMResponse(text=text, provider=self.name, model=self.model,
                           prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"),
                           raw={"finish_reason": choice.get("finish_reason")})


class DashScopeProvider(OpenAICompatProvider):
    """阿里云百炼（DashScope）的 Qwen-VL 系列。

    百炼提供 OpenAI 兼容端点，因此直接复用上面的实现，只换 base_url / key / 默认模型。
    对国内车队项目，数据不出境 + 中文场景理解是这条后端的主要理由。
    """

    name = "dashscope"
    api_key_attr = "dashscope_api_key"
    key_env_name = "DASHSCOPE_API_KEY"

    @property
    def default_model(self) -> str:
        return "qwen-vl-max-latest"

    @property
    def base_url(self) -> str:
        return self.settings.dashscope_base_url.rstrip("/")

    def _invoke(self, images, system, user) -> VLMResponse:
        self._require_key()
        body = {
            "model": self.model,
            "messages": self._build_messages(images, system, user),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }  # 百炼部分模型不接受 response_format，这里不传，靠 prompt 约束 JSON
        data = self._post(f"{self.base_url}/chat/completions",
                          {"Authorization": f"Bearer {self.api_key}",
                           "Content-Type": "application/json"}, body)
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        if isinstance(text, list):  # 部分版本返回 content 数组
            text = "".join(seg.get("text", "") for seg in text if isinstance(seg, dict))
        usage = data.get("usage", {})
        return VLMResponse(text=text, provider=self.name, model=self.model,
                           prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"),
                           raw={"finish_reason": choice.get("finish_reason")})
