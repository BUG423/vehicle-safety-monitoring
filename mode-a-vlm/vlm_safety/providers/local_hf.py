"""本地开源 VLM 后端（transformers）。

存在的理由不是「省钱」，而是**数据不出车、不出机房**：车内影像属于个人信息，
不少车队/政企客户根本不允许上公有云。本地后端让模式A 在私有化环境里仍然成立，
代价是需要一张显卡常驻（见 DESIGN.md 的成本对比）。

默认模型 Qwen2.5-VL-7B-Instruct：中文指令跟随好、支持多图、7B 在单卡 A100 上
bf16 推理约 16GB 显存，是这个场景里性价比合适的档位。

加载是**惰性**的：只有真正发起第一次推理时才占显存，
这样 FastAPI 服务在 provider=local 但尚未收到请求时不会白占一张卡。
"""
from __future__ import annotations

import io
import logging
import os
import threading
import time
from typing import Sequence

from PIL import Image

from ..imaging import PreparedImage
from .base import VLMProvider, VLMResponse

log = logging.getLogger(__name__)


class LocalHFProvider(VLMProvider):
    """用 transformers 在本地跑开源 VLM。"""

    name = "local"
    simulated = False

    _lock = threading.Lock()

    @property
    def default_model(self) -> str:
        return "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.model_path = settings.local_model_path or self.model
        self._model = None
        self._processor = None
        self._load_error: str | None = None
        self._load_seconds: float | None = None

    # ---- 加载 ----
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._load_error:
            raise RuntimeError(self._load_error)
        with self._lock:
            if self._model is not None:
                return
            t0 = time.perf_counter()
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                         "float32": torch.float32}.get(self.settings.local_dtype, torch.bfloat16)
                log.info("加载本地 VLM: %s -> %s", self.model_path, self.settings.local_device)
                self._processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    max_pixels=self.settings.local_max_pixels,
                )
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path, dtype=dtype,
                ).to(self.settings.local_device).eval()
                self._load_seconds = time.perf_counter() - t0
                log.info("本地 VLM 加载完成，用时 %.1fs", self._load_seconds)
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"本地模型加载失败: {type(exc).__name__}: {exc}"
                log.error(self._load_error)
                raise RuntimeError(self._load_error) from exc

    # ---- 推理 ----
    def _invoke(self, images: Sequence[PreparedImage], system: str, user: str) -> VLMResponse:
        self._ensure_loaded()
        import torch

        pil_images = [Image.open(io.BytesIO(im.jpeg)).convert("RGB") for im in images]
        content: list[dict] = []
        for i, _ in enumerate(pil_images):
            if len(pil_images) > 1:
                content.append({"type": "text", "text": f"第 {i + 1} 帧："})
            content.append({"type": "image"})
        content.append({"type": "text", "text": user})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=pil_images, return_tensors="pt")
        inputs = {k: (v.to(self.settings.local_device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=self.settings.temperature > 0,
                temperature=max(self.settings.temperature, 1e-5),
            )
        latency = (time.perf_counter() - t0) * 1000
        in_len = inputs["input_ids"].shape[1]
        gen = out[:, in_len:]
        answer = self._processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return VLMResponse(
            text=answer, provider=self.name, model=os.path.basename(str(self.model_path).rstrip("/")),
            latency_ms=latency, prompt_tokens=int(in_len), completion_tokens=int(gen.shape[1]),
            raw={"device": self.settings.local_device, "load_seconds": self._load_seconds},
        )

    def health(self) -> dict:
        d = super().health()
        d["model"] = str(self.model_path)
        exists = os.path.isdir(str(self.model_path))
        if self._load_error:
            d["ready"] = False
            d["detail"] = self._load_error
        elif self._model is not None:
            d["detail"] = f"已加载到 {self.settings.local_device}，加载耗时 {self._load_seconds:.1f}s"
        else:
            d["detail"] = ("权重目录存在，将在首次请求时惰性加载" if exists
                           else "尚未加载（首次请求时会从本地路径或 HF Hub 拉取）")
        return d

    def close(self) -> None:
        if self._model is not None:
            try:
                import torch

                del self._model
                self._model = None
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
