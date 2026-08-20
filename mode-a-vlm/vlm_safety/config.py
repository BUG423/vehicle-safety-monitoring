"""模式A 的运行配置 —— 全部通过环境变量注入，不引入配置文件依赖。

设计意图：同一份代码在「无 key 的演示机」「有云端 key 的甲方环境」「本地 A100」
三种场景下只靠环境变量切换，不改代码、不改镜像。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """一次运行的全部可调参数。"""

    # ---- VLM 后端 ----
    provider: str = "mock"           # mock | anthropic | openai | dashscope | local
    model: str = ""                  # 留空则由各 provider 用自己的默认模型
    temperature: float = 0.0         # 结构化抽取任务固定用 0，降低幻觉与不稳定
    max_tokens: int = 1200
    timeout_s: float = 60.0
    # 自一致性投票次数：>1 时同一张图多次采样，取多数票（成本翻倍，换稳定性）
    self_consistency: int = 1

    # ---- 各后端凭据 ----
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    local_model_path: str = ""       # 本地权重目录或 HF repo id
    local_device: str = "cuda:0"
    local_dtype: str = "bfloat16"
    local_max_pixels: int = 768 * 28 * 28   # Qwen2.5-VL 的视觉 token 上限控制

    # ---- 图像预处理（直接影响 token 成本）----
    max_image_edge: int = 896        # 上传前缩放到最长边不超过该值
    jpeg_quality: int = 80

    # ---- 业务上下文 ----
    vehicle_id: str = "DEMO-VEHICLE-001"
    # 车内提醒的最低等级：INFO 会很吵，默认 warn
    cabin_min_severity: str = "warn"
    backend_webhook: str = ""        # 留空则后台通道只打印到 stdout（演示用）
    alert_spool_dir: str = ".alert_spool"

    # ---- 隐私 ----
    # 上云前是否对人脸做马赛克（模式A 的默认合规姿态；本地/私有化部署可关闭）
    blur_faces: bool = False
    # 事件里是否携带缩略图 base64（关闭后后台只拿到结构化结论，影像留在车端）
    attach_thumbnail: bool = True
    thumbnail_edge: int = 256

    extra: dict = field(default_factory=dict)

    @property
    def has_cloud_key(self) -> bool:
        return bool(self.anthropic_api_key or self.openai_api_key or self.dashscope_api_key)


def load_settings(**overrides) -> Settings:
    """从环境变量构造 Settings，再用关键字参数覆盖。"""
    s = Settings(
        provider=_env("VSM_VLM_PROVIDER", "mock").lower() or "mock",
        model=_env("VSM_VLM_MODEL"),
        temperature=_env_float("VSM_VLM_TEMPERATURE", 0.0),
        max_tokens=_env_int("VSM_VLM_MAX_TOKENS", 1200),
        timeout_s=_env_float("VSM_VLM_TIMEOUT_S", 60.0),
        self_consistency=max(1, _env_int("VSM_VLM_SELF_CONSISTENCY", 1)),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        anthropic_base_url=_env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        dashscope_api_key=_env("DASHSCOPE_API_KEY"),
        dashscope_base_url=_env("DASHSCOPE_BASE_URL",
                                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        local_model_path=_env("VSM_LOCAL_MODEL_PATH"),
        local_device=_env("VSM_LOCAL_DEVICE", "cuda:0"),
        local_dtype=_env("VSM_LOCAL_DTYPE", "bfloat16"),
        local_max_pixels=_env_int("VSM_LOCAL_MAX_PIXELS", 768 * 28 * 28),
        max_image_edge=_env_int("VSM_MAX_IMAGE_EDGE", 896),
        jpeg_quality=_env_int("VSM_JPEG_QUALITY", 80),
        vehicle_id=_env("VSM_VEHICLE_ID", "DEMO-VEHICLE-001"),
        cabin_min_severity=_env("VSM_CABIN_MIN_SEVERITY", "warn").lower(),
        backend_webhook=_env("VSM_BACKEND_WEBHOOK"),
        alert_spool_dir=_env("VSM_ALERT_SPOOL", ".alert_spool"),
        blur_faces=_env_bool("VSM_BLUR_FACES", False),
        attach_thumbnail=_env_bool("VSM_ATTACH_THUMBNAIL", True),
        thumbnail_edge=_env_int("VSM_THUMBNAIL_EDGE", 256),
    )
    for k, v in overrides.items():
        if hasattr(s, k):
            setattr(s, k, v)
        else:
            s.extra[k] = v
    return s
