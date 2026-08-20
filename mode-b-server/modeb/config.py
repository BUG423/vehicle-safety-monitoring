"""模式B 全局配置 —— 集中在一处，避免参数散落。

所有阈值都可以通过环境变量覆盖，便于在不同车型/摄像头上现场调参而不用改代码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _s(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


@dataclass
class PerceptionConfig:
    """感知层参数。"""

    backend: str = field(default_factory=lambda: _s("MODEB_BACKEND", "auto"))
    """auto | torchvision | mock —— auto 表示优先真模型，失败自动降级到 mock。"""

    device: str = field(default_factory=lambda: _s("MODEB_DEVICE", "cuda:0"))
    person_score_thr: float = field(default_factory=lambda: _f("MODEB_PERSON_THR", 0.75))
    object_score_thr: float = field(default_factory=lambda: _f("MODEB_OBJECT_THR", 0.45))
    keypoint_score_thr: float = field(default_factory=lambda: _f("MODEB_KP_THR", 3.0))
    """Keypoint R-CNN 输出的关键点分数是未归一化 logit，经验阈值取 3.0。"""

    infer_short_side: int = field(default_factory=lambda: _i("MODEB_INFER_SHORT_SIDE", 480))
    """推理前把画面短边缩放到该尺寸，直接决定吞吐。"""

    max_batch: int = field(default_factory=lambda: _i("MODEB_MAX_BATCH", 8))
    batch_wait_ms: float = field(default_factory=lambda: _f("MODEB_BATCH_WAIT_MS", 12.0))

    enable_object_detector: bool = field(default_factory=lambda: _b("MODEB_OBJECT_DET", True))
    """是否加载第二个 COCO 目标检测器（手机等物体）。关掉可换取约 40% 吞吐。"""


@dataclass
class RuleConfig:
    """逐帧原始判定阈值。这些数字是工程经验起点，不是标注集上调出来的。"""

    driver_side: str = field(default_factory=lambda: _s("MODEB_DRIVER_SIDE", "left"))
    """驾驶位在画面的哪一侧（左舵车正对镜头时通常在画面左半边）。"""

    seatbelt_score_thr: float = field(default_factory=lambda: _f("MODEB_BELT_THR", 0.10))
    eye_open_thr: float = field(default_factory=lambda: _f("MODEB_EYE_OPEN_THR", 0.28))
    perclos_window_s: float = field(default_factory=lambda: _f("MODEB_PERCLOS_WIN", 20.0))
    perclos_thr: float = field(default_factory=lambda: _f("MODEB_PERCLOS_THR", 0.40))
    yaw_distract_deg: float = field(default_factory=lambda: _f("MODEB_YAW_THR", 35.0))
    pitch_distract_deg: float = field(default_factory=lambda: _f("MODEB_PITCH_THR", 28.0))
    phone_near_head_px_ratio: float = field(default_factory=lambda: _f("MODEB_PHONE_HEAD_R", 1.6))
    blur_var_thr: float = field(default_factory=lambda: _f("MODEB_BLUR_VAR", 12.0))
    dark_mean_thr: float = field(default_factory=lambda: _f("MODEB_DARK_MEAN", 18.0))
    max_occupancy: int = field(default_factory=lambda: _i("MODEB_MAX_OCCUPANCY", 5))
    speeding_tolerance_kmh: float = field(default_factory=lambda: _f("MODEB_SPEED_TOL", 5.0))

    enable_smoking_proxy: bool = field(default_factory=lambda: _b("MODEB_SMOKING_PROXY", False))
    """COCO 没有香烟类别，真模型后端无法直接检测抽烟。
    打开后使用「手到嘴 + 无手机」的姿态代理，误报率高，默认关闭。见 DESIGN.md 第 3.6 节。"""

    enable_hands_off_proxy: bool = field(default_factory=lambda: _b("MODEB_HANDS_OFF_PROXY", False))
    """双手脱离方向盘的代理判定同样不可靠（方向盘位置未标定），默认关闭。"""


@dataclass
class ServerConfig:
    host: str = field(default_factory=lambda: _s("MODEB_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _i("MODEB_PORT", 8080))
    db_path: str = field(default_factory=lambda: _s("MODEB_DB", "runs/modeb.db"))
    evidence_dir: str = field(default_factory=lambda: _s("MODEB_EVIDENCE_DIR", "runs/evidence"))
    heartbeat_timeout_s: float = field(default_factory=lambda: _f("MODEB_HB_TIMEOUT", 20.0))
    max_ws_queue: int = field(default_factory=lambda: _i("MODEB_WS_QUEUE", 200))


@dataclass
class Config:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONFIG = Config()
