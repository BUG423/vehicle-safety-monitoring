"""嵌入式端运行参数。

所有阈值集中在这里，方便按车型 / 摄像头安装位置做标定，
也方便 OTA 时只下发一个配置文件而不用重刷固件。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PKG_DIR = MODULE_DIR.parent


@dataclass
class EdgeConfig:
    # ---- 车辆标识 ----
    vehicle_id: str = "京A·12345"

    # ---- 采集（嵌入式约束的核心）----
    frame_width: int = 320          # DMS 摄像头下采样后的处理分辨率
    frame_height: int = 240
    target_fps: float = 10.0        # 主循环限帧率，超出即 sleep 让出 CPU
    num_threads: int = 1            # 推理线程数，模拟 RK3588 只给 1~2 个大核

    # ---- 模型 ----
    models_dir: Path = field(default_factory=lambda: PKG_DIR / "models")
    face_score_thr: float = 0.7
    object_stride: int = 5          # 目标检测降频：每 N 帧跑一次
    object_score_thr: float = 0.35
    use_int8_detector: bool = True  # 优先用 INT8 权重（嵌入式默认）

    # ---- 疲劳（PERCLOS）----
    perclos_window_s: float = 30.0  # PERCLOS 统计窗口
    perclos_thr: float = 0.30       # 窗口内闭眼帧占比阈值
    eye_close_ratio: float = 0.62   # 相对个体基线的闭眼判定系数
    baseline_warmup_s: float = 5.0  # 个体基线标定时长
    yawn_ratio: float = 0.55        # 嘴部张开比阈值（打哈欠）

    # ---- 分心（头部姿态）----
    yaw_thr_deg: float = 32.0
    pitch_thr_deg: float = 24.0

    # ---- 安全带 ----
    belt_line_min_len: float = 0.35     # 归一化最短带体线段长度
    belt_angle_range: tuple = (20.0, 70.0)  # 斜跨肩带与水平线的夹角范围（度）

    # ---- 车辆信号 ----
    speed_tolerance: float = 1.03       # 超过限速 3% 才算超速（GPS/OBD 误差裕量）
    harsh_accel_thr: float = 3.5        # m/s^2，急加速/急刹阈值

    # ---- 遮挡 ----
    blur_thr: float = 12.0              # 拉普拉斯方差低于此值判定镜头遮挡/失效

    # ---- 告警 ----
    backend_url: str = "http://127.0.0.1:18080/api/events"
    backend_timeout_s: float = 2.0
    retry_interval_s: float = 3.0       # 断网补传扫描间隔（车载实机建议 30~60s）
    spool_dir: Path = field(default_factory=lambda: PKG_DIR / "runtime" / "spool")
    audio_dir: Path = field(default_factory=lambda: PKG_DIR / "runtime" / "audio")
    evidence_dir: Path = field(default_factory=lambda: PKG_DIR / "runtime" / "evidence")
    event_log: Path = field(default_factory=lambda: PKG_DIR / "runtime" / "events.jsonl")

    # ---- 证据 ----
    evidence_jpeg_quality: int = 55     # 证据缩略图质量，权衡 4G 流量
    evidence_max_side: int = 240

    def ensure_dirs(self) -> None:
        for d in (self.spool_dir, self.audio_dir, self.evidence_dir, self.event_log.parent):
            Path(d).mkdir(parents=True, exist_ok=True)

    def to_json(self) -> str:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "EdgeConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for k in ("models_dir", "spool_dir", "audio_dir", "evidence_dir", "event_log"):
            if k in raw:
                raw[k] = Path(raw[k])
        if "belt_angle_range" in raw:
            raw["belt_angle_range"] = tuple(raw["belt_angle_range"])
        return cls(**raw)
