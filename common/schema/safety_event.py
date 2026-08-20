"""车辆安全监测 —— 统一安全事件模型。

设计原则：
1. 只依赖标准库（dataclasses/enum/json），车载嵌入式端也能直接引入，不引入 pydantic 等重依赖。
2. 三种技术模式产生的事件结构完全一致，后台无需区分来源即可入库、告警、统计；
   `mode` 字段仅用于横向对比与灰度切换。
3. `evidence` 与 `raw_signals` 保留各模式特有的中间产物（VLM 原始回答、检测框、传感器读数），
   便于事后复核与误报归因。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

try:  # 允许以包方式或独立文件方式引入
    from .violation_types import DetectionMode, Severity, SubjectRole, ViolationType
except ImportError:  # pragma: no cover
    from violation_types import DetectionMode, Severity, SubjectRole, ViolationType

SCHEMA_VERSION = "1.0"


@dataclass
class Subject:
    """事件涉及的人 / 车。"""

    role: SubjectRole = SubjectRole.UNKNOWN
    seat: str | None = None          # "driver" | "front_passenger" | "rear_left" | ...
    track_id: int | None = None      # 同一路视频内的跟踪 ID
    person_id: str | None = None     # 已完成身份识别时的员工/驾驶员编号


@dataclass
class Evidence:
    """证据材料 —— 后台复核与申诉的依据。"""

    frame_uri: str | None = None       # 关键帧存储路径 / 对象存储 URL
    frame_b64: str | None = None       # 小尺寸缩略图（base64），便于告警直接携带
    clip_uri: str | None = None        # 事件前后视频片段
    bbox: list[float] | None = None    # [x1, y1, x2, y2] 归一化坐标
    captured_at: float | None = None   # 证据帧的采集时间戳


@dataclass
class SafetyEvent:
    """一条安全违规事件。"""

    violation: ViolationType
    vehicle_id: str
    mode: DetectionMode

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)          # 事件确认时间（Unix 秒）
    severity: Severity | None = None                       # 缺省时取违规类型的默认等级
    subject: Subject = field(default_factory=Subject)
    confidence: float = 1.0                                # 0~1，确认时的置信度
    duration_s: float = 0.0                                # 违规持续时长
    evidence: Evidence = field(default_factory=Evidence)
    raw_signals: dict[str, Any] = field(default_factory=dict)  # PERCLOS、车速、VLM 原文等
    message: str = ""                                      # 面向人的中文描述
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.violation, str):
            self.violation = ViolationType(self.violation)
        if self.severity is None:
            self.severity = self.violation.default_severity
        elif isinstance(self.severity, str):
            self.severity = Severity(self.severity)
        if isinstance(self.mode, str):
            self.mode = DetectionMode(self.mode)
        if self.subject.role is SubjectRole.UNKNOWN:
            self.subject.role = self.violation.default_role
        if not self.message:
            self.message = self._default_message()

    def _default_message(self) -> str:
        seat = f"（{self.subject.seat}）" if self.subject.seat else ""
        dur = f"，持续 {self.duration_s:.1f} 秒" if self.duration_s >= 1 else ""
        return f"{self.violation.label_zh}{seat}{dur}"

    # ---- 序列化 ----
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["violation"] = self.violation.value
        d["severity"] = self.severity.value
        d["mode"] = self.mode.value
        d["subject"]["role"] = self.subject.role.value
        return d

    def to_json(self, *, drop_b64: bool = False) -> str:
        d = self.to_dict()
        if drop_b64:
            d["evidence"]["frame_b64"] = None
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SafetyEvent:
        d = dict(d)
        subject = Subject(**{**d.pop("subject", {}), "role": SubjectRole(d.get("subject", {}).get("role", "unknown"))}) \
            if isinstance(d.get("subject"), dict) else Subject()
        evidence = Evidence(**d.pop("evidence", {})) if isinstance(d.get("evidence"), dict) else Evidence()
        d.pop("subject", None)
        d.pop("evidence", None)
        d.pop("schema_version", None)
        return cls(subject=subject, evidence=evidence, **d)


@dataclass
class VehicleContext:
    """车辆侧上下文 —— 超速等事件必须依赖这些非视觉信号。"""

    vehicle_id: str
    speed_kmh: float | None = None
    speed_limit_kmh: float | None = None
    gear: str | None = None            # P/R/N/D
    engine_on: bool | None = None
    gps: tuple[float, float] | None = None
    seatbelt_switch: dict[str, bool] = field(default_factory=dict)  # 座位 -> 是否扣合（车身总线信号）
    ts: float = field(default_factory=time.time)

    @property
    def is_moving(self) -> bool:
        return bool(self.speed_kmh and self.speed_kmh > 3.0)
