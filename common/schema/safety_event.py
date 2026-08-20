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
from dataclasses import asdict, dataclass, field, fields
from typing import Any

try:  # 允许以包方式或独立文件方式引入
    from .violation_types import Decision, DetectionMode, Severity, SubjectRole, ViolationType
except ImportError:  # pragma: no cover
    from violation_types import Decision, DetectionMode, Severity, SubjectRole, ViolationType

SCHEMA_VERSION = "1.3"

SEAT_LABELS_ZH: dict[str, str] = {
    "driver": "主驾",
    "front_passenger": "副驾",
    "rear_left": "后排左",
    "rear_middle": "后排中",
    "rear_right": "后排右",
}


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
    clip_range_s: tuple[float, float] | None = None   # 片段相对事件时刻的起止秒，如 (-5.0, 3.0)
    bbox: list[float] | None = None    # [x1, y1, x2, y2] 归一化坐标
    captured_at: float | None = None   # 证据帧的采集时间戳
    model_version: str | None = None   # 产出该判定的模型标识，灰度期用于把误报归因到具体版本
    evidence_text: str | None = None
    """模型给出的文字依据（如「肩部可见浅色斜跨织带」）。

    可结构化查询，且不含影像——是隐私友好的复核方式：管理者读文字即可复核，无需调阅车内录像。"""


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
    decision: Decision = Decision.CONFIRMED                # 确认违规 / 判不了
    undecidable_reason: str | None = None                  # decision 为 UNDECIDABLE 时说明原因
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
        if isinstance(self.decision, str):
            self.decision = Decision(self.decision)
        if self.subject.role is SubjectRole.UNKNOWN:
            self.subject.role = self.violation.default_role
        if not self.message:
            self.message = self._default_message()

    def _default_message(self) -> str:
        seat_zh = SEAT_LABELS_ZH.get(self.subject.seat or "", self.subject.seat or "")
        seat = f"（{seat_zh}）" if seat_zh else ""
        if self.decision is Decision.UNDECIDABLE:
            why = f"：{self.undecidable_reason}" if self.undecidable_reason else ""
            return f"{self.violation.label_zh}{seat} 无法判定{why}"
        dur = f"，持续 {self.duration_s:.1f} 秒" if self.duration_s >= 1 else ""
        return f"{self.violation.label_zh}{seat}{dur}"

    # ---- 序列化 ----
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["violation"] = self.violation.value
        d["severity"] = self.severity.value
        d["mode"] = self.mode.value
        d["decision"] = self.decision.value
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
        raw_subject = d.pop("subject", None)
        if isinstance(raw_subject, dict):
            raw_subject = dict(raw_subject)
            role = raw_subject.pop("role", "unknown")
            subject = Subject(role=SubjectRole(role) if isinstance(role, str) else role,
                              **raw_subject)
        else:
            subject = Subject()
        raw_evidence = d.pop("evidence", None)
        if isinstance(raw_evidence, dict):
            ev_known = {f.name for f in fields(Evidence)}
            evidence = Evidence(**{k: v for k, v in raw_evidence.items() if k in ev_known})
        else:
            evidence = Evidence()
        d.pop("schema_version", None)
        # 宽进严出：后台落库、复核流程必然会给事件附加自己的元数据（review_status、
        # 处理人、工单号…）。这些字段随事件回流到下游时不应让反序列化崩溃 ——
        # 它们不属于契约，忽略即可，但产出方仍必须严格按契约写。
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
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
    seat_capacity: int | None = None   # 核定载客数，超员判定的统一口径（来自行驶证，非视觉推断）
    road_type: str | None = None       # highway/urban/rural/parking，限速与告警口径随路况不同
    ts: float = field(default_factory=time.time)

    @property
    def is_moving(self) -> bool:
        return bool(self.speed_kmh and self.speed_kmh > 3.0)
