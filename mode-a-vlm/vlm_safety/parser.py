"""把 VLM 的自由文本回答解析成受控的结构化观测。

大模型输出 JSON 有三种典型失败，这里逐一兜底：

  1. **包了 markdown 代码块** 或前后有寒暄 -> 括号配平扫描，只取第一个完整 JSON；
  2. **JSON 语法不合法**（尾逗号、单引号、注释、中文标点）-> 逐条修复后重试；
  3. **字段/枚举越界**（编了个 "maybe_fastened"）-> 白名单校验，越界一律降级为 unknown。

第 3 条是安全性的关键：宁可丢一次检出，也不能让一个没定义的状态穿到告警层。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from . import prompts as P

# 中文/口语座位名 -> 标准枚举
_SEAT_ALIASES = {
    "driver": "driver", "driver_seat": "driver", "主驾": "driver", "驾驶位": "driver",
    "驾驶员": "driver", "司机": "driver",
    "front_passenger": "front_passenger", "passenger": "front_passenger",
    "副驾": "front_passenger", "副驾驶": "front_passenger", "前排乘客": "front_passenger",
    "rear_left": "rear_left", "后排左": "rear_left", "后左": "rear_left",
    "rear_middle": "rear_middle", "后排中": "rear_middle",
    "rear_right": "rear_right", "后排右": "rear_right", "后右": "rear_right",
}

_ATTR_FIELDS = {
    "seatbelt": P.SEATBELT_STATES,
    "eyes": P.EYE_STATES,
    "mouth": P.MOUTH_STATES,
    "gaze": P.GAZE_STATES,
    "phone": P.PHONE_STATES,
    "smoking": P.SMOKING_STATES,
    "hands": P.HANDS_STATES,
}


@dataclass
class Attr:
    """一个视觉属性的观测结果。"""

    state: str = "unknown"
    evidence: str = ""
    confidence: float = 0.0
    coerced: bool = False       # 是否因越界被强制降级为 unknown

    @property
    def known(self) -> bool:
        return self.state not in ("unknown", "occluded")

    def to_dict(self) -> dict:
        return {"state": self.state, "evidence": self.evidence,
                "confidence": round(self.confidence, 3), "coerced": self.coerced}


@dataclass
class Occupant:
    seat: str = "unknown"
    person_present: bool = True
    apparent_age_group: str = "unknown"
    attrs: dict[str, Attr] = field(default_factory=dict)

    def attr(self, name: str) -> Attr:
        return self.attrs.get(name, Attr())

    def to_dict(self) -> dict:
        return {"seat": self.seat, "person_present": self.person_present,
                "apparent_age_group": self.apparent_age_group,
                **{k: v.to_dict() for k, v in self.attrs.items()}}


@dataclass
class FrameObservation:
    """一帧画面的完整观测。"""

    view: str = "unknown"
    image_quality: str = "unknown"
    persons_visible: int = 0
    occupants: list[Occupant] = field(default_factory=list)
    notes: str = ""
    frame_index: int = 0
    ts: float = 0.0             # 该帧的时间戳（多帧序列时由调用方填）

    def by_seat(self, seat: str) -> Occupant | None:
        for o in self.occupants:
            if o.seat == seat:
                return o
        return None

    def to_dict(self) -> dict:
        return {"view": self.view, "image_quality": self.image_quality,
                "persons_visible": self.persons_visible, "notes": self.notes,
                "frame_index": self.frame_index, "ts": self.ts,
                "occupants": [o.to_dict() for o in self.occupants]}


@dataclass
class ParseResult:
    frames: list[FrameObservation] = field(default_factory=list)
    ok: bool = False
    error: str | None = None
    repaired: bool = False           # 是否经过语法修复才解析成功
    coerced_fields: list[str] = field(default_factory=list)  # 被降级的越界字段
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "error": self.error, "repaired": self.repaired,
                "coerced_fields": self.coerced_fields,
                "frames": [f.to_dict() for f in self.frames]}


# ---------------------------------------------------------------------------
# 第 1、2 步：把文本抠成合法 JSON
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _extract_balanced(text: str) -> str | None:
    """扫描第一个配平的 {...} 或 [...]，忽略字符串内的括号。"""
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repair(text: str) -> str:
    """修复模型常犯的 JSON 语法错误。"""
    t = text
    t = re.sub(r"//[^\n]*", "", t)                       # 行注释
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)          # 块注释
    t = t.replace("“", '"').replace("”", '"').replace("：", ":").replace("，", ",")
    t = re.sub(r",\s*([}\]])", r"\1", t)                 # 尾逗号
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)
    t = re.sub(r"\bNone\b", "null", t)
    t = re.sub(r"\bNaN\b", "null", t)
    return t


def load_json_lenient(text: str) -> tuple[Any | None, bool, str | None]:
    """返回 (对象, 是否经过修复, 错误信息)。"""
    if not text or not text.strip():
        return None, False, "模型返回为空"
    candidate = _strip_fences(text)
    for repaired, payload in ((False, candidate), (True, _repair(candidate))):
        try:
            return json.loads(payload), repaired, None
        except json.JSONDecodeError:
            pass
        block = _extract_balanced(payload)
        if block:
            try:
                return json.loads(block), repaired, None
            except json.JSONDecodeError as exc:
                last = str(exc)
                continue
    return None, False, f"JSON 解析失败：{last if 'last' in dir() else '未找到合法 JSON 片段'}"


# ---------------------------------------------------------------------------
# 第 3 步：白名单校验与归一化
# ---------------------------------------------------------------------------
def _clamp_conf(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


def _norm_attr(raw: Any, allowed: list[str], coerced: list[str], path: str) -> Attr:
    if isinstance(raw, str):          # 模型偷懒直接给了字符串
        raw = {"state": raw}
    if not isinstance(raw, dict):
        return Attr()
    state = str(raw.get("state", "unknown")).strip().lower()
    a = Attr(state=state,
             evidence=str(raw.get("evidence", "") or "").strip(),
             confidence=_clamp_conf(raw.get("confidence", 0.0)))
    if a.state not in allowed:
        coerced.append(f"{path}.state={a.state!r}")
        a.state = "unknown"
        a.coerced = True
        a.confidence = 0.0
    if a.known and not a.evidence:
        # 有结论但说不出可见依据 -> 按规则第 3 条降级，这是最有效的反幻觉过滤器之一
        coerced.append(f"{path}: 缺少 evidence，降级为 unknown")
        a.state = "unknown"
        a.coerced = True
        a.confidence = 0.0
    return a


def _norm_occupant(raw: dict, coerced: list[str], idx: int) -> Occupant:
    seat_raw = str(raw.get("seat", "unknown")).strip().lower()
    seat = _SEAT_ALIASES.get(seat_raw, seat_raw if seat_raw in P.SEATS else "unknown")
    if seat != seat_raw and seat_raw not in P.SEATS:
        coerced.append(f"occupants[{idx}].seat={seat_raw!r} -> {seat}")
    age = str(raw.get("apparent_age_group", "unknown")).strip().lower()
    if age not in P.AGE_GROUPS:
        coerced.append(f"occupants[{idx}].apparent_age_group={age!r}")
        age = "unknown"
    present = raw.get("person_present", True)
    occ = Occupant(seat=seat, person_present=bool(present) if isinstance(present, bool) else str(present).lower() != "false",
                   apparent_age_group=age)
    for name, allowed in _ATTR_FIELDS.items():
        occ.attrs[name] = _norm_attr(raw.get(name), allowed, coerced, f"occupants[{idx}].{name}")
    return occ


def _norm_frame(raw: dict, coerced: list[str], idx: int) -> FrameObservation:
    scene = raw.get("scene") if isinstance(raw.get("scene"), dict) else {}
    view = str(scene.get("view", "unknown")).strip().lower()
    if view not in P.VIEWS:
        coerced.append(f"frames[{idx}].scene.view={view!r}")
        view = "unknown"
    quality = str(scene.get("image_quality", "unknown")).strip().lower()
    if quality not in P.IMAGE_QUALITY:
        coerced.append(f"frames[{idx}].scene.image_quality={quality!r}")
        quality = "unknown"
    try:
        persons = int(scene.get("persons_visible", 0))
    except (TypeError, ValueError):
        persons = 0

    raw_occ = raw.get("occupants")
    occupants: list[Occupant] = []
    if isinstance(raw_occ, list):
        for i, o in enumerate(raw_occ):
            if isinstance(o, dict):
                occ = _norm_occupant(o, coerced, i)
                if occ.person_present:
                    occupants.append(occ)
    return FrameObservation(view=view, image_quality=quality,
                            persons_visible=max(persons, len(occupants)),
                            occupants=occupants,
                            notes=str(raw.get("notes", "") or "").strip(),
                            frame_index=int(scene.get("frame_index", idx) or idx))


def parse_vlm_output(text: str, *, expected_frames: int = 1,
                     frame_timestamps: list[float] | None = None) -> ParseResult:
    """解析 VLM 回答。任何异常都收敛成 ``ok=False`` 的结果，不抛出。"""
    obj, repaired, err = load_json_lenient(text)
    if obj is None:
        return ParseResult(ok=False, error=err, raw_text=text)

    coerced: list[str] = []
    raw_frames: list[dict]
    if isinstance(obj, dict) and isinstance(obj.get("frames"), list):
        raw_frames = [f for f in obj["frames"] if isinstance(f, dict)]
    elif isinstance(obj, list):
        raw_frames = [f for f in obj if isinstance(f, dict)]
    elif isinstance(obj, dict):
        raw_frames = [obj]
    else:
        return ParseResult(ok=False, error="顶层结构既不是对象也不是数组", raw_text=text)

    if not raw_frames:
        return ParseResult(ok=False, error="未解析到任何帧", raw_text=text)

    frames = [_norm_frame(f, coerced, i) for i, f in enumerate(raw_frames)]
    if frame_timestamps:
        for f, ts in zip(frames, frame_timestamps):
            f.ts = ts
    if expected_frames > 1 and len(frames) != expected_frames:
        coerced.append(f"帧数不符：期望 {expected_frames}，实际 {len(frames)}")
    return ParseResult(frames=frames, ok=True, repaired=repaired,
                       coerced_fields=coerced, raw_text=text)
