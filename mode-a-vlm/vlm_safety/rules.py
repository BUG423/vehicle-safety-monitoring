"""视觉属性 -> 违规判定的确定性规则层。

**这一层刻意不交给大模型。** 理由有三：

  1. **可审计**：甲方问「为什么判我未系安全带」，答案必须是一条能写进合同的规则，
     而不是「模型认为」。这里每条判定都能回溯到具体属性 + 模型给出的文字依据。
  2. **可调**：不同车队对分心、抽烟的容忍口径不同，改阈值不需要重写 prompt、重测模型。
  3. **可测**：规则层是纯函数，能被单测覆盖；prompt 的行为没法单测。

同时这里守住模式A 的能力边界：
  - 超速 / 急加速：**只**来自 ``VehicleContext``（OBD/GPS），图像给不出，规则层也不去猜。
  - 疲劳：单帧只有「本帧是否闭眼」这种瞬时量，PERCLOS 必须跨帧算，见 ``evaluate_sequence``。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ._common import Severity, SubjectRole, VehicleContext, ViolationType
from .parser import FrameObservation, Occupant

# 驾驶位以外的座位
_PASSENGER_SEATS = {"front_passenger", "rear_left", "rear_middle", "rear_right"}


@dataclass
class Detection:
    """一次逐帧原始判定（尚未经过防误报确认）。"""

    violation: ViolationType
    hit: bool                      # 本帧是否命中
    confidence: float
    seat: str = "unknown"
    role: SubjectRole = SubjectRole.UNKNOWN
    evidence: str = ""             # 模型给出的可见依据（原文）
    source: str = "vision"         # vision | signal | fusion
    frame_index: int = 0
    ts: float = 0.0

    @property
    def key(self) -> str:
        """确认器里区分不同主体的 key —— 同一违规不同座位互不干扰。"""
        return f"{self.seat}"

    def to_dict(self) -> dict:
        return {"violation": self.violation.value, "hit": self.hit,
                "confidence": round(self.confidence, 3), "seat": self.seat,
                "role": self.role.value, "evidence": self.evidence,
                "source": self.source, "frame_index": self.frame_index}


@dataclass
class Undecidable:
    """明确「判不了」的项 —— 必须显式呈现，不能被当成「合规」。

    车队安全检查里最危险的不是误报，是把「看不清」悄悄当成「没问题」。
    """

    violation: ViolationType
    seat: str
    reason: str

    def to_dict(self) -> dict:
        return {"violation": self.violation.value, "seat": self.seat, "reason": self.reason}


@dataclass
class RuleConfig:
    """判定口径 —— 甲方可调。"""

    seat_capacity: int = 5              # 核载人数，用于超员判定
    min_confidence: float = 0.55        # 低于该置信度的视觉结论不参与判定
    distraction_gaze_states: tuple[str, ...] = ("down", "left", "right")
    # 疲劳：PERCLOS（闭眼帧占比）阈值与所需最短观察窗
    perclos_threshold: float = 0.4
    perclos_min_window_s: float = 6.0
    perclos_min_frames: int = 6
    yawn_ratio_threshold: float = 0.3
    # 只有车辆处于可行驶状态时才判「双手脱离方向盘/分心」
    require_moving_for_driving_rules: bool = True


@dataclass
class RuleOutput:
    detections: list[Detection] = field(default_factory=list)
    undecidable: list[Undecidable] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _conf_ok(conf: float, cfg: RuleConfig) -> bool:
    return conf >= cfg.min_confidence


def _role_for(seat: str) -> SubjectRole:
    if seat == "driver":
        return SubjectRole.DRIVER
    if seat in _PASSENGER_SEATS:
        return SubjectRole.PASSENGER
    return SubjectRole.UNKNOWN


def _seatbelt_rules(occ: Occupant, cfg: RuleConfig, out: RuleOutput, frame: FrameObservation,
                    ctx: VehicleContext | None) -> None:
    a = occ.attr("seatbelt")
    is_driver = occ.seat == "driver"
    vt = ViolationType.DRIVER_NO_SEATBELT if is_driver else ViolationType.PASSENGER_NO_SEATBELT
    if occ.seat == "unknown":
        out.undecidable.append(Undecidable(vt, occ.seat, "座位无法确定，安全带归属不明"))
        return
    if a.state == "not_fastened" and _conf_ok(a.confidence, cfg):
        ev = a.evidence
        # 与车身总线的安全带扣合开关交叉校验：
        # 开关说「已扣」但画面看不到织带 —— 这正是「插扣欺骗」（把插扣插好、人坐在带子外面）
        # 的典型特征，是视觉相对开关信号的核心增量价值。
        if ctx and ctx.seatbelt_switch.get(occ.seat) is True:
            ev += "；车身开关显示已扣合但画面无织带，疑似插扣欺骗"
        out.detections.append(Detection(vt, True, a.confidence, occ.seat, _role_for(occ.seat), ev,
                                        frame_index=frame.frame_index, ts=frame.ts))
    elif a.state == "fastened":
        out.detections.append(Detection(vt, False, a.confidence, occ.seat, _role_for(occ.seat),
                                        a.evidence, frame_index=frame.frame_index, ts=frame.ts))
    else:
        out.undecidable.append(Undecidable(vt, occ.seat,
                                           f"安全带状态 {a.state}（置信度 {a.confidence:.2f}），画面不足以判定"))


def _driver_rules(occ: Occupant, cfg: RuleConfig, out: RuleOutput, frame: FrameObservation,
                  ctx: VehicleContext | None) -> None:
    """仅对驾驶位生效的规则。"""
    moving = bool(ctx and ctx.is_moving)
    driving_applicable = moving or not cfg.require_moving_for_driving_rules

    phone = occ.attr("phone")
    if phone.state in ("held_to_ear", "held_in_view") and _conf_ok(phone.confidence, cfg):
        out.detections.append(Detection(ViolationType.DRIVER_PHONE_USE, True, phone.confidence,
                                        occ.seat, SubjectRole.DRIVER, phone.evidence,
                                        frame_index=frame.frame_index, ts=frame.ts))
    elif phone.state in ("not_visible", "on_lap_or_mount"):
        out.detections.append(Detection(ViolationType.DRIVER_PHONE_USE, False, phone.confidence,
                                        occ.seat, SubjectRole.DRIVER, phone.evidence,
                                        frame_index=frame.frame_index, ts=frame.ts))

    smoke = occ.attr("smoking")
    if smoke.state == "cigarette_visible" and _conf_ok(smoke.confidence, cfg):
        out.detections.append(Detection(ViolationType.DRIVER_SMOKING, True, smoke.confidence,
                                        occ.seat, SubjectRole.DRIVER, smoke.evidence,
                                        frame_index=frame.frame_index, ts=frame.ts))
    elif smoke.state == "not_visible":
        out.detections.append(Detection(ViolationType.DRIVER_SMOKING, False, smoke.confidence,
                                        occ.seat, SubjectRole.DRIVER, smoke.evidence,
                                        frame_index=frame.frame_index, ts=frame.ts))

    hands = occ.attr("hands")
    if hands.state == "none_on_wheel" and _conf_ok(hands.confidence, cfg):
        if driving_applicable:
            out.detections.append(Detection(ViolationType.DRIVER_HANDS_OFF_WHEEL, True,
                                            hands.confidence, occ.seat, SubjectRole.DRIVER,
                                            hands.evidence, frame_index=frame.frame_index, ts=frame.ts))
        else:
            out.notes.append("双手脱离方向盘：车辆未在行驶（或无车速信号），按静态检查不计违规")

    gaze = occ.attr("gaze")
    if gaze.state in cfg.distraction_gaze_states and _conf_ok(gaze.confidence, cfg):
        if driving_applicable:
            out.detections.append(Detection(ViolationType.DRIVER_DISTRACTION, True, gaze.confidence,
                                            occ.seat, SubjectRole.DRIVER, gaze.evidence,
                                            frame_index=frame.frame_index, ts=frame.ts))
        else:
            out.notes.append("视线偏离：车辆未在行驶，分心判定不适用")
    elif gaze.state == "forward":
        out.detections.append(Detection(ViolationType.DRIVER_DISTRACTION, False, gaze.confidence,
                                        occ.seat, SubjectRole.DRIVER, gaze.evidence,
                                        frame_index=frame.frame_index, ts=frame.ts))


def evaluate_frame(frame: FrameObservation, *, cfg: RuleConfig | None = None,
                   ctx: VehicleContext | None = None) -> RuleOutput:
    """单帧规则判定。不含任何需要时序的结论。"""
    cfg = cfg or RuleConfig()
    out = RuleOutput()

    # --- 画面本身是否可用 ---
    if frame.image_quality == "lens_blocked":
        out.detections.append(Detection(ViolationType.SYSTEM_CAMERA_BLOCKED, True, 0.9,
                                        "unknown", SubjectRole.VEHICLE,
                                        frame.notes or "画面被遮挡", "vision",
                                        frame.frame_index, frame.ts))
        return out
    if frame.view == "not_a_vehicle_cabin":
        out.notes.append("输入不是车内画面，跳过全部安全检查")
        return out
    if frame.image_quality in ("dark", "blurry", "overexposed"):
        out.notes.append(f"图像质量 {frame.image_quality}，判定置信度整体下调")

    # --- 驾驶位有没有人 ---
    driver = frame.by_seat("driver")
    if driver is None:
        if frame.view in ("cabin_front", "unknown"):
            engine_on = bool(ctx and ctx.engine_on)
            out.detections.append(Detection(ViolationType.DRIVER_ABSENT, True, 0.7, "driver",
                                            SubjectRole.DRIVER,
                                            "画面中驾驶位未见人员" + ("（发动机已启动）" if engine_on else ""),
                                            frame_index=frame.frame_index, ts=frame.ts))

    # --- 逐乘员 ---
    for occ in frame.occupants:
        _seatbelt_rules(occ, cfg, out, frame, ctx)
        if occ.seat == "driver":
            _driver_rules(occ, cfg, out, frame, ctx)
        if occ.seat == "front_passenger" and occ.apparent_age_group == "child":
            out.detections.append(Detection(ViolationType.PASSENGER_CHILD_FRONT_SEAT, True, 0.75,
                                            occ.seat, SubjectRole.PASSENGER,
                                            "副驾乘员体型/面部特征判断为儿童",
                                            frame_index=frame.frame_index, ts=frame.ts))

    # --- 超员 ---
    if frame.persons_visible > cfg.seat_capacity:
        out.detections.append(Detection(ViolationType.PASSENGER_OVERLOAD, True, 0.7, "unknown",
                                        SubjectRole.PASSENGER,
                                        f"画面可见 {frame.persons_visible} 人，核载 {cfg.seat_capacity} 人",
                                        frame_index=frame.frame_index, ts=frame.ts))

    # --- 单帧判不了的：疲劳 ---
    if driver is not None:
        out.undecidable.append(Undecidable(
            ViolationType.DRIVER_FATIGUE, "driver",
            "疲劳需要 PERCLOS 等时序指标，单帧只能给出瞬时闭眼/哈欠状态，不构成判定"))

    return out


# ---------------------------------------------------------------------------
# 时序：PERCLOS
# ---------------------------------------------------------------------------
@dataclass
class PerclosResult:
    perclos: float
    closed_frames: int
    valid_frames: int
    yawn_ratio: float
    window_s: float
    sufficient: bool          # 观察窗是否够长、样本是否够多
    reason: str = ""

    def to_dict(self) -> dict:
        return {"perclos": round(self.perclos, 3), "closed_frames": self.closed_frames,
                "valid_frames": self.valid_frames, "yawn_ratio": round(self.yawn_ratio, 3),
                "window_s": round(self.window_s, 2), "sufficient": self.sufficient,
                "reason": self.reason}


def compute_perclos(frames: list[FrameObservation], *, seat: str = "driver",
                    cfg: RuleConfig | None = None) -> PerclosResult:
    """按 PERCLOS 口径统计闭眼帧占比。

    PERCLOS(P80) 的工业定义是「单位时间内眼睑遮盖瞳孔超过 80% 的时间占比」。
    VLM 给不出眼睑开度百分比，只能给 open/partially_closed/closed 三档，
    因此这里用 closed + 0.5×partially_closed 作为近似 —— 这是**近似而非等价**，
    必须在报告里写明（见 DESIGN.md「疲劳判定怎么办」）。
    """
    cfg = cfg or RuleConfig()
    closed = valid = yawns = mouth_valid = 0
    for f in frames:
        occ = f.by_seat(seat)
        if occ is None:
            continue
        eyes = occ.attr("eyes")
        if eyes.state == "closed":
            closed += 1
            valid += 1
        elif eyes.state == "partially_closed":
            closed += 0.5
            valid += 1
        elif eyes.state == "open":
            valid += 1
        mouth = occ.attr("mouth")
        if mouth.state in ("normal", "yawning"):
            mouth_valid += 1
            if mouth.state == "yawning":
                yawns += 1

    ts = [f.ts for f in frames if f.ts]
    window = (max(ts) - min(ts)) if len(ts) >= 2 else 0.0
    perclos = (closed / valid) if valid else 0.0
    yawn_ratio = (yawns / mouth_valid) if mouth_valid else 0.0

    reason = ""
    sufficient = True
    if valid < cfg.perclos_min_frames:
        sufficient, reason = False, f"有效帧仅 {valid} 帧（需 ≥{cfg.perclos_min_frames}）"
    elif window < cfg.perclos_min_window_s:
        sufficient, reason = False, f"观察窗仅 {window:.1f}s（需 ≥{cfg.perclos_min_window_s}s）"
    return PerclosResult(perclos=perclos, closed_frames=int(closed), valid_frames=valid,
                         yawn_ratio=yawn_ratio, window_s=window, sufficient=sufficient,
                         reason=reason)


def evaluate_sequence(frames: list[FrameObservation], *, cfg: RuleConfig | None = None,
                      ctx: VehicleContext | None = None) -> tuple[list[RuleOutput], PerclosResult | None]:
    """多帧序列判定：逐帧规则 + 跨帧 PERCLOS 疲劳判定。"""
    cfg = cfg or RuleConfig()
    per_frame = [evaluate_frame(f, cfg=cfg, ctx=ctx) for f in frames]

    perclos = compute_perclos(frames, cfg=cfg)
    has_driver = any(f.by_seat("driver") for f in frames)
    if not has_driver:
        return per_frame, None

    last = per_frame[-1]
    # 把单帧阶段登记的「疲劳判不了」换成真正的时序结论
    last.undecidable = [u for u in last.undecidable if u.violation is not ViolationType.DRIVER_FATIGUE]
    for out in per_frame[:-1]:
        out.undecidable = [u for u in out.undecidable if u.violation is not ViolationType.DRIVER_FATIGUE]

    if not perclos.sufficient:
        last.undecidable.append(Undecidable(ViolationType.DRIVER_FATIGUE, "driver",
                                            f"时序样本不足：{perclos.reason}"))
        return per_frame, perclos

    # 疲劳判定**逐帧**下发给确认器，而不是把整段窗口打包成一条聚合结论。
    #
    # 这一点踩过坑：最早的实现把 PERCLOS 算完只在最后一帧产出一条 Detection，
    # 结果 ViolationConfirmer 只收到 1 个样本，duration 恒为 0，永远达不到
    # min_duration_s，疲劳事件一条也发不出来。
    #
    # 正确做法是让契约层的滑窗投票自己去积累时间：逐帧喂「本帧是否闭眼」，
    # 由 window_s / hit_ratio / min_duration_s 决定何时确认。这样模式A 与
    # 模式B/C 用的是同一套确认口径，三条路线的告警延迟才可比。
    # PERCLOS 则退居为**证据与诊断量**，写进事件的 raw_signals 供后台复核。
    evidence_tpl = (f"PERCLOS={perclos.perclos:.2f}（{perclos.closed_frames}/{perclos.valid_frames} "
                    f"帧闭眼，窗口 {perclos.window_s:.1f}s），哈欠帧占比 {perclos.yawn_ratio:.2f}")
    for frame, out in zip(frames, per_frame):
        occ = frame.by_seat(seat_key := "driver")
        if occ is None:
            continue
        eyes = occ.attr("eyes")
        if eyes.state not in ("open", "closed", "partially_closed"):
            continue        # 看不清就不投票，避免把「不确定」稀释成「正常」
        hit = eyes.state in ("closed", "partially_closed")
        out.detections.append(Detection(
            ViolationType.DRIVER_FATIGUE, hit,
            min(0.95, max(0.5, eyes.confidence)), seat_key, SubjectRole.DRIVER,
            f"本帧眼睛={eyes.state}；{eyes.evidence}；窗口统计 {evidence_tpl}",
            source="fusion", frame_index=frame.frame_index, ts=frame.ts))
    return per_frame, perclos


# ---------------------------------------------------------------------------
# 非视觉信号：超速 / 急加速。图像永远给不出这两个结论。
# ---------------------------------------------------------------------------
def evaluate_vehicle_signals(ctx: VehicleContext | None) -> list[Detection]:
    """从 OBD/GPS 上下文判定车辆状态类违规。

    模式A 的能力边界在这里被显式固化：这些 Detection 的 ``source`` 是 ``signal``，
    与 VLM 无关。没有 ``VehicleContext`` 就没有超速结论 —— 不允许从图像"推测"。
    """
    out: list[Detection] = []
    if ctx is None:
        return out
    if ctx.speed_kmh is not None and ctx.speed_limit_kmh:
        over = ctx.speed_kmh - ctx.speed_limit_kmh
        if over > 0:
            out.append(Detection(ViolationType.VEHICLE_SPEEDING, True,
                                 min(0.99, 0.7 + over / 100), "unknown", SubjectRole.VEHICLE,
                                 f"OBD/GPS 车速 {ctx.speed_kmh:.0f} km/h，路段限速 "
                                 f"{ctx.speed_limit_kmh:.0f} km/h，超出 {over:.0f} km/h",
                                 source="signal", ts=ctx.ts))
    return out
