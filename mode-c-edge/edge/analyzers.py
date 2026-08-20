"""违规判定规则 —— 把感知层的标量变成「这一帧算不算违规」的布尔值。

刻意与感知层、确认层都解耦：
  感知层给「看见什么」→ 本模块给「这一帧像不像违规」→ 确认层给「要不要告警」。

两条贯穿全模块的设计：

1. **一律相对个体基线，不用绝对阈值。**
   眼睛开合度的绝对值随人（单/双眼皮）、随摄像头安装角度、随模型版本变化，
   写死阈值必然在换车或换人后失效。这里统一用 `AdaptiveBaseline`：
   开机后几秒内标定该乘员的睁眼基线，之后按比例判闭眼。

2. **有车身总线信号时，总线优先。**
   安全带扣合是 CAN 上的一个开关量，比任何视觉算法都准。视觉只在
   拿不到该信号的车上兜底，且两者冲突时以总线为准（见 `SeatbeltRule`）。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from common import Severity, SubjectRole, VehicleContext, ViolationType

from .perception import DRIVER, Perception, SeatObs


@dataclass
class Hit:
    """一帧上对某个（违规类型, 主体）的原始判定。"""

    violation: ViolationType
    key: str                      # 确认器的分组键，一般用座位名
    hit: bool
    confidence: float = 1.0
    seat: str | None = None
    role: SubjectRole = SubjectRole.UNKNOWN
    signals: dict = field(default_factory=dict)   # 写进事件的 raw_signals，用于事后归因


class AdaptiveBaseline:
    """滑动的个体基线：用高分位数代表「正常睁眼」的水平。

    用分位数而不是均值，是因为闭眼样本会把均值拉低，越困基线越低，
    最后变成「越困越判不出困」—— 这是 PERCLOS 实现里最常见的坑。
    """

    def __init__(self, warmup_s: float, window_s: float = 60.0, q: float = 0.85) -> None:
        self.warmup_s = warmup_s
        self.window_s = window_s
        self.q = q
        self._buf: Deque[tuple[float, float]] = deque()
        self._t0: float | None = None

    def update(self, value: float, now: float) -> None:
        if self._t0 is None:
            self._t0 = now
        self._buf.append((now, value))
        while self._buf and now - self._buf[0][0] > self.window_s:
            self._buf.popleft()

    @property
    def ready(self) -> bool:
        return (self._t0 is not None and self._buf
                and self._buf[-1][0] - self._t0 >= self.warmup_s)

    def value(self) -> float | None:
        if not self._buf:
            return None
        vals = sorted(v for _, v in self._buf)
        idx = min(len(vals) - 1, int(self.q * (len(vals) - 1)))
        return vals[idx]


class PerclosTracker:
    """PERCLOS：统计窗口内「眼睛闭合」的时间占比。

    行业通用的疲劳指标（PERCLOS P80）。闭眼判据用相对基线的比例，
    窗口内占比超过阈值即认为处于疲劳态。
    """

    def __init__(self, window_s: float, close_ratio: float, warmup_s: float) -> None:
        self.window_s = window_s
        self.close_ratio = close_ratio
        self.baseline = AdaptiveBaseline(warmup_s=warmup_s, window_s=max(window_s, 60.0))
        self._buf: Deque[tuple[float, bool]] = deque()
        self._closed_since: float | None = None
        self.closed_run_s = 0.0          # 当前这次连续闭眼已持续多久

    def update(self, eye_open: float, now: float) -> tuple[float, bool]:
        """返回 (PERCLOS, 本帧是否闭眼)，并维护连续闭眼时长 `closed_run_s`。"""
        self.baseline.update(eye_open, now)
        base = self.baseline.value()
        closed = bool(base is not None and self.baseline.ready
                      and eye_open < base * self.close_ratio)
        if closed:
            if self._closed_since is None:
                self._closed_since = now
            self.closed_run_s = now - self._closed_since
        else:
            self._closed_since = None
            self.closed_run_s = 0.0
        self._buf.append((now, closed))
        while self._buf and now - self._buf[0][0] > self.window_s:
            self._buf.popleft()
        perclos = sum(1 for _, c in self._buf if c) / len(self._buf) if self._buf else 0.0
        return perclos, closed


class SeatState:
    """一个座位的时序状态。"""

    def __init__(self, cfg) -> None:
        self.perclos = PerclosTracker(cfg.perclos_window_s, cfg.eye_close_ratio,
                                      cfg.baseline_warmup_s)
        self.mouth = AdaptiveBaseline(warmup_s=cfg.baseline_warmup_s)
        self.belt = AdaptiveBaseline(warmup_s=cfg.baseline_warmup_s, q=0.9)


class RuleEngine:
    """把一帧 `Perception` + `VehicleContext` 变成一组 `Hit`。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._seats: dict[str, SeatState] = {}
        self._last_speed: tuple[float, float] | None = None   # (ts, speed_kmh)
        self._sharp = AdaptiveBaseline(warmup_s=cfg.baseline_warmup_s, window_s=60.0, q=0.85)

    def _state(self, seat: str) -> SeatState:
        return self._seats.setdefault(seat, SeatState(self.cfg))

    # ------------------------------------------------------------------
    def evaluate(self, p: Perception, ctx: VehicleContext | None) -> list[Hit]:
        hits: list[Hit] = []
        now = p.ts

        # ---- 镜头遮挡/失效：先判它，遮挡时其他视觉结论都不可信 ----
        # 判据是「清晰度相对自身历史基线塌陷」+「画面里一个人都没有」，
        # 而不是拿绝对方差跟固定阈值比 —— 后者会把素色背景的正常画面判成遮挡（实测踩过）。
        self._sharp.update(p.sharpness, now)
        base = self._sharp.value()
        anyone = any(o.present for o in p.seats.values())
        blocked = bool(
            p.sharpness < self.cfg.blur_abs_floor
            or (self._sharp.ready and base and not anyone
                and p.sharpness < base * self.cfg.blur_rel_ratio))
        hits.append(Hit(ViolationType.SYSTEM_CAMERA_BLOCKED, key="camera", hit=blocked,
                        confidence=1.0, role=SubjectRole.VEHICLE,
                        signals={"sharpness": round(p.sharpness, 2),
                                 "sharpness_baseline": round(base, 2) if base else None,
                                 "anyone_present": anyone}))

        for seat, obs in p.seats.items():
            hits.extend(self._seat_hits(seat, obs, now, blocked, ctx))

        # ---- 驾驶位无人（发动机已启动时才有意义）----
        driver = p.seats.get(DRIVER)
        if ctx is not None and ctx.engine_on:
            hits.append(Hit(ViolationType.DRIVER_ABSENT, key=DRIVER,
                            hit=not (driver and driver.present), confidence=0.9,
                            seat=DRIVER, role=SubjectRole.DRIVER,
                            signals={"engine_on": True}))

        hits.extend(self._vehicle_hits(ctx, now))
        return hits

    # ------------------------------------------------------------------
    def _seat_hits(self, seat: str, obs: SeatObs, now: float, blocked: bool,
                   ctx: VehicleContext | None) -> list[Hit]:
        cfg = self.cfg
        st = self._state(seat)
        is_driver = seat == DRIVER
        role = SubjectRole.DRIVER if is_driver else SubjectRole.PASSENGER
        out: list[Hit] = []

        if not obs.present or blocked:
            # 人不在或看不清：不产出违规判定，但也不能当成「合规」——
            # 交给确认器的滑窗自然衰减，避免中途遮挡导致误报或漏报翻转。
            return out

        # ---- 安全带 ----
        belt_hit, belt_sig = self._seatbelt(seat, obs, st, ctx, now)
        if belt_hit is not None:
            v = ViolationType.DRIVER_NO_SEATBELT if is_driver else ViolationType.PASSENGER_NO_SEATBELT
            out.append(Hit(v, key=seat, hit=belt_hit, confidence=belt_sig.pop("_conf", 0.8),
                           seat=seat, role=role, signals=belt_sig))

        if not is_driver:
            return out    # 疲劳/分心/手机只对驾驶员判定

        # ---- 疲劳：PERCLOS + 哈欠 ----
        if obs.eye_open is not None:
            perclos, closed = st.perclos.update(obs.eye_open, now)
            yawn = False
            if obs.mouth_open is not None:
                st.mouth.update(obs.mouth_open, now)
                mb = st.mouth.value()
                yawn = bool(mb is not None and st.mouth.ready and obs.mouth_open > mb / max(cfg.yawn_ratio, 1e-3))
            # 三条互补的疲劳判据：
            #   微睡眠  —— 连续闭眼超过 micro_sleep_s，最危险也最该秒级报警
            #   PERCLOS —— 窗口内闭眼时间占比，反映累积困倦
            #   闭眼+哈欠 —— 组合证据，降低单一指标的误报
            micro = st.perclos.closed_run_s >= cfg.micro_sleep_s
            fatigued = micro or perclos >= cfg.perclos_thr or (closed and yawn)
            out.append(Hit(ViolationType.DRIVER_FATIGUE, key=seat, hit=fatigued,
                           confidence=min(1.0, 0.5 + max(perclos, 0.4 if micro else 0.0)),
                           seat=seat, role=role,
                           signals={"perclos": round(perclos, 3),
                                    "closed_run_s": round(st.perclos.closed_run_s, 2),
                                    "micro_sleep": micro,
                                    "eye_open": round(obs.eye_open, 4),
                                    "eye_baseline": (round(st.perclos.baseline.value(), 4)
                                                     if st.perclos.baseline.value() else None),
                                    "closed": closed, "yawn": yawn}))

        # ---- 分心：头部姿态偏离前方 ----
        if obs.yaw_deg is not None and obs.pitch_deg is not None:
            off = abs(obs.yaw_deg) > cfg.yaw_thr_deg or abs(obs.pitch_deg) > cfg.pitch_thr_deg
            out.append(Hit(ViolationType.DRIVER_DISTRACTION, key=seat, hit=off, confidence=0.8,
                           seat=seat, role=role,
                           signals={"yaw_deg": round(obs.yaw_deg, 1),
                                    "pitch_deg": round(obs.pitch_deg, 1)}))

        # ---- 手机 ----
        if obs.phone_score is not None:
            out.append(Hit(ViolationType.DRIVER_PHONE_USE, key=seat,
                           hit=obs.phone_score >= cfg.phone_score_thr,
                           confidence=min(1.0, 0.5 + 0.5 * obs.phone_score),
                           seat=seat, role=role,
                           signals={"phone_score": round(obs.phone_score, 4)}))

        # ---- 抽烟：当前没有可用的开源香烟检测模型，不产出判定 ----
        if obs.smoke_score is not None:
            out.append(Hit(ViolationType.DRIVER_SMOKING, key=seat,
                           hit=obs.smoke_score >= 0.5, confidence=obs.smoke_score,
                           seat=seat, role=role,
                           signals={"smoke_score": round(obs.smoke_score, 4)}))
        return out

    # ------------------------------------------------------------------
    def _seatbelt(self, seat: str, obs: SeatObs, st: SeatState,
                  ctx: VehicleContext | None, now: float):
        """总线信号优先，视觉兜底。返回 (是否未系, 证据字典) 或 (None, {})。"""
        sig: dict = {}
        can = None
        if ctx is not None and ctx.seatbelt_switch:
            can = ctx.seatbelt_switch.get(seat)
        if can is not None:
            sig.update({"source": "can_bus", "seatbelt_switch": bool(can), "_conf": 0.99})
            if obs.belt_score is not None:
                sig["belt_score_vision"] = round(obs.belt_score, 3)
            return (not can), sig

        if obs.belt_score is None:
            return None, {}
        st.belt.update(obs.belt_score, now)
        base = st.belt.value()
        # 视觉基线：以该乘员「系着时」的带体线强度为参照，低于其一半判未系
        thr = max(0.25, (base or 0.0) * 0.5)
        sig.update({"source": "vision", "belt_score": round(obs.belt_score, 3),
                    "belt_threshold": round(thr, 3), "_conf": 0.75})
        return obs.belt_score < thr, sig

    # ------------------------------------------------------------------
    def _vehicle_hits(self, ctx: VehicleContext | None, now: float) -> list[Hit]:
        if ctx is None:
            return []
        out: list[Hit] = []
        if ctx.speed_kmh is not None and ctx.speed_limit_kmh:
            over = ctx.speed_kmh > ctx.speed_limit_kmh * self.cfg.speed_tolerance
            out.append(Hit(ViolationType.VEHICLE_SPEEDING, key="vehicle", hit=over,
                           confidence=0.99, role=SubjectRole.VEHICLE,
                           signals={"speed_kmh": round(ctx.speed_kmh, 1),
                                    "speed_limit_kmh": ctx.speed_limit_kmh,
                                    "source": "obd_can"}))
        if ctx.speed_kmh is not None:
            harsh = False
            accel = 0.0
            if self._last_speed is not None:
                dt = now - self._last_speed[0]
                if dt > 1e-3:
                    accel = (ctx.speed_kmh - self._last_speed[1]) / 3.6 / dt
                    harsh = abs(accel) > self.cfg.harsh_accel_thr
            self._last_speed = (now, ctx.speed_kmh)
            out.append(Hit(ViolationType.VEHICLE_HARSH_DRIVING, key="vehicle", hit=harsh,
                           confidence=0.9, role=SubjectRole.VEHICLE,
                           signals={"accel_mps2": round(accel, 2), "source": "obd_can"}))
        return out
