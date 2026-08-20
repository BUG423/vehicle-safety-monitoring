"""违规确认状态机 —— 把「逐帧原始判定」收敛为「可告警的事件」。

单帧模型输出直接告警会产生大量误报（眨眼被判疲劳、手抬起被判打电话），
车载场景下误报比漏报更致命：司机会关掉系统。

本模块提供三种模式共用的确认逻辑：
  1. 滑动窗口投票：窗口内命中率超过阈值才算「进入违规态」；
  2. 最短持续时长：违规态需保持 min_duration_s 才首次上报；
  3. 冷却与升级：同一违规在 cooldown_s 内不重复上报；持续不改正则升级严重等级；
  4. 恢复确认：连续 release_ratio 以下才判定「已恢复」，避免状态抖动。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

try:
    from .violation_types import Severity, ViolationType
except ImportError:  # pragma: no cover
    from violation_types import Severity, ViolationType


@dataclass
class ConfirmRule:
    """单个违规类型的确认规则。"""

    window_s: float = 3.0          # 滑动窗口长度（秒）
    hit_ratio: float = 0.6         # 窗口内命中比例阈值
    min_duration_s: float = 2.0    # 首次上报所需最短持续时长
    cooldown_s: float = 30.0       # 同类事件重复上报的冷却时间
    escalate_after_s: float | None = 60.0   # 持续超过该时长升级为 CRITICAL
    release_ratio: float = 0.2     # 命中比例低于该值判定为恢复


# 按违规类型定制：疲劳需要长窗口，未系安全带可较快确认，超速由信号直接判定
DEFAULT_RULES: dict[ViolationType, ConfirmRule] = {
    ViolationType.DRIVER_FATIGUE: ConfirmRule(window_s=8.0, hit_ratio=0.5, min_duration_s=4.0,
                                              cooldown_s=60.0, escalate_after_s=30.0),
    ViolationType.DRIVER_DISTRACTION: ConfirmRule(window_s=4.0, hit_ratio=0.7, min_duration_s=3.0,
                                                  cooldown_s=45.0),
    ViolationType.DRIVER_NO_SEATBELT: ConfirmRule(window_s=3.0, hit_ratio=0.7, min_duration_s=2.0,
                                                  cooldown_s=60.0, escalate_after_s=20.0),
    ViolationType.PASSENGER_NO_SEATBELT: ConfirmRule(window_s=3.0, hit_ratio=0.7, min_duration_s=2.0,
                                                     cooldown_s=90.0),
    ViolationType.DRIVER_PHONE_USE: ConfirmRule(window_s=4.0, hit_ratio=0.6, min_duration_s=2.5,
                                                cooldown_s=45.0),
    ViolationType.DRIVER_SMOKING: ConfirmRule(window_s=6.0, hit_ratio=0.5, min_duration_s=4.0,
                                              cooldown_s=120.0),
    ViolationType.VEHICLE_SPEEDING: ConfirmRule(window_s=5.0, hit_ratio=0.8, min_duration_s=3.0,
                                                cooldown_s=30.0, escalate_after_s=15.0),
}
FALLBACK_RULE = ConfirmRule()


@dataclass
class _TrackState:
    samples: Deque[tuple[float, bool]] = field(default_factory=deque)  # (ts, hit)
    active_since: float | None = None
    last_report_ts: float | None = None
    reported: bool = False


@dataclass
class Confirmation:
    """确认器的一次输出。"""

    violation: ViolationType
    should_alert: bool          # 是否应触发一次新告警
    active: bool                # 当前是否处于违规态
    duration_s: float
    confidence: float
    severity: Severity
    released: bool = False      # 本次是否从违规态恢复


class ViolationConfirmer:
    """多违规类型、多主体（按 key 区分座位/track）的确认器。

    典型用法::

        confirmer = ViolationConfirmer()
        for frame in stream:
            hit = model.predict(frame)          # 逐帧原始判定
            c = confirmer.update(ViolationType.DRIVER_FATIGUE, hit.is_violating,
                                 confidence=hit.score, key="driver")
            if c.should_alert:
                alerter.dispatch(build_event(c))
    """

    def __init__(self, rules: dict[ViolationType, ConfirmRule] | None = None) -> None:
        self._rules = {**DEFAULT_RULES, **(rules or {})}
        self._states: dict[tuple[ViolationType, str], _TrackState] = {}

    def rule_for(self, violation: ViolationType) -> ConfirmRule:
        return self._rules.get(violation, FALLBACK_RULE)

    def update(
        self,
        violation: ViolationType,
        hit: bool,
        *,
        confidence: float = 1.0,
        key: str = "default",
        now: float | None = None,
    ) -> Confirmation:
        now = time.time() if now is None else now
        rule = self.rule_for(violation)
        state = self._states.setdefault((violation, key), _TrackState())

        state.samples.append((now, hit))
        while state.samples and now - state.samples[0][0] > rule.window_s:
            state.samples.popleft()

        total = len(state.samples)
        hits = sum(1 for _, h in state.samples if h)
        ratio = hits / total if total else 0.0

        released = False
        if ratio >= rule.hit_ratio:
            if state.active_since is None:
                state.active_since = now
        elif ratio <= rule.release_ratio:
            if state.active_since is not None:
                released = True
            state.active_since = None
            state.reported = False

        active = state.active_since is not None
        duration = now - state.active_since if active else 0.0

        severity = violation.default_severity
        if active and rule.escalate_after_s and duration >= rule.escalate_after_s:
            severity = Severity.CRITICAL

        should_alert = False
        if active and duration >= rule.min_duration_s:
            cooled = state.last_report_ts is None or (now - state.last_report_ts) >= rule.cooldown_s
            if not state.reported or cooled:
                should_alert = True
                state.reported = True
                state.last_report_ts = now

        return Confirmation(
            violation=violation,
            should_alert=should_alert,
            active=active,
            duration_s=duration,
            confidence=round(ratio * confidence, 4) if active else round(confidence, 4),
            severity=severity,
            released=released,
        )

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._states.clear()
        else:
            for k in [k for k in self._states if k[1] == key]:
                del self._states[k]
