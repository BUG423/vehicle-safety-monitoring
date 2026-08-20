"""告警链路装配 —— 直接复用 ``common/alerting``，不另造轮子。

模式A 在契约层之上只补了两件与本路线相关的事：
  1. 车内提醒需要能被 H5 页面消费（而不是打印到 stdout），因此把 CabinPrompt 收进内存环形缓冲，
     由 ``/api/alerts`` 吐给前端；
  2. 后台上报支持 Webhook（``VSM_BACKEND_WEBHOOK``），未配置时退化为记录到内存 + stdout，
     保证无外部依赖也能演示完整链路。
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ._common import (
    AlertDispatcher, BackendChannel, CabinPrompt, InCabinChannel, SafetyEvent, Severity,
)
from .config import Settings


@dataclass
class AlertRecord:
    """一条告警流水（车内或后台），供演示页展示。"""

    channel: str
    ts: float
    severity: str
    violation: str
    text: str
    delivered: bool = True
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"channel": self.channel, "ts": self.ts, "severity": self.severity,
                "violation": self.violation, "text": self.text,
                "delivered": self.delivered, "detail": self.detail}


class AlertRecorder:
    """内存环形缓冲，记录两条通道的全部投递结果。"""

    def __init__(self, maxlen: int = 200) -> None:
        self.records: deque[AlertRecord] = deque(maxlen=maxlen)

    def add(self, rec: AlertRecord) -> None:
        self.records.append(rec)

    def recent(self, limit: int = 50) -> list[dict]:
        return [r.to_dict() for r in list(self.records)[-limit:]][::-1]

    def clear(self) -> None:
        self.records.clear()


def build_dispatcher(settings: Settings, recorder: AlertRecorder | None = None
                     ) -> tuple[AlertDispatcher, AlertRecorder]:
    """按配置组装「车内 + 后台」双通道分发器。"""
    recorder = recorder or AlertRecorder()
    min_sev = {"info": Severity.INFO, "warn": Severity.WARN,
               "critical": Severity.CRITICAL}.get(settings.cabin_min_severity, Severity.WARN)

    _pending: dict[str, SafetyEvent] = {}

    def cabin_sink(prompt: CabinPrompt) -> None:
        ev = _pending.get("current")
        recorder.add(AlertRecord(
            channel="in_cabin", ts=time.time(), severity=prompt.severity,
            violation=ev.violation.value if ev else "", text=prompt.text,
            detail=prompt.to_dict()))
        print(f"[车内提醒][{prompt.severity}] {prompt.text} "
              f"(重复{prompt.repeat}次 蜂鸣={prompt.beep})")

    def backend_sender(event: SafetyEvent) -> bool:
        payload = event.to_json(drop_b64=not settings.attach_thumbnail)
        ok = True
        detail: dict[str, Any] = {"target": settings.backend_webhook or "stdout"}
        if settings.backend_webhook:
            try:
                import requests

                r = requests.post(settings.backend_webhook,
                                  data=payload.encode("utf-8"),
                                  headers={"Content-Type": "application/json"}, timeout=5)
                ok = r.status_code < 400
                detail["status"] = r.status_code
            except Exception as exc:  # noqa: BLE001 - 断网属常态，交给通道落盘补传
                ok = False
                detail["error"] = f"{type(exc).__name__}: {exc}"
        else:
            print(f"[后台告警] {event.to_json(drop_b64=True)}")
        recorder.add(AlertRecord(
            channel="backend", ts=time.time(), severity=event.severity.value,
            violation=event.violation.value, text=event.message,
            delivered=ok, detail=detail))
        return ok

    cabin = InCabinChannel(sink=cabin_sink, min_severity=min_sev)
    backend = BackendChannel(sender=backend_sender, spool_dir=settings.alert_spool_dir)

    class _TrackingDispatcher(AlertDispatcher):
        """在分发前把当前事件挂到闭包上，好让 cabin_sink 拿到违规类型。"""

        def dispatch(self, event: SafetyEvent) -> dict[str, bool]:
            _pending["current"] = event
            try:
                return super().dispatch(event)
            finally:
                _pending.pop("current", None)

    return _TrackingDispatcher([cabin, backend]), recorder
