"""统一告警分发 —— 需求 (c)：对当事人和后台都要有告警信息。

两条通道彼此独立、互为兜底：

  车内通道 InCabinChannel  —— 让当事人立刻知道（语音 TTS / 蜂鸣 / 屏幕横幅）
  后台通道 BackendChannel  —— 让管理者知道（HTTP Webhook / MQTT / WebSocket / 本地队列）

断网时后台通道自动落盘为待发队列，恢复后补传，保证事件不丢失
（车载场景隧道/地库断网是常态，这一点在三种技术模式下都必须成立）。
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

try:
    from ..schema.safety_event import SafetyEvent
    from ..schema.violation_types import Severity
except ImportError:  # pragma: no cover - 独立运行
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "schema"))
    from safety_event import SafetyEvent
    from violation_types import Severity


class AlertChannel(ABC):
    """告警通道基类。"""

    name = "base"

    @abstractmethod
    def send(self, event: SafetyEvent) -> bool:
        """返回是否投递成功。实现必须自己吞掉异常，不得中断检测主循环。"""


# --------------------------------------------------------------------------
# 车内通道：提醒当事人
# --------------------------------------------------------------------------
@dataclass
class CabinPrompt:
    """一次车内提醒指令 —— 由车机/嵌入式端的播报模块消费。"""

    text: str              # 语音播报文本
    repeat: int            # 重复次数
    beep: bool             # 是否伴随蜂鸣
    banner_color: str      # 屏幕横幅颜色
    severity: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class InCabinChannel(AlertChannel):
    """车内提醒通道。

    按严重等级决定打扰强度 —— INFO 不打扰，WARN 提醒一次，CRITICAL 循环播报 + 蜂鸣。
    `sink` 由各模式注入：嵌入式端接 TTS 芯片，服务器端接 WebSocket 推到车机页面。
    """

    name = "in_cabin"

    _STYLE = {
        Severity.INFO: (0, False, "#3b82f6"),
        Severity.WARN: (1, False, "#f59e0b"),
        Severity.CRITICAL: (3, True, "#ef4444"),
    }

    def __init__(self, sink: Callable[[CabinPrompt], None] | None = None,
                 min_severity: Severity = Severity.WARN) -> None:
        self._sink = sink or self._default_sink
        self._min = min_severity

    @staticmethod
    def _default_sink(prompt: CabinPrompt) -> None:
        print(f"[车内提醒][{prompt.severity}] {prompt.text} (重复{prompt.repeat}次 蜂鸣={prompt.beep})")

    def build_prompt(self, event: SafetyEvent) -> CabinPrompt:
        repeat, beep, color = self._STYLE[event.severity]
        text = self._speech_text(event)
        return CabinPrompt(text=text, repeat=repeat, beep=beep,
                           banner_color=color, severity=event.severity.value)

    @staticmethod
    def _speech_text(event: SafetyEvent) -> str:
        """播报语要短、要给出可执行动作 —— 行车中司机没有阅读长句的余裕。"""
        from violation_types import ViolationType as VT  # 局部导入避免循环
        action = {
            VT.DRIVER_NO_SEATBELT: "请系好安全带",
            VT.PASSENGER_NO_SEATBELT: "请提醒乘客系好安全带",
            VT.DRIVER_FATIGUE: "检测到疲劳，请就近停车休息",
            VT.DRIVER_DISTRACTION: "请注意前方道路",
            VT.DRIVER_PHONE_USE: "行车中请勿使用手机",
            VT.DRIVER_SMOKING: "车内请勿吸烟",
            VT.DRIVER_HANDS_OFF_WHEEL: "请双手握好方向盘",
            VT.VEHICLE_SPEEDING: "您已超速，请减速",
        }.get(event.violation)
        return action or event.message

    def send(self, event: SafetyEvent) -> bool:
        if event.severity.rank < self._min.rank:
            return True  # 低于阈值不打扰驾驶员，但不算失败
        try:
            self._sink(self.build_prompt(event))
            return True
        except Exception as exc:  # noqa: BLE001 - 告警失败不能拖垮检测主循环
            print(f"[InCabinChannel] 播报失败: {exc}")
            return False


# --------------------------------------------------------------------------
# 后台通道：上报管理端，带断网重传
# --------------------------------------------------------------------------
class BackendChannel(AlertChannel):
    """后台上报通道，内置异步队列 + 断网落盘补传。"""

    name = "backend"

    def __init__(
        self,
        sender: Callable[[SafetyEvent], bool] | None = None,
        *,
        spool_dir: str | os.PathLike[str] | None = None,
        max_queue: int = 1000,
        retry_interval_s: float = 10.0,
    ) -> None:
        self._sender = sender or self._default_sender
        self._spool = Path(spool_dir) if spool_dir else Path(".alert_spool")
        self._spool.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue[SafetyEvent] = queue.Queue(maxsize=max_queue)
        self._retry_interval = retry_interval_s
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="backend-alert")
        self._worker.start()

    @staticmethod
    def _default_sender(event: SafetyEvent) -> bool:
        print(f"[后台告警] {event.to_json(drop_b64=True)}")
        return True

    def send(self, event: SafetyEvent) -> bool:
        try:
            self._q.put_nowait(event)
            return True
        except queue.Full:
            self._spool_event(event)   # 队列满也不丢事件
            return False

    def _spool_event(self, event: SafetyEvent) -> None:
        path = self._spool / f"{int(event.ts * 1000)}_{event.event_id[:8]}.json"
        try:
            path.write_text(event.to_json(), encoding="utf-8")
        except OSError as exc:  # noqa: BLE001
            print(f"[BackendChannel] 落盘失败: {exc}")

    def _run(self) -> None:
        last_retry = 0.0
        while not self._stop.is_set():
            try:
                event = self._q.get(timeout=1.0)
            except queue.Empty:
                event = None
            if event is not None:
                ok = False
                try:
                    ok = bool(self._sender(event))
                except Exception as exc:  # noqa: BLE001 - 网络异常属常态
                    print(f"[BackendChannel] 上报异常: {exc}")
                if not ok:
                    self._spool_event(event)
            now = time.time()
            if now - last_retry >= self._retry_interval:
                last_retry = now
                self._flush_spool()

    def _flush_spool(self) -> None:
        """网络恢复后补传落盘事件。"""
        for path in sorted(self._spool.glob("*.json"))[:50]:
            try:
                event = SafetyEvent.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                path.unlink(missing_ok=True)
                continue
            try:
                if self._sender(event):
                    path.unlink(missing_ok=True)
                else:
                    break   # 仍然不通，下轮再试
            except Exception:  # noqa: BLE001
                break

    @property
    def pending(self) -> int:
        return self._q.qsize() + len(list(self._spool.glob("*.json")))

    def close(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._worker.join(timeout=timeout)


class AlertDispatcher:
    """把一条事件同时投递到车内与后台 —— 需求 (c) 的统一入口。"""

    def __init__(self, channels: Iterable[AlertChannel] | None = None) -> None:
        self.channels: list[AlertChannel] = list(channels) if channels else [
            InCabinChannel(), BackendChannel()
        ]

    def dispatch(self, event: SafetyEvent) -> dict[str, bool]:
        return {ch.name: ch.send(event) for ch in self.channels}

    def close(self) -> None:
        for ch in self.channels:
            if hasattr(ch, "close"):
                ch.close()
