"""WebSocket 连接管理 —— 后台 → 看板 / 后台 → 车机 两条实时推送。

需求 (c) 要求告警既到当事人、也到后台。模式B 的形态是：
推理在后台完成，因此**车内提醒也必须由后台推回车机**（`cabin:<vehicle_id>` 主题），
这与模式C（本地推理本地播报）是根本区别，也是模式B 最大的软肋：
网络断了，车内提醒同样断。这一点在 DESIGN.md 的风险一节里如实展开。

推送策略：每个连接一个有界队列，**慢客户端不阻塞快客户端**，队列满就丢最旧消息。
一个卡住的看板页面不能拖垮整个车队的告警链路。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import WebSocket


class _Client:
    __slots__ = ("ws", "topic", "queue", "dropped", "connected_at")

    def __init__(self, ws: WebSocket, topic: str, maxsize: int) -> None:
        self.ws = ws
        self.topic = topic
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self.connected_at = time.time()


class WSHub:
    """按主题广播。主题约定：

        dashboard            车队看板，收全量事件与车辆状态
        cabin:<vehicle_id>   某辆车的车机，只收该车的车内提醒
    """

    def __init__(self, max_queue: int = 200) -> None:
        self._clients: dict[str, set[_Client]] = {}
        self._lock = asyncio.Lock()
        self.max_queue = max_queue
        self.sent = 0
        self.dropped = 0

    async def connect(self, ws: WebSocket, topic: str) -> _Client:
        await ws.accept()
        client = _Client(ws, topic, self.max_queue)
        async with self._lock:
            self._clients.setdefault(topic, set()).add(client)
        return client

    async def disconnect(self, client: _Client) -> None:
        async with self._lock:
            self._clients.get(client.topic, set()).discard(client)

    async def pump(self, client: _Client) -> None:
        """把该连接队列里的消息发出去，直到连接断开。"""
        while True:
            msg = await client.queue.get()
            await client.ws.send_text(msg)

    def broadcast(self, topic: str, payload: dict[str, Any]) -> int:
        """线程安全的非阻塞广播。返回投递到的连接数。

        注意：本方法会被**非 asyncio 线程**（推理调度器的后处理线程）调用，
        所以内部不能 await，只能往 asyncio.Queue 里 put_nowait。
        """
        msg = json.dumps(payload, ensure_ascii=False)
        n = 0
        for client in list(self._clients.get(topic, ())):
            try:
                client.queue.put_nowait(msg)
                n += 1
            except asyncio.QueueFull:
                try:            # 丢最旧的一条，保证新告警一定能进去
                    client.queue.get_nowait()
                    client.queue.put_nowait(msg)
                    client.dropped += 1
                    self.dropped += 1
                    n += 1
                except Exception:  # noqa: BLE001
                    pass
        self.sent += n
        return n

    def counts(self) -> dict[str, int]:
        return {t: len(cs) for t, cs in self._clients.items() if cs}

    def stats(self) -> dict[str, Any]:
        return {"topics": self.counts(), "sent": self.sent, "dropped": self.dropped}
