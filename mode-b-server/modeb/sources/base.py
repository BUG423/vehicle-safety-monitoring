"""视频源抽象。

模式B 的接入方式有两大类，它们的差别在于**谁主动**：

  拉流 (pull)  —— 后台主动去拉：RTSP / GB28181 / 本地文件 / 本机摄像头
                  适合车队已有部标 808/1078 平台、车端只是普通 IPC 的场景
  推流 (push)  —— 车端主动推：WebRTC / HTTP 分片上传 / WebSocket 帧上传
                  适合车端在 NAT 后面、没有公网 IP 的场景（绝大多数营运车辆）

两类源都归一到同一个 `FrameSource` 接口，调度器不需要知道帧从哪来。
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass
class Frame:
    """一帧待推理的画面。"""

    vehicle_id: str
    image: np.ndarray            # BGR, HxWx3, uint8
    ts: float = field(default_factory=time.time)      # 车端采集时间戳
    seq: int = 0                 # 该路视频内的帧序号
    recv_ts: float = field(default_factory=time.time)  # 后台收到的时间戳
    meta: dict[str, Any] = field(default_factory=dict)  # 车速/GPS 等随帧上传的车身信号

    @property
    def uplink_delay_ms(self) -> float:
        """车端采集到后台收到的时间差 —— 推流链路的真实网络延迟。

        注意：只有在车端与后台时钟同步（NTP）时才有意义，否则会出现负值。
        """
        return (self.recv_ts - self.ts) * 1000.0

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


class FrameSource(abc.ABC):
    """帧源统一接口。

    实现要求：
      - `read()` 返回一帧或 None（None 表示暂时无帧，不代表结束）
      - `is_open` 为 False 表示该源已结束/断开，调度器会回收它
      - 实现自己吞掉异常，不得让调度器崩溃
    """

    kind = "base"

    def __init__(self, vehicle_id: str) -> None:
        self.vehicle_id = vehicle_id
        self._seq = 0
        self._opened = True
        self.stats: dict[str, Any] = {"frames": 0, "dropped": 0, "errors": 0}

    @property
    def is_open(self) -> bool:
        return self._opened

    @abc.abstractmethod
    def read(self) -> Frame | None:
        """取一帧。非阻塞或短阻塞，不允许长时间卡住调度线程。"""

    def close(self) -> None:
        self._opened = False

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def __iter__(self) -> Iterator[Frame]:
        while self.is_open:
            frame = self.read()
            if frame is not None:
                yield frame

    def describe(self) -> dict[str, Any]:
        return {"vehicle_id": self.vehicle_id, "kind": self.kind,
                "open": self.is_open, **self.stats}
