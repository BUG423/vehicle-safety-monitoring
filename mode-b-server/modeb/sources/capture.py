"""拉流源 —— 本地视频文件 / 本机摄像头 / RTSP。

用 OpenCV 的 VideoCapture 统一处理，因为这三者在 OpenCV 里是同一个 API：

    "demo.mp4"                      本地文件
    0 / 1                           本机摄像头设备号
    "rtsp://user:pw@ip:554/stream"  网络摄像机或车载 DVR 的 RTSP 流

读文件时默认按原始帧率节流（`realtime=True`），否则一段 30 秒的视频会在 2 秒内跑完，
测出来的「并发路数」是假的。压测时把 `realtime` 关掉可以打满 GPU。
"""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

from .base import Frame, FrameSource


class OpenCVSource(FrameSource):
    """基于 cv2.VideoCapture 的拉流源。"""

    kind = "opencv"

    def __init__(
        self,
        vehicle_id: str,
        uri: str | int,
        *,
        realtime: bool = True,
        loop: bool = True,
        target_fps: float | None = None,
        resize_short_side: int | None = None,
        meta_provider: Any = None,
    ) -> None:
        super().__init__(vehicle_id)
        self.uri = uri
        self.loop = loop
        self.realtime = realtime
        self.resize_short_side = resize_short_side
        self.meta_provider = meta_provider

        self._cap = cv2.VideoCapture(uri)
        if not self._cap.isOpened():
            self._opened = False
            raise RuntimeError(f"无法打开视频源: {uri}")

        native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.fps = target_fps or (native_fps if 1.0 < native_fps < 120.0 else 25.0)
        self._interval = 1.0 / self.fps
        self._next_due = time.time()
        self.stats["fps_nominal"] = round(self.fps, 2)

    def read(self) -> Frame | None:
        if not self._opened:
            return None

        if self.realtime:
            now = time.time()
            if now < self._next_due:
                time.sleep(min(self._next_due - now, 0.01))
                return None
            # 落后太多时直接对齐到当前时刻，避免积压后疯狂追帧
            self._next_due = max(self._next_due + self._interval, now - self._interval)

        ok, img = self._cap.read()
        if not ok or img is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self._cap.read()
            if not ok or img is None:
                self.close()
                return None

        if self.resize_short_side:
            img = _resize_short_side(img, self.resize_short_side)

        self.stats["frames"] += 1
        meta = self.meta_provider() if callable(self.meta_provider) else {}
        return Frame(vehicle_id=self.vehicle_id, image=img, seq=self._next_seq(), meta=meta or {})

    def close(self) -> None:
        super().close()
        try:
            self._cap.release()
        except Exception:  # noqa: BLE001
            pass


class SyntheticSource(FrameSource):
    """合成源 —— 没有真实车载素材时用来跑通链路与压测。

    画面里画一个可控的「人形」：肩、髋、头、眼睛、手，
    并按脚本切换「系没系安全带 / 闭没闭眼 / 拿没拿手机 / 转头角度」。
    真模型（Keypoint R-CNN）**认不出**这种简笔画，因此合成源只适合：
      1) 配 Mock 检测器跑通端到端链路；
      2) 做纯吞吐压测（推理照样真跑，只是检不出人）。
    真模型的效果验证必须用真实视频，见 README「实测到什么程度」。
    """

    kind = "synthetic"

    def __init__(self, vehicle_id: str, *, fps: float = 15.0, size: tuple[int, int] = (640, 480),
                 seed: int | None = None) -> None:
        super().__init__(vehicle_id)
        self.fps = fps
        self.size = size
        self._interval = 1.0 / fps
        self._next_due = time.time()
        self._rng = np.random.default_rng(seed if seed is not None else abs(hash(vehicle_id)) % 2**31)
        self._t0 = time.time()

    def read(self) -> Frame | None:
        now = time.time()
        if now < self._next_due:
            time.sleep(min(self._next_due - now, 0.01))
            return None
        self._next_due = max(self._next_due + self._interval, now - self._interval)

        w, h = self.size
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        t = now - self._t0
        # 缓慢起伏的背景，避免 camera_blocked 规则把纯色画面误判为遮挡
        img[:, :, 0] = 40 + (20 * np.sin(t / 3.0)).astype(np.uint8)
        cv2.rectangle(img, (0, int(h * 0.75)), (w, h), (60, 60, 70), -1)
        for cx, label in ((int(w * 0.28), "driver"), (int(w * 0.72), "passenger")):
            cv2.circle(img, (cx, int(h * 0.30)), int(h * 0.10), (170, 160, 150), -1)
            cv2.rectangle(img, (cx - int(w * 0.10), int(h * 0.42)),
                          (cx + int(w * 0.10), int(h * 0.78)), (90, 110, 150), -1)
            cv2.putText(img, label, (cx - 30, int(h * 0.95)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (220, 220, 220), 1)
        cv2.putText(img, f"SYNTHETIC {self.vehicle_id} t={t:5.1f}s", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        img += self._rng.integers(0, 6, size=img.shape, dtype=np.uint8)

        self.stats["frames"] += 1
        return Frame(vehicle_id=self.vehicle_id, image=img, seq=self._next_seq())


class PushSource(FrameSource):
    """推流源 —— 车端主动把帧推给后台（HTTP POST / WebSocket）。

    后台的 HTTP/WS 接口收到帧后调用 `offer()` 放进有界队列，
    调度器从队列里 `read()`。队列满时**丢最旧的帧**：实时监测里
    「处理一帧 3 秒前的画面」没有意义，宁可丢帧也要保时效。
    """

    kind = "push"

    def __init__(self, vehicle_id: str, *, maxlen: int = 4) -> None:
        super().__init__(vehicle_id)
        self._buf: list[Frame] = []
        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._last_offer = time.time()

    def offer(self, image: np.ndarray, *, ts: float | None = None,
              meta: dict[str, Any] | None = None) -> bool:
        """由 HTTP/WS 接口调用。返回 False 表示发生了丢帧。"""
        frame = Frame(vehicle_id=self.vehicle_id, image=image, seq=self._next_seq(),
                      ts=ts if ts is not None else time.time(), meta=meta or {})
        dropped = False
        with self._lock:
            self._buf.append(frame)
            while len(self._buf) > self._maxlen:
                self._buf.pop(0)
                self.stats["dropped"] += 1
                dropped = True
            self._last_offer = time.time()
            self.stats["frames"] += 1
        return not dropped

    def read(self) -> Frame | None:
        with self._lock:
            if not self._buf:
                return None
            return self._buf.pop(0)

    @property
    def idle_s(self) -> float:
        return time.time() - self._last_offer

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["pending"] = len(self._buf)
        d["idle_s"] = round(self.idle_s, 1)
        return d


def _resize_short_side(img: np.ndarray, short_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = short_side / min(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


class ThreadedSource(FrameSource):
    """给拉流源套一个独立解码线程。

    为什么必须有它：`InferenceScheduler` 的攒批循环是**单线程**的，
    如果在这个线程里逐个调用 `VideoCapture.read()`，N 路视频的 H.264 解码
    就全部串行排在 GPU 前面。实测 16 路以上时 GPU 利用率不到 10%，
    瓶颈完全在 CPU 侧 —— 解码必须挪出攒批循环。

    生产环境更进一步应该用 NVDEC 硬解（`cv2.cudacodec` 或 PyNvCodec），
    把解码也放到 GPU 上，CPU 只做调度。
    """

    kind = "threaded"

    def __init__(self, inner: FrameSource, *, maxlen: int = 3) -> None:
        super().__init__(inner.vehicle_id)
        self.inner = inner
        self.kind = f"threaded:{inner.kind}"
        self._buf: list[Frame] = []
        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name=f"src-{inner.vehicle_id}")
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set() and self.inner.is_open:
            try:
                f = self.inner.read()
            except Exception:  # noqa: BLE001
                self.stats["errors"] += 1
                time.sleep(0.05)
                continue
            if f is None:
                continue
            with self._lock:
                self._buf.append(f)
                while len(self._buf) > self._maxlen:
                    self._buf.pop(0)
                    self.stats["dropped"] += 1
                self.stats["frames"] += 1
        self._opened = False

    def read(self) -> Frame | None:
        with self._lock:
            return self._buf.pop(0) if self._buf else None

    @property
    def is_open(self) -> bool:
        return self._opened and (self.inner.is_open or bool(self._buf))

    def close(self) -> None:
        self._stop.set()
        super().close()
        self.inner.close()

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["inner"] = self.inner.describe()
        with self._lock:
            d["pending"] = len(self._buf)
        return d
