"""车内本地告警 + 后台上报 —— 模式C 相对模式B 的核心差异所在。

差异只有一句话：**车内这条通道不经过网络。**

  模式B：帧上行 → 云端推理 → 判定 → 下行推送 → 车机播报
         链路里有两次广域网往返，隧道/地库/山区断网时这条链路直接断掉。
  模式C：本地推理 → 本地判定 → 本地播报
         断网只影响后台那条通道，且事件落盘等恢复后补传，一条不丢。

本模块提供两个注入给 `common.alerting` 的实现：
  * `LocalVoiceSink`  —— 车内播报（TTS + 蜂鸣），并记录「采集 → 播报」的端到端延迟
  * `HttpBackendSender` —— 后台上报（HTTP POST），并记录「事件生成 → 后台确认」的延迟
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

from common import CabinPrompt, SafetyEvent


# --------------------------------------------------------------------------
# 车内播报
# --------------------------------------------------------------------------
@dataclass
class VoiceRecord:
    ts: float
    text: str
    severity: str
    repeat: int
    beep: bool
    engine: str
    wav: str | None = None


class LocalVoiceSink:
    """车内语音 + 蜂鸣播报。

    播报引擎按可用性降级，**每次都如实记录用的是哪一档**：
      1. `espeak-ng` / `espeak`：真离线 TTS，实车推荐（几 MB，无需联网）
      2. 无 TTS 引擎时：仍然真实合成蜂鸣 WAV（CRITICAL 级必须有听得见的提示音），
         语音内容以文本形式落盘 —— 这正是本机（无声卡、无 TTS 引擎）的实际情况

    实车方案见 DESIGN.md：RK3588 直接跑 espeak-ng 或接 SYN6288 语音芯片。
    """

    def __init__(self, audio_dir: Path, *, enable_wav: bool = True) -> None:
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.enable_wav = enable_wav
        self.records: list[VoiceRecord] = []
        self._tts = shutil.which("espeak-ng") or shutil.which("espeak")
        self.engine = Path(self._tts).name if self._tts else "beep_wav+text_log"
        self._n = 0

    def _beep_wav(self, severity: str) -> str | None:
        """真实合成一段提示音并写成 WAV（不是占位文件，可以直接播放）。"""
        if not self.enable_wav:
            return None
        self._n += 1
        path = self.audio_dir / f"alert_{self._n:04d}_{severity}.wav"
        sr = 16000
        # CRITICAL 用双音急促提示，WARN 用单音短提示
        pattern = [(1400, 0.12), (0, 0.06), (1400, 0.12)] if severity == "critical" else [(880, 0.18)]
        frames = bytearray()
        for freq, dur in pattern:
            for i in range(int(sr * dur)):
                v = 0 if freq == 0 else int(16000 * math.sin(2 * math.pi * freq * i / sr))
                frames += struct.pack("<h", v)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(bytes(frames))
        return str(path)

    def __call__(self, prompt: CabinPrompt) -> None:
        wav = self._beep_wav(prompt.severity) if prompt.beep or prompt.severity == "warn" else None
        if self._tts:
            try:
                subprocess.run([self._tts, "-v", "zh", prompt.text],
                               check=False, timeout=5,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:  # noqa: BLE001 - 播报失败不能中断检测主循环
                pass
        self.records.append(VoiceRecord(ts=time.time(), text=prompt.text,
                                        severity=prompt.severity, repeat=prompt.repeat,
                                        beep=prompt.beep, engine=self.engine, wav=wav))
        tag = {"info": "提示", "warn": "警告", "critical": "严重"}.get(prompt.severity, prompt.severity)
        print(f"  🔊 [车内·{tag}] {prompt.text}"
              + (f"（重复{prompt.repeat}次 + 蜂鸣）" if prompt.beep else ""))


# --------------------------------------------------------------------------
# 后台上报
# --------------------------------------------------------------------------
@dataclass
class SendStat:
    ok: int = 0
    fail: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def avg_ms(self) -> float | None:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else None


class HttpBackendSender:
    """向后台 POST 一条事件；失败返回 False，交给 `BackendChannel` 落盘补传。

    实车上这里会换成 MQTT over TLS（4G 下比 HTTP 省流量、断线重连成熟），
    但接口语义不变 —— 返回 bool，失败即落盘。
    """

    def __init__(self, url: str, timeout_s: float = 2.0, *, drop_b64: bool = False) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.drop_b64 = drop_b64
        self.stat = SendStat()
        self.online = True          # 演示用的「模拟断网」开关

    def __call__(self, event: SafetyEvent) -> bool:
        if not self.online:
            self.stat.fail += 1
            return False
        body = event.to_json(drop_b64=self.drop_b64).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                ok = 200 <= r.status < 300
        except (urllib.error.URLError, OSError, TimeoutError):
            self.stat.fail += 1
            return False
        dt = (time.perf_counter() - t0) * 1000
        if ok:
            self.stat.ok += 1
            self.stat.latencies_ms.append(dt)
        else:
            self.stat.fail += 1
        return ok


class JsonlSink:
    """把事件同时写一份本地 JSONL —— 既是本地存证，也是 bench 打分的输入。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self.n = 0

    def write(self, event: SafetyEvent) -> None:
        self._f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())      # 车载断电是常态，事件必须立刻落盘
        self.n += 1

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:  # noqa: BLE001
            pass
