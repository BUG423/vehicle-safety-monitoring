"""车端 agent —— 把车上的画面推给后台，并接收后台下发的车内提醒。

这是模式B 在车上唯一需要跑的东西：**只做采集、编码、上传**，不做任何推理。
车上因此只需要一个几百块的 4G/5G DTU 或工控机，不需要 NPU 算力盒子 ——
这正是模式B 对比车载嵌入式方案的成本优势所在。

两种上传通道：
    --transport ws     WebSocket 长连接推 JPEG（默认，省头开销、抖动小）
    --transport http   逐帧 HTTP POST（最容易穿透企业网络与各种代理）

用法::

    python3 -m modeb.tools.vehicle_agent --server http://127.0.0.1:8080 \\
        --vehicle 京A12345 --video ../bench/clips/scenario_a.mp4 --fps 10 --quality 70
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modeb.sources.capture import OpenCVSource, SyntheticSource  # noqa: E402


def _open_source(args) -> object:
    if args.video in ("synthetic", "", None):
        return SyntheticSource(args.vehicle, fps=args.fps)
    uri: str | int = args.video
    if str(args.video).isdigit():
        uri = int(args.video)
    return OpenCVSource(args.vehicle, uri, realtime=True, loop=args.loop,
                        target_fps=args.fps, resize_short_side=args.short_side)


def run_http(args, src) -> None:
    import urllib.request
    base = args.server.rstrip("/")
    url = f"{base}/api/v1/vehicles/{args.vehicle}/frame"
    sent = 0
    t0 = time.time()
    while src.is_open and (args.duration <= 0 or time.time() - t0 < args.duration):
        f = src.read()
        if f is None:
            continue
        ok, buf = cv2.imencode(".jpg", f.image, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        if not ok:
            continue
        meta = {"ts": f.ts, "seq": f.seq, **_vehicle_signals(args, f.seq)}
        req = urllib.request.Request(url, data=buf.tobytes(), method="POST",
                                     headers={"Content-Type": "image/jpeg",
                                              "X-Meta": json.dumps(meta)})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                r.read()
            sent += 1
        except Exception as exc:  # noqa: BLE001 - 断网是车载常态，重试即可
            print(f"[agent] 上传失败（将继续重试）: {exc}")
            time.sleep(0.5)
        if sent % 50 == 0 and sent:
            print(f"[agent] 已上传 {sent} 帧 ({sent/(time.time()-t0):.1f} fps)")
    print(f"[agent] 结束，共上传 {sent} 帧")


def run_ws(args, src) -> None:
    import asyncio
    import websockets

    base = args.server.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    url = f"{base}/ws/ingest/{args.vehicle}"

    async def pump() -> None:
        sent, t0 = 0, time.time()
        async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
            print(f"[agent] 已连接 {url}")
            while src.is_open and (args.duration <= 0 or time.time() - t0 < args.duration):
                f = src.read()
                if f is None:
                    await asyncio.sleep(0.002)
                    continue
                ok, buf = cv2.imencode(".jpg", f.image,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                if not ok:
                    continue
                await ws.send(json.dumps({"ts": f.ts, "seq": f.seq,
                                          **_vehicle_signals(args, f.seq)}))
                await ws.send(buf.tobytes())
                sent += 1
                if sent % 100 == 0:
                    dt = time.time() - t0
                    print(f"[agent] 已上传 {sent} 帧 ({sent/dt:.1f} fps, "
                          f"{sent*len(buf)/dt/1024:.0f} KB/s)")
        print(f"[agent] 结束，共上传 {sent} 帧")

    asyncio.run(pump())


def _vehicle_signals(args, seq: int) -> dict:
    """模拟 OBD/GPS 随帧上传。超速判定只能来自这里，不能来自图像。"""
    if not args.with_obd:
        return {}
    speed = 45.0 + 35.0 * abs(((seq // 30) % 8) - 4) / 4.0
    return {"speed_kmh": round(speed, 1), "speed_limit_kmh": 60.0,
            "gear": "D", "engine_on": True}


def run_cabin_listener(args) -> None:
    """另起一线程订阅车内提醒 —— 演示「后台 → 车机」这条回程链路真的通。"""
    import asyncio
    import websockets

    base = args.server.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    url = f"{base}/ws/cabin/{args.vehicle}"

    async def listen() -> None:
        while True:
            try:
                async with websockets.connect(url) as ws:
                    async for msg in ws:
                        m = json.loads(msg)
                        if m.get("type") == "cabin_prompt":
                            print(f"\n  ★ [车内播报] {m['text']}  "
                                  f"(重复{m['repeat']}次 蜂鸣={m['beep']} 等级={m['severity']})\n")
            except Exception:  # noqa: BLE001
                await asyncio.sleep(1.5)

    asyncio.run(listen())


def main() -> int:
    ap = argparse.ArgumentParser(description="模式B 车端 agent：采集 + 上传 + 接收车内提醒")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--vehicle", default="DEMO-001")
    ap.add_argument("--video", default="synthetic", help="视频文件 / 摄像头设备号 / rtsp:// / synthetic")
    ap.add_argument("--transport", default="ws", choices=["ws", "http"])
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--quality", type=int, default=70, help="JPEG 质量，直接决定上行带宽")
    ap.add_argument("--short-side", type=int, default=None, help="上传前缩放短边")
    ap.add_argument("--duration", type=float, default=0, help="运行秒数，0 表示不限")
    ap.add_argument("--loop", action="store_true", default=True)
    ap.add_argument("--with-obd", action="store_true", help="随帧上传模拟车速信号")
    ap.add_argument("--no-cabin", action="store_true", help="不订阅车内提醒")
    ap.add_argument("--plate", default=None)
    ap.add_argument("--driver", default=None)
    args = ap.parse_args()

    import urllib.request
    body = json.dumps({"vehicle_id": args.vehicle, "plate": args.plate or args.vehicle,
                       "driver_name": args.driver, "source_kind": f"{args.transport}_push"}
                      ).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{args.server.rstrip('/')}/api/v1/vehicles/register", data=body,
            headers={"Content-Type": "application/json"}), timeout=5).read()
        print(f"[agent] 车辆 {args.vehicle} 已注册")
    except Exception as exc:  # noqa: BLE001
        print(f"[agent] 注册失败（继续尝试推流）: {exc}")

    if not args.no_cabin:
        threading.Thread(target=run_cabin_listener, args=(args,), daemon=True).start()

    src = _open_source(args)
    print(f"[agent] 视频源: {src.describe()}")
    (run_ws if args.transport == "ws" else run_http)(args, src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
