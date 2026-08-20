"""离线跑一段视频，把事件写成 JSONL —— 用于 bench/score.py 打分与回归。

它跑的是**完整链路**：感知 → 逐帧判定 → ViolationConfirmer → SafetyEvent → 告警分发，
和在线服务用的是同一份 VehiclePipeline，只是帧来自文件而不是网络。

用法::

    # 合成素材（bench）走 cartoon 后端 —— 评的是链路与时序逻辑
    python3 -m modeb.tools.run_clip --video ../bench/clips/scenario_a.mp4 \\
        --backend cartoon --fps 15 --out runs/mode_b_events.jsonl

    # 真实素材走 yolo 后端 —— 评的是真模型能力
    python3 -m modeb.tools.run_clip --video real.mp4 --backend yolo --face
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import AlertDispatcher, BackendChannel, InCabinChannel  # noqa: E402

from modeb.config import Config  # noqa: E402
from modeb.engine.pipeline import VehiclePipeline  # noqa: E402
from modeb.perception import build_detector, build_face_module, fallback_log  # noqa: E402
from modeb.sources.base import Frame  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="离线跑一段视频并输出 SafetyEvent JSONL")
    ap.add_argument("--video", required=True)
    ap.add_argument("--backend", default="cartoon",
                    help="yolo | torchvision | cartoon | mock | auto")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--short-side", type=int, default=None,
                    help="推理输入短边。手机这类小目标对它极敏感：480 时置信 0.43、640 时 0.75")
    ap.add_argument("--face", action="store_true", help="启用 MediaPipe 人脸模块（真实 EAR/PERCLOS）")
    ap.add_argument("--fps", type=float, default=0.0, help="覆盖视频帧率，用于换算 clip_t")
    ap.add_argument("--stride", type=int, default=1, help="抽帧间隔，1 表示逐帧")
    ap.add_argument("--vehicle", default="CLIP-001")
    ap.add_argument("--out", default="runs/mode_b_events.jsonl")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.perception.backend = args.backend
    cfg.perception.device = args.device
    if args.short_side:
        cfg.perception.infer_short_side = args.short_side
    det = build_detector(cfg.perception)
    if not args.quiet:
        print(f"[run_clip] 后端: {det.describe()}")
        for f in fallback_log():
            print(f"[run_clip] 降级记录: {f['backend']} -> {f['reason']}")

    face = None
    if args.face:
        face, err = build_face_module()
        print(f"[run_clip] 人脸模块: {'已启用' if face else '未启用 -> ' + str(err)}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"打不开视频: {args.video}", file=sys.stderr)
        return 2
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 15.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", encoding="utf-8")

    events_seen: list = []
    cabin_prompts: list = []
    dispatcher = AlertDispatcher(channels=[
        InCabinChannel(sink=lambda p: cabin_prompts.append(p)),
        BackendChannel(sender=lambda e: True, spool_dir=str(out_path.parent / ".spool")),
    ])

    def on_event(ev) -> None:
        events_seen.append(ev)
        fh.write(ev.to_json(drop_b64=True) + "\n")
        if not args.quiet:
            t = ev.raw_signals.get("clip_t")
            print(f"  [事件] t={t:>6}s  {ev.violation.label_zh:<12} "
                  f"{ev.severity.value:<8} 持续{ev.duration_s:.1f}s 置信{ev.confidence}")

    d = det.describe()
    version = f"{d.get('name')}/{d.get('pose_model') or d.get('kind')}" + ("+face478" if face else "")
    pipe = VehiclePipeline(args.vehicle, cfg, dispatcher=dispatcher, face_module=face,
                           on_event=on_event, model_version=version)

    idx = 0
    t_start = time.time()
    processed = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if idx % args.stride:
            idx += 1
            continue
        clip_t = idx / fps
        # 用「视频内时间」当事件时间轴，确认器的窗口/时长才对得上素材节奏，
        # 而不是被离线跑片的墙钟速度带偏
        frame = Frame(vehicle_id=args.vehicle, image=img, ts=time.time(), seq=idx,
                      meta={"clip_t": clip_t})
        res = det.infer(img, vehicle_id=args.vehicle)
        res.ts = clip_t
        pipe.process(frame, res)
        idx += 1
        processed += 1

    cap.release()
    fh.close()
    dispatcher.close()

    wall = time.time() - t_start
    print(f"\n[run_clip] 处理 {processed} 帧 / 视频 {idx / fps:.1f}s，"
          f"墙钟 {wall:.1f}s（{processed / max(wall, 1e-6):.1f} FPS）")
    print(f"[run_clip] 事件 {len(events_seen)} 条 -> {out_path}")
    print(f"[run_clip] 车内播报 {len(cabin_prompts)} 次")
    if face is not None:
        print(f"[run_clip] 人脸模块命中 {pipe.rules.face_hit} 帧 / 失败 {pipe.rules.face_miss} 帧")
    stats = pipe.stats.to_dict()
    print(f"[run_clip] 单帧推理均值 {stats['infer_ms_avg']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
