"""模式C 车载端命令行入口。

典型用法::

    # 1) 真模型跑真实人脸视频（先 tools/fetch_models.py + tools/fetch_assets.py）
    python3 -m edge.main --backend onnx --source assets/head-pose-face-detection-female.mp4

    # 2) 跑公共基准素材（合成卡通，用规则后端评告警链路）
    python3 -m edge.main --backend rule --source ../bench/clips/scenario_a.mp4 \
        --width 640 --height 480 --fps 15 --obd cruise --events-out runs/mode_c_events.jsonl

    # 3) 断网补传演示：第 10~25 秒之间后台不可达
    python3 -m edge.main --backend rule --source ../bench/clips/scenario_a.mp4 \
        --offline-from 10 --offline-to 25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import edge  # noqa: F401  —— 触发 sys.path 引导，使 common 可引入

from .alerting import HttpBackendSender
from .config import EdgeConfig
from .engine import EdgeEngine
from .obd import build_obd
from .perception import build_backend


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="模式C：车载嵌入式端安全监测")
    ap.add_argument("--source", required=True, help="视频文件路径，或摄像头序号（如 0）")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "rule", "mock"],
                    help="感知后端：onnx=真模型 / rule=经典CV规则 / mock=脚本信号")
    ap.add_argument("--obd", default="demo", choices=["off", "cruise", "demo", "no_belt_bus"],
                    help="模拟 OBD/CAN 档案")
    ap.add_argument("--vehicle-id", default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None, help="目标处理帧率")
    ap.add_argument("--threads", type=int, default=None, help="推理线程数（模拟嵌入式大核数）")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--realtime", action="store_true", help="按真实时间限帧（实车模式）")
    ap.add_argument("--events-out", default=None, help="事件 JSONL 输出路径")
    ap.add_argument("--backend-url", default=None)
    ap.add_argument("--no-backend", action="store_true", help="不上报后台，只跑本地链路")
    ap.add_argument("--no-evidence-file", action="store_true", help="证据只留 base64，不落 JPEG")
    ap.add_argument("--offline-from", type=float, default=None, help="第几秒开始模拟断网")
    ap.add_argument("--offline-to", type=float, default=None, help="第几秒恢复联网")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = EdgeConfig()
    for k_arg, k_cfg in (("vehicle_id", "vehicle_id"), ("width", "frame_width"),
                         ("height", "frame_height"), ("fps", "target_fps"),
                         ("threads", "num_threads"), ("backend_url", "backend_url")):
        v = getattr(args, k_arg)
        if v is not None:
            setattr(cfg, k_cfg, v)
    cfg.ensure_dirs()

    source = int(args.source) if args.source.isdigit() else args.source
    backend = build_backend(args.backend, cfg)
    desc = backend.describe()
    print("=" * 78)
    print(f"感知后端: {desc['name']}   真实模型: {'是' if desc.get('real_model') else '否'}")
    for k, v in (desc.get("capabilities") or {}).items():
        print(f"  - {k}: {v}")
    if desc.get("limitation"):
        print(f"  ! 局限: {desc['limitation']}")
    print(f"处理规格: {cfg.frame_width}x{cfg.frame_height} @ {cfg.target_fps}fps, "
          f"推理线程={cfg.num_threads}")
    print("=" * 78)

    sender = None
    if args.no_backend:
        sender = lambda ev: True   # noqa: E731 - 只跑本地链路时后台恒成功
    else:
        sender = HttpBackendSender(cfg.backend_url, cfg.backend_timeout_s)

    engine = EdgeEngine(cfg, backend, obd=build_obd(args.obd, cfg.vehicle_id),
                        events_path=Path(args.events_out) if args.events_out else None,
                        sender=sender, save_evidence=not args.no_evidence_file)

    hook = None
    if args.offline_from is not None and isinstance(sender, HttpBackendSender):
        state = {"cut": False, "restored": False}

        def hook(t: float) -> None:               # noqa: ANN001
            if not state["cut"] and t >= args.offline_from:
                sender.online = False
                state["cut"] = True
                print(f"\n>>> t={t:.1f}s 模拟断网：后台通道不可达，事件应转为落盘\n")
            elif (state["cut"] and not state["restored"]
                  and args.offline_to is not None and t >= args.offline_to):
                sender.online = True
                state["restored"] = True
                print(f"\n>>> t={t:.1f}s 网络恢复：落盘事件应开始补传\n")

    try:
        engine.run(source, max_frames=args.max_frames, realtime=args.realtime,
                   verbose=not args.quiet, frame_hook=hook)
        pending = engine.flush(timeout_s=max(8.0, cfg.retry_interval_s * 3))
    finally:
        summary = engine.stats.summary()
        engine.close()

    print("\n" + "=" * 78)
    print("运行统计")
    for k, v in summary.items():
        print(f"  {k:<24s} {v}")
    print(f"  {'车内播报次数':<24s} {len(engine.voice.records)}（引擎={engine.voice.engine}）")
    if isinstance(sender, HttpBackendSender):
        st = sender.stat
        print(f"  {'后台上报成功/失败':<24s} {st.ok}/{st.fail}"
              + (f"，平均往返 {st.avg_ms:.1f} ms" if st.avg_ms else ""))
    print(f"  {'待补传事件':<24s} {pending}")
    print(f"  事件文件: {engine.jsonl.path}（{engine.jsonl.n} 条）")
    ok = all(r.get("in_cabin", False) for r in engine.dispatch_results)
    print(f"  车内通道全部投递成功: {'是' if ok else '否'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
