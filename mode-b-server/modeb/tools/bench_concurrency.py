"""单卡多路并发扩容曲线 —— 模式B 最需要回答的问题。

模式B 相对车载嵌入式方案的全部说服力都压在一个数字上：**一块 GPU 能带多少台车**。
带得越多，单车成本越低；带不动，集中式就没有意义。

本脚本在**进程内**直接跑调度器（不经过 HTTP/WebSocket），因此测的是纯推理侧的上限：

    for N in 1 4 8 16 ...:
        起 N 路视频源 → 共享一个检测器 → 跑 duration 秒
        记录：总吞吐、单路实际 FPS、GPU 每帧耗时、端到端延迟 P50/P95、GPU 利用率/显存

「端到端延迟」= 帧生成时刻 → 该帧的判定与告警处理完成，
不含车端编码与网络上行（那部分由 DESIGN.md 的带宽测算单独讨论）。

用法::

    python3 -m modeb.tools.bench_concurrency --backend yolo --levels 1,4,8,16 \\
        --duration 20 --video ../bench/clips/scenario_a.mp4
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modeb.config import Config  # noqa: E402
from modeb.engine.pipeline import VehiclePipeline  # noqa: E402
from modeb.engine.scheduler import InferenceScheduler  # noqa: E402
from modeb.perception import build_detector, build_face_module  # noqa: E402
from modeb.sources.capture import OpenCVSource, SyntheticSource  # noqa: E402


def gpu_probe(index: int = 0) -> dict:
    """用 nvidia-smi 采一次真实的 GPU 利用率与显存 —— 不估算，只报实测。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={index}",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
        util, used, total = [int(x.strip()) for x in out.split(",")]
        return {"gpu_util_pct": util, "mem_used_mb": used, "mem_total_mb": total}
    except Exception:  # noqa: BLE001
        return {"gpu_util_pct": None, "mem_used_mb": None, "mem_total_mb": None}


def run_level(n: int, args, detector, face) -> dict:
    cfg = Config()
    cfg.perception.backend = args.backend
    cfg.perception.device = args.device
    cfg.perception.max_batch = args.max_batch
    cfg.perception.infer_short_side = args.short_side

    events: list = []
    sched = InferenceScheduler(
        detector, cfg,
        pipeline_factory=lambda vid: VehiclePipeline(
            vid, cfg, dispatcher=None, face_module=face,
            on_event=lambda e: events.append(e)),
        post_workers=args.post_workers)
    sched.start()

    for i in range(n):
        vid = f"BENCH-{i:03d}"
        if args.video and args.video != "synthetic":
            src = OpenCVSource(vid, args.video, realtime=not args.stress, loop=True,
                               target_fps=args.fps, resize_short_side=args.short_side_src)
        else:
            src = SyntheticSource(vid, fps=args.fps)
        sched.add_source(src)

    time.sleep(args.warmup)
    sched.stats.__init__()          # 丢掉热身阶段的统计
    for vid in [f"BENCH-{i:03d}" for i in range(n)]:
        p = sched.pipeline(vid)
        if p:
            p.stats.__init__()

    t0 = time.time()
    gpu_samples = []
    while time.time() - t0 < args.duration:
        time.sleep(1.0)
        gpu_samples.append(gpu_probe(args.gpu_index))
    elapsed = time.time() - t0

    pipes = [sched.pipeline(f"BENCH-{i:03d}") for i in range(n)]
    per_stream_fps = [round(p.stats.frames / elapsed, 2) for p in pipes if p]
    p50 = [p.stats.percentile(50) for p in pipes if p and p.stats.frames]
    p95 = [p.stats.percentile(95) for p in pipes if p and p.stats.frames]
    s = sched.stats.to_dict()
    utils = [g["gpu_util_pct"] for g in gpu_samples if g["gpu_util_pct"] is not None]
    mems = [g["mem_used_mb"] for g in gpu_samples if g["mem_used_mb"] is not None]

    sched.stop()
    return {
        "n_streams": n,
        "elapsed_s": round(elapsed, 1),
        "total_fps": round(sum(per_stream_fps), 2),
        "per_stream_fps_avg": round(statistics.mean(per_stream_fps), 2) if per_stream_fps else 0,
        "per_stream_fps_min": min(per_stream_fps) if per_stream_fps else 0,
        "target_fps": args.fps,
        "gpu_ms_per_frame": s["gpu_ms_per_frame_avg"],
        "batch_size_avg": s["batch_size_avg"],
        "post_ms_avg": s["post_ms_avg"],
        "e2e_p50_ms": round(statistics.mean(p50), 1) if p50 else None,
        "e2e_p95_ms": round(max(p95), 1) if p95 else None,
        "dropped_post": s["dropped"],
        "events": len(events),
        "gpu_util_pct_avg": round(statistics.mean(utils), 1) if utils else None,
        "gpu_util_pct_max": max(utils) if utils else None,
        "mem_used_mb": max(mems) if mems else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="单卡多路并发扩容曲线实测")
    ap.add_argument("--backend", default="yolo")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=10.0, help="每路目标帧率")
    ap.add_argument("--video", default="synthetic")
    ap.add_argument("--short-side", type=int, default=480, help="推理输入短边")
    ap.add_argument("--short-side-src", type=int, default=None, help="读入后先缩放（模拟车端已降分辨率）")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--post-workers", type=int, default=3)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--face", action="store_true", help="同时启用 MediaPipe 人脸模块（会显著降吞吐）")
    ap.add_argument("--stress", action="store_true", help="关掉帧率节流，纯打满 GPU 测上限")
    ap.add_argument("--out", default="runs/bench_concurrency.json")
    args = ap.parse_args()

    cfg = Config()
    cfg.perception.backend = args.backend
    cfg.perception.device = args.device
    cfg.perception.infer_short_side = args.short_side
    cfg.perception.max_batch = args.max_batch
    detector = build_detector(cfg.perception)
    print(f"[bench] 检测器: {json.dumps(detector.describe(), ensure_ascii=False)}")

    face = None
    if args.face:
        face, err = build_face_module()
        print(f"[bench] 人脸模块: {'已启用' if face else err}")

    base = gpu_probe(args.gpu_index)
    print(f"[bench] 起测前 GPU: {base}")

    rows = []
    for n in [int(x) for x in args.levels.split(",") if x.strip()]:
        print(f"\n[bench] === {n} 路并发，{args.duration:.0f} 秒 ===")
        r = run_level(n, args, detector, face)
        rows.append(r)
        print(f"  总吞吐 {r['total_fps']} fps | 单路 {r['per_stream_fps_avg']}/{args.fps} fps "
              f"| GPU {r['gpu_ms_per_frame']} ms/帧 | 批 {r['batch_size_avg']} "
              f"| 端到端 P50 {r['e2e_p50_ms']}ms P95 {r['e2e_p95_ms']}ms "
              f"| GPU利用率 {r['gpu_util_pct_avg']}% | 显存 {r['mem_used_mb']}MB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "detector": detector.describe(),
                               "baseline_gpu": base, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'路数':>4} {'总吞吐':>9} {'单路fps':>9} {'GPU ms/帧':>10} {'批':>6} "
          f"{'P50 ms':>8} {'P95 ms':>8} {'GPU%':>6} {'显存MB':>8}")
    for r in rows:
        print(f"{r['n_streams']:>4} {r['total_fps']:>9.1f} {r['per_stream_fps_avg']:>9.2f} "
              f"{r['gpu_ms_per_frame']:>10.2f} {r['batch_size_avg']:>6.1f} "
              f"{(r['e2e_p50_ms'] or 0):>8.1f} {(r['e2e_p95_ms'] or 0):>8.1f} "
              f"{(r['gpu_util_pct_avg'] or 0):>6.1f} {(r['mem_used_mb'] or 0):>8}")
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
