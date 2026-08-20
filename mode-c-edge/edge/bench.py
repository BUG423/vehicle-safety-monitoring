"""算力受限实测 —— 这条路线最有说服力的证据。

回答一个问题：**这套 DMS 算法能不能塞进 RK3588 / Jetson Orin Nano 这一档？**

方法：把推理线程限制到 1~4 个（模拟嵌入式只有几个可用大核），
在真实拍摄的人脸视频上，逐模型和端到端各测一遍延迟、吞吐、常驻内存。

    python3 -m edge.bench --source assets/head-pose-face-detection-female.mp4

注意两条纪律：
  1. 只用**真实素材**测性能。`bench/` 的合成卡通帧上人脸检测器检不到目标，
     跑出来的数字既不代表精度也不代表真实负载。
  2. 本文件只输出**实测值**。从 x86 折算到 ARM 的推论一律写在 DESIGN.md 并明确标注「估算」。
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

import edge  # noqa: F401 —— sys.path 引导
from .config import EdgeConfig
from .perception.onnx_dms import OnnxDmsBackend


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        for line in open("/proc/self/status", encoding="utf-8"):
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return 0.0


def _cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _stat(xs: list[float]) -> dict:
    s = sorted(xs)
    return {
        "n": len(s),
        "mean_ms": round(statistics.fmean(s), 2),
        "p50_ms": round(s[len(s) // 2], 2),
        "p95_ms": round(s[min(len(s) - 1, int(0.95 * (len(s) - 1)))], 2),
        "max_ms": round(s[-1], 2),
    }


def load_frames(source: str, n: int, size: tuple[int, int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"打不开素材: {source}")
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.resize(f, size))
    cap.release()
    if not out:
        raise RuntimeError("素材里一帧都读不出来")
    while len(out) < n:                 # 素材不够长就循环，保证各配置样本数一致
        out.append(out[len(out) % max(1, len(out) // 2 or 1)])
    return out[:n]


def bench_stage(name: str, fn, frames, warmup: int = 5) -> dict:
    for f in frames[:warmup]:
        fn(f)
    lat = []
    for f in frames:
        t0 = time.perf_counter()
        fn(f)
        lat.append((time.perf_counter() - t0) * 1000)
    d = _stat(lat)
    d["fps"] = round(1000.0 / d["mean_ms"], 1) if d["mean_ms"] > 0 else None
    d["stage"] = name
    return d


def run(source: str, *, n_frames: int, thread_list: list[int],
        sizes: list[tuple[int, int]]) -> dict:
    report: dict = {
        "环境": {
            "cpu": _cpu_name(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "onnxruntime": __import__("onnxruntime").__version__,
            "素材": Path(source).name,
            "说明": "真实拍摄的人脸视频；合成卡通素材不用于性能实测",
        },
        "模型": {},
        "分阶段": [],
        "端到端": [],
    }

    rss0 = _rss_mb()
    cfg = EdgeConfig()
    cfg.num_threads = thread_list[0]
    cfg.frame_width, cfg.frame_height = sizes[0]
    be = OnnxDmsBackend(cfg)
    rss_loaded = _rss_mb()
    desc = be.describe()
    report["模型"] = desc["models"]
    report["内存"] = {
        "解释器+numpy+opencv+ort_基线_MB": round(rss0, 1),
        "三模型全部载入后_MB": round(rss_loaded, 1),
        "模型带来的增量_MB": round(rss_loaded - rss0, 1),
    }

    for size in sizes:
        frames = load_frames(source, n_frames, size)
        for th in thread_list:
            cfg2 = EdgeConfig()
            cfg2.num_threads = th
            cfg2.frame_width, cfg2.frame_height = size
            b = OnnxDmsBackend(cfg2)
            tag = f"{size[0]}x{size[1]}@{th}线程"

            # --- 分阶段：人脸检测 / 关键点 / 目标检测 ---
            b._det.setInputSize(size)
            s_face = bench_stage("人脸检测 YuNet", lambda f: b._det.detect(f), frames)

            _, faces = b._det.detect(frames[0])
            if faces is not None and len(faces):
                x, y, w_, h_ = faces[0][:4]
                bbox = (int(x), int(y), int(x + w_), int(y + h_))
            else:   # 素材首帧没检到脸时用画面中心框，只为测负载
                bbox = (size[0] // 3, size[1] // 4, size[0] * 2 // 3, size[1] * 3 // 4)
            s_lmk = bench_stage("关键点 2d106det", lambda f: b._landmarks(f, bbox), frames)
            s_obj = (bench_stage("目标检测 NanoDet", lambda f: b._detect_objects(f), frames)
                     if b._obj is not None else None)

            for s in (s_face, s_lmk, s_obj):
                if s:
                    report["分阶段"].append({"配置": tag, **s})

            # --- 端到端（含降频调度的目标检测）---
            rss_a = _rss_mb()
            for i, f in enumerate(frames[:5]):
                b.process(f, i)
            lat = []
            for i, f in enumerate(frames):
                t0 = time.perf_counter()
                b.process(f, i)
                lat.append((time.perf_counter() - t0) * 1000)
            rss_b = _rss_mb()
            d = _stat(lat)
            report["端到端"].append({
                "配置": tag, "分辨率": f"{size[0]}x{size[1]}", "线程": th,
                **d, "fps": round(1000.0 / d["mean_ms"], 1),
                "目标检测降频": f"每 {cfg2.object_stride} 帧一次",
                "推理期间RSS_MB": round(max(rss_a, rss_b), 1),
            })
            b.close()
    be.close()
    return report


def render(rep: dict) -> str:
    L = ["", "=" * 96, "模式C 算力受限实测（全部为本机真跑数字，未做任何折算）", "=" * 96]
    e = rep["环境"]
    L.append(f"CPU: {e['cpu']}")
    L.append(f"运行时: opencv {e['opencv']} / onnxruntime {e['onnxruntime']} / Python {e['python']}")
    L.append(f"素材: {e['素材']}（{e['说明']}）")
    L.append("")
    L.append("模型体积:")
    for k, v in rep["模型"].items():
        if v:
            L.append(f"  {k:<12s} {v['file']:<46s} {v['size_kb']:>8.1f} KB  [{v['runtime']}]")
    m = rep["内存"]
    L.append("")
    L.append(f"常驻内存: 基线 {m['解释器+numpy+opencv+ort_基线_MB']} MB "
             f"→ 载入三模型后 {m['三模型全部载入后_MB']} MB "
             f"（模型增量 {m['模型带来的增量_MB']} MB）")

    L += ["", "-" * 96, "分阶段单帧延迟", "-" * 96,
          f"{'配置':<20s}{'阶段':<22s}{'均值ms':>9s}{'p50':>9s}{'p95':>9s}{'单阶段FPS':>11s}"]
    for r in rep["分阶段"]:
        L.append(f"{r['配置']:<20s}{r['stage']:<22s}{r['mean_ms']:>9.2f}{r['p50_ms']:>9.2f}"
                 f"{r['p95_ms']:>9.2f}{r['fps']:>11.1f}")

    L += ["", "-" * 96, "端到端（人脸检测 + 关键点 + 降频目标检测 + 清晰度）", "-" * 96,
          f"{'配置':<20s}{'均值ms':>9s}{'p95ms':>9s}{'FPS':>8s}{'RSS_MB':>10s}"]
    for r in rep["端到端"]:
        L.append(f"{r['配置']:<20s}{r['mean_ms']:>9.2f}{r['p95_ms']:>9.2f}"
                 f"{r['fps']:>8.1f}{r['推理期间RSS_MB']:>10.1f}")
    L.append("=" * 96)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="模式C 算力受限性能实测")
    ap.add_argument("--source", default="assets/head-pose-face-detection-female.mp4")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--sizes", default="320x240,640x480")
    ap.add_argument("--json-out", default="runtime/bench_perf.json")
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    sizes = [tuple(int(x) for x in s.split("x")) for s in args.sizes.split(",")]
    threads = [int(x) for x in args.threads.split(",")]
    rep = run(args.source, n_frames=args.frames, thread_list=threads, sizes=sizes)
    txt = render(rep)
    print(txt)
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 已写入 {p}")
    if args.md_out:
        Path(args.md_out).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
