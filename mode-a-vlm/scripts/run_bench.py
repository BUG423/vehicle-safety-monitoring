"""在 bench/ 公共基准上跑模式A，输出可被 bench/score.py 打分的事件 JSONL。

模式A 与模式B/C 在这里有一个**结构性差别**，必须说清楚：
模式B/C 是逐帧实时处理（15fps），模式A 不可能也不应该这样做 —— 每帧一次云端调用
在延迟和成本上都不成立（见 DESIGN.md 的测算）。所以模式A 采用**抽帧 + 批量多帧**：
每 ``--stride`` 秒取一帧，每 ``--batch`` 帧打一次 VLM 调用，事件时间戳按帧在视频中的真实时刻回填。
这样 score.py 算出的告警延迟对模式A 才有意义 —— 它反映的正是「抽帧间隔 + 云端往返」的真实代价。

**关于合成素材的重要提醒**：bench 的卡通驾驶舱不适合评估 VLM 的感知精度。
实测中 VLM 会把卡通人物描述成「坐在桌子后面的人」、把安全带看成「一根细长的棍子」。
本脚本跑出来的低检出率**不代表 VLM 在真实照片上的能力**，它恰恰是「合成素材不能用来评 VLM」
这一结论的量化证据。真实照片上的表现见 ``evalset/``。

用法::

    python3 mode-a-vlm/scripts/run_bench.py --clip bench/clips/scenario_a.mp4 \\
        --provider siliconflow --stride 2.0 --batch 4 --out runs/mode_a_events.jsonl
    python3 bench/score.py --truth bench/clips/scenario_a_truth.json \\
        --events runs/mode_a_events.jsonl --label 模式A
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm_safety import SafetyPipeline, load_settings   # noqa: E402


def extract_frames(clip: Path, stride_s: float) -> list[tuple[float, bytes]]:
    """按固定时间间隔抽帧，返回 [(视频内秒, JPEG字节)]。"""
    import cv2

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频 {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, int(round(stride_s * fps)))
    out: list[tuple[float, bytes]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if ok2:
                out.append((idx / fps, buf.tobytes()))
        idx += 1
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="模式A 跑公共评测基准")
    ap.add_argument("--clip", default="bench/clips/scenario_a.mp4")
    ap.add_argument("--out", default="runs/mode_a_events.jsonl")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--stride", type=float, default=2.0, help="抽帧间隔（秒）")
    ap.add_argument("--batch", type=int, default=4, help="每次 VLM 调用送几帧")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 帧（调试用）")
    args = ap.parse_args()

    overrides = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    settings = load_settings(**overrides)
    pipe = SafetyPipeline(settings)
    print(f"[配置] provider={pipe.provider.name} model={pipe.provider.model} "
          f"模拟={pipe.provider.simulated} stride={args.stride}s batch={args.batch}")

    frames = extract_frames(Path(args.clip), args.stride)
    if args.limit:
        frames = frames[:args.limit]
    print(f"[抽帧] 共 {len(frames)} 帧（原视频按 {args.stride}s 间隔抽样）")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    n_events = n_calls = 0
    total_ms = 0.0
    t_wall = time.time()

    with outp.open("w", encoding="utf-8") as f:
        for i in range(0, len(frames), args.batch):
            chunk = frames[i:i + args.batch]
            clip_ts = [t for t, _ in chunk]
            blobs = [b for _, b in chunk]
            # base_ts 用视频内相对秒作为"时间轴原点"，确认器的滑窗因此按视频时间推进
            res = pipe.analyze(blobs, policy="temporal", frame_interval_s=args.stride,
                               base_ts=clip_ts[0], dispatch=False)
            n_calls += 1
            total_ms += res.timings_ms.get("vlm", 0.0)
            for ev in res.events:
                d = ev.to_dict()
                d["evidence"]["frame_b64"] = None
                # score.py 优先读 raw_signals.clip_t
                d["raw_signals"]["clip_t"] = round(ev.ts, 2)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_events += 1
            # 区分「调用失败」和「解析失败」—— 前者是超时/网络，后者才是模型输出格式问题，
            # 混在一起会把「多帧批量把请求撑爆了 60 秒超时」误诊成「模型不会输出 JSON」。
            if res.vlm and not res.vlm.ok:
                status = f"调用失败({(res.vlm.error or '')[:48]})"
            elif res.parse and not res.parse.ok:
                status = f"解析失败({(res.parse.error or '')[:32]})"
            else:
                status = "OK"
            print(f"  [{clip_ts[0]:5.1f}s-{clip_ts[-1]:5.1f}s] "
                  f"VLM {res.timings_ms.get('vlm', 0):6.0f}ms  "
                  f"事件 {len(res.events)}  {status}"
                  + (f"  {[e.violation.value for e in res.events]}" if res.events else ""))

    pipe.close()
    print(f"\n共 {n_calls} 次 VLM 调用，平均 {total_ms / max(1, n_calls):.0f}ms/次，"
          f"墙钟 {time.time() - t_wall:.0f}s，产出 {n_events} 条事件 -> {outp}")
    print("下一步：python3 bench/score.py --truth bench/clips/scenario_a_truth.json "
          f"--events {outp} --label 模式A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
