"""按同一份 ground truth 给某条技术路线的事件输出打分。

评的是三件对选型真正重要的事：

  检出率   标注的违规区间里，有多少被告警覆盖到
  告警延迟 违规开始到首次告警的秒数 —— 这是三条路线差异最大的指标
  误报     无违规区间内产生的告警，以及被眨眼/短暂低头这类干扰项骗到的次数

用法::

    python3 bench/score.py --truth bench/clips/scenario_a_truth.json \\
                           --events runs/mode_c_events.jsonl --label 模式C

事件文件为 JSONL，每行一个 `SafetyEvent.to_dict()`。时间取 `raw_signals.clip_t`
（视频内相对秒）；没有该字段时用 `ts - t0`，t0 取 `--t0` 或所有事件的最小 ts。
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Hit:
    violation: str
    start_s: float
    end_s: float
    detected: bool
    latency_s: float | None      # 违规开始 → 首次告警
    n_alerts: int


def load_events(path: Path, t0: float | None) -> list[tuple[float, str]]:
    """返回 [(相对秒, 违规类型)]。"""
    rows = []
    raw = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            raw.append(json.loads(line))
    if not raw:
        return rows
    base = t0
    if base is None:
        base = min(r.get("ts", 0.0) for r in raw)
    for r in raw:
        clip_t = (r.get("raw_signals") or {}).get("clip_t")
        t = float(clip_t) if clip_t is not None else float(r.get("ts", 0.0)) - base
        rows.append((t, r["violation"]))
    return sorted(rows)


def score(truth: dict, events: list[tuple[float, str]], *, grace_s: float = 1.0) -> dict:
    segments = truth["segments"]
    duration = float(truth["duration_s"])

    hits: list[Hit] = []
    matched_idx: set[int] = set()
    for seg in segments:
        v, a, b = seg["violation"], float(seg["start_s"]), float(seg["end_s"])
        # 告警允许落在区间内，或区间结束后 grace_s 内（确认逻辑本身有延迟）
        inside = [(i, t) for i, (t, ev) in enumerate(events)
                  if ev == v and a <= t <= b + grace_s]
        matched_idx.update(i for i, _ in inside)
        first = min((t for _, t in inside), default=None)
        hits.append(Hit(violation=v, start_s=a, end_s=b,
                        detected=first is not None,
                        latency_s=round(first - a, 2) if first is not None else None,
                        n_alerts=len(inside)))

    false_alarms = [(t, v) for i, (t, v) in enumerate(events) if i not in matched_idx]

    # 干扰项归因：误报是否落在眨眼/短暂低头窗口附近
    distractors = truth.get("distractors", {})
    def _near(t: float, ranges, pad: float = 2.0) -> bool:
        return any(a - pad <= t <= b + pad for a, b in ranges)

    fooled_by_blink = [x for x in false_alarms
                       if x[1] == "driver.fatigue" and _near(x[0], distractors.get("blinks", []))]
    fooled_by_glance = [x for x in false_alarms
                        if x[1] == "driver.distraction" and _near(x[0], distractors.get("glances", []))]

    detected = [h for h in hits if h.detected]
    lats = [h.latency_s for h in detected if h.latency_s is not None]
    return {
        "n_segments": len(hits),
        "n_detected": len(detected),
        "detect_rate": round(len(detected) / len(hits), 3) if hits else 0.0,
        "latency_avg_s": round(sum(lats) / len(lats), 2) if lats else None,
        "latency_max_s": round(max(lats), 2) if lats else None,
        "n_events": len(events),
        "n_false_alarms": len(false_alarms),
        "false_alarms_per_min": round(len(false_alarms) / (duration / 60.0), 2) if duration else 0.0,
        "fooled_by_blink": len(fooled_by_blink),
        "fooled_by_glance": len(fooled_by_glance),
        "per_segment": [h.__dict__ for h in hits],
        "false_alarm_list": [{"t": round(t, 2), "violation": v} for t, v in false_alarms],
    }


def _w(text: str) -> int:
    """终端显示宽度：中日韩字符按 2 列计。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _w(text))


def render(label: str, s: dict) -> str:
    lines = [f"\n=== {label} ===",
             f"检出率      {s['n_detected']}/{s['n_segments']}  ({s['detect_rate']:.0%})"]
    if s["latency_avg_s"] is not None:
        lines.append(f"告警延迟    平均 {s['latency_avg_s']:.2f}s / 最慢 {s['latency_max_s']:.2f}s")
    else:
        lines.append("告警延迟    —（无检出）")
    lines.append(f"误报        {s['n_false_alarms']} 次  ({s['false_alarms_per_min']:.2f} 次/分钟)")
    if s["fooled_by_blink"] or s["fooled_by_glance"]:
        lines.append(f"  其中被干扰项骗到  眨眼 {s['fooled_by_blink']} 次 / 短暂低头 {s['fooled_by_glance']} 次")
    lines.append("")
    rows = [(f"{h['violation']} [{h['start_s']:.0f}-{h['end_s']:.0f}s]",
             "✓" if h["detected"] else "✗",
             f"{h['latency_s']:.2f}s" if h["latency_s"] is not None else "—",
             str(h["n_alerts"])) for h in s["per_segment"]]
    c0 = max([_w("违规区间")] + [_w(r[0]) for r in rows]) + 2
    lines.append(f"{_pad('违规区间', c0)}{_pad('检出', 6)}{_pad('延迟', 9)}告警次数")
    for r in rows:
        lines.append(f"{_pad(r[0], c0)}{_pad(r[1], 6)}{_pad(r[2], 9)}{r[3]}")
    if s["false_alarm_list"]:
        lines.append("\n误报明细:")
        for fa in s["false_alarm_list"][:12]:
            lines.append(f"  t={fa['t']:>6.2f}s  {fa['violation']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="按 ground truth 给一条技术路线的事件输出打分")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--events", required=True, help="JSONL，每行一个 SafetyEvent.to_dict()")
    ap.add_argument("--label", default="未命名路线")
    ap.add_argument("--t0", type=float, default=None, help="视频起点的绝对时间戳")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    events = load_events(Path(args.events), args.t0)
    s = score(truth, events)
    print(render(args.label, s))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"label": args.label, **s},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.json_out}")


if __name__ == "__main__":
    main()
