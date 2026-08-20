"""命令行端到端入口：图片 -> VLM -> 事件 -> 双通道告警。

用法::

    # 无 key：走 mock，全链路照样跑通
    python3 mode-a-vlm/scripts/run_pipeline.py mode-a-vlm/testdata/driver_no_seatbelt.jpg

    # 有 key：真实云端模型
    source /root/.config/vsm/env
    python3 mode-a-vlm/scripts/run_pipeline.py path/to/cabin.jpg --provider siliconflow

    # 多帧序列（走时序确认 + PERCLOS）
    python3 mode-a-vlm/scripts/run_pipeline.py mode-a-vlm/testdata/sequence_fatigue/*.jpg \\
        --policy temporal --interval 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_safety import SafetyPipeline, load_settings          # noqa: E402
from vlm_safety._common import VehicleContext                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="模式A 端到端检测")
    ap.add_argument("images", nargs="+", help="一张或多张图片（多张按时间先后排列）")
    ap.add_argument("--provider", default=None, help="auto|mock|siliconflow|anthropic|openai|dashscope|local")
    ap.add_argument("--model", default=None)
    ap.add_argument("--policy", default="auto", choices=["auto", "instant", "temporal"])
    ap.add_argument("--interval", type=float, default=1.0, help="多帧之间的时间间隔（秒）")
    ap.add_argument("--vehicle-id", default=None)
    ap.add_argument("--speed", type=float, default=None, help="OBD 车速 km/h（用于超速判定）")
    ap.add_argument("--speed-limit", type=float, default=None, help="路段限速 km/h")
    ap.add_argument("--scenario", default=None, help="仅 mock 后端：强制指定场景")
    ap.add_argument("--no-dispatch", action="store_true", help="只检测不发告警")
    ap.add_argument("--json-out", default=None, help="把完整结果写成 JSON")
    ap.add_argument("--events-jsonl", default=None, help="把事件追加写成 JSONL（bench 打分用）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    overrides = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    if args.vehicle_id:
        overrides["vehicle_id"] = args.vehicle_id
    if args.scenario:
        overrides["scenario"] = args.scenario
    settings = load_settings(**overrides)

    provider = None
    if settings.resolve_provider() == "mock" and args.scenario:
        from vlm_safety.providers.mock import MockProvider
        provider = MockProvider(settings, scenario=args.scenario)

    pipe = SafetyPipeline(settings, provider=provider)
    print(f"[配置] provider={pipe.provider.name} model={pipe.provider.model} "
          f"模拟={pipe.provider.simulated}")

    blobs = [Path(p).read_bytes() for p in args.images]
    ctx = None
    if args.speed is not None:
        ctx = VehicleContext(vehicle_id=settings.vehicle_id, speed_kmh=args.speed,
                             speed_limit_kmh=args.speed_limit, engine_on=True)

    res = pipe.analyze(blobs, vehicle_ctx=ctx, policy=args.policy,
                       frame_interval_s=args.interval, dispatch=not args.no_dispatch)

    if not args.quiet:
        print("\n===== 检测结果 =====")
        print(f"策略: {res.policy}   帧数: {len(blobs)}   "
              f"模拟输出: {'是（无真实视觉理解）' if res.simulated else '否（真实模型）'}")
        print(f"耗时(ms): {json.dumps({k: round(v, 1) for k, v in res.timings_ms.items()}, ensure_ascii=False)}")
        if res.vlm:
            print(f"token: prompt={res.vlm.prompt_tokens} completion={res.vlm.completion_tokens}")
        if res.perclos:
            print(f"PERCLOS: {json.dumps(res.perclos.to_dict(), ensure_ascii=False)}")
        print(f"\n事件 {len(res.events)} 条：")
        for e in res.events:
            print(f"  [{e.severity.value:8s}] {e.violation.label_zh:<12s} "
                  f"座位={e.subject.seat or '-'} 置信={e.confidence:.2f} "
                  f"依据={e.raw_signals.get('evidence_text', '')[:60]}")
        if res.undecidable:
            print(f"\n判不了（显式呈现，不算合规） {len(res.undecidable)} 条：")
            for u in res.undecidable:
                print(f"  - {u.violation.label_zh}（{u.seat}）：{u.reason}")
        if res.notes:
            print("\n备注：")
            for n in res.notes:
                print(f"  - {n}")
        if res.alerts:
            print("\n告警投递：")
            for a in res.alerts:
                print(f"  {a['violation']} -> {a['channels']}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整结果已写入 {args.json_out}")
    if args.events_jsonl:
        with open(args.events_jsonl, "a", encoding="utf-8") as f:
            for e in res.events:
                f.write(e.to_json(drop_b64=True) + "\n")

    pipe.close()
    return 0 if res.ok or not blobs else 1


if __name__ == "__main__":
    raise SystemExit(main())
