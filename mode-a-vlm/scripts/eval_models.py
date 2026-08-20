"""在真实照片验证集上横向对比多个 VLM 模型。

评什么（三个甲方最关心、且人工能客观标注的轴）：
  1. **场景判别**  能不能识别出「这根本不是车内座舱」——越界输入必须被挡住，不能硬编结论；
  2. **驾驶员安全带**  模式A 的头号业务指标；
  3. **手机使用**  典型的危险驾驶行为。

怎么算分（这套口径比单一 accuracy 更贴合车载场景）：
  - 正确        模型结论 == 人工标注
  - 过度自信    人工都判不了（undecidable），模型却给了确定结论 —— 车载场景下这是最危险的错误
  - 保守        人工能判定，模型给 unknown/occluded —— 漏检，但不会打扰司机，可接受
  - 错误        模型给了与人工相反的确定结论

用法::

    source /root/.config/vsm/env
    python3 mode-a-vlm/scripts/eval_models.py --models Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-32B-Instruct
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
from vlm_safety.cost import PRICE_TABLE                # noqa: E402

VERDICT_OK, VERDICT_OVER, VERDICT_CONS, VERDICT_WRONG = "正确", "过度自信", "保守", "错误"


def _scene_pred(frame) -> str:
    if frame is None:
        return "unknown"
    return {"cabin_front": "cabin", "cabin_rear": "cabin", "exterior": "exterior",
            "not_a_vehicle_cabin": "not_cabin"}.get(frame.view, "unknown")


def _belt_pred(frame) -> str:
    if frame is None:
        return "unknown"
    occ = frame.by_seat("driver") or (frame.occupants[0] if frame.occupants else None)
    if occ is None:
        return "unknown"
    st = occ.attr("seatbelt").state
    return {"fastened": "fastened", "not_fastened": "not_fastened"}.get(st, "unknown")


def _phone_pred(frame) -> str:
    if frame is None:
        return "unknown"
    states = [o.attr("phone").state for o in frame.occupants]
    if any(s in ("held_to_ear", "held_in_view") for s in states):
        return "yes"
    if states and all(s in ("not_visible", "on_lap_or_mount") for s in states):
        return "no"
    if not states:
        return "no"          # 画面无人 -> 无人用手机
    return "unknown"


def judge(pred: str, truth: str) -> str:
    if truth == "undecidable":
        return VERDICT_OK if pred == "unknown" else VERDICT_OVER
    if pred == "unknown":
        return VERDICT_CONS
    return VERDICT_OK if pred == truth else VERDICT_WRONG


def run_model(model: str, items: list[dict], img_dir: Path, *, provider: str,
              verbose: bool = True) -> dict:
    settings = load_settings(provider=provider, model=model)
    pipe = SafetyPipeline(settings)
    rows, lat, ptok, ctok = [], [], [], []
    parse_fail = 0

    for it in items:
        path = img_dir / f"{it['id']}.jpg"
        if not path.exists():
            continue
        res = pipe.analyze([path.read_bytes()], policy="instant", dispatch=False)
        frame = res.parse.frames[0] if (res.parse and res.parse.ok and res.parse.frames) else None
        if frame is None:
            parse_fail += 1
        lat.append(res.timings_ms.get("vlm", 0.0))
        if res.vlm and res.vlm.prompt_tokens:
            ptok.append(res.vlm.prompt_tokens)
        if res.vlm and res.vlm.completion_tokens:
            ctok.append(res.vlm.completion_tokens)
        row = {
            "id": it["id"], "desc": it["desc"],
            "scene": {"pred": _scene_pred(frame), "truth": it["labels"]["scene"]},
            "belt": {"pred": _belt_pred(frame), "truth": it["labels"]["driver_seatbelt"]},
            "phone": {"pred": _phone_pred(frame), "truth": it["labels"]["phone"]},
            "events": [e.violation.value for e in res.events],
            "parse_ok": frame is not None,
            "vlm_error": res.vlm.error if res.vlm else "无响应",
            "latency_ms": round(res.timings_ms.get("vlm", 0.0), 1),
        }
        for axis in ("scene", "belt", "phone"):
            row[axis]["verdict"] = judge(row[axis]["pred"], row[axis]["truth"])
        rows.append(row)
        if verbose:
            marks = " ".join(f"{a}:{row[a]['verdict']}" for a in ("scene", "belt", "phone"))
            print(f"  [{it['id']}] {row['latency_ms']:7.0f}ms  {marks}   {it['desc'][:26]}")

    pipe.close()

    summary: dict = {"model": model, "n": len(rows), "parse_fail": parse_fail,
                     "latency_ms_avg": round(sum(lat) / len(lat), 1) if lat else 0,
                     "latency_ms_max": round(max(lat), 1) if lat else 0,
                     "prompt_tokens_avg": round(sum(ptok) / len(ptok)) if ptok else 0,
                     "completion_tokens_avg": round(sum(ctok) / len(ctok)) if ctok else 0}
    for axis in ("scene", "belt", "phone"):
        cnt: dict[str, int] = {}
        for r in rows:
            cnt[r[axis]["verdict"]] = cnt.get(r[axis]["verdict"], 0) + 1
        summary[axis] = cnt
        summary[f"{axis}_正确率"] = round(cnt.get(VERDICT_OK, 0) / len(rows), 3) if rows else 0
    price = PRICE_TABLE.get(model)
    if price:
        summary["usd_per_call"] = round(
            (summary["prompt_tokens_avg"] * price.input_per_mtok
             + summary["completion_tokens_avg"] * price.output_per_mtok) / 1e6, 6)
    return {"summary": summary, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="真实照片验证集上的多模型对比")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-VL-8B-Instruct"])
    ap.add_argument("--provider", default="siliconflow")
    ap.add_argument("--manifest", default=str(ROOT / "evalset" / "manifest.json"))
    ap.add_argument("--images", default=str(ROOT / "evalset" / "images"))
    ap.add_argument("--out", default=str(ROOT / "evalset" / "results.json"))
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    items = man["items"]
    img_dir = Path(args.images)
    if not any(img_dir.glob("*.jpg")):
        print(f"没有图片，请先跑 scripts/fetch_evalset.py")
        return 1

    all_res = []
    for model in args.models:
        print(f"\n===== {model} =====")
        t0 = time.time()
        r = run_model(model, items, img_dir, provider=args.provider)
        r["summary"]["wall_seconds"] = round(time.time() - t0, 1)
        all_res.append(r)
        s = r["summary"]
        print(f"  -> 场景 {s['scene_正确率']:.0%} | 安全带 {s['belt_正确率']:.0%} | "
              f"手机 {s['phone_正确率']:.0%} | 平均 {s['latency_ms_avg']:.0f}ms | "
              f"解析失败 {s['parse_fail']}/{s['n']}")

    print("\n===== 汇总 =====")
    hdr = f"{'模型':<34}{'场景':>7}{'安全带':>8}{'手机':>7}{'过度自信':>9}{'平均延迟':>10}{'解析失败':>9}"
    print(hdr)
    print("-" * 84)
    for r in all_res:
        s = r["summary"]
        over = sum(s[a].get(VERDICT_OVER, 0) for a in ("scene", "belt", "phone"))
        print(f"{s['model']:<34}{s['scene_正确率']:>7.0%}{s['belt_正确率']:>8.0%}"
              f"{s['phone_正确率']:>7.0%}{over:>9d}{s['latency_ms_avg']:>9.0f}ms{s['parse_fail']:>9d}")

    Path(args.out).write_text(json.dumps(
        {"manifest": args.manifest, "results": all_res}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n详细结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
