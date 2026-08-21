"""三条路线与混合部署的总拥有成本（TCO）测算。

**全部为估算**，参数取自各路线 DESIGN.md 的测算，来源标在 `SOURCES` 里。
实际报价必须走询价，本模型的用途是看**结构和敏感性**，不是给出准确报价。

回答的核心问题：混合部署是不是三套成本相加？
结论：不是，但它有一个会失控的变量 —— 云端复核调用率。

    python3 tools/cost_model.py                     # 默认 500 台车 / 3 年
    python3 tools/cost_model.py --fleet 100 --years 2
    python3 tools/cost_model.py --review-per-day 40 # 边缘侧误报多时的成本
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

SOURCES = {
    "C_硬件": "mode-c-edge/DESIGN.md §8.1  RK3588 方案含安装工时",
    "C_月费": "mode-c-edge/DESIGN.md §8.1  4G ¥8 + 后台分摊 ¥4",
    "B_硬件": "mode-c-edge/DESIGN.md §8.2  传输盒方案（无车端算力）",
    "B_月费": "mode-c-edge/DESIGN.md §8.3  含帧上传流量与 GPU 分摊",
    "A_复核": "mode-a-vlm/DESIGN.md §6   20 次/天/车 → $158/月/100车",
    "A_发车前": "mode-a-vlm/DESIGN.md §6   2 次/天/车 → $13/月/100车",
}

USD_CNY = 7.2   # 汇率，估算用


@dataclass
class Plan:
    name: str
    capex_per_vehicle: float      # 单车一次性（硬件 + 安装）
    opex_per_vehicle: float       # 单车月费
    note: str
    offline_capable: bool
    fleet_dashboard: bool
    rule_agility: str             # 新增违规类型的代价


def build_plans(review_per_day: float, precheck_per_day: float = 2.0) -> list[Plan]:
    """按调用率构造四种方案。

    模式A 的成本按调用次数线性缩放：DESIGN.md 实测 20 次/天/车 = $158/月/100车，
    即每次调用约 $0.00263。
    """
    usd_per_call = 158 / 100 / 30 / 20          # $/次
    cny_per_call = usd_per_call * USD_CNY

    a_review_cny = cny_per_call * review_per_day * 30
    a_precheck_cny = cny_per_call * precheck_per_day * 30

    return [
        Plan("模式A 纯云端（发车前检查）", 0, a_precheck_cny,
             "无车端硬件；只做发车前静态检查，不做行车中监测", False, False, "改提示词"),
        Plan("模式B 纯后台", 1100, 103,
             "车端只装传输盒；后台 GPU 集中推理", False, True, "需训练模型"),
        Plan("模式C 纯车载", 1365, 12,
             "算力在车上；后台只收事件", True, False, "需训练 + OTA"),
        Plan("混合 C+B+A", 1365, 8 + 8 + a_review_cny,
             f"车端检测(C) + 后台汇总(B，无GPU) + 云端复核(A，{review_per_day:.0f}次/天)",
             True, True, "车端需训练；新规则可先由云端顶上"),
    ]


def tco(plan: Plan, fleet: int, years: int) -> dict:
    months = years * 12
    capex = plan.capex_per_vehicle * fleet
    opex = plan.opex_per_vehicle * fleet * months
    return {"capex": capex, "opex": opex, "total": capex + opex,
            "per_vehicle_month": (capex + opex) / fleet / months}


def render(fleet: int, years: int, review_per_day: float) -> str:
    plans = build_plans(review_per_day)
    rows = [(p, tco(p, fleet, years)) for p in plans]
    L = [f"\n{'='*86}",
         f"总拥有成本测算（估算）  车队 {fleet} 台 · 周期 {years} 年 · 云端复核 {review_per_day:.0f} 次/天/车",
         f"{'='*86}",
         f"{'方案':<26}{'一次性':>12}{'运营':>14}{'合计':>14}{'折合单车月':>12}",
         "-"*86]
    for p, t in rows:
        L.append(f"{p.name:<24}{t['capex']/10000:>10.1f}万{t['opex']/10000:>12.1f}万"
                 f"{t['total']/10000:>12.1f}万{t['per_vehicle_month']:>11.0f}元")
    L.append("-"*86)

    base = rows[2][1]["total"]                    # 以纯车载为基准
    hyb = rows[3][1]["total"]
    b_total = rows[1][1]["total"]
    L.append(f"\n混合 vs 纯车载C：贵 {(hyb-base)/10000:.1f} 万（{(hyb/base-1)*100:+.0f}%）"
             f"   混合 vs 纯后台B：省 {(b_total-hyb)/10000:.1f} 万（{(1-hyb/b_total)*100:.0f}%）")

    L.append(f"\n{'能力对比':<26}{'断网可用':>10}{'车队看板':>10}   新增违规类型")
    L.append("-"*86)
    for p in plans:
        L.append(f"{p.name:<24}{'是' if p.offline_capable else '否':>10}"
                 f"{'是' if p.fleet_dashboard else '否':>10}   {p.rule_agility}")
    return "\n".join(L)


def sensitivity(fleet: int, years: int) -> str:
    """复核调用率是混合方案唯一会失控的变量 —— 它由边缘侧误报率决定。"""
    L = [f"\n{'='*86}",
         "敏感性：云端复核调用率（= 边缘侧判为可疑的事件率）",
         f"{'='*86}",
         f"{'复核次数/天/车':>16}{'混合单车月费':>14}{'混合3年合计':>14}{'相对纯车载C':>14}   评价",
         "-"*86]
    base = tco(build_plans(0)[2], fleet, years)["total"]
    for r in (0, 5, 10, 20, 40, 80, 160):
        p = build_plans(r)[3]
        t = tco(p, fleet, years)
        ratio = t["total"] / base
        verdict = ("复核几乎不花钱" if r <= 5 else
                   "成本可接受" if r <= 20 else
                   "开始显著" if r <= 40 else
                   "复核费超过硬件摊销" if r <= 80 else "失控，必须限流")
        L.append(f"{r:>16}{p.opex_per_vehicle:>13.0f}元{t['total']/10000:>13.1f}万"
                 f"{ratio:>13.2f}x   {verdict}")
    L.append("-"*86)
    L.append("每次复核调用约 ¥0.019（4 帧）。20 次/天/车 = ¥11.4/月/车，已接近车端 4G 流量费的 1.5 倍。")
    return "\n".join(L)


def breakeven(years: int) -> str:
    """只在**提供行车中监测**的方案之间比较。

    模式A 单独一列作参考：它只做发车前静态检查，不监测行车过程，
    与另外三者能力不对等，按金额横向比会得出「A 永远最便宜」这种无意义的结论。
    """
    L = [f"\n{'='*86}", "规模敏感性：3 年合计（估算）", f"{'='*86}",
         f"{'车队规模':>10}{'模式B':>12}{'模式C':>12}{'混合':>12}   最省"
         f"      │{'模式A':>10}（不监测行车，仅参考）"]
    L.append("-"*86)
    for fleet in (5, 10, 30, 50, 100, 300, 500, 1000):
        plans = build_plans(20)
        a, b, c, h = [tco(p, fleet, years)["total"] / 10000 for p in plans]
        comparable = {"模式B": b, "模式C": c, "混合": h}
        best = min(comparable, key=comparable.get)
        L.append(f"{fleet:>10}{b:>11.1f}万{c:>11.1f}万{h:>11.1f}万{best:>8}"
                 f"      │{a:>9.1f}万")
    L.append("-"*86)
    L.append("纯车载C 在任何规模下都是最省的 —— 因为混合是在它之上叠加能力，不是替代它。")
    L.append("所以问题不是「混合是不是更贵」（一定更贵），而是「多花的钱买到的能力值不值」。")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="三路线与混合部署的 TCO 测算（估算）")
    ap.add_argument("--fleet", type=int, default=500)
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--review-per-day", type=float, default=20,
                    help="每车每天的云端复核调用次数，由边缘侧可疑事件率决定")
    args = ap.parse_args()

    print(render(args.fleet, args.years, args.review_per_day))
    print(sensitivity(args.fleet, args.years))
    print(breakeven(args.years))
    print(f"\n{'='*86}\n参数来源（均为估算，实际报价需询价）")
    for k, v in SOURCES.items():
        print(f"  {k:<10} {v}")
    print(f"  汇率        1 USD = {USD_CNY} CNY")


if __name__ == "__main__":
    main()
