"""成本与延迟测算。

**价格口径声明**：下表是各家公开标价的**量级参考**，用于选型阶段的数量级比较，
不是报价依据。签约前必须以当期官网价格 / 商务折扣 / 实际账单为准；
本环境无 API key，也无法用真实账单校准，这一点在 DESIGN.md 里同样写明。

之所以把它写成代码而不是文档里的一张表：调用量、分辨率、帧数一变，结论就变，
让它可执行才能在需求变化时立刻重算（``python3 -m vlm_safety.cost``）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Price:
    """每百万 token 的价格（美元）。"""

    input_per_mtok: float
    output_per_mtok: float
    note: str = ""


#: 公开标价量级参考（2026 年中口径，USD/MTok）
PRICE_TABLE: dict[str, Price] = {
    "claude-sonnet-4-5": Price(3.0, 15.0, "Anthropic 标价"),
    "claude-haiku-4-5": Price(1.0, 5.0, "Anthropic 标价，轻量档"),
    "gpt-4o-mini": Price(0.15, 0.60, "OpenAI 标价，轻量档"),
    "gpt-4o": Price(2.5, 10.0, "OpenAI 标价"),
    "qwen-vl-max-latest": Price(0.42, 1.26, "阿里百炼标价折算（约 ¥3/¥9 每百万 token，汇率 7.2）"),
    "qwen-vl-plus": Price(0.21, 0.63, "阿里百炼标价折算，性价比档"),
    "local-qwen2.5-vl-7b": Price(0.0, 0.0, "本地推理，成本体现在显卡折旧而非 token"),
}

#: 本地部署的摊销成本假设
LOCAL_GPU_COST_PER_HOUR_USD = 1.2      # 一张 A10/A100 级别显卡的云上租用价量级
LOCAL_THROUGHPUT_CALLS_PER_HOUR = 900  # 7B VLM 单卡、单张 896px 图、约 4s/次的保守吞吐


@dataclass
class CostEstimate:
    model: str
    calls_per_day: int
    input_tokens_per_call: int
    output_tokens_per_call: int
    usd_per_call: float
    usd_per_day: float
    usd_per_month: float
    note: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def estimate(model: str, *, calls_per_day: int, input_tokens: int, output_tokens: int) -> CostEstimate:
    price = PRICE_TABLE.get(model)
    if price is None:
        price = Price(0.0, 0.0, "未知模型，按 0 计（请补充价格）")
    per_call = (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1e6
    return CostEstimate(model=model, calls_per_day=calls_per_day,
                        input_tokens_per_call=input_tokens, output_tokens_per_call=output_tokens,
                        usd_per_call=per_call, usd_per_day=per_call * calls_per_day,
                        usd_per_month=per_call * calls_per_day * 30, note=price.note)


def estimate_local(*, calls_per_day: int) -> CostEstimate:
    """本地部署：成本 = 显卡小时价 / 单卡每小时可处理次数。"""
    per_call = LOCAL_GPU_COST_PER_HOUR_USD / LOCAL_THROUGHPUT_CALLS_PER_HOUR
    return CostEstimate(model="local-qwen2.5-vl-7b", calls_per_day=calls_per_day,
                        input_tokens_per_call=0, output_tokens_per_call=0,
                        usd_per_call=per_call, usd_per_day=per_call * calls_per_day,
                        usd_per_month=per_call * calls_per_day * 30,
                        note=f"按 ${LOCAL_GPU_COST_PER_HOUR_USD}/卡·小时、"
                             f"{LOCAL_THROUGHPUT_CALLS_PER_HOUR} 次/卡·小时 摊销")


#: 三种典型用法的调用量假设（每车每天）
USAGE_PROFILES: dict[str, dict] = {
    "发车前一次性检查": {
        "calls_per_vehicle_per_day": 2,     # 早班发车 + 换班各一次
        "frames_per_call": 3,               # 主驾特写 + 全舱 + 后排
        "desc": "车辆出场前拍 2~3 张图做静态安全核查，这是模式A 最合适的落点",
    },
    "边缘上报后云端复核": {
        "calls_per_vehicle_per_day": 20,    # 车端初筛出的可疑事件量级
        "frames_per_call": 4,
        "desc": "车端设备初筛，只把可疑帧上云复核，云端只承担 1~5% 的帧量",
    },
    "7x24 逐帧实时": {
        "calls_per_vehicle_per_day": 28800,  # 8 小时 × 1fps
        "frames_per_call": 1,
        "desc": "反例：论证模式A 不适合做实时逐帧监控",
    },
}


def fleet_report(*, fleet_size: int = 100, model: str = "qwen-vl-max-latest",
                 input_tokens_per_frame: int = 1100, prompt_tokens: int = 900,
                 output_tokens: int = 320) -> list[dict]:
    """给出车队级别的月度成本量级对比。"""
    rows = []
    for name, prof in USAGE_PROFILES.items():
        frames = prof["frames_per_call"]
        in_tok = prompt_tokens + frames * input_tokens_per_frame
        calls_day = prof["calls_per_vehicle_per_day"] * fleet_size
        cloud = estimate(model, calls_per_day=calls_day, input_tokens=in_tok,
                         output_tokens=output_tokens)
        local = estimate_local(calls_per_day=calls_day)
        rows.append({
            "用法": name, "说明": prof["desc"],
            "每车每天调用": prof["calls_per_vehicle_per_day"], "每次帧数": frames,
            "车队每天调用": calls_day,
            "云端每次(USD)": round(cloud.usd_per_call, 5),
            "云端每月(USD)": round(cloud.usd_per_month, 2),
            "本地每月(USD)": round(local.usd_per_month, 2),
            "本地需卡数": max(1, round(calls_day / (LOCAL_THROUGHPUT_CALLS_PER_HOUR * 24) + 0.49)),
        })
    return rows


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(fleet_report(), ensure_ascii=False, indent=2))
