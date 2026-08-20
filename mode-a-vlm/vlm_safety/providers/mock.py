"""Mock Provider —— 无任何 API key 时保证全链路可演示。

**诚实声明**：本 provider 不做任何视觉理解。它按下面两条规则合成一段
「格式与真实 VLM 完全一致」的 JSON 回答：

  1. 若调用方通过 ``scenario`` 显式指定场景（或图片里嵌了场景标签，见
     ``scripts/gen_test_images.py``），就返回该场景对应的预置观测；
  2. 否则用图片字节的 SHA-256 做确定性哈希选一个场景 —— 同一张图永远得到
     同一个结果，便于回归测试，但**与图片内容无关**。

它的价值在于：Mock 之后的每一环（JSON 解析、枚举校验、规则映射、防误报确认、
事件构造、双通道告警、H5 展示）都是真实代码，可以被真实地跑通和测试。
把 provider 换成 anthropic/openai/dashscope/local，下游一行不改。
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Sequence

from ..imaging import PreparedImage
from .base import VLMProvider, VLMResponse

#: 预置场景 —— 覆盖甲方需求里的主要违规组合
SCENARIOS: dict[str, dict] = {
    "all_clear": {
        "_desc": "驾驶员与乘客均系安全带，状态正常",
        "scene": {"view": "cabin_front", "image_quality": "good", "persons_visible": 2},
        "occupants": [
            {
                "seat": "driver", "person_present": True, "apparent_age_group": "adult",
                "seatbelt": {"state": "fastened", "evidence": "左肩至右腰可见深色斜向织带", "confidence": 0.93},
                "eyes": {"state": "open", "evidence": "双眼睁开，瞳孔清晰", "confidence": 0.9},
                "mouth": {"state": "normal", "evidence": "嘴部闭合", "confidence": 0.88},
                "gaze": {"state": "forward", "evidence": "面部朝向正前方", "confidence": 0.85},
                "phone": {"state": "not_visible", "evidence": "双手区域无手持设备", "confidence": 0.9},
                "smoking": {"state": "not_visible", "evidence": "口部及手部未见烟支", "confidence": 0.92},
                "hands": {"state": "both_on_wheel", "evidence": "两手分别位于方向盘两侧", "confidence": 0.87},
            },
            {
                "seat": "front_passenger", "person_present": True, "apparent_age_group": "adult",
                "seatbelt": {"state": "fastened", "evidence": "右肩可见斜跨织带", "confidence": 0.9},
                "eyes": {"state": "open", "evidence": "双眼睁开", "confidence": 0.8},
                "mouth": {"state": "normal", "evidence": "嘴部闭合", "confidence": 0.8},
                "gaze": {"state": "forward", "evidence": "面部朝前", "confidence": 0.7},
                "phone": {"state": "not_visible", "evidence": "未见手机", "confidence": 0.85},
                "smoking": {"state": "not_visible", "evidence": "未见烟支", "confidence": 0.9},
                "hands": {"state": "unknown", "evidence": "非驾驶位不适用", "confidence": 0.5},
            },
        ],
        "notes": "光照良好，两名乘员均清晰可见。",
    },
    "driver_no_seatbelt": {
        "_desc": "驾驶员未系安全带",
        "scene": {"view": "cabin_front", "image_quality": "good", "persons_visible": 1},
        "occupants": [
            {
                "seat": "driver", "person_present": True, "apparent_age_group": "adult",
                "seatbelt": {"state": "not_fastened", "evidence": "肩部与胸前清晰可见，无任何织带经过", "confidence": 0.91},
                "eyes": {"state": "open", "evidence": "双眼睁开", "confidence": 0.88},
                "mouth": {"state": "normal", "evidence": "嘴部闭合", "confidence": 0.85},
                "gaze": {"state": "forward", "evidence": "面部朝前", "confidence": 0.8},
                "phone": {"state": "not_visible", "evidence": "未见手机", "confidence": 0.88},
                "smoking": {"state": "not_visible", "evidence": "未见烟支", "confidence": 0.9},
                "hands": {"state": "both_on_wheel", "evidence": "双手握方向盘", "confidence": 0.84},
            }
        ],
        "notes": "驾驶位安全带插扣区域未见织带。",
    },
    "phone_use": {
        "_desc": "驾驶员手持手机贴耳",
        "scene": {"view": "cabin_front", "image_quality": "good", "persons_visible": 1},
        "occupants": [
            {
                "seat": "driver", "person_present": True, "apparent_age_group": "adult",
                "seatbelt": {"state": "fastened", "evidence": "可见斜跨织带", "confidence": 0.86},
                "eyes": {"state": "open", "evidence": "双眼睁开", "confidence": 0.85},
                "mouth": {"state": "normal", "evidence": "嘴部张合，似在说话", "confidence": 0.7},
                "gaze": {"state": "left", "evidence": "头部向左偏转", "confidence": 0.75},
                "phone": {"state": "held_to_ear", "evidence": "左手持矩形深色物体紧贴左耳", "confidence": 0.89},
                "smoking": {"state": "not_visible", "evidence": "未见烟支", "confidence": 0.9},
                "hands": {"state": "one_on_wheel", "evidence": "仅右手在方向盘上", "confidence": 0.86},
            }
        ],
        "notes": "手机为矩形深色物体，贴于左耳位置。",
    },
    "smoking": {
        "_desc": "驾驶员抽烟",
        "scene": {"view": "cabin_front", "image_quality": "good", "persons_visible": 1},
        "occupants": [
            {
                "seat": "driver", "person_present": True, "apparent_age_group": "adult",
                "seatbelt": {"state": "fastened", "evidence": "可见斜跨织带", "confidence": 0.88},
                "eyes": {"state": "open", "evidence": "双眼睁开", "confidence": 0.87},
                "mouth": {"state": "normal", "evidence": "嘴角叼有细长白色物体", "confidence": 0.8},
                "gaze": {"state": "forward", "evidence": "面部朝前", "confidence": 0.82},
                "phone": {"state": "not_visible", "evidence": "未见手机", "confidence": 0.88},
                "smoking": {"state": "cigarette_visible", "evidence": "嘴角细长白色柱体，端部有亮点", "confidence": 0.83},
                "hands": {"state": "one_on_wheel", "evidence": "右手在方向盘，左手抬至面部", "confidence": 0.8},
            }
        ],
        "notes": "白色柱状物端部有亮点，符合点燃状态。",
    },
    "fatigue": {
        "_desc": "驾驶员闭眼 + 哈欠（单帧只能给瞬时状态）",
        "scene": {"view": "cabin_front", "image_quality": "good", "persons_visible": 1},
        "occupants": [
            {
                "seat": "driver", "person_present": True, "apparent_age_group": "adult",
                "seatbelt": {"state": "fastened", "evidence": "可见斜跨织带", "confidence": 0.9},
                "eyes": {"state": "closed", "evidence": "上下眼睑闭合，未见瞳孔", "confidence": 0.86},
                "mouth": {"state": "yawning", "evidence": "口部大幅张开呈椭圆", "confidence": 0.81},
                "gaze": {"state": "occluded", "evidence": "闭眼状态无法判断视线", "confidence": 0.4},
                "phone": {"state": "not_visible", "evidence": "未见手机", "confidence": 0.9},
                "smoking": {"state": "not_visible", "evidence": "未见烟支", "confidence": 0.9},
                "hands": {"state": "both_on_wheel", "evidence": "双手握方向盘", "confidence": 0.85},
            }
        ],
        "notes": "本帧眼睛闭合。单帧无法判定疲劳，需下游按时序计算 PERCLOS。",
    },
    "passenger_no_seatbelt": {
        "_desc": "后排乘客未系安全带 + 儿童坐副驾",
        "scene": {"view": "cabin_rear", "image_quality": "good", "persons_visible": 3},
        "occupants": [
            {
                "seat": "driver", "person_present": True, "apparent_age_group": "adult",
                "seatbelt": {"state": "fastened", "evidence": "可见斜跨织带", "confidence": 0.9},
                "eyes": {"state": "open", "evidence": "双眼睁开", "confidence": 0.85},
                "mouth": {"state": "normal", "evidence": "嘴部闭合", "confidence": 0.85},
                "gaze": {"state": "forward", "evidence": "面部朝前", "confidence": 0.8},
                "phone": {"state": "not_visible", "evidence": "未见手机", "confidence": 0.88},
                "smoking": {"state": "not_visible", "evidence": "未见烟支", "confidence": 0.9},
                "hands": {"state": "both_on_wheel", "evidence": "双手握方向盘", "confidence": 0.82},
            },
            {
                "seat": "front_passenger", "person_present": True, "apparent_age_group": "child",
                "seatbelt": {"state": "not_fastened", "evidence": "肩部可见，无织带", "confidence": 0.84},
                "eyes": {"state": "open", "evidence": "双眼睁开", "confidence": 0.8},
                "mouth": {"state": "normal", "evidence": "嘴部闭合", "confidence": 0.75},
                "gaze": {"state": "right", "evidence": "头部朝向车窗", "confidence": 0.7},
                "phone": {"state": "not_visible", "evidence": "未见手机", "confidence": 0.8},
                "smoking": {"state": "not_visible", "evidence": "未见烟支", "confidence": 0.9},
                "hands": {"state": "unknown", "evidence": "非驾驶位不适用", "confidence": 0.5},
            },
            {
                "seat": "rear_left", "person_present": True, "apparent_age_group": "adult",
                "seatbelt": {"state": "not_fastened", "evidence": "后排座椅肩部区域清晰，未见织带", "confidence": 0.78},
                "eyes": {"state": "open", "evidence": "双眼睁开", "confidence": 0.7},
                "mouth": {"state": "normal", "evidence": "嘴部闭合", "confidence": 0.7},
                "gaze": {"state": "down", "evidence": "低头", "confidence": 0.65},
                "phone": {"state": "held_in_view", "evidence": "手中持发光矩形物体", "confidence": 0.7},
                "smoking": {"state": "not_visible", "evidence": "未见烟支", "confidence": 0.85},
                "hands": {"state": "unknown", "evidence": "非驾驶位不适用", "confidence": 0.5},
            },
        ],
        "notes": "副驾为儿童体型；后排左侧乘员未系安全带。",
    },
    "camera_blocked": {
        "_desc": "镜头被遮挡",
        "scene": {"view": "unknown", "image_quality": "lens_blocked", "persons_visible": 0},
        "occupants": [],
        "notes": "画面整体均匀暗色，疑似镜头被遮挡或贴纸覆盖。",
    },
    "empty_cabin": {
        "_desc": "驾驶位无人",
        "scene": {"view": "cabin_front", "image_quality": "good", "persons_visible": 0},
        "occupants": [],
        "notes": "座舱内未见人员。",
    },
    "not_a_cabin": {
        "_desc": "上传的不是车内照片（越界输入）",
        "scene": {"view": "not_a_vehicle_cabin", "image_quality": "good", "persons_visible": 0},
        "occupants": [],
        "notes": "画面不是车辆座舱，无法进行安全检查。",
    },
}

#: 哈希兜底时的候选池（不含越界场景，避免随机命中造成困惑）
_HASH_POOL = ["all_clear", "driver_no_seatbelt", "phone_use", "smoking",
              "fatigue", "passenger_no_seatbelt"]


def scenario_names() -> list[str]:
    return list(SCENARIOS.keys())


def pick_scenario(seed_bytes: bytes) -> str:
    """按图片字节确定性地选一个场景 —— 同图同果，可回归。"""
    h = hashlib.sha256(seed_bytes).digest()
    return _HASH_POOL[h[0] % len(_HASH_POOL)]


def _degrade_for_frame(payload: dict, idx: int, n: int, rng: random.Random) -> dict:
    """多帧场景下让各帧之间有合理抖动。

    真实模型逐帧输出必然抖动（眨眼、遮挡、光照），Mock 也必须抖，
    否则下游的滑窗投票 / PERCLOS 计算就永远跑在理想输入上，测不出问题。
    """
    out = json.loads(json.dumps(payload))
    for occ in out.get("occupants", []):
        eyes = occ.get("eyes", {})
        if eyes.get("state") == "closed":
            # 疲劳场景：约 70% 的帧闭眼，剩余帧睁眼 —— 模拟真实 PERCLOS 波形
            if rng.random() > 0.7:
                eyes["state"] = "open"
                eyes["evidence"] = "本帧双眼睁开"
        elif eyes.get("state") == "open":
            # 正常场景：约 8% 的帧因眨眼被判闭眼 —— 用来验证防误报确认器
            if rng.random() < 0.08:
                eyes["state"] = "closed"
                eyes["evidence"] = "本帧眼睑闭合（瞬时眨眼）"
        for key in ("seatbelt", "phone", "smoking"):
            item = occ.get(key, {})
            if isinstance(item.get("confidence"), (int, float)):
                item["confidence"] = round(min(0.99, max(0.3, item["confidence"] + rng.uniform(-0.06, 0.06))), 3)
    out.setdefault("scene", {})["frame_index"] = idx
    return out


class MockProvider(VLMProvider):
    """规则化模拟后端。无 key 环境下的默认选择。"""

    name = "mock"
    simulated = True

    @property
    def default_model(self) -> str:
        return "mock-rules-v1"

    def __init__(self, settings, scenario: str | None = None) -> None:
        super().__init__(settings)
        self.scenario = scenario or settings.extra.get("scenario")
        self.latency_ms = float(settings.extra.get("mock_latency_ms", 120))

    def _payload_for(self, images) -> tuple[dict | list[dict], str]:
        seed = images[0].jpeg if images else b"empty"
        name = self.scenario if self.scenario in SCENARIOS else pick_scenario(seed)
        base = SCENARIOS[name]
        payload = {k: v for k, v in base.items() if not k.startswith("_")}
        if len(images) <= 1:
            return payload, name
        rng = random.Random(hashlib.sha256(seed).digest()[:8])
        return [_degrade_for_frame(payload, i, len(images), rng) for i in range(len(images))], name

    def _invoke(self, images, system, user) -> VLMResponse:
        time.sleep(self.latency_ms / 1000.0)   # 模拟网络/推理延迟，让前端 loading 态可见
        payload, name = self._payload_for(images)
        if isinstance(payload, list):
            body: dict = {"frames": payload}
        else:
            body = payload
        text = json.dumps(body, ensure_ascii=False, indent=2)
        return VLMResponse(
            text=text, provider=self.name, model=self.model, simulated=True,
            prompt_tokens=sum(im.est_vision_tokens() for im in images) + len(system + user) // 4,
            completion_tokens=len(text) // 3,
            raw={"scenario": name, "note": "模拟输出，未做任何视觉理解"},
        )

    def health(self) -> dict:
        d = super().health()
        d["detail"] = f"规则化模拟，可用场景：{', '.join(SCENARIOS)}"
        return d
