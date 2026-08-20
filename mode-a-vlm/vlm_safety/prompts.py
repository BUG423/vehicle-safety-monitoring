"""Prompt 与输出 JSON Schema。

核心反幻觉策略：**让 VLM 只做「看得见的视觉属性描述」，不做「违规判定」。**

如果直接问模型「这个司机违规了吗」，模型会把常识先验当成观察结果
（例如看到方向盘就顺口说「司机在开车所以应该系安全带，判定未系」）。
改成让它逐座位回答「肩部是否有斜跨织带」这类可证伪的问题后：

  1. 每个字段都是封闭枚举 + 必填 ``evidence`` 文字依据，模型必须指出画面里的可见线索；
  2. 允许并鼓励输出 ``unknown`` / ``occluded``，把「不确定」从错误答案变成合法答案；
  3. 违规判定放在 ``rules.py`` 的确定性代码里，可单测、可审计、可按甲方口径调阈值；
  4. 超速这类图像上根本不存在的信息，schema 里压根没有对应字段，模型无从编造。
"""
from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# 枚举白名单 —— 解析阶段用它校验模型输出，越界值一律降级为 unknown
# ---------------------------------------------------------------------------
SEATS = ["driver", "front_passenger", "rear_left", "rear_middle", "rear_right", "unknown"]
SEATBELT_STATES = ["fastened", "not_fastened", "occluded", "unknown"]
EYE_STATES = ["open", "closed", "partially_closed", "occluded", "unknown"]
MOUTH_STATES = ["normal", "yawning", "occluded", "unknown"]
GAZE_STATES = ["forward", "down", "left", "right", "up", "occluded", "unknown"]
PHONE_STATES = ["held_to_ear", "held_in_view", "on_lap_or_mount", "not_visible", "unknown"]
SMOKING_STATES = ["cigarette_visible", "not_visible", "unknown"]
HANDS_STATES = ["both_on_wheel", "one_on_wheel", "none_on_wheel", "occluded", "unknown"]
AGE_GROUPS = ["adult", "child", "unknown"]
IMAGE_QUALITY = ["good", "dark", "blurry", "overexposed", "lens_blocked", "unknown"]
VIEWS = ["cabin_front", "cabin_rear", "exterior", "not_a_vehicle_cabin", "unknown"]

def _enum(values: list[str]) -> str:
    """把枚举渲染成 ``"a|b|c"`` 形式的**字符串**。

    这里踩过一个真实的坑：最早 schema 里把枚举直接写成 JSON 数组
    （``"view": ["cabin_front", "cabin_rear", ...]``）。Qwen3-VL-32B 与 GLM-4.5V
    会把它当成「这个字段的值就是一个数组」，原样回填 ``"view": ["cabin_front"]``，
    于是每个字段都被白名单判为越界、全部降级 unknown —— 在验证集上表现为
    「大模型还不如小模型」。8B 恰好没犯这个错，差距完全是 schema 呈现方式造成的假象。

    改成竖线分隔的字符串后，模型不再有「这里应该填数组」的错觉。
    解析器同时也做了单元素数组的兼容（见 parser._unwrap），双保险。
    """
    return "|".join(values)


# 输出 JSON 的形状说明（同时作为给模型看的 schema 和给解析器用的白名单来源）
OUTPUT_SCHEMA: dict = {
    "scene": {
        "view": _enum(VIEWS),
        "image_quality": _enum(IMAGE_QUALITY),
        "persons_visible": "整数，画面中可见的人数",
    },
    "occupants": [
        {
            "seat": _enum(SEATS),
            "person_present": "true 或 false",
            "apparent_age_group": _enum(AGE_GROUPS),
            "seatbelt": {"state": _enum(SEATBELT_STATES), "evidence": "字符串", "confidence": "0~1 小数"},
            "eyes": {"state": _enum(EYE_STATES), "evidence": "字符串", "confidence": "0~1 小数"},
            "mouth": {"state": _enum(MOUTH_STATES), "evidence": "字符串", "confidence": "0~1 小数"},
            "gaze": {"state": _enum(GAZE_STATES), "evidence": "字符串", "confidence": "0~1 小数"},
            "phone": {"state": _enum(PHONE_STATES), "evidence": "字符串", "confidence": "0~1 小数"},
            "smoking": {"state": _enum(SMOKING_STATES), "evidence": "字符串", "confidence": "0~1 小数"},
            "hands": {"state": _enum(HANDS_STATES), "evidence": "字符串", "confidence": "0~1 小数"},
        }
    ],
    "notes": "字符串，可选，写下你不确定的地方",
}

SYSTEM_PROMPT = """你是车队安全检查系统的视觉核查模块。你的唯一职责是**客观描述车内图片中可见的视觉事实**。

铁律：
1. 你不做违规判定，不写「违规」「合规」「应该」之类的结论。判定由下游规则引擎负责。
2. 只描述你在这张图片里**真正看得见**的东西。看不见、被遮挡、太暗、分辨率不够 —— 一律填 "unknown" 或 "occluded"。填 unknown 不会被扣分，猜错会导致误报并让整套系统被弃用。
3. 每个判断都必须在 evidence 字段用一句中文写出画面依据（例如「左肩到右腰有一条深色斜向织带」）。写不出可见依据的，state 必须是 unknown。
4. 不要推理画面之外的信息：车速、行驶状态、这个人平时的习惯、图片拍摄时间，你都不知道。
5. 严格输出 JSON，不要 markdown 代码块，不要任何解释性文字。

安全带的判据（这是最容易出错的一项，请严格执行）：
- fastened：能看到一条从肩部斜跨到腰侧的织带，且织带贴合身体。
- not_fastened：肩部区域清晰可见且确认没有织带经过。
- occluded：人物被方向盘/衣物/角度遮挡，或图像太暗看不清肩部 —— 这种情况非常常见，不要硬猜。
"""

USER_PROMPT_TEMPLATE = """请核查这{n_images_desc}车内图片，逐个可见座位输出视觉属性。

严格按下面的 JSON 结构输出。**竖线分隔的是候选值清单，请从中挑一个填进去，
不要把整串或数组填回来**；字段名与枚举值必须完全一致，不得新增枚举值：
{schema}

补充要求：
- occupants 数组只包含**画面中确实看到有人**的座位；没人的座位不要写进去。
- 如果画面里没有任何人，occupants 输出空数组 []。
- 如果这根本不是一张车内照片，scene.view 填 "not_a_vehicle_cabin"，occupants 输出 []。
- confidence 是你对该项判断的把握程度（0~1 的小数），把握不足就给低分，不要一律填 0.9。
{extra}
只输出 JSON。"""

# 多帧序列任务的追加说明：疲劳判定必须靠时序，单帧只能给瞬时状态
MULTI_FRAME_EXTRA = """- 这是按时间先后排列的 {n} 帧连续画面。请为**每一帧**分别输出一份上述结构，
  放进顶层数组 "frames" 中：{{"frames": [{{...第1帧...}}, {{...第2帧...}}]}}。
- 每帧独立描述，不要把后面帧看到的东西写进前面帧。
- 眼睛闭合、哈欠这类瞬时状态请如实逐帧标注，下游会用它们计算 PERCLOS，你不要自己下疲劳结论。"""

SINGLE_FRAME_EXTRA = """- 这是单张静态图片。你无法判断疲劳程度、注视持续时间、车辆是否在行驶，不要尝试。"""


def build_user_prompt(n_frames: int = 1) -> str:
    """按单帧 / 多帧生成用户提示词。"""
    if n_frames > 1:
        return USER_PROMPT_TEMPLATE.format(
            n_images_desc=f"组（共 {n_frames} 帧）",
            schema=json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2),
            extra=MULTI_FRAME_EXTRA.format(n=n_frames),
        )
    return USER_PROMPT_TEMPLATE.format(
        n_images_desc="张",
        schema=json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2),
        extra=SINGLE_FRAME_EXTRA,
    )


# 给不支持 system role 的后端用的合并版
def build_merged_prompt(n_frames: int = 1) -> str:
    return SYSTEM_PROMPT + "\n\n" + build_user_prompt(n_frames)
