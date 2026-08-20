"""生成测试用的合成驾驶舱图片（纯 PIL，无外部素材依赖）。

**诚实声明**：这是卡通化的合成图，**不能用来评估真实感知精度**。
它的用途是让全链路（上传 -> provider -> 解析 -> 规则 -> 确认 -> 告警 -> 前端）
有稳定、可复现、可入库的输入。真实精度评测需要 DMD / StateFarm 这类真实标注数据集，
见 ``bench/README.md`` 与 ``DESIGN.md``。

用法::

    python3 mode-a-vlm/scripts/gen_test_images.py --out mode-a-vlm/testdata
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W, H = 640, 480

SKIN = (214, 176, 148)
SHIRT = (96, 108, 132)
BELT = (38, 38, 44)
DASH = (38, 34, 30)
ROOF = (52, 46, 40)
BG = (28, 28, 28)


def _person(d: ImageDraw.ImageDraw, cx: int, *, belt: bool, eyes_closed: bool = False,
            yawning: bool = False, phone: bool = False, cigarette: bool = False,
            child: bool = False, gaze: str = "forward") -> None:
    scale = 0.72 if child else 1.0
    cy = 250 + (30 if child else 0)
    bw, bh = int(78 * scale), int(90 * scale)
    hr = int(48 * scale)

    # 身体
    d.ellipse([cx - bw, cy + 30, cx + bw, cy + 30 + bh * 2], fill=SHIRT)
    # 安全带（从左肩斜跨到右腰）
    if belt:
        d.line([(cx - bw + 12, cy + 40), (cx + bw - 24, cy + 30 + bh)], fill=BELT, width=11)
    # 头
    dx = {"forward": 0, "left": -18, "right": 18, "down": 0}.get(gaze, 0)
    dy = 14 if gaze == "down" else 0
    d.ellipse([cx - hr + dx, cy - hr * 2 + dy, cx + hr + dx, cy + dy], fill=SKIN)
    # 眼睛
    ey = cy - hr - 6 + dy
    for ox in (-18, 18):
        ex = cx + dx + int(ox * scale)
        if eyes_closed:
            d.line([(ex - 10, ey), (ex + 10, ey)], fill=(40, 40, 40), width=4)
        else:
            d.ellipse([ex - 9, ey - 8, ex + 9, ey + 8], fill=(250, 250, 250))
            d.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=(30, 30, 30))
    # 嘴
    my = cy - 16 + dy
    if yawning:
        d.ellipse([cx + dx - 16, my - 12, cx + dx + 16, my + 18], fill=(80, 30, 30))
    else:
        d.line([(cx + dx - 14, my), (cx + dx + 14, my)], fill=(120, 60, 60), width=4)
    # 手机
    if phone:
        d.rounded_rectangle([cx + dx - hr - 26, ey - 14, cx + dx - hr - 6, ey + 34],
                            radius=4, fill=(20, 20, 24), outline=(90, 160, 220), width=2)
    # 香烟
    if cigarette:
        d.line([(cx + dx + 14, my), (cx + dx + 46, my - 8)], fill=(245, 245, 235), width=5)
        d.ellipse([cx + dx + 44, my - 12, cx + dx + 52, my - 4], fill=(255, 120, 40))


def _cabin(d: ImageDraw.ImageDraw, *, dark: bool = False) -> None:
    d.rectangle([0, 0, W, 150], fill=ROOF)
    d.rectangle([0, 380, W, H], fill=DASH)
    d.line([(0, 380), (W, 380)], fill=(70, 64, 56), width=2)
    # 方向盘
    d.ellipse([100, 330, 280, 470], outline=(70, 70, 78), width=12)


def make(scenario: str) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    if scenario == "camera_blocked":
        img = Image.new("RGB", (W, H), (22, 22, 24))
        ImageDraw.Draw(img).rectangle([0, 0, W, H], fill=(26, 24, 22))
        return img
    if scenario == "not_a_cabin":
        img = Image.new("RGB", (W, H), (120, 170, 210))
        d2 = ImageDraw.Draw(img)
        d2.rectangle([0, 320, W, H], fill=(96, 140, 84))
        d2.ellipse([440, 40, 560, 160], fill=(250, 235, 120))
        return img

    _cabin(d)
    if scenario == "empty_cabin":
        return img

    opts = {
        "all_clear": (dict(belt=True), dict(belt=True)),
        "driver_no_seatbelt": (dict(belt=False), None),
        "phone_use": (dict(belt=True, phone=True, gaze="left"), None),
        "smoking": (dict(belt=True, cigarette=True), None),
        "fatigue": (dict(belt=True, eyes_closed=True, yawning=True), None),
        "passenger_no_seatbelt": (dict(belt=True), dict(belt=False, child=True)),
    }.get(scenario, (dict(belt=True), dict(belt=True)))

    if opts[0]:
        _person(d, 190, **opts[0])
    if opts[1]:
        _person(d, 450, **opts[1])
    return img


def main() -> None:
    from vlm_safety.providers.mock import SCENARIOS

    ap = argparse.ArgumentParser(description="生成模式A 的合成测试图片")
    ap.add_argument("--out", default="mode-a-vlm/testdata")
    ap.add_argument("--sequence", type=int, default=8,
                    help="额外生成一段 N 帧的疲劳序列，用于验证 PERCLOS 与防误报确认")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in SCENARIOS:
        p = out / f"{name}.jpg"
        make(name).save(p, quality=88)
        print(f"生成 {p}")

    if args.sequence > 0:
        seq = out / "sequence_fatigue"
        seq.mkdir(exist_ok=True)
        for i in range(args.sequence):
            # 约 70% 的帧闭眼 —— 与 mock provider 的抖动模型对齐
            closed = (i % 10) not in (2, 7, 9)
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            _cabin(d)
            _person(d, 190, belt=True, eyes_closed=closed, yawning=(i % 5 == 0))
            _person(d, 450, belt=True)
            img.save(seq / f"frame_{i:02d}.jpg", quality=88)
        print(f"生成 {args.sequence} 帧序列于 {seq}")


if __name__ == "__main__":
    main()
