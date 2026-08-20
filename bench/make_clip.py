"""生成带 ground truth 标注的合成驾驶舱视频，供三条路线跑同一份输入。

为什么需要它
------------
三条技术路线如果各自用各自的素材做演示，产出的数字不可比，选型就没有依据。
本脚本生成一段**时间轴完全已知**的合成视频：什么时刻开始未系安全带、什么时刻开始闭眼，
全部写进 `*_truth.json`。三条路线跑完后用 `bench/score.py` 对同一份 ground truth 打分。

诚实的能力边界
--------------
合成的卡通驾驶舱**不能替代真实数据集**。真实人脸/人体检测器在这段视频上大概率检不到目标，
因此它评测的是「告警链路与时序逻辑」——检出延迟、误报抑制、事件语义是否一致——
而**不是感知模型的真实精度**。后者需要 DMD、StateFarm 分心驾驶等真实标注数据集，
见 `bench/README.md` 的说明。

用法::

    python3 bench/make_clip.py --out bench/clips/scenario_a
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 640, 480, 15


@dataclass
class Segment:
    """一段有标注的违规区间（秒）。"""

    violation: str
    start_s: float
    end_s: float


# 场景脚本：刻意包含「短暂眨眼」和「短暂低头」作为干扰项，
# 用来检验各路线的防误报能力 —— 它们不应产生告警。
DEFAULT_SCRIPT: list[Segment] = [
    Segment("driver.no_seatbelt", 2.0, 14.0),     # 发车时未系安全带，12 秒后系上
    Segment("driver.fatigue", 20.0, 34.0),        # 持续闭眼 14 秒
    Segment("driver.phone_use", 40.0, 48.0),      # 拿起手机 8 秒
    Segment("passenger.no_seatbelt", 6.0, 52.0),  # 副驾全程未系
]
DURATION_S = 55.0

# 干扰项：不应触发告警的短暂动作（写入 truth 的 distractors 字段）
BLINKS = [(5.0, 5.2), (9.5, 9.65), (16.0, 16.2), (25.5, 25.6), (37.0, 37.15), (50.0, 50.2)]
GLANCES = [(11.0, 11.6), (30.0, 30.5), (44.0, 44.7)]   # 短暂低头看仪表


def _active(segs: list[Segment], name: str, t: float) -> bool:
    return any(s.violation == name and s.start_s <= t < s.end_s for s in segs)


def _in_ranges(ranges: list[tuple[float, float]], t: float) -> bool:
    return any(a <= t < b for a, b in ranges)


def draw_frame(t: float, segs: list[Segment]) -> np.ndarray:
    """画一帧简化驾驶舱：驾驶员在左，乘客在右。"""
    img = np.full((H, W, 3), 28, np.uint8)

    # 车内背景：挡风玻璃 + 仪表台
    cv2.rectangle(img, (0, 0), (W, 150), (52, 46, 40), -1)          # 车顶
    cv2.rectangle(img, (0, 380), (W, H), (38, 34, 30), -1)          # 仪表台
    cv2.line(img, (0, 380), (W, 380), (70, 64, 56), 2)

    for who, cx, role in (("driver", 190, "driver"), ("passenger", 450, "passenger")):
        is_driver = who == "driver"
        cy = 250
        # 身体
        cv2.ellipse(img, (cx, cy + 120), (78, 90), 0, 180, 360, (96, 108, 132), -1)
        # 头
        cv2.ellipse(img, (cx, cy), (52, 62), 0, 0, 360, (168, 152, 136), -1)

        # 眼睛：疲劳时闭合，眨眼时短暂闭合
        closed = False
        if is_driver:
            closed = _active(segs, "driver.fatigue", t) or _in_ranges(BLINKS, t)
        eye_y = cy - 8
        for ex in (cx - 20, cx + 20):
            if closed:
                cv2.line(img, (ex - 11, eye_y), (ex + 11, eye_y), (40, 34, 30), 3)
            else:
                cv2.ellipse(img, (ex, eye_y), (11, 7), 0, 0, 360, (250, 250, 250), -1)
                cv2.circle(img, (ex, eye_y), 4, (30, 26, 22), -1)
        # 嘴
        cv2.ellipse(img, (cx, cy + 28), (16, 8), 0, 0, 180, (110, 76, 70), 2)

        # 头部姿态：短暂低头看仪表（干扰项，不应告警）
        if is_driver and _in_ranges(GLANCES, t):
            cv2.line(img, (cx - 30, cy + 50), (cx + 30, cy + 50), (60, 54, 48), 2)

        # 安全带：从左肩到右腰的斜带
        belt_key = "driver.no_seatbelt" if is_driver else "passenger.no_seatbelt"
        if not _active(segs, belt_key, t):
            cv2.line(img, (cx - 52, cy + 55), (cx + 46, cy + 175), (232, 226, 214), 9)
            cv2.line(img, (cx - 52, cy + 55), (cx + 46, cy + 175), (150, 146, 138), 2)

        # 手机
        if is_driver and _active(segs, "driver.phone_use", t):
            px, py = cx + 58, cy + 10
            cv2.rectangle(img, (px, py), (px + 26, py + 46), (40, 44, 52), -1)
            cv2.rectangle(img, (px + 3, py + 4), (px + 23, py + 40), (120, 190, 235), -1)

    # 方向盘
    cv2.ellipse(img, (190, 430), (76, 30), 0, 180, 360, (58, 54, 50), 6)

    # 时间戳水印（便于人工核对）
    cv2.putText(img, f"t={t:5.1f}s", (W - 128, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="生成带 ground truth 的合成驾驶舱视频")
    ap.add_argument("--out", default="bench/clips/scenario_a", help="输出前缀（不含扩展名）")
    ap.add_argument("--duration", type=float, default=DURATION_S)
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    video_path = out.with_suffix(".mp4")

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, H))
    if not writer.isOpened():   # 某些环境缺 mp4 编码器，退回 AVI
        video_path = out.with_suffix(".avi")
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"),
                                 args.fps, (W, H))
    n = int(args.duration * args.fps)
    for i in range(n):
        writer.write(draw_frame(i / args.fps, DEFAULT_SCRIPT))
    writer.release()

    truth = {
        "clip": video_path.name,
        "fps": args.fps,
        "duration_s": args.duration,
        "segments": [asdict(s) for s in DEFAULT_SCRIPT],
        "distractors": {
            "blinks": BLINKS,       # 正常眨眼，不应触发疲劳告警
            "glances": GLANCES,     # 短暂低头看仪表，不应触发分心告警
        },
        "note": "合成素材，用于评测告警链路与时序逻辑，不用于评测感知模型的真实精度。",
    }
    truth_path = out.parent / f"{out.name}_truth.json"
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"视频: {video_path}  ({n} 帧 @{args.fps}fps)")
    print(f"标注: {truth_path}  ({len(DEFAULT_SCRIPT)} 段违规, "
          f"{len(BLINKS)} 次眨眼干扰, {len(GLANCES)} 次低头干扰)")


if __name__ == "__main__":
    main()
