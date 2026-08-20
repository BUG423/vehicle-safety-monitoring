"""用**真实人物照片**合成一段驾驶舱风格的视频，供真模型端到端演示。

为什么需要它
------------
`bench/` 的卡通素材上真模型检出 0 个人体（已实测），所以那段素材只能评链路。
而公开的真实素材里又没有驾驶舱视频。折中方案：把真实人像按驾驶舱构图排布，
加上可控的运动与脚本化的安全带，生成一段**每一帧都是真实人脸/人体像素**的视频。

因此这段素材上的下列结论是真的：
  - 人体检测与 17 点关键点是真模型在真实人像上跑出来的
  - 人脸 478 点、EAR、头部姿态是 MediaPipe 在真实人脸上跑出来的
  - 安全带判定读的是真实图像上的斜带对比度

不真的部分（必须说清楚）：
  - 拍摄条件（光照、视角、镜头畸变、红外补光）与真实舱内摄像头不同
  - 人是静态照片，没有真实的头部转动与眨眼，所以疲劳/分心在这段素材上
    只能验证「不误报」，无法验证「能检出」
  - 安全带是画上去的矩形斜带，不是真实织带的纹理

用法::

    python3 -m modeb.tools.fetch_samples
    python3 -m modeb.tools.make_real_demo_clip --out samples/real_cockpit.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parents[2]
W, H, FPS = 960, 540, 15


@dataclass
class Segment:
    violation: str
    start_s: float
    end_s: float


# 只脚本化「画得出来」的两项。疲劳/分心不脚本化 —— 真实照片眼睛始终睁着、正脸，
# 因此这两项在本素材上的正确行为是**不告警**，这本身就是一次误报检验。
SCRIPT = [
    Segment("driver.no_seatbelt", 0.0, 16.0),
    Segment("passenger.no_seatbelt", 0.0, 30.0),
    Segment("driver.phone_use", 24.0, 34.0),
]
DURATION = 40.0


def _fit(img: np.ndarray, h: int) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


def _paste(dst: np.ndarray, src: np.ndarray, cx: int, cy: int) -> tuple[int, int, int, int]:
    h, w = src.shape[:2]
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = x1 + w, y1 + h
    sx1, sy1 = max(0, -x1), max(0, -y1)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(dst.shape[1], x2), min(dst.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0
    dst[y1:y2, x1:x2] = src[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
    return x1, y1, x2, y2


def _active(t: float, name: str) -> bool:
    return any(s.violation == name and s.start_s <= t < s.end_s for s in SCRIPT)


def _anchors(base: np.ndarray) -> dict:
    """在合成好的底图上跑一次真模型姿态估计，拿到肩/髋/头的真实位置。

    安全带与手机必须画在**解剖学上正确的位置**，否则测的就不是「能不能看出安全带」，
    而是「贴图碰巧落在采样线上没有」。跑一次检测拿真实关键点，是最省事也最诚实的做法。
    检测器不可用时退回按人体框比例估计，并在输出里注明。
    """
    try:
        from modeb.config import PerceptionConfig
        from modeb.perception import assign_seats, build_detector
        cfg = PerceptionConfig()
        cfg.backend, cfg.person_score_thr = "yolo", 0.5
        det = build_detector(cfg)
        r = det.infer(base)
        assign_seats(r.persons, r.width, r.height)
        out = {"source": "yolo11-pose"}
        for pobs in r.persons:
            if pobs.seat not in ("driver", "front_passenger"):
                continue
            ls = pobs.kp("left_shoulder", 3.0) or pobs.kp("left_shoulder", 0.5)
            rs = pobs.kp("right_shoulder", 3.0) or pobs.kp("right_shoulder", 0.5)
            lh = pobs.kp("left_hip", 3.0) or pobs.kp("left_hip", 0.5)
            rh = pobs.kp("right_hip", 3.0) or pobs.kp("right_hip", 0.5)
            nose = pobs.kp("nose", 0.5)
            if ls is None or rs is None:
                continue
            sw = abs(ls[0] - rs[0]) or 60.0
            bottom = min(pobs.box.y2, H - 6.0)
            drop = min(1.5 * sw, max(40.0, bottom - max(ls[1], rs[1])))
            if lh is None:
                lh = (ls[0], ls[1] + drop)
            if rh is None:
                rh = (rs[0], rs[1] + drop)
            lh = (lh[0], min(lh[1], H - 6.0))
            rh = (rh[0], min(rh[1], H - 6.0))
            out[pobs.seat] = {"ls": ls, "rs": rs, "lh": lh, "rh": rh,
                              "nose": nose or ((ls[0] + rs[0]) / 2, ls[1] - sw * 0.7),
                              "sw": sw, "box": pobs.box.as_list()}
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"  （姿态锚点不可用，退回比例估计: {exc}）")
        return {"source": "proportional"}


def _fallback_anchor(cx: float) -> dict:
    return {"ls": (cx + W * .055, H * .40), "rs": (cx - W * .055, H * .40),
            "lh": (cx + W * .045, H * .86), "rh": (cx - W * .045, H * .86),
            "nose": (cx, H * .27), "sw": W * .11, "box": None}


def _compose(drv, psg, t: float) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    frame = np.full((H, W, 3), 26, np.uint8)
    cv2.rectangle(frame, (0, 0), (W, 62), (46, 42, 38), -1)         # 车顶
    cv2.rectangle(frame, (0, H - 74), (W, H), (34, 31, 28), -1)     # 仪表台
    jx = int(5 * math.sin(t * 1.7))
    jy = int(4 * math.sin(t * 1.1 + 1.0))
    dc = (int(W * 0.27) + jx, int(H * 0.47) + jy)
    pc = (int(W * 0.74) - jx, int(H * 0.47) - jy)
    _paste(frame, drv, *dc)
    _paste(frame, psg, *pc)
    return frame, dc, pc


def _draw_belt(frame: np.ndarray, a: dict, dx: float, dy: float) -> None:
    """沿「一侧肩 → 对侧髋」画织带 —— 和真实三点式安全带的走向一致。"""
    top = (int(a["rs"][0] + dx), int(a["rs"][1] + dy))
    bot = (int(a["lh"][0] + dx), int(a["lh"][1] + dy))
    cv2.line(frame, top, bot, (226, 222, 214), 15)
    cv2.line(frame, top, bot, (146, 144, 138), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description="用真实人像合成驾驶舱风格视频")
    ap.add_argument("--samples", default=str(_HERE / "samples"))
    ap.add_argument("--out", default=str(_HERE / "samples" / "real_cockpit.mp4"))
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    root = Path(args.samples)
    driver_img = cv2.imread(str(root / "grace_hopper_517x606.jpg"))
    if driver_img is None:
        print("缺少 samples/grace_hopper_517x606.jpg，请先运行 tools/fetch_samples.py", file=sys.stderr)
        return 2
    pass_src = root / "zidane.jpg"
    passenger_img = cv2.imread(str(pass_src)) if pass_src.exists() else None
    if passenger_img is None:
        passenger_img = cv2.flip(driver_img, 1)
    else:
        passenger_img = passenger_img[:, : passenger_img.shape[1] // 2]

    phone_img = cv2.imread(str(root / "phone_calling.jpg"))
    phone_crop = None
    if phone_img is not None:
        h, w = phone_img.shape[:2]
        phone_crop = _fit(phone_img[int(h * .12):int(h * .62), int(w * .18):int(w * .82)], 150)

    dh = int(H * 0.80)
    drv = _fit(driver_img, dh)
    psg = _fit(passenger_img, dh)

    print("在底图上估计肩/髋/头锚点…")
    base, dc0, pc0 = _compose(drv, psg, 0.0)
    anchors = _anchors(base)
    print(f"  锚点来源: {anchors.get('source')}  座位: "
          f"{[k for k in anchors if k not in ('source',)]}")
    a_drv = anchors.get("driver") or _fallback_anchor(dc0[0])
    a_psg = anchors.get("front_passenger") or _fallback_anchor(pc0[0])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not writer.isOpened():
        out = out.with_suffix(".avi")
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (W, H))

    n = int(args.duration * args.fps)
    for i in range(n):
        t = i / args.fps
        frame, dc, pc = _compose(drv, psg, t)
        if not _active(t, "driver.no_seatbelt"):
            _draw_belt(frame, a_drv, dc[0] - dc0[0], dc[1] - dc0[1])
        if not _active(t, "passenger.no_seatbelt"):
            _draw_belt(frame, a_psg, pc[0] - pc0[0], pc[1] - pc0[1])

        # 手机：贴到司机面部旁边（接打电话的典型位置），用的是真实手机照片的裁片
        if phone_crop is not None and _active(t, "driver.phone_use"):
            nx, ny = a_drv["nose"]
            _paste(frame, phone_crop,
                   int(nx + 0.42 * a_drv["sw"] + (dc[0] - dc0[0])),
                   int(ny + 0.05 * a_drv["sw"] + (dc[1] - dc0[1])))

        cv2.putText(frame, f"REAL-PORTRAIT COMPOSITE  t={t:5.1f}s", (10, H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()

    truth = {"clip": out.name, "fps": args.fps, "duration_s": args.duration,
             "segments": [asdict(s) for s in SCRIPT],
             "distractors": {"blinks": [], "glances": []},
             "anchor_source": anchors.get("source"),
             "note": "真实人像合成的驾驶舱风格素材：人体/人脸/关键点是真的，"
                     "拍摄条件与真实舱内摄像头不同；疲劳与分心未脚本化，"
                     "正确行为是不告警（误报检验）。"}
    tp = out.parent / f"{out.stem}_truth.json"
    tp.write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"视频: {out} ({n} 帧 @{args.fps}fps {W}x{H})")
    print(f"标注: {tp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
