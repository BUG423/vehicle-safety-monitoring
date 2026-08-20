"""真模型能力验证 —— 在**真实素材**上跑，与合成素材的链路评测严格分开。

它回答三个问题，并且只报实测到的东西：
  1. YOLO11-pose / Keypoint R-CNN 在真实画面上能不能检出人体与 17 点关键点
  2. MediaPipe Face Landmarker 能不能给出 478 点、EAR、头部姿态
  3. 真实素材上的单帧推理耗时是多少（FPS）

明确不回答：安全带/疲劳/分心的真实准确率。那需要真实驾驶舱标注数据集
（DMD、StateFarm、AUC Distracted Driver），本仓库没有，也不该拿别的数字冒充。

用法::  python3 -m modeb.tools.verify_real_models --backend yolo
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modeb.config import PerceptionConfig  # noqa: E402
from modeb.engine.rules import _head_roi  # noqa: E402
from modeb.perception import analyzers as A  # noqa: E402
from modeb.perception import build_detector, build_face_module  # noqa: E402
from modeb.perception.base import assign_seats  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="在真实素材上验证真模型")
    ap.add_argument("--backend", default="yolo", choices=["yolo", "torchvision"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dir", default=str(_HERE / "samples"))
    ap.add_argument("--video-frames", type=int, default=120)
    ap.add_argument("--short-side", type=int, default=480)
    ap.add_argument("--person-thr", type=float, default=None,
                    help="人体检出阈值。远景监控画面里人很小，默认 0.75 会全部漏掉，这一点本身就是有价值的实测结论")
    ap.add_argument("--out", default="runs/verify_real_models.json")
    args = ap.parse_args()

    cfg = PerceptionConfig()
    cfg.backend, cfg.device, cfg.infer_short_side = args.backend, args.device, args.short_side
    if args.person_thr is not None:
        cfg.person_score_thr = args.person_thr
    det = build_detector(cfg)
    face, face_err = build_face_module()
    print(f"检测器  {json.dumps(det.describe(), ensure_ascii=False)}")
    print(f"人脸模块 {json.dumps(face.describe(), ensure_ascii=False) if face else face_err}\n")

    root = Path(args.dir)
    report: dict = {"backend": det.describe(), "person_score_thr": cfg.person_score_thr,
                    "face_module": face.describe() if face else {"error": face_err},
                    "images": [], "video": None}

    # ---- 静态真实照片 ----
    print("=== 真实照片 ===")
    for name in ("grace_hopper_517x606.jpg", "bus.jpg", "zidane.jpg", "phone_calling.jpg"):
        p = root / name
        if not p.exists():
            print(f"  跳过 {name}（未下载，运行 tools/fetch_samples.py）")
            continue
        img = cv2.imread(str(p))
        r = det.infer(img)
        assign_seats(r.persons, r.width, r.height)
        row = {"file": name, "size": [r.width, r.height], "persons": len(r.persons),
               "objects": [(o.label, round(o.score, 2)) for o in r.objects],
               "infer_ms": r.infer_ms, "faces": []}
        print(f"  {name:<26} {r.width}x{r.height}  人体 {len(r.persons)}  "
              f"物体 {row['objects']}  推理 {r.infer_ms:.1f}ms")
        for pobs in r.persons[:3]:
            pose = A.head_pose(pobs, r.width, r.height, cfg.keypoint_score_thr)
            kp_ok = int((pobs.keypoints[:, 2] >= cfg.keypoint_score_thr).sum()) if pobs.keypoints is not None else 0
            entry = {"score": round(pobs.score, 3), "kp_valid": kp_ok, "solvepnp_pose": pose}
            if face is not None:
                roi = _head_roi(img, pobs, cfg.keypoint_score_thr)
                m = face.analyze(roi) if roi is not None else None
                if m:
                    entry["face"] = m.to_dict()
                    print(f"      人体score={pobs.score:.2f} 有效关键点={kp_ok}/17 "
                          f"solvePnP头姿={pose}")
                    print(f"      MediaPipe: EAR={m.ear:.3f} 闭眼概率={max(m.blink_left,m.blink_right):.3f} "
                          f"yaw={m.yaw}° pitch={m.pitch}° 张嘴={m.mouth_open:.3f} -> 闭眼判定={m.eyes_closed}")
                else:
                    print(f"      人体score={pobs.score:.2f} 有效关键点={kp_ok}/17 头姿={pose} （人脸未检出）")
            row["faces"].append(entry)
        report["images"].append(row)

    # ---- 真实视频吞吐 ----
    vp = root / "vtest.avi"
    if vp.exists():
        print("\n=== 真实视频（OpenCV vtest.avi，768x576 真实行人）===")
        cap = cv2.VideoCapture(str(vp))
        lat, npersons = [], []
        n = 0
        while n < args.video_frames:
            ok, img = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            r = det.infer(img)
            lat.append((time.perf_counter() - t0) * 1000.0)
            npersons.append(len(r.persons))
            n += 1
        cap.release()
        if lat:
            report["video"] = {
                "file": "vtest.avi", "frames": n,
                "latency_ms_mean": round(statistics.mean(lat), 2),
                "latency_ms_p95": round(sorted(lat)[int(len(lat) * .95) - 1], 2),
                "fps_single_stream": round(1000.0 / statistics.mean(lat), 1),
                "persons_mean": round(statistics.mean(npersons), 2),
                "persons_max": max(npersons),
                "frames_with_person_pct": round(100.0 * sum(1 for x in npersons if x) / n, 1),
            }
            v = report["video"]
            print(f"  {n} 帧  单帧 {v['latency_ms_mean']}ms (P95 {v['latency_ms_p95']}ms)"
                  f"  → 单路 {v['fps_single_stream']} FPS")
            print(f"  平均检出 {v['persons_mean']} 人/帧，最多 {v['persons_max']} 人，"
                  f"{v['frames_with_person_pct']}% 的帧至少检出 1 人")
    else:
        print("\n（未找到 samples/vtest.avi，运行 tools/fetch_samples.py 下载）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    print("注意：以上验证的是「真模型在真实画面上确实工作」，"
          "不是安全带/疲劳/分心的真实准确率——那需要真实驾驶舱标注数据集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
