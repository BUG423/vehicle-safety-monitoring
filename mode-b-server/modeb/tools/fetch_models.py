"""下载真模型权重到 mode-b-server/models/。

权重不入版本库（见 .gitignore），部署时跑一次这个脚本即可。
所有下载源都在公网、无需登录：
  - YOLO11        github.com/ultralytics/assets releases
  - Face Mesh     storage.googleapis.com/mediapipe-models
  - Keypoint R-CNN / Faster R-CNN 由 torchvision 自己下到 ~/.cache/torch

用法::  python3 -m modeb.tools.fetch_models [--all]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "yolo11n-pose.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
    "yolo11s-pose.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}
OPTIONAL = {"yolo11s-pose.pt"}


def main() -> int:
    ap = argparse.ArgumentParser(description="下载模式B 所需的开源预训练权重")
    ap.add_argument("--all", action="store_true", help="连可选权重一起下")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parents[2] / "models"))
    ap.add_argument("--torchvision", action="store_true", help="顺便预热 torchvision 的权重缓存")
    args = ap.parse_args()

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, url in MODELS.items():
        if name in OPTIONAL and not args.all:
            continue
        path = out / name
        if path.exists() and path.stat().st_size > 1024:
            print(f"  已存在  {name}  ({path.stat().st_size/1e6:.1f} MB)")
            continue
        print(f"  下载中  {name} <- {url}")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  完成    {name}  ({path.stat().st_size/1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  失败    {name}: {exc}", file=sys.stderr)

    if args.torchvision:
        try:
            from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                                      keypointrcnn_resnet50_fpn)
            keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
            print("  完成    torchvision keypointrcnn 权重已缓存")
        except Exception as exc:  # noqa: BLE001
            print(f"  失败    torchvision 权重: {exc}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
