"""下载模式C 使用的开源轻量模型。

全部是**公开可下载的、已训练好的**端侧模型，权重不入库（见 .gitignore），
用时执行本脚本拉取 —— 这也正是实车上 OTA 下发模型的方式。

    python3 mode-c-edge/tools/fetch_models.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

MODELS: dict[str, dict] = {
    "face_detection_yunet_2023mar.onnx": {
        "url": f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "desc": "YuNet 人脸检测（含 5 关键点），OpenCV Zoo，端侧专用",
        "size": 232589,
    },
    "2d106det.onnx": {
        "url": "https://huggingface.co/artemonlysuno/2d106det/resolve/main/2d106det.onnx",
        "desc": "insightface 106 点人脸关键点，192x192 输入",
        "size": 5030888,
    },
    "object_detection_nanodet_2022nov.onnx": {
        "url": f"{ZOO}/object_detection_nanodet/object_detection_nanodet_2022nov.onnx",
        "desc": "NanoDet-Plus-m COCO 80 类 FP32（含 cell phone）",
        "size": 3800954,
    },
    "object_detection_nanodet_2022nov_int8.onnx": {
        "url": f"{ZOO}/object_detection_nanodet/object_detection_nanodet_2022nov_int8.onnx",
        "desc": "NanoDet-Plus-m INT8 量化版（嵌入式默认）",
        "size": 1031424,
    },
}


def fetch(name: str, spec: dict, out_dir: Path, force: bool = False) -> bool:
    dst = out_dir / name
    if dst.exists() and not force and dst.stat().st_size == spec["size"]:
        print(f"  [已存在] {name} ({dst.stat().st_size / 1024:.0f} KB)")
        return True
    print(f"  [下载中] {name} — {spec['desc']}")
    try:
        with urllib.request.urlopen(spec["url"], timeout=180) as r, open(dst, "wb") as f:
            f.write(r.read())
    except Exception as exc:  # noqa: BLE001
        print(f"  [失败] {name}: {exc}")
        return False
    got = dst.stat().st_size
    ok = got == spec["size"]
    print(f"  [{'完成' if ok else '大小不符'}] {name} {got / 1024:.0f} KB"
          + ("" if ok else f"（期望 {spec['size'] / 1024:.0f} KB）"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="下载模式C 的端侧模型权重")
    ap.add_argument("--out", default=str(MODELS_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"模型目录: {out}")
    ok = all([fetch(n, s, out, args.force) for n, s in MODELS.items()])
    print("\n全部就绪。" if ok else "\n部分模型未就绪，请检查网络后重试。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
