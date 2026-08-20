"""下载真实人脸测试素材。

`bench/` 的合成卡通素材只能评告警链路，**不能评真实模型的性能与精度**
（实测：真实人脸/关键点模型在卡通帧上无判别力）。因此真模型的 FPS / 延迟 / 内存
必须在真实拍摄的人脸视频上测，本脚本拉取这样一段公开素材。

素材来源：intel-iot-devkit/sample-videos，Intel 官方发布的 DMS 类演示片段
（正对镜头的头部转动视频，768x432，与车载 DMS 的取景非常接近）。

    python3 mode-c-edge/tools/fetch_assets.py
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
BASE = "https://github.com/intel-iot-devkit/sample-videos/raw/master"
CLIPS = {
    "head-pose-face-detection-female.mp4": 15628037,
    "head-pose-face-detection-male.mp4": 15522596,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="下载真实人脸测试视频")
    ap.add_argument("--out", default=str(ASSETS_DIR))
    ap.add_argument("--only-first", action="store_true", help="只下第一段（省流量）")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = list(CLIPS.items())[:1] if args.only_first else list(CLIPS.items())
    ok = True
    for name, size in items:
        dst = out / name
        if dst.exists() and dst.stat().st_size == size:
            print(f"  [已存在] {name}")
            continue
        print(f"  [下载中] {name} ({size / 1e6:.1f} MB)")
        try:
            with urllib.request.urlopen(f"{BASE}/{name}", timeout=300) as r, open(dst, "wb") as f:
                f.write(r.read())
            print(f"  [完成] {name} {dst.stat().st_size / 1e6:.1f} MB")
        except Exception as exc:  # noqa: BLE001
            print(f"  [失败] {name}: {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
