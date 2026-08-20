"""下载公开的真实素材，用于验证真模型（而不是用合成卡通冒充真实精度）。

素材来源都是公网可直接下载的公开测试资源：
  vtest.avi   OpenCV 官方示例视频，768x576，真实行人（github.com/opencv/opencv）
  bus/zidane  Ultralytics 官方示例图，真实多人场景
  grace_hopper torchvision 测试资源，真实单人正脸
  phone_calling 真实手持手机通话照片（Wikimedia Commons），验证 COCO cell phone 类

它们**不是驾驶舱素材**，不能用来评安全带/疲劳的真实精度；
它们能验证的是：人体检测、17 点关键点、人脸 478 点、EAR、头部姿态在真实画面上确实工作。
真正的精度评测需要 DMD / StateFarm / AUC Distracted Driver 等真实标注数据集。
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

SAMPLES = {
    "vtest.avi": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
    "grace_hopper_517x606.jpg":
        "https://raw.githubusercontent.com/pytorch/vision/main/test/assets/encode_jpeg/grace_hopper_517x606.jpg",
    # 用于验证 COCO `cell phone` 类在真实画面上的检出（Wikimedia Commons，自由许可）
    "phone_calling.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Hand-smartphone-technology-calling.jpg/960px-Hand-smartphone-technology-calling.jpg",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="下载真实素材（用于真模型验证）")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parents[2] / "samples"))
    args = ap.parse_args()
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, url in SAMPLES.items():
        p = out / name
        if p.exists() and p.stat().st_size > 1024:
            print(f"  已存在  {name}")
            continue
        try:
            print(f"  下载中  {name}")
            urllib.request.urlretrieve(url, p)
            print(f"  完成    {name} ({p.stat().st_size/1e6:.2f} MB)")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  失败    {name}: {exc}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
