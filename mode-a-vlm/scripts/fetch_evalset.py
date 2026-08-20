"""拉取真实照片验证集。

图片来自 Wikimedia Commons（各自的开放许可，见 ``evalset/manifest.json``）。
**图片本身不入库**：一是署名/许可要求随图分发时更麻烦，二是避免仓库体积膨胀。
需要时用本脚本按 manifest 重新拉取，结果落在 ``evalset/images/``（已被 .gitignore 忽略）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="拉取真实照片验证集")
    ap.add_argument("--manifest", default=str(ROOT / "evalset" / "manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "evalset" / "images"))
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for item in man["items"]:
        dst = out / f"{item['id']}.jpg"
        if dst.exists() and dst.stat().st_size > 10_000:
            print(f"[跳过] {dst.name} 已存在")
            ok += 1
            continue
        r = subprocess.run(["curl", "-sSL", "-A", "vsm-mode-a-eval/1.0", "--max-time", "60",
                            "-o", str(dst), item["url"]])
        size = dst.stat().st_size if dst.exists() else 0
        if r.returncode == 0 and size > 10_000:
            print(f"[OK]   {dst.name} {size // 1024}KB  {item['license']}  {item['desc']}")
            ok += 1
        else:
            print(f"[失败] {dst.name} ({size}B) {item['commons']}")
            fail += 1
    print(f"\n完成：成功 {ok} 张，失败 {fail} 张 -> {out}")
    print("图片版权归原作者，许可见 manifest.json；仅用于本项目的技术验证。")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
