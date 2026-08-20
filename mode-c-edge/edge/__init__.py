"""模式C：车载嵌入式设备端安全监测。

设计约束（全部按嵌入式实机来写，不是「服务器上跑得动就行」）：
  * 输入分辨率固定为 320x240 / 640x480 这一档，不接 1080p；
  * 主循环限帧率（默认 10 FPS），空闲时主动 sleep 让出 CPU；
  * 依赖只有 numpy / opencv / onnxruntime，事件与告警全部复用 `common/`（纯标准库）；
  * 重模型（目标检测）降频调度，不是每帧都跑；
  * 内存中不缓存整段视频，证据只留一帧 JPEG 缩略图。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `common` 包可被引入（仓库根目录），同时把 common/schema 加进 sys.path，
# 兼容契约层内部的裸模块名导入写法。
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _REPO_ROOT / "common" / "schema"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

__all__ = ["REPO_ROOT"]
REPO_ROOT = _REPO_ROOT
