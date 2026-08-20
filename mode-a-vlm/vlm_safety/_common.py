"""统一的契约层导入入口。

`common/` 是三条技术路线共用的契约层，位于仓库根目录。模式A 既可能被以
`python -m mode-a-vlm...` 方式运行，也可能被 uvicorn 从任意工作目录拉起，
因此这里统一把仓库根目录塞进 sys.path，避免每个模块各写一份路径拼接。
"""
from __future__ import annotations

import sys
from pathlib import Path

# mode-a-vlm/vlm_safety/_common.py -> mode-a-vlm/vlm_safety -> mode-a-vlm -> 仓库根
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.schema.violation_types import (  # noqa: E402
    DetectionMode, Severity, SubjectRole, ViolationType,
)
from common.schema.safety_event import (  # noqa: E402
    Evidence, SafetyEvent, Subject, VehicleContext, SCHEMA_VERSION,
)
from common.schema.confirmation import (  # noqa: E402
    Confirmation, ConfirmRule, ViolationConfirmer,
)
from common.alerting.channels import (  # noqa: E402
    AlertChannel, AlertDispatcher, BackendChannel, CabinPrompt, InCabinChannel,
)

__all__ = [
    "REPO_ROOT",
    "DetectionMode", "Severity", "SubjectRole", "ViolationType",
    "Evidence", "SafetyEvent", "Subject", "VehicleContext", "SCHEMA_VERSION",
    "Confirmation", "ConfirmRule", "ViolationConfirmer",
    "AlertChannel", "AlertDispatcher", "BackendChannel", "CabinPrompt", "InCabinChannel",
]
