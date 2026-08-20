"""运行环境兼容层 —— 让 MediaPipe 在无图形栈的服务器容器里跑起来。

背景（如实记录，便于复现）：
本机的 `mediapipe 1.0.1` 只提供 Tasks API，其 C 绑定 `libmediapipe.so` 链接了
`libGLESv2.so.2`。容器里没有安装 Mesa，`apt-get install libgles2` 又因为根分区写满而失败，
于是 `FaceLandmarker.create_from_options()` 直接抛 `OSError: libGLESv2.so.2`。

解决办法：MediaPipe 只是链接了这个符号，人脸推理本身走 CPU 的 XNNPACK，
并不真的需要 GPU 图形能力。因此把系统上任意一个 ABI 兼容的 GLESv2 实现
（本机是 Playwright 自带的 Chromium ANGLE）软链到 `libmediapipe.so` 的 RPATH 目录即可。

这个 shim 是**幂等**的，重复调用无副作用；找不到候选库就返回 False，
调用方据此降级到不依赖人脸关键点的路径，而不是崩溃。
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

_GLES_CANDIDATE_GLOBS = [
    "/usr/lib/x86_64-linux-gnu/libGLESv2.so*",
    "/usr/lib64/libGLESv2.so*",
    "/root/.cache/ms-playwright/*/chrome-linux/libGLESv2.so",
    "/opt/google/chrome/libGLESv2.so",
    os.path.expanduser("~/.cache/ms-playwright/*/chrome-linux/libGLESv2.so"),
]
_EGL_CANDIDATE_GLOBS = [
    "/usr/lib/x86_64-linux-gnu/libEGL.so.1",
    "/usr/lib64/libEGL.so.1",
    "/root/.cache/ms-playwright/*/chrome-linux/libEGL.so",
]

_status: dict[str, object] = {"applied": False, "reason": None, "gles": None}


def _first(globs: list[str]) -> str | None:
    for pattern in globs:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                return path
    return None


def _mediapipe_rpath_dirs() -> list[Path]:
    """libmediapipe.so 的 RPATH 展开后的两个 `_solib_k8/...` 目录。"""
    try:
        import mediapipe  # noqa: F401
    except Exception:  # noqa: BLE001
        return []
    site = Path(sys.modules["mediapipe"].__file__).resolve().parents[2]   # site-packages
    base = site.parent / "_solib_k8"                                      # RPATH 里的 ../../../..
    return [
        base / "_U_S_Sthird_Uparty_SGL_Snative_CGLESv3___Uthird_Uparty_SGL_Snative",
        base / "_U_S_Sthird_Uparty_SGL_Snative_CEGL___Uthird_Uparty_SGL_Snative",
    ]


def ensure_mediapipe_runtime() -> bool:
    """确保 MediaPipe 的 C 绑定能被加载。返回是否可用。"""
    if _status["applied"]:
        return bool(_status["gles"]) or _status["reason"] is None

    try:
        import ctypes
        ctypes.CDLL("libGLESv2.so.2")
        _status.update(applied=True, gles="system", reason=None)
        return True
    except OSError:
        pass

    gles = _first(_GLES_CANDIDATE_GLOBS)
    egl = _first(_EGL_CANDIDATE_GLOBS)
    if gles is None:
        _status.update(applied=True, gles=None,
                       reason="系统上找不到任何 libGLESv2 实现，MediaPipe 人脸模块不可用")
        return False

    linked = 0
    for d in _mediapipe_rpath_dirs():
        try:
            d.mkdir(parents=True, exist_ok=True)
            _symlink(gles, d / "libGLESv2.so.2")
            if egl:
                _symlink(egl, d / "libEGL.so.1")
            linked += 1
        except OSError as exc:  # noqa: BLE001 - site-packages 只读时会走到这里
            _status["reason"] = f"无法在 {d} 创建软链: {exc}"

    if linked == 0:
        _status.update(applied=True, gles=None)
        return False

    _status.update(applied=True, gles=gles, reason=None)
    return True


def _symlink(src: str, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        return
    dst.symlink_to(src)


def runtime_status() -> dict[str, object]:
    return dict(_status)
