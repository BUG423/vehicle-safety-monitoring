"""图像预处理 —— 直接决定 token 成本、延迟和隐私风险。

三件事：
  1. **缩放**：VLM 的图片 token 数随像素面积增长。1080p 原图和 896px 长边的图，
     对「肩上有没有安全带」这类判断的信息量几乎一样，但 token 数差 4~6 倍。
  2. **脱敏**：可选的人脸马赛克。车内影像上公有云涉及个人生物识别信息，
     若甲方走公有云，必须先脱敏（安全带/手机/香烟的判断不依赖人脸细节，
     但疲劳判断依赖眼睛，所以脱敏与疲劳能力是互斥的，见 DESIGN.md）。
  3. **缩略图**：事件证据里带一张小图，后台复核时不用回源拉原始影像。
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field

from PIL import Image, ImageFilter

log = logging.getLogger(__name__)

_FACE_CASCADE = None
_FACE_CASCADE_TRIED = False


@dataclass
class PreparedImage:
    """送进 VLM 之前的图片及其元信息。"""

    jpeg: bytes                       # 归一化后的 JPEG 字节
    width: int
    height: int
    original_width: int
    original_height: int
    faces_blurred: int = 0            # 实际打码的人脸数量
    blur_requested: bool = False
    blur_available: bool = True       # 人脸检测器是否可用（不可用时如实标记）
    notes: list[str] = field(default_factory=list)

    @property
    def b64(self) -> str:
        return base64.b64encode(self.jpeg).decode("ascii")

    @property
    def data_uri(self) -> str:
        return f"data:image/jpeg;base64,{self.b64}"

    def est_vision_tokens(self) -> int:
        """粗估视觉 token 数。

        主流 VLM（Claude / GPT-4o / Qwen-VL）都按 patch 数计费，量级约为
        像素数 / 750（Claude 官方口径）。这里用同一口径给出量级估计，
        用于 DESIGN.md 的成本测算与运行时的成本可观测性。
        """
        return int(self.width * self.height / 750)


def _load_face_cascade():
    global _FACE_CASCADE, _FACE_CASCADE_TRIED
    if _FACE_CASCADE_TRIED:
        return _FACE_CASCADE
    _FACE_CASCADE_TRIED = True
    try:
        import os

        import cv2

        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            log.warning("人脸级联分类器加载为空，脱敏功能不可用")
            return None
        _FACE_CASCADE = cascade
    except Exception as exc:  # noqa: BLE001 - 缺少 opencv 属于可降级情况
        log.warning("人脸检测不可用（%s），脱敏功能降级为不打码", exc)
        _FACE_CASCADE = None
    return _FACE_CASCADE


def _blur_faces(img: Image.Image) -> tuple[Image.Image, int, bool]:
    """对检测到的人脸做高斯模糊。返回 (图片, 打码数量, 检测器是否可用)。"""
    cascade = _load_face_cascade()
    if cascade is None:
        return img, 0, False
    try:
        import cv2
        import numpy as np

        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
        if len(faces) == 0:
            return img, 0, True
        out = img.copy()
        for (x, y, w, h) in faces:
            box = (int(x), int(y), int(x + w), int(y + h))
            region = out.crop(box).filter(ImageFilter.GaussianBlur(radius=max(6, w // 6)))
            out.paste(region, box)
        return out, len(faces), True
    except Exception as exc:  # noqa: BLE001
        log.warning("人脸脱敏执行失败: %s", exc)
        return img, 0, False


def prepare_image(
    raw: bytes | Image.Image,
    *,
    max_edge: int = 896,
    jpeg_quality: int = 80,
    blur_faces: bool = False,
) -> PreparedImage:
    """把任意输入图片归一化成送进 VLM 的 JPEG。"""
    if isinstance(raw, Image.Image):
        img = raw
    else:
        img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    ow, oh = img.size

    notes: list[str] = []
    if max(ow, oh) > max_edge:
        scale = max_edge / max(ow, oh)
        img = img.resize((max(1, int(ow * scale)), max(1, int(oh * scale))), Image.LANCZOS)
        notes.append(f"缩放 {ow}x{oh} -> {img.size[0]}x{img.size[1]}（省 token）")

    n_faces, blur_ok = 0, True
    if blur_faces:
        img, n_faces, blur_ok = _blur_faces(img)
        notes.append(f"人脸脱敏：{'打码 %d 张' % n_faces if blur_ok else '检测器不可用，未打码'}")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return PreparedImage(
        jpeg=buf.getvalue(),
        width=img.size[0], height=img.size[1],
        original_width=ow, original_height=oh,
        faces_blurred=n_faces, blur_requested=blur_faces, blur_available=blur_ok,
        notes=notes,
    )


def thumbnail_b64(raw: bytes | Image.Image, *, edge: int = 256, quality: int = 60) -> str:
    """生成事件证据用的小缩略图（base64，不含 data URI 前缀）。"""
    img = raw if isinstance(raw, Image.Image) else Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    img.thumbnail((edge, edge), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_data_uri(value: str) -> bytes:
    """接受 `data:image/jpeg;base64,xxx` 或裸 base64。"""
    if value.startswith("data:"):
        _, _, payload = value.partition(",")
    else:
        payload = value
    return base64.b64decode(payload)
