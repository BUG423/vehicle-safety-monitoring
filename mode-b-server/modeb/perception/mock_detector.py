"""可插拔的降级检测器 —— 真模型跑不通、或素材是合成画面时使用。

两个实现，用途完全不同，**不要混用**：

`CartoonCockpitDetector`（registry 名 "cartoon"）
    针对 `bench/` 的合成驾驶舱素材与本项目 SyntheticSource 的经典 CV 解析器：
    按颜色/形状找到头部与躯干，再把它们**折算成 COCO 17 点关键点**。
    折算之后，下游的 analyzers（安全带斜带对比度、眼睛开合、头姿、手机）
    与真模型路径**完全是同一份代码**。
    它真的在读像素（眨眼会让眼部暗区坍缩、安全带画上去对比度就上来），
    不读 ground truth 脚本 —— 否则评测就没有意义了。
    但它**只在合成卡通画面上成立**，对真实车载画面完全无效。

`ScriptedMockDetector`（registry 名 "mock"）
    完全不看画面，按一段确定性脚本产出行为。用于：
      - 没有任何素材时冒烟测试整条链路
      - 车队压测（几十路虚拟车灌数据到看板）
    它的输出**不是检测结果**，任何精度数字都不能来自它。
"""
from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from .base import (BBox, COCO_KP, Detector, KP_INDEX, ObjectDet, PerceptionResult, PersonObs)
from .torchvision_detector import frame_quality

_KP_STRONG = 9.0     # 折算关键点给一个明确高于阈值的分数
_KP_WEAK = 0.5


class CartoonCockpitDetector(Detector):
    """合成驾驶舱画面的经典 CV 解析器（非深度模型）。"""

    name = "cartoon"
    thread_safe = True

    # 合成素材里用到的颜色（BGR），带容差匹配
    SKIN = np.array([168, 152, 136], dtype=np.int16)
    SKIN_ALT = np.array([170, 160, 150], dtype=np.int16)
    BODY = np.array([96, 108, 132], dtype=np.int16)
    BODY_ALT = np.array([90, 110, 150], dtype=np.int16)
    PHONE_SCREEN = np.array([120, 190, 235], dtype=np.int16)

    def __init__(self, *, skin_tol: int = 26, body_tol: int = 30, min_head_area: int = 400) -> None:
        self.skin_tol = skin_tol
        self.body_tol = body_tol
        self.min_head_area = min_head_area

    def infer_batch(self, images: list[np.ndarray], *, vehicle_ids: list[str] | None = None
                    ) -> list[PerceptionResult]:
        now = time.time()
        out = []
        for img in images:
            t0 = time.perf_counter()
            res = self._infer_one(img, now)
            res.infer_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            out.append(res)
        return out

    def _infer_one(self, img: np.ndarray, now: float) -> PerceptionResult:
        h, w = img.shape[:2]
        res = PerceptionResult(ts=now, width=w, height=h, backend=self.name)
        res.frame_stats = frame_quality(img)

        heads = self._blobs(img, [self.SKIN, self.SKIN_ALT], self.skin_tol, self.min_head_area)
        bodies = self._blobs(img, [self.BODY, self.BODY_ALT], self.body_tol, self.min_head_area * 2)
        phones = self._blobs(img, [self.PHONE_SCREEN], 40, 120)

        for hx1, hy1, hx2, hy2 in heads:
            body = self._match_body(bodies, (hx1 + hx2) / 2, hy2)
            person = self._build_person(img, (hx1, hy1, hx2, hy2), body)
            if person is not None:
                res.persons.append(person)

        for px1, py1, px2, py2 in phones:
            res.objects.append(ObjectDet(label="cell phone", score=0.9,
                                         box=BBox(px1, py1, px2, py2)))
        return res

    # -- 颜色连通域 ---------------------------------------------------------
    def _blobs(self, img: np.ndarray, colors: list[np.ndarray], tol: int, min_area: int
               ) -> list[tuple[float, float, float, float]]:
        a = img.astype(np.int16)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for c in colors:
            d = np.abs(a - c.reshape(1, 1, 3)).max(axis=2)
            mask |= (d <= tol).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append((float(x), float(y), float(x + bw), float(y + bh)))
        boxes.sort(key=lambda b: b[0])
        return boxes

    @staticmethod
    def _match_body(bodies: list[tuple[float, float, float, float]], cx: float, head_bottom: float):
        best, best_d = None, 1e9
        for b in bodies:
            bcx = (b[0] + b[2]) / 2
            if b[3] < head_bottom:            # 躯干必须在头部下方
                continue
            d = abs(bcx - cx)
            if d < best_d:
                best, best_d = b, d
        return best if best_d < 160 else None

    # -- 头/躯干 → COCO 17 点 ----------------------------------------------
    def _build_person(self, img: np.ndarray, head, body) -> PersonObs | None:
        """由头部色块的位置与尺寸折算出 COCO 17 点。

        合成驾驶舱里人体按固定比例绘制（肩、髋相对头部的位置是常数），
        所以「检出头 → 推出肩/髋」在这段素材上是成立的几何关系。
        注意：**检测器只负责定位，不负责判定**——安全带有没有、眼睛睁没睁，
        全部交给 analyzers 去读那几块 ROI 的像素，与真模型路径共用同一份代码。
        """
        hx1, hy1, hx2, hy2 = head
        hw, hh = hx2 - hx1, hy2 - hy1
        if hw < 12 or hh < 12:
            return None
        hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2

        kp = np.zeros((17, 3), dtype=np.float32)

        def put(name: str, x: float, y: float, s: float = _KP_STRONG) -> None:
            kp[KP_INDEX[name]] = (x, y, s)

        # 眼位：先按人体比例落点，再在小窗口内往「与皮肤色差最大处」微调
        eye_y = hy1 + 0.44 * hh
        for name, sign in (("right_eye", -1), ("left_eye", +1)):
            ex, ey = self._refine_eye(img, hcx + sign * 0.19 * hw, eye_y, 0.13 * hw, 0.10 * hh)
            put(name, ex, ey)
        put("nose", hcx, hy1 + 0.62 * hh)
        put("right_ear", hx1 + 0.03 * hw, hcy)
        put("left_ear", hx2 - 0.03 * hw, hcy)

        # 肩 / 髋：安全带锚点相对头部的比例（对照 bench/make_clip.py 的画法标定）
        sh_y = hcy + 0.44 * hh
        hip_y = hcy + 1.41 * hh
        put("right_shoulder", hcx - 0.50 * hw, sh_y)
        put("left_shoulder", hcx + 0.50 * hw, sh_y)
        put("right_hip", hcx - 0.44 * hw, hip_y)
        put("left_hip", hcx + 0.44 * hw, hip_y)

        if body is not None:
            bx1, by1, bx2, by2 = body
            box = BBox(min(hx1, bx1), hy1, max(hx2, bx2), max(by2, hip_y))
        else:
            box = BBox(hx1 - 0.6 * hw, hy1, hx2 + 0.6 * hw, hip_y)
        return PersonObs(box=box, score=0.9, keypoints=kp)

    @staticmethod
    def _refine_eye(img: np.ndarray, cx: float, cy: float, rx: float, ry: float
                    ) -> tuple[float, float]:
        """在候选点周围找「与局部中值色差最大」的像素团中心。

        睁眼时是白色巩膜 + 深色瞳孔，闭眼时是一条深色横线 —— 两者都是高色差区域，
        因此用「色差最大」而不是「找白色」，闭眼帧才不会因为定位失败被整帧跳过。
        """
        x1, y1 = int(cx - rx), int(cy - ry)
        x2, y2 = int(cx + rx), int(cy + ry)
        roi = img[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
            return cx, cy
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.int16)
        diff = np.abs(gray - int(np.median(gray))).astype(np.uint8)
        if int(diff.max()) < 25:
            return cx, cy
        ys, xs = np.where(diff >= diff.max() * 0.6)
        return float(max(0, x1) + xs.mean()), float(max(0, y1) + ys.mean())

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": "classical-cv",
                "note": "仅对合成卡通驾驶舱有效，对真实画面无效", "thread_safe": True}


class ScriptedMockDetector(Detector):
    """脚本化 Mock —— 不看画面，按确定性脚本产出行为。

    只用于冒烟测试和车队压测。任何精度指标都不能来自它。
    """

    name = "mock"
    thread_safe = True

    def __init__(self, *, seed: int = 0, period_s: float = 60.0, violation_rate: float = 0.35) -> None:
        self.seed = seed
        self.period = period_s
        self.rate = violation_rate
        self._t0 = time.time()

    def infer_batch(self, images: list[np.ndarray], *, vehicle_ids: list[str] | None = None
                    ) -> list[PerceptionResult]:
        now = time.time()
        vids = vehicle_ids or [""] * len(images)
        out = []
        for img, vid in zip(images, vids):
            h, w = img.shape[:2]
            res = PerceptionResult(ts=now, width=w, height=h, backend=self.name)
            res.frame_stats = {"mean": 90.0, "std": 40.0, "lap_var": 200.0}
            phase = ((now - self._t0) / self.period + (abs(hash(vid)) % 1000) / 1000.0) % 1.0

            for idx, seat_x in enumerate((0.28, 0.72)):
                cx, cy = w * seat_x, h * 0.42
                sw, sh = w * 0.16, h * 0.30
                box = BBox(cx - sw, cy - sh, cx + sw, cy + sh * 1.6)
                kp = _mock_keypoints(cx, cy, sw, sh)
                res.persons.append(PersonObs(box=box, score=0.95, keypoints=kp))

            # 用相位窗口驱动行为，保证同一辆车的事件是可复现的
            res.extra["scripted"] = _scripted_behaviour(phase, self.rate)
            if res.extra["scripted"].get("phone"):
                cx, cy = w * 0.28, h * 0.30
                res.objects.append(ObjectDet(label="cell phone", score=0.88,
                                             box=BBox(cx + 20, cy - 10, cx + 46, cy + 36)))
            out.append(res)
        return out

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": "scripted",
                "note": "不读画面，仅用于冒烟测试与压测", "thread_safe": True}


def _mock_keypoints(cx: float, cy: float, sw: float, sh: float) -> np.ndarray:
    kp = np.zeros((17, 3), dtype=np.float32)
    pts = {
        "nose": (cx, cy - sh * 0.55), "left_eye": (cx + sw * 0.16, cy - sh * 0.68),
        "right_eye": (cx - sw * 0.16, cy - sh * 0.68), "left_ear": (cx + sw * 0.38, cy - sh * 0.60),
        "right_ear": (cx - sw * 0.38, cy - sh * 0.60), "left_shoulder": (cx + sw * 0.72, cy - sh * 0.05),
        "right_shoulder": (cx - sw * 0.72, cy - sh * 0.05), "left_hip": (cx + sw * 0.50, cy + sh * 0.95),
        "right_hip": (cx - sw * 0.50, cy + sh * 0.95), "left_wrist": (cx + sw * 0.60, cy + sh * 0.55),
        "right_wrist": (cx - sw * 0.60, cy + sh * 0.55),
    }
    for name, (x, y) in pts.items():
        kp[KP_INDEX[name]] = (x, y, _KP_STRONG)
    return kp


def _scripted_behaviour(phase: float, rate: float) -> dict[str, bool]:
    """把 [0,1) 的相位映射到一组行为。窗口宽度由 rate 缩放。"""
    def win(start: float, width: float) -> bool:
        return start <= phase < start + width * rate / 0.35
    return {
        "driver_no_seatbelt": win(0.02, 0.14),
        "fatigue": win(0.22, 0.13),
        "phone": win(0.42, 0.09),
        "distraction": win(0.56, 0.08),
        "passenger_no_seatbelt": win(0.70, 0.16),
        "smoking": win(0.88, 0.06),
    }
