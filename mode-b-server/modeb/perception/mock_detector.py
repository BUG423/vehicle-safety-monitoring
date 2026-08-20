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
        hx1, hy1, hx2, hy2 = head
        hw, hh = hx2 - hx1, hy2 - hy1
        if hw < 12 or hh < 12:
            return None
        hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2

        kp = np.zeros((17, 3), dtype=np.float32)

        def put(name: str, x: float, y: float, s: float = _KP_STRONG) -> None:
            i = KP_INDEX[name]
            kp[i] = (x, y, s)

        # 面部：眼睛用画面里真实的眼部区域定位（见 _find_eyes），找不到就按几何摆放
        eyes = self._find_eyes(img, head)
        if eyes is not None:
            (rex, rey), (lex, ley) = eyes
        else:
            rex, rey = hcx - 0.20 * hw, hcy - 0.10 * hh
            lex, ley = hcx + 0.20 * hw, hcy - 0.10 * hh
        put("right_eye", rex, rey)
        put("left_eye", lex, ley)
        put("nose", (rex + lex) / 2, (rey + ley) / 2 + 0.16 * hh)
        put("right_ear", hx1 + 0.03 * hw, hcy)
        put("left_ear", hx2 - 0.03 * hw, hcy)

        if body is not None:
            bx1, by1, bx2, by2 = body
            bw = bx2 - bx1
            put("right_shoulder", bx1 + 0.10 * bw, by1 + 0.06 * (by2 - by1))
            put("left_shoulder", bx2 - 0.10 * bw, by1 + 0.06 * (by2 - by1))
            put("right_hip", bx1 + 0.22 * bw, by2 - 0.05 * (by2 - by1))
            put("left_hip", bx2 - 0.22 * bw, by2 - 0.05 * (by2 - by1))
            box = BBox(min(hx1, bx1), hy1, max(hx2, bx2), by2)
        else:
            put("right_shoulder", hcx - 0.9 * hw, hy2 + 0.35 * hh, _KP_WEAK)
            put("left_shoulder", hcx + 0.9 * hw, hy2 + 0.35 * hh, _KP_WEAK)
            box = BBox(hx1 - 0.6 * hw, hy1, hx2 + 0.6 * hw, hy2 + 2.4 * hh)

        return PersonObs(box=box, score=0.9, keypoints=kp)

    def _find_eyes(self, img: np.ndarray, head) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """在头部 ROI 上半部找两个左右对称的「眼位」。

        睁眼时是白色巩膜块，闭眼时是一条深色横线 —— 两种情况都要能定位，
        否则闭眼帧会因为「找不到眼睛」而被跳过，疲劳就永远检不出来。
        因此这里用「与周围皮肤色差最大的区域」而不是「白色区域」来定位。
        """
        hx1, hy1, hx2, hy2 = [int(v) for v in head]
        roi = img[max(0, hy1):hy2, max(0, hx1):hx2]
        if roi.size == 0 or roi.shape[0] < 12 or roi.shape[1] < 12:
            return None
        rh, rw = roi.shape[:2]
        band = roi[int(0.18 * rh):int(0.62 * rh), :]
        if band.size == 0:
            return None
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        med = float(np.median(gray))
        diff = np.abs(gray.astype(np.int16) - med).astype(np.uint8)
        _, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 4 or bh < 1 or bw > 0.5 * rw:
                continue
            cands.append((x + bw / 2, y + bh / 2, bw * bh))
        if len(cands) < 2:
            return None
        cands.sort(key=lambda c: c[2], reverse=True)
        top = sorted(cands[:4], key=lambda c: c[0])
        left_c, right_c = top[0], top[-1]
        if right_c[0] - left_c[0] < 0.18 * rw:
            return None
        ox, oy = max(0, hx1), max(0, hy1) + int(0.18 * rh)
        return (ox + left_c[0], oy + left_c[1]), (ox + right_c[0], oy + right_c[1])

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
