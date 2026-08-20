"""把「画面里有什么」翻译成「行为指标」。

这一层是纯几何 / 图像处理，不含深度模型，因此**真模型后端和合成素材后端共用同一套代码**——
这正是模式B 能够先用合成素材验证时序链路、再换真实素材验证感知精度的原因。

每个分析器都明确标注了它的可信度等级：

  [真实]   有明确物理依据、在真实画面上成立（头部姿态 solvePnP、摄像头遮挡判定）
  [代理]   有依据但不是行业标准做法，未在标注集上评测（安全带斜带对比度、眼睛开合度）
  [不可用] 当前模型栈无法支撑，只能靠姿态代理，默认关闭（抽烟、双手脱离方向盘）

不要把 [代理] 的输出当成产品指标向甲方汇报。生产环境需要：
  - 安全带：专用安全带分割/检测模型（需自采数据标注，公开数据集极少）
  - 疲劳：68/478 点人脸对齐模型算标准 EAR + PERCLOS（本环境 mediapipe 装不上，见 README）
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .base import ObjectDet, PersonObs

# ---------------------------------------------------------------------------
# 1. 安全带 —— [代理] 斜向织带的对比度
# ---------------------------------------------------------------------------
def seatbelt_score(img: np.ndarray, person: PersonObs, kp_thr: float = 1.0) -> tuple[float, list[float] | None]:
    """检测躯干上是否存在「肩 → 对侧髋」的斜向织带。

    原理：三点式安全带在图像上永远是一条从一侧肩部斜拉到对侧髋部的**连续窄带**，
    它与衣服之间必然存在亮度或色度差（无论深色带配浅衣还是反过来）。
    因此沿两条候选对角线采样，与其左右平行偏移线做对比度比较，取较强的一条。

    返回 (score, 带状区域 bbox)。score ∈ [0,1]，越大越像有安全带。

    局限（必须如实说明）：
      - 深色衣服 + 深色安全带时对比度会塌掉；夜间红外画面同理
      - 「把安全带插扣插上但人坐在带子外面」这种规避行为，图像上看是系着的，检不出来
      - 未在任何标注数据集上评测过 —— 阈值 0.28 是在合成素材上定的经验值
    """
    ls = person.kp("left_shoulder", kp_thr)
    rs = person.kp("right_shoulder", kp_thr)
    lh = person.kp("left_hip", kp_thr)
    rh = person.kp("right_hip", kp_thr)
    if ls is None or rs is None:
        return 0.0, None
    # 髋部常被方向盘/仪表台挡住，缺失时用肩宽外推一个躯干高度
    shoulder_w = abs(ls[0] - rs[0]) or 1.0
    if lh is None:
        lh = (ls[0], ls[1] + 1.6 * shoulder_w)
    if rh is None:
        rh = (rs[0], rs[1] + 1.6 * shoulder_w)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    offset = max(3.0, 0.10 * shoulder_w)

    best, best_pts = 0.0, None
    for a, b in ((ls, rh), (rs, lh)):
        s = _line_contrast(gray, a, b, offset, w, h)
        if s > best:
            best, best_pts = s, (a, b)

    box = None
    if best_pts is not None:
        (ax, ay), (bx, by) = best_pts
        box = [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]
    return best, box


def _line_contrast(gray: np.ndarray, a: tuple[float, float], b: tuple[float, float],
                   offset: float, w: int, h: int, n: int = 24) -> float:
    """线上采样值与左右平行偏移线的平均对比度，归一化到 [0,1]。"""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 8:
        return 0.0
    nx, ny = -dy / length, dx / length          # 单位法向量

    ts = np.linspace(0.18, 0.82, n)             # 掐掉两端（肩关节/髋关节本身就有边缘）
    xs = ax + ts * dx
    ys = ay + ts * dy
    on = _sample(gray, xs, ys, w, h)
    l1 = _sample(gray, xs + nx * offset, ys + ny * offset, w, h)
    l2 = _sample(gray, xs - nx * offset, ys - ny * offset, w, h)
    valid = ~(np.isnan(on) | np.isnan(l1) | np.isnan(l2))
    if valid.sum() < n * 0.5:
        return 0.0
    on, l1, l2 = on[valid], l1[valid], l2[valid]

    diff = np.abs(on - (l1 + l2) / 2.0)
    # 织带是连续的：要求沿线对比度稳定（用中位数而非均值，抗局部高光）
    consistency = float(np.median(diff))
    return float(np.clip(consistency / 110.0, 0.0, 1.0))


def _sample(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> np.ndarray:
    xi = np.round(xs).astype(int)
    yi = np.round(ys).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full(xs.shape, np.nan, dtype=np.float64)
    out[ok] = gray[yi[ok], xi[ok]]
    return out


# ---------------------------------------------------------------------------
# 2. 眼睛开合 / PERCLOS —— [代理]
# ---------------------------------------------------------------------------
def eye_openness(img: np.ndarray, person: PersonObs, kp_thr: float = 1.0) -> float | None:
    """眼睛开合度代理指标，∈[0,1]，越小越接近闭眼。

    标准做法是 EAR = 眼睑上下距离 / 眼角水平距离，需要 6 个眼睑轮廓点。
    COCO 关键点每只眼只有 1 个中心点，**给不出 EAR**。

    这里的替代做法：在眼睛 ROI 内用 Otsu 分出暗区（瞳孔 + 睫毛线），
    量它的「高 / 宽」。睁眼时暗区近似圆（比值 ≈ 1），闭眼时坍缩成一条横线（比值 ≈ 0.15）。
    这个量与 EAR 单调相关，但**不是 EAR**，且没有在标注集上标定过。

    返回 None 表示这一帧拿不到可用的眼部 ROI（人脸侧过去、关键点分数太低）。
    """
    le = person.kp("left_eye", kp_thr)
    re = person.kp("right_eye", kp_thr)
    if le is None or re is None:
        return None
    inter = math.hypot(le[0] - re[0], le[1] - re[1])
    if inter < 8:
        return None

    vals = []
    for ex, ey in (le, re):
        rw = int(max(4, 0.42 * inter))
        rh = int(max(3, 0.30 * inter))
        x1, y1 = int(ex - rw), int(ey - rh)
        x2, y2 = int(ex + rw), int(ey + rh)
        roi = img[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 4:
            continue
        v = _dark_region_ratio(roi)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals))


def _dark_region_ratio(roi: np.ndarray) -> float | None:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if gray.std() < 6:            # 对比度过低（过曝/全黑），判不了
        return None
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    _, _, bw, bh = cv2.boundingRect(c)
    if bw < 2:
        return None
    return float(np.clip(bh / bw, 0.0, 1.2))


@dataclass
class PerclosMeter:
    """PERCLOS —— 单位时间内眼睛闭合的时间占比，疲劳检测的行业通用指标。

    真正的 PERCLOS(P80) 定义是「瞳孔被眼睑遮挡超过 80% 的时间占比」，
    这里用上面的开合代理替代，因此叫 [代理] PERCLOS。
    """

    window_s: float = 20.0
    closed_thr: float = 0.28
    _buf: deque = field(default_factory=deque)

    def update(self, openness: float | None, now: float) -> float:
        if openness is not None:
            self._buf.append((now, openness < self.closed_thr))
        while self._buf and now - self._buf[0][0] > self.window_s:
            self._buf.popleft()
        if not self._buf:
            return 0.0
        return sum(1 for _, c in self._buf if c) / len(self._buf)

    @property
    def samples(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# 3. 头部姿态 —— [真实] solvePnP
# ---------------------------------------------------------------------------
# 通用成人头部 3D 模型点（毫米，图像坐标习惯：X 右，Y 下，Z 朝镜头外为负）
_HEAD_3D = np.array([
    (0.0, 0.0, 0.0),          # 鼻尖
    (32.0, -34.0, -35.0),     # 左眼（解剖左，成像在画面右侧）
    (-32.0, -34.0, -35.0),    # 右眼
    (78.0, -18.0, -98.0),     # 左耳
    (-78.0, -18.0, -98.0),    # 右耳
], dtype=np.float64)
_HEAD_KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]


def head_pose(person: PersonObs, width: int, height: int, kp_thr: float = 1.0
              ) -> tuple[float, float, float] | None:
    """用 5 个面部关键点解 PnP，返回 (yaw, pitch, roll)，单位度。

    yaw > 0 表示头转向画面右侧。相机内参用「焦距 ≈ 图像宽度」的常用近似——
    没有标定过的舱内摄像头只能这样估，绝对角度会有偏差，但**相对变化是可靠的**，
    分心判定只需要相对变化。

    耳朵被遮挡（大角度侧脸）时会退化到几何近似，此时精度显著下降。
    """
    pts2d, pts3d = [], []
    for i, name in enumerate(_HEAD_KP):
        p = person.kp(name, kp_thr)
        if p is not None:
            pts2d.append(p)
            pts3d.append(_HEAD_3D[i])
    if len(pts2d) < 4:
        return _geometric_yaw_fallback(person, kp_thr)

    f = float(width)
    cam = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(np.array(pts3d), np.array(pts2d, dtype=np.float64), cam,
                               np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return _geometric_yaw_fallback(person, kp_thr)
    R, _ = cv2.Rodrigues(rvec)
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
        roll = 0.0
    # solvePnP 在近似平面点集上偶尔会翻到 ±180 的等价解，折回可读区间
    pitch = _wrap180(pitch)
    if abs(pitch) > 90:
        pitch = _wrap180(180.0 - pitch)
    return round(_wrap180(yaw), 1), round(pitch, 1), round(_wrap180(roll), 1)


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _geometric_yaw_fallback(person: PersonObs, kp_thr: float) -> tuple[float, float, float] | None:
    """退化方案：鼻尖相对双耳（或双眼）中点的水平偏移 → 近似 yaw。"""
    nose = person.kp("nose", kp_thr)
    if nose is None:
        return None
    lear, rear = person.kp("left_ear", kp_thr), person.kp("right_ear", kp_thr)
    le, re = person.kp("left_eye", kp_thr), person.kp("right_eye", kp_thr)
    ref = [p for p in (lear, rear, le, re) if p is not None]
    if len(ref) < 2:
        return None
    mid_x = sum(p[0] for p in ref) / len(ref)
    mid_y = sum(p[1] for p in ref) / len(ref)
    span = max(abs(ref[0][0] - ref[-1][0]), 1.0)
    yaw = math.degrees(math.atan2(nose[0] - mid_x, span * 0.8))
    pitch = math.degrees(math.atan2(nose[1] - mid_y, span * 0.8)) - 20.0
    return round(yaw, 1), round(pitch, 1), 0.0


# ---------------------------------------------------------------------------
# 4. 手机 —— [真实] COCO cell phone + 手/头几何关系
# ---------------------------------------------------------------------------
def phone_signal(person: PersonObs, objects: list[ObjectDet], kp_thr: float = 1.0,
                 near_ratio: float = 1.6) -> dict[str, Any]:
    """判定「拿着手机」。

    两条互补证据：
      1) COCO 检出 `cell phone`，且框中心落在头部或手腕附近 —— [真实]
      2) 手腕抬到耳朵旁边（接打电话的典型姿态）—— [代理]，单独不足以定罪

    只有 (1) 成立、或 (1)(2) 同时成立时才算命中，避免把「挠头」判成打电话。
    """
    head = _head_circle(person, kp_thr)
    wrists = [p for p in (person.kp("left_wrist", kp_thr), person.kp("right_wrist", kp_thr)) if p]

    phone_hit, phone_score, phone_box = False, 0.0, None
    if head is not None:
        hx, hy, hr = head
        for o in objects:
            if o.label != "cell phone":
                continue
            d_head = math.hypot(o.box.cx - hx, o.box.cy - hy)
            d_wrist = min([math.hypot(o.box.cx - wx, o.box.cy - wy) for wx, wy in wrists], default=1e9)
            if d_head < near_ratio * hr or d_wrist < near_ratio * hr:
                phone_hit = True
                if o.score > phone_score:
                    phone_score, phone_box = o.score, o.box.as_list()

    hand_to_ear = False
    if head is not None:
        hx, hy, hr = head
        for wx, wy in wrists:
            if math.hypot(wx - hx, wy - hy) < 1.25 * hr:
                hand_to_ear = True
    return {"phone_detected": phone_hit, "phone_score": round(phone_score, 3),
            "phone_box": phone_box, "hand_to_ear": hand_to_ear}


def hand_to_mouth(person: PersonObs, kp_thr: float = 1.0) -> bool:
    """[代理] 手靠近嘴部 —— 抽烟/喝水/打哈欠都会命中，单独用误报极高。"""
    nose = person.kp("nose", kp_thr)
    head = _head_circle(person, kp_thr)
    if nose is None or head is None:
        return False
    _, _, hr = head
    mouth = (nose[0], nose[1] + 0.35 * hr)
    for w in (person.kp("left_wrist", kp_thr), person.kp("right_wrist", kp_thr)):
        if w and math.hypot(w[0] - mouth[0], w[1] - mouth[1]) < 0.75 * hr:
            return True
    return False


def _head_circle(person: PersonObs, kp_thr: float) -> tuple[float, float, float] | None:
    """用面部关键点估一个头部外接圆 (cx, cy, r)。"""
    pts = [p for p in (person.kp(n, kp_thr) for n in _HEAD_KP) if p is not None]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    r = max(max(xs) - min(xs), max(ys) - min(ys), 8.0) * 0.85
    return cx, cy, r


# ---------------------------------------------------------------------------
# 5. 摄像头遮挡 —— [真实]
# ---------------------------------------------------------------------------
def camera_blocked(frame_stats: dict[str, float], blur_var_thr: float = 12.0,
                   dark_mean_thr: float = 18.0) -> tuple[bool, str]:
    """摄像头被贴住 / 完全失焦 / 夜间无补光。

    实际运营里这是**触发最频繁**的一条：司机为了躲监控贴胶带、挂毛巾，
    比疲劳驾驶常见得多。所以它必须是一等公民，而不是附属功能。
    """
    lap = frame_stats.get("lap_var", 999.0)
    mean = frame_stats.get("mean", 128.0)
    std = frame_stats.get("std", 64.0)
    if mean < dark_mean_thr:
        return True, "画面过暗（遮挡或夜间无补光）"
    if lap < blur_var_thr and std < 20:
        return True, "画面无纹理（镜头被遮挡或严重失焦）"
    return False, ""
