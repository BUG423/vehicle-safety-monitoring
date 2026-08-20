"""逐帧原始判定 —— 把感知输出翻译成「这一帧是否违规」。

这一层输出的是**噪声很大的逐帧判定**，绝对不能直接告警：
眨一次眼、手抬一下、检测框抖一下都会翻转结果。
收敛工作交给 common 的 `ViolationConfirmer`（滑窗投票 + 最短时长 + 冷却 + 升级）。

判定与确认刻意分开，是因为三条技术路线的判定层不同、确认层必须相同——
否则三条路线的告警时序不可比，选型时说不清差异到底来自感知还是来自策略。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import Decision, SubjectRole, VehicleContext, ViolationType  # noqa: E402

from ..config import RuleConfig  # noqa: E402
from ..perception import analyzers as A  # noqa: E402
from ..perception.base import PerceptionResult, PersonObs  # noqa: E402
from ..perception.face_module import FaceMetrics, MediaPipeFaceModule  # noqa: E402


@dataclass
class RawHit:
    """一帧、一个主体、一个违规类型上的原始判定。

    `decidable=False` 表示**这一帧根本判不了**（遮挡、目标未入镜、关键点缺失），
    与 `hit=False`（判了，没违规）是两件完全不同的事。
    安全检查里把「判不了」当成「合规」是最危险的失败模式：
    发车前摄像头被挡住，后台如果只是「没收到安全带事件」，会被读成检查通过而放行。
    """

    violation: ViolationType
    key: str                       # 确认器的分组键：座位 / "frame" / "vehicle"
    hit: bool
    confidence: float = 1.0
    seat: str | None = None
    track_id: int | None = None
    role: SubjectRole = SubjectRole.UNKNOWN
    bbox: list[float] | None = None
    signals: dict[str, Any] = field(default_factory=dict)
    decidable: bool = True
    reason: str = ""               # decidable=False 时说明为什么判不了
    evidence_text: str = ""        # 面向复核者的文字依据，不含影像

    @property
    def decision(self) -> Decision:
        return Decision.CONFIRMED if self.decidable else Decision.UNDECIDABLE


class ViolationRuleEngine:
    """一辆车一个实例 —— 内部持有 PERCLOS 等需要跨帧累计的状态。"""

    def __init__(self, cfg: RuleConfig | None = None, *,
                 face_module: MediaPipeFaceModule | None = None,
                 kp_thr: float = 3.0) -> None:
        self.cfg = cfg or RuleConfig()
        self.face = face_module
        self.kp_thr = kp_thr
        self._perclos: dict[str, A.PerclosMeter] = {}
        self.face_hit = 0      # 人脸模块成功解析的帧数（用于如实报告它到底起没起作用）
        self.face_miss = 0

    # -- 主入口 -------------------------------------------------------------
    def evaluate(self, res: PerceptionResult, img: np.ndarray,
                 ctx: VehicleContext | None = None) -> list[RawHit]:
        hits: list[RawHit] = []
        frame_hits = self._frame_level(res)
        hits.extend(frame_hits)
        blocked = frame_hits[0].hit

        if blocked:
            # 镜头被挡住时，所有依赖画面的检查都是「判不了」，不能当成「没违规」。
            # 这条不上报给驾驶员（他已经收到遮挡告警了），但必须让后台看到
            # 「这台车的安全带/疲劳检查没完成」。
            hits.extend(self._all_undecidable(res, "摄像头被遮挡或画面无有效纹理"))
            return hits

        for p in res.persons:
            hits.extend(self._person_level(res, img, p))
        hits.extend(self._occupancy(res, ctx))
        if ctx is not None:
            hits.extend(self._vehicle_level(ctx))
        return hits

    _VISUAL_CHECKS = (
        (ViolationType.DRIVER_NO_SEATBELT, "driver", SubjectRole.DRIVER),
        (ViolationType.DRIVER_FATIGUE, "driver", SubjectRole.DRIVER),
        (ViolationType.DRIVER_DISTRACTION, "driver", SubjectRole.DRIVER),
        (ViolationType.DRIVER_PHONE_USE, "driver", SubjectRole.DRIVER),
        (ViolationType.PASSENGER_NO_SEATBELT, "front_passenger", SubjectRole.PASSENGER),
    )

    def _all_undecidable(self, res: PerceptionResult, reason: str) -> list[RawHit]:
        return [RawHit(v, key, hit=False, confidence=0.0, seat=key, role=role,
                       decidable=False, reason=reason,
                       evidence_text=f"未能完成「{v.label_zh}」检查：{reason}",
                       signals={"undecidable_reason": reason,
                                **{k: round(x, 2) for k, x in res.frame_stats.items()}})
                for v, key, role in self._VISUAL_CHECKS]

    # -- 整帧级 -------------------------------------------------------------
    def _frame_level(self, res: PerceptionResult) -> list[RawHit]:
        blocked, why = A.camera_blocked(res.frame_stats, self.cfg.blur_var_thr, self.cfg.dark_mean_thr)
        return [RawHit(ViolationType.SYSTEM_CAMERA_BLOCKED, "frame", blocked,
                       confidence=1.0 if blocked else 0.0, role=SubjectRole.VEHICLE,
                       evidence_text=why or "画面亮度与纹理正常",
                       signals={"reason": why,
                                **{k: round(v, 2) for k, v in res.frame_stats.items()}})]

    def _occupancy(self, res: PerceptionResult, ctx: VehicleContext | None = None) -> list[RawHit]:
        n = res.occupancy
        # 核定载客数以行驶证为准（契约层 1.2 的 seat_capacity），没有才退回配置默认值
        cap = (ctx.seat_capacity if ctx is not None and ctx.seat_capacity else self.cfg.max_occupancy)
        out = [RawHit(ViolationType.PASSENGER_OVERLOAD, "vehicle", n > cap,
                      confidence=1.0, role=SubjectRole.PASSENGER,
                      evidence_text=f"画面内检出 {n} 人，核定载客 {cap} 人",
                      signals={"occupancy": n, "seat_capacity": cap,
                               "capacity_source": "行驶证" if (ctx and ctx.seat_capacity) else "配置默认值"})]
        # 驾驶位无人：只在画面里确实检出了人（说明摄像头正常）却没有司机时才判，
        # 否则夜间/遮挡导致的「检不到人」会被错报成「司机不在」
        driver_missing = res.driver() is None
        out.append(RawHit(ViolationType.DRIVER_ABSENT, "driver", n > 0 and driver_missing,
                          confidence=1.0, role=SubjectRole.DRIVER,
                          evidence_text=f"画面内 {n} 人，驾驶位无人" if driver_missing else "驾驶位有人",
                          signals={"occupancy": n}))
        if driver_missing:
            # 驾驶位没人 → 驾驶员的四项检查全部判不了（而不是「都合规」）
            reason = "驾驶位未检出人员（未入镜/严重遮挡/夜间失效）" if n == 0 else "驾驶位未检出人员"
            for v, key, role in self._VISUAL_CHECKS:
                if role is SubjectRole.DRIVER:
                    out.append(RawHit(v, key, hit=False, confidence=0.0, seat=key, role=role,
                                      decidable=False, reason=reason,
                                      evidence_text=f"未能完成「{v.label_zh}」检查：{reason}",
                                      signals={"undecidable_reason": reason, "occupancy": n}))
        if not any(p.seat == "front_passenger" for p in res.persons):
            # 副驾没人不是违规，也不该产出「乘客未系带=合规」的记录 —— 什么都不报
            pass
        return out

    # -- 人员级 -------------------------------------------------------------
    def _person_level(self, res: PerceptionResult, img: np.ndarray, p: PersonObs) -> list[RawHit]:
        seat = p.seat
        is_driver = seat == "driver"
        role = SubjectRole.DRIVER if is_driver else SubjectRole.PASSENGER
        key = seat if seat != "unknown" else f"track{p.track_id}"
        bbox = p.box.normalized(res.width, res.height)
        hits: list[RawHit] = []

        # --- 安全带 ---
        v = ViolationType.DRIVER_NO_SEATBELT if is_driver else ViolationType.PASSENGER_NO_SEATBELT
        shoulders_ok = (p.kp("left_shoulder", self.kp_thr) is not None
                        and p.kp("right_shoulder", self.kp_thr) is not None)
        if not shoulders_ok:
            # 肩部关键点拿不到就没有躯干 ROI，安全带**无从判起**——
            # 这时候报「合规」等于放行一台没检查过的车
            reason = "肩部关键点不可见（侧身/遮挡/画面截断）"
            hits.append(RawHit(v, key, hit=False, confidence=0.0, seat=seat,
                               track_id=p.track_id, role=role, bbox=bbox,
                               decidable=False, reason=reason,
                               evidence_text=f"未能完成「{v.label_zh}」检查：{reason}",
                               signals={"undecidable_reason": reason}))
        else:
            belt, belt_box = A.seatbelt_score(img, p, self.kp_thr)
            no_belt = belt < self.cfg.seatbelt_score_thr
            hits.append(RawHit(v, key, no_belt,
                               confidence=float(np.clip(1.0 - belt / max(self.cfg.seatbelt_score_thr, 1e-6), 0, 1)),
                               seat=seat, track_id=p.track_id, role=role, bbox=bbox,
                               evidence_text=("躯干未见斜跨织带" if no_belt
                                              else f"躯干可见斜跨织带（对比度 {belt:.2f}）"),
                               signals={"belt_score": round(belt, 3), "belt_roi": belt_box,
                                        "method": "斜带对比度[代理]"}))

        if not is_driver:
            return hits   # 乘客只查安全带；对乘客做疲劳/分心判定既无意义也不合规

        # --- 人脸精细指标（可用时优先，把疲劳从[代理]升级为标准 EAR/PERCLOS） ---
        face = self._face_metrics(img, p)

        # --- 疲劳：PERCLOS ---
        if face is not None:
            openness = 0.0 if face.eyes_closed else 1.0
            eye_signals = face.to_dict()
        else:
            o = A.eye_openness(img, p, self.kp_thr)
            openness = o
            eye_signals = {"eye_open_proxy": None if o is None else round(o, 3),
                           "source": "eye_roi_proxy[代理]"}
        meter = self._perclos.setdefault(key, A.PerclosMeter(self.cfg.perclos_window_s,
                                                            self.cfg.eye_open_thr))
        closed_now = None if openness is None else (openness < self.cfg.eye_open_thr)
        perclos = meter.update(openness, res.ts)
        # 这一帧只回答「此刻眼睛是不是闭着的」，不做任何时间上的聚合。
        # 「闭多久才算疲劳」「眨眼算不算」全部交给 ViolationConfirmer
        # （8 秒窗 / 命中率 50% / 最短 4 秒）——时间聚合只做一次，做两次会把告警延迟叠加。
        # PERCLOS 单独作为一路证据：即使每次闭眼都很短，只要窗口内闭眼占比超阈值
        # （微睡眠的典型表现），同样算命中。
        fatigue = bool(closed_now) or perclos >= self.cfg.perclos_thr
        if closed_now is None:
            reason = "眼部区域不可见（低头/侧脸/戴墨镜/光线不足）"
            hits.append(RawHit(ViolationType.DRIVER_FATIGUE, key, hit=False, confidence=0.0,
                               seat=seat, track_id=p.track_id, role=role, bbox=bbox,
                               decidable=False, reason=reason,
                               evidence_text=f"未能完成「疲劳」检查：{reason}",
                               signals={"undecidable_reason": reason, **eye_signals}))
        else:
            hits.append(RawHit(ViolationType.DRIVER_FATIGUE, key, fatigue,
                               confidence=round(float(perclos), 3), seat=seat,
                               track_id=p.track_id, role=role, bbox=bbox,
                               evidence_text=(f"近 {self.cfg.perclos_window_s:.0f} 秒闭眼占比 "
                                              f"{perclos:.0%}" + ("，当前闭眼" if closed_now else "，当前睁眼")),
                               signals={"perclos": round(perclos, 3), "samples": meter.samples,
                                        **eye_signals}))

        # --- 分心：头部姿态 ---
        if face is not None:
            yaw, pitch, pose_src = face.yaw, face.pitch, "mediapipe_matrix[真实]"
        else:
            pose = A.head_pose(p, res.width, res.height, self.kp_thr)
            yaw, pitch = (pose[0], pose[1]) if pose else (None, None)
            pose_src = "solvePnP_5pt[真实]"
        if yaw is None:
            reason = "面部关键点不足，头部姿态无法解算"
            hits.append(RawHit(ViolationType.DRIVER_DISTRACTION, key, hit=False, confidence=0.0,
                               seat=seat, track_id=p.track_id, role=role, bbox=bbox,
                               decidable=False, reason=reason,
                               evidence_text=f"未能完成「分心」检查：{reason}",
                               signals={"undecidable_reason": reason, "source": pose_src}))
        else:
            distracted = (abs(yaw) > self.cfg.yaw_distract_deg
                          or abs(pitch) > self.cfg.pitch_distract_deg)
            hits.append(RawHit(ViolationType.DRIVER_DISTRACTION, key, bool(distracted),
                               confidence=0.9 if distracted else 0.1, seat=seat,
                               track_id=p.track_id, role=role, bbox=bbox,
                               evidence_text=f"头部偏航 {yaw:.0f}°、俯仰 {pitch:.0f}°",
                               signals={"yaw": yaw, "pitch": pitch, "source": pose_src}))

        # --- 手机 ---
        ph = A.phone_signal(p, res.objects, self.kp_thr, self.cfg.phone_near_head_px_ratio)
        phone_hit = bool(ph["phone_detected"])
        phone_bbox = bbox
        if ph["phone_box"]:
            b = ph["phone_box"]
            phone_bbox = [round(b[0] / res.width, 4), round(b[1] / res.height, 4),
                          round(b[2] / res.width, 4), round(b[3] / res.height, 4)]
        if A._head_circle(p, self.kp_thr) is None:
            reason = "头部位置无法圈定，手机与头/手的相对关系判不了"
            hits.append(RawHit(ViolationType.DRIVER_PHONE_USE, key, hit=False, confidence=0.0,
                               seat=seat, track_id=p.track_id, role=role, bbox=bbox,
                               decidable=False, reason=reason,
                               evidence_text=f"未能完成「使用手机」检查：{reason}",
                               signals={"undecidable_reason": reason}))
        else:
            hits.append(RawHit(ViolationType.DRIVER_PHONE_USE, key, phone_hit,
                               confidence=float(ph["phone_score"]) if phone_hit else 0.0,
                               seat=seat, track_id=p.track_id, role=role, bbox=phone_bbox,
                               evidence_text=("检出手机且位于头部/手腕附近" if phone_hit
                                              else "未在头部或手腕附近检出手机"),
                               signals=ph))

        # --- 抽烟 / 双手脱离方向盘：默认关闭，理由见 config.RuleConfig ---
        if self.cfg.enable_smoking_proxy:
            smoking = A.hand_to_mouth(p, self.kp_thr) and not phone_hit
            hits.append(RawHit(ViolationType.DRIVER_SMOKING, key, smoking, confidence=0.4,
                               seat=seat, track_id=p.track_id, role=role, bbox=bbox,
                               signals={"method": "手到嘴姿态代理[不可用-高误报]"}))
        if self.cfg.enable_hands_off_proxy:
            wrists = [p.kp(n, self.kp_thr) for n in ("left_wrist", "right_wrist")]
            hands_off = all(w is None for w in wrists)
            hits.append(RawHit(ViolationType.DRIVER_HANDS_OFF_WHEEL, key, hands_off, confidence=0.3,
                               seat=seat, track_id=p.track_id, role=role, bbox=bbox,
                               signals={"method": "手腕关键点缺失代理[不可用-方向盘未标定]"}))
        return hits

    def _face_metrics(self, img: np.ndarray, p: PersonObs) -> FaceMetrics | None:
        if self.face is None:
            return None
        roi = _head_roi(img, p, self.kp_thr)
        if roi is None:
            self.face_miss += 1
            return None
        m = self.face.analyze(roi)
        if m is None:
            self.face_miss += 1
            return None
        self.face_hit += 1
        return m

    # -- 车辆信号级（非视觉） ------------------------------------------------
    def _vehicle_level(self, ctx: VehicleContext) -> list[RawHit]:
        over = (ctx.speed_kmh is not None and ctx.speed_limit_kmh is not None
                and ctx.speed_kmh > ctx.speed_limit_kmh + self.cfg.speeding_tolerance_kmh)
        if ctx.speed_kmh is None or ctx.speed_limit_kmh is None:
            reason = "缺少车速或限速信号（OBD/GPS 未接入）"
            return [RawHit(ViolationType.VEHICLE_SPEEDING, "vehicle", hit=False, confidence=0.0,
                           role=SubjectRole.VEHICLE, decidable=False, reason=reason,
                           evidence_text=f"未能完成「超速」检查：{reason}",
                           signals={"undecidable_reason": reason})]
        return [RawHit(ViolationType.VEHICLE_SPEEDING, "vehicle", bool(over), confidence=1.0,
                       role=SubjectRole.VEHICLE,
                       evidence_text=f"车速 {ctx.speed_kmh:.0f} km/h，限速 {ctx.speed_limit_kmh:.0f} km/h",
                       signals={"speed_kmh": ctx.speed_kmh, "limit_kmh": ctx.speed_limit_kmh,
                                "source": "OBD/GPS，非视觉"})]


def _head_roi(img: np.ndarray, p: PersonObs, kp_thr: float, pad: float = 0.75) -> np.ndarray | None:
    """裁出头部 ROI 交给人脸模块 —— 整帧丢进去脸太小检不到，裁出来才有分辨率。"""
    pts = [q for q in (p.kp(n, kp_thr) for n in
                       ("nose", "left_eye", "right_eye", "left_ear", "right_ear")) if q is not None]
    h, w = img.shape[:2]
    if len(pts) >= 2:
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        r = max(max(xs) - min(xs), max(ys) - min(ys), 16.0) * (1.0 + pad)
    else:
        cx, cy = p.box.cx, p.box.y1 + 0.18 * p.box.h
        r = max(p.box.w * 0.5, 24.0)
    x1, y1 = int(max(0, cx - r)), int(max(0, cy - r))
    x2, y2 = int(min(w, cx + r)), int(min(h, cy + r))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return img[y1:y2, x1:x2]
