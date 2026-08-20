"""真模型感知后端 —— 嵌入式算力档位的 DMS 三件套。

选型理由（详见 DESIGN.md「模型怎么塞进去」）：

===================  ========  ============  ==========================================
模型                 权重大小  输入           为什么选它
===================  ========  ============  ==========================================
YuNet                227 KB    320x240       专为端侧设计的人脸检测器，同时给 5 个关键点，
                                             OpenCV 原生支持，RKNN/TensorRT 都能直接转
insightface 2d106det 4.8 MB    192x192       106 点关键点，覆盖眼轮廓 → 能算真 EAR；
                                             只跑在人脸框内，与画面分辨率解耦
NanoDet-Plus-m       3.6 MB    416x416       COCO 80 类，含 cell phone；ONNX 可直转 RKNN。
                                             最重的一档，因此**降频调度**不是每帧都跑
===================  ========  ============  ==========================================

两条实测得出的运行时选择（数字见 README 性能表，均为本机真跑）：

* **NanoDet 用 onnxruntime 而不是 OpenCV DNN**：同一个 ONNX，单线程 27.1 ms vs 112.4 ms，
  差 4.2 倍。OpenCV DNN 的通用 CPU 后端对这类模型没有优化到位。
* **CPU 上不要用 INT8 权重**：INT8 版单线程 78.0 ms，比 FP32 的 27.1 ms 还慢 2.9 倍。
  ONNX 的 QDQ 量化图在 CPU 上要反复反量化，收益为负；
  **量化的收益要到 NPU（RKNN / TensorRT）上才兑现**，那里有原生 INT8 算力。
  因此 `use_int8_detector` 默认为 False，转 RKNN 时再打开。

人脸检测仍走 OpenCV 的 `FaceDetectorYN`，因为 YuNet 的解码与 NMS 已经封装在里面，
自己在 onnxruntime 上重写后处理没有收益（实测 320x240 单线程 5.3 ms，不是瓶颈）。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import DRIVER, FRONT_PASSENGER, ObjectObs, Perception, PerceptionBackend, SeatObs, now_ts

# --------------------------------------------------------------------------
# insightface 2d106 关键点分组（本仓库实测确认，见 README「关键点语义」）
# --------------------------------------------------------------------------
LMK_CONTOUR = range(0, 33)
LMK_EYE_L = range(33, 43)      # 图像左侧的眼睛（=被拍者的右眼）
LMK_BROW_L = range(43, 52)
LMK_MOUTH = range(52, 72)
LMK_NOSE = range(72, 87)
LMK_EYE_R = range(87, 97)      # 图像右侧的眼睛
LMK_BROW_R = range(97, 106)
LMK_NOSE_TIP = 86
LMK_CHIN = 0

COCO_NAMES = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
    "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat "
    "baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl "
    "banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant "
    "bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink "
    "refrigerator book clock vase scissors teddy_bear hair_drier toothbrush"
).split()

# 与「驾驶中使用手机」相关的 COCO 类别。COCO 没有香烟类别 —— 抽烟检测需要自训练模型，
# 本后端不产出 smoke_score，README 中如实说明。
PHONE_LABELS = {"cell_phone", "remote"}

# 通用人脸 3D 模型（毫米），用于 solvePnP 估头姿。取自 OpenCV 经典 6 点模型。
_FACE_3D = np.array([
    (0.0, 0.0, 0.0),          # 鼻尖
    (0.0, -63.6, -12.5),      # 下巴
    (-43.3, 32.7, -26.0),     # 左眼外角
    (43.3, 32.7, -26.0),      # 右眼外角
    (-28.9, -28.9, -24.1),    # 左嘴角
    (28.9, -28.9, -24.1),     # 右嘴角
], dtype=np.float64)


def _spread_ratio(pts: np.ndarray) -> float:
    """点云在次主轴 / 主主轴上的展布比 —— 与关键点排列顺序无关的开合度。

    对眼轮廓：睁眼约 0.35~0.45，闭眼降到 0.15 以下；
    比经典 EAR（依赖固定的 6 点顺序）对模型换代更鲁棒。
    """
    q = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    p = q @ vt.T
    major = p[:, 0].max() - p[:, 0].min()
    minor = p[:, 1].max() - p[:, 1].min()
    return float(minor / major) if major > 1e-6 else 0.0


class OnnxDmsBackend(PerceptionBackend):
    """YuNet + 2d106det + NanoDet 组合后端。"""

    name = "onnx_dms"

    def __init__(self, cfg, *, enable_objects: bool = True) -> None:
        self.cfg = cfg
        md = Path(cfg.models_dir)
        self._face_path = md / "face_detection_yunet_2023mar.onnx"
        self._lmk_path = md / "2d106det.onnx"
        nd = ("object_detection_nanodet_2022nov_int8.onnx" if cfg.use_int8_detector
              else "object_detection_nanodet_2022nov.onnx")
        self._obj_path = md / nd
        missing = [p.name for p in (self._face_path, self._lmk_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"缺少模型 {missing}，先执行： python3 mode-c-edge/tools/fetch_models.py")

        cv2.setNumThreads(cfg.num_threads)
        self._det = cv2.FaceDetectorYN.create(
            str(self._face_path), "", (cfg.frame_width, cfg.frame_height),
            score_threshold=cfg.face_score_thr, nms_threshold=0.3, top_k=8)

        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.num_threads
        so.inter_op_num_threads = 1
        self._lmk = ort.InferenceSession(str(self._lmk_path), so,
                                         providers=["CPUExecutionProvider"])
        self._lmk_in = self._lmk.get_inputs()[0].name

        self._obj = None
        if enable_objects and self._obj_path.exists():
            self._obj = ort.InferenceSession(str(self._obj_path), so,
                                             providers=["CPUExecutionProvider"])
            self._obj_in = self._obj.get_inputs()[0].name
            self._obj_mean = np.array([103.53, 116.28, 123.675], np.float32).reshape(1, 1, 3)
            self._obj_std = np.array([57.375, 57.12, 58.395], np.float32).reshape(1, 1, 3)
            self._obj_strides = (8, 16, 32)
            self._obj_proj = np.arange(8, dtype=np.float32)
            self._obj_anchors = []
            for st in self._obj_strides:
                n = 416 // st
                xv, yv = np.meshgrid(np.arange(n) * st, np.arange(n) * st)
                self._obj_anchors.append(
                    np.column_stack((xv.ravel() + 0.5 * (st - 1), yv.ravel() + 0.5 * (st - 1))))
        self._last_objects: list[ObjectObs] = []

    # ---------------- 关键点 ----------------
    def _landmarks(self, img: np.ndarray, bbox) -> np.ndarray:
        """在人脸框上做 insightface 标准的相似变换裁剪，再回投到原图坐标。"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        s = 192.0 / (max(x2 - x1, y2 - y1) * 1.5)
        M = np.array([[s, 0, -s * cx + 96], [0, s, -s * cy + 96]], np.float32)
        crop = cv2.warpAffine(img, M, (192, 192), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(crop, 1.0, (192, 192), (0, 0, 0), swapRB=True)
        pred = self._lmk.run(None, {self._lmk_in: blob})[0][0].reshape(-1, 2)
        pred = (pred + 1.0) * 96.0
        IM = cv2.invertAffineTransform(M)
        return pred @ IM[:, :2].T + IM[:, 2]

    def _head_pose(self, pts: np.ndarray, shape) -> tuple[float, float, float] | tuple[None, None, None]:
        h, w = shape
        eye_l = pts[list(LMK_EYE_L)]
        eye_r = pts[list(LMK_EYE_R)]
        mouth = pts[list(LMK_MOUTH)]
        img_pts = np.array([
            pts[LMK_NOSE_TIP],
            pts[LMK_CHIN],
            eye_l[eye_l[:, 0].argmin()],       # 左眼外角
            eye_r[eye_r[:, 0].argmax()],       # 右眼外角
            mouth[mouth[:, 0].argmin()],
            mouth[mouth[:, 0].argmax()],
        ], dtype=np.float64)
        f = float(w)
        cam = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)
        ok, rvec, _ = cv2.solvePnP(_FACE_3D, img_pts, cam, np.zeros((4, 1)),
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None
        R, _ = cv2.Rodrigues(rvec)
        sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
        yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        roll = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        # solvePnP 给的是相机→模型的旋转，折算成「正视为 0」的直观角度
        roll = roll - 180.0 if roll > 90 else (roll + 180.0 if roll < -90 else roll)
        return yaw, pitch, roll

    # ---------------- 安全带（视觉）----------------
    @staticmethod
    def _belt_score(img: np.ndarray, head_bbox, cfg) -> float:
        """在人脸框正下方的躯干 ROI 里找斜跨的亮带。

        这是没有车身总线安全带信号时的兜底手段，精度未在标注数据集上验证过，
        务必与 CAN 扣合开关信号联合判定（见 analyzers.SeatbeltRule）。
        """
        x1, y1, x2, y2 = head_bbox
        fw, fh = x2 - x1, y2 - y1
        h, w = img.shape[:2]
        rx1, rx2 = int(max(0, x1 - fw)), int(min(w, x2 + fw))
        ry1, ry2 = int(min(h - 1, y2 + 0.15 * fh)), int(min(h, y2 + 2.2 * fh))
        if ry2 - ry1 < 16 or rx2 - rx1 < 16:
            return 0.0
        roi = img[ry1:ry2, rx1:rx2]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thr = max(120, int(np.percentile(g, 90)))
        mask = ((g > thr).astype(np.uint8)) * 255
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=22,
                                minLineLength=int(cfg.belt_line_min_len * max(roi.shape)),
                                maxLineGap=12)
        lo, hi = cfg.belt_angle_range
        best = 0.0
        if lines is not None:
            for lx1, ly1, lx2, ly2 in lines[:, 0]:
                dx, dy = float(lx2 - lx1), float(ly2 - ly1)
                a = abs(np.degrees(np.arctan2(dy, dx)))
                a = min(a, 180.0 - a)
                if lo <= a <= hi:
                    best = max(best, float(np.hypot(dx, dy)) / max(roi.shape))
        return min(best, 1.0)

    # ---------------- 目标检测 ----------------
    def _detect_objects(self, img: np.ndarray) -> list[ObjectObs]:
        h, w = img.shape[:2]
        r = cv2.resize(img, (416, 416)).astype(np.float32)
        r = (r - self._obj_mean) / self._obj_std
        outs = self._obj.run(None, {self._obj_in: cv2.dnn.blobFromImage(r)})
        # 输出按「通道数 + 空间尺寸」配对，不依赖 ONNX 里的输出顺序
        cls = sorted((o for o in outs if o.shape[-1] == 80), key=lambda o: -o.shape[1])
        box = sorted((o for o in outs if o.shape[-1] == 32), key=lambda o: -o.shape[1])
        B, S = [], []
        for st, c, b, a in zip(self._obj_strides, cls, box, self._obj_anchors):
            c, b = c[0], b[0]
            e = np.exp(b.reshape(-1, 8))
            d = ((e / e.sum(1, keepdims=True)) @ self._obj_proj).reshape(-1, 4) * st
            B.append(np.column_stack([a[:, 0] - d[:, 0], a[:, 1] - d[:, 1],
                                      a[:, 0] + d[:, 2], a[:, 1] + d[:, 3]]))
            S.append(c)
        B, S = np.concatenate(B), np.concatenate(S)
        cid, conf = S.argmax(1), S.max(1)
        wh = B.copy()
        wh[:, 2:] -= wh[:, :2]
        keep = cv2.dnn.NMSBoxes(wh.tolist(), conf.tolist(), self.cfg.object_score_thr, 0.6)
        out: list[ObjectObs] = []
        sx, sy = w / 416.0, h / 416.0
        for i in np.array(keep).flatten() if len(keep) else []:
            x1, y1, x2, y2 = B[i]
            out.append(ObjectObs(label=COCO_NAMES[cid[i]], score=float(conf[i]),
                                 bbox=(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))))
        return out

    # ---------------- 主流程 ----------------
    def process(self, frame: np.ndarray, frame_idx: int, ts: float | None = None) -> Perception:
        ts = now_ts(ts)
        h, w = frame.shape[:2]
        lat: dict[str, float] = {}

        t0 = time.perf_counter()
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame)
        lat["face_det"] = (time.perf_counter() - t0) * 1000

        seats: dict[str, SeatObs] = {}
        faces = [] if faces is None else sorted(faces, key=lambda f: f[0])
        # 座位归属：驾驶位在图像左侧（左舵车、DMS 相机装在中控上方朝后）
        seat_names = [DRIVER, FRONT_PASSENGER]
        t0 = time.perf_counter()
        for i, f in enumerate(faces[:2]):
            seat = seat_names[i] if i < len(seat_names) else f"extra_{i}"
            x, y, fw_, fh_ = f[:4]
            bbox = (int(x), int(y), int(x + fw_), int(y + fh_))
            obs = SeatObs(seat=seat, present=True, score=float(f[14]), head_bbox=bbox)
            try:
                pts = self._landmarks(frame, bbox)
                obs.eye_open = (_spread_ratio(pts[list(LMK_EYE_L)]) +
                                _spread_ratio(pts[list(LMK_EYE_R)])) / 2.0
                obs.mouth_open = _spread_ratio(pts[list(LMK_MOUTH)])
                obs.yaw_deg, obs.pitch_deg, obs.roll_deg = self._head_pose(pts, (h, w))
            except Exception:  # noqa: BLE001 - 单帧关键点失败不能中断主循环
                pass
            obs.belt_score = self._belt_score(frame, bbox, self.cfg)
            seats[seat] = obs
        lat["landmark"] = (time.perf_counter() - t0) * 1000

        # 目标检测降频调度：这是最重的一档，每帧都跑会直接吃掉整个算力预算
        if self._obj is not None and frame_idx % max(1, self.cfg.object_stride) == 0:
            t0 = time.perf_counter()
            self._last_objects = self._detect_objects(frame)
            lat["object_det"] = (time.perf_counter() - t0) * 1000
        objects = list(self._last_objects)

        # 手机证据：手机框与头部框有交叠或紧邻
        for obs in seats.values():
            if obs.head_bbox is None:
                continue
            best = 0.0
            for o in objects:
                if o.label in PHONE_LABELS and _near(o.bbox, obs.head_bbox):
                    best = max(best, o.score)
            obs.phone_score = best

        t0 = time.perf_counter()
        sharp = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        lat["sharpness"] = (time.perf_counter() - t0) * 1000

        return Perception(ts=ts, frame_idx=frame_idx, frame_shape=(h, w), backend=self.name,
                          seats=seats, objects=objects, sharpness=sharp, latency_ms=lat)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "real_model": True,
            "models": {
                "face_det": {"file": self._face_path.name, "runtime": "OpenCV DNN",
                             "size_kb": round(self._face_path.stat().st_size / 1024, 1)},
                "landmark": {"file": self._lmk_path.name, "runtime": "onnxruntime",
                             "size_kb": round(self._lmk_path.stat().st_size / 1024, 1)},
                "object_det": ({"file": self._obj_path.name, "runtime": "onnxruntime",
                                "size_kb": round(self._obj_path.stat().st_size / 1024, 1)}
                               if self._obj is not None else None),
            },
            "capabilities": {
                "眼睛开合/PERCLOS": "真实（106 点眼轮廓）",
                "哈欠": "真实（嘴部关键点展布比）",
                "头部姿态": "真实（solvePnP 6 点）",
                "手机": "真实（NanoDet COCO cell_phone 类）",
                "安全带": "视觉启发式（霍夫斜带线），精度未在标注集验证，建议与 CAN 扣合信号联判",
                "抽烟": "未实现 —— COCO 无香烟类别，需自训练模型",
            },
            "threads": self.cfg.num_threads,
        }


def _near(a, b, pad_ratio: float = 0.6) -> bool:
    """两框是否交叠或紧邻（按 b 的尺寸外扩后判交叠）。"""
    bx1, by1, bx2, by2 = b
    pw, ph = (bx2 - bx1) * pad_ratio, (by2 - by1) * pad_ratio
    bx1, by1, bx2, by2 = bx1 - pw, by1 - ph, bx2 + pw, by2 + ph
    ax1, ay1, ax2, ay2 = a
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)
