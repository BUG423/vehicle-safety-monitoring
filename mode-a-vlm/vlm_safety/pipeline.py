"""端到端流水线：图片 -> VLM -> 结构化观测 -> 规则判定 -> 防误报确认 -> 事件 -> 双通道告警。

两种确认策略（这是模式A 与另外两条路线最大的结构性差别）：

``instant``  发车前静态检查。车是停着的，检查目的就是「合规才准出场」，
             没有时序可言，命中即出事件。**但需要时序才能成立的违规（疲劳、分心）
             在这个策略下不会产生事件**，而是进入 ``undecidable`` 显式呈现。

``temporal`` 多帧序列 / 边缘上报的可疑片段复核。逐帧判定喂进 ``ViolationConfirmer``
             （契约层共用），走滑窗投票 + 最短持续 + 冷却 + 升级，与模式B/C 完全一致的口径。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import prompts as P
from ._common import (
    DetectionMode, Evidence, SafetyEvent, Severity, Subject, VehicleContext,
    ViolationConfirmer, ViolationType,
)
from .alerting import AlertRecorder, build_dispatcher
from .config import Settings, load_settings
from .imaging import PreparedImage, prepare_image, thumbnail_b64
from .parser import ParseResult, parse_vlm_output
from .providers import VLMProvider, VLMResponse, get_provider
from .rules import (
    Detection, PerclosResult, RuleConfig, Undecidable, evaluate_frame, evaluate_sequence,
    evaluate_vehicle_signals,
)


@dataclass
class PipelineResult:
    """一次检测的全部产物 —— 事件、证据、原始回答、成本、耗时。"""

    events: list[SafetyEvent] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    undecidable: list[Undecidable] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    parse: ParseResult | None = None
    vlm: VLMResponse | None = None
    perclos: PerclosResult | None = None
    images: list[PreparedImage] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    policy: str = "instant"
    timings_ms: dict[str, float] = field(default_factory=dict)
    simulated: bool = True

    @property
    def ok(self) -> bool:
        return bool(self.vlm and self.vlm.ok and self.parse and self.parse.ok)

    def to_dict(self, *, include_raw_text: bool = True) -> dict:
        vlm = self.vlm.to_dict() if self.vlm else None
        if vlm and not include_raw_text:
            vlm.pop("text", None)
        return {
            "ok": self.ok,
            "simulated": self.simulated,
            "policy": self.policy,
            "events": [e.to_dict() for e in self.events],
            "detections": [d.to_dict() for d in self.detections],
            "undecidable": [u.to_dict() for u in self.undecidable],
            "notes": self.notes,
            "observations": self.parse.to_dict() if self.parse else None,
            "vlm": vlm,
            "perclos": self.perclos.to_dict() if self.perclos else None,
            "images": [{"width": im.width, "height": im.height,
                        "original": f"{im.original_width}x{im.original_height}",
                        "est_vision_tokens": im.est_vision_tokens(),
                        "faces_blurred": im.faces_blurred,
                        "blur_requested": im.blur_requested,
                        "blur_available": im.blur_available,
                        "notes": im.notes} for im in self.images],
            "alerts": self.alerts,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }


class SafetyPipeline:
    """模式A 的主流程。线程安全性：确认器有状态，多车共用时请按车实例化。"""

    def __init__(self, settings: Settings | None = None, *,
                 provider: VLMProvider | None = None,
                 rule_config: RuleConfig | None = None,
                 confirmer: ViolationConfirmer | None = None,
                 dispatcher=None, recorder: AlertRecorder | None = None) -> None:
        self.settings = settings or load_settings()
        self.provider = provider or get_provider(self.settings)
        self.rule_config = rule_config or RuleConfig()
        self.confirmer = confirmer or ViolationConfirmer()
        if dispatcher is None:
            dispatcher, recorder = build_dispatcher(self.settings, recorder)
        self.dispatcher = dispatcher
        self.recorder = recorder or AlertRecorder()

    # ------------------------------------------------------------------
    def analyze(
        self,
        images: Sequence[bytes],
        *,
        vehicle_ctx: VehicleContext | None = None,
        policy: str = "auto",
        frame_interval_s: float = 1.0,
        base_ts: float | None = None,
        dispatch: bool = True,
        vehicle_id: str | None = None,
    ) -> PipelineResult:
        """跑一次完整检测。

        images  按时间先后排列的原始图片字节（单张 = 静态检查，多张 = 序列复核）
        policy  ``auto`` | ``instant`` | ``temporal``
        """
        t_start = time.perf_counter()
        result = PipelineResult()
        if not images:
            result.notes.append("没有输入图片")
            return result

        n = len(images)
        result.policy = policy if policy in ("instant", "temporal") else ("temporal" if n > 1 else "instant")

        # --- 1. 预处理（缩放 / 可选脱敏），直接决定 token 成本 ---
        t0 = time.perf_counter()
        prepared = [prepare_image(raw, max_edge=self.settings.max_image_edge,
                                  jpeg_quality=self.settings.jpeg_quality,
                                  blur_faces=self.settings.blur_faces) for raw in images]
        result.images = prepared
        result.timings_ms["preprocess"] = (time.perf_counter() - t0) * 1000

        # --- 2. 调 VLM ---
        system = P.SYSTEM_PROMPT
        user = P.build_user_prompt(n)
        t0 = time.perf_counter()
        vlm = self.provider.analyze(prepared, system, user)
        result.timings_ms["vlm"] = (time.perf_counter() - t0) * 1000
        result.vlm = vlm
        result.simulated = vlm.simulated
        if not vlm.ok:
            result.notes.append(f"VLM 调用失败：{vlm.error or '返回为空'}")
            result.timings_ms["total"] = (time.perf_counter() - t_start) * 1000
            return result

        # --- 3. 解析 + 白名单归一化 ---
        base_ts = base_ts if base_ts is not None else time.time()
        timestamps = [base_ts + i * frame_interval_s for i in range(n)]
        t0 = time.perf_counter()
        parsed = parse_vlm_output(vlm.text, expected_frames=n, frame_timestamps=timestamps)
        result.timings_ms["parse"] = (time.perf_counter() - t0) * 1000
        result.parse = parsed
        if not parsed.ok:
            # 解析失败的兜底：不产生任何违规事件（宁可漏，不可错），但明确记录，
            # 并把原始回答留在结果里供人工复核。
            result.notes.append(f"结构化解析失败：{parsed.error}；本次不产生任何违规事件")
            result.timings_ms["total"] = (time.perf_counter() - t_start) * 1000
            return result
        if parsed.repaired:
            result.notes.append("模型输出 JSON 语法不合法，已自动修复后解析")
        if parsed.coerced_fields:
            result.notes.append(f"{len(parsed.coerced_fields)} 个字段越界或缺依据，已降级为 unknown")

        # 帧数对不上时按实际帧对齐时间戳
        for i, f in enumerate(parsed.frames):
            if not f.ts:
                f.ts = timestamps[min(i, len(timestamps) - 1)]

        # --- 4. 规则判定 ---
        t0 = time.perf_counter()
        if len(parsed.frames) > 1:
            per_frame, perclos = evaluate_sequence(parsed.frames, cfg=self.rule_config, ctx=vehicle_ctx)
            result.perclos = perclos
        else:
            per_frame = [evaluate_frame(parsed.frames[0], cfg=self.rule_config, ctx=vehicle_ctx)]
        signal_dets = evaluate_vehicle_signals(vehicle_ctx)
        if signal_dets:
            per_frame[-1].detections.extend(signal_dets)
        elif vehicle_ctx is None:
            result.notes.append("未提供 VehicleContext：超速/急加速类违规本次不参与判定"
                                "（这类结论只能来自 OBD/GPS，图像给不出）")
        result.timings_ms["rules"] = (time.perf_counter() - t0) * 1000

        for out in per_frame:
            result.detections.extend(out.detections)
            result.undecidable.extend(out.undecidable)
            result.notes.extend(out.notes)

        # --- 5. 确认 + 建事件 ---
        vid = vehicle_id or self.settings.vehicle_id
        thumb = None
        if self.settings.attach_thumbnail:
            thumb = thumbnail_b64(prepared[-1].jpeg, edge=self.settings.thumbnail_edge)

        if result.policy == "instant":
            events = self._events_instant(per_frame, vid, thumb, parsed, vlm, timestamps[-1])
        else:
            events = self._events_temporal(per_frame, vid, thumb, parsed, vlm, result.perclos)
        result.events = events

        # --- 6. 双通道告警 ---
        if dispatch and events:
            t0 = time.perf_counter()
            for ev in events:
                res = self.dispatcher.dispatch(ev)
                result.alerts.append({"event_id": ev.event_id, "violation": ev.violation.value,
                                      "channels": res})
            result.timings_ms["alert"] = (time.perf_counter() - t0) * 1000

        result.timings_ms["total"] = (time.perf_counter() - t_start) * 1000
        return result

    # ------------------------------------------------------------------
    def _build_event(self, det: Detection, vehicle_id: str, thumb: str | None,
                     parsed: ParseResult, vlm: VLMResponse, *,
                     severity: Severity | None = None, duration_s: float = 0.0,
                     confidence: float | None = None,
                     extra_signals: dict[str, Any] | None = None) -> SafetyEvent:
        seat = det.seat if det.seat != "unknown" else None
        raw: dict[str, Any] = {
            "detector": "mode_a_vlm",
            "vlm_provider": vlm.provider,
            "vlm_model": vlm.model,
            "vlm_simulated": vlm.simulated,     # 诚实标注：模拟输出会一路带到后台
            "vlm_latency_ms": round(vlm.latency_ms, 1),
            "evidence_text": det.evidence,      # 模型给出的可见依据原文，供人工复核
            "detection_source": det.source,
            "frame_index": det.frame_index,
            "parse_repaired": parsed.repaired,
            "coerced_fields": parsed.coerced_fields[:10],
        }
        if extra_signals:
            raw.update(extra_signals)
        return SafetyEvent(
            violation=det.violation,
            vehicle_id=vehicle_id,
            mode=DetectionMode.VLM,
            severity=severity,
            subject=Subject(role=det.role, seat=seat),
            confidence=round(confidence if confidence is not None else det.confidence, 4),
            duration_s=duration_s,
            evidence=Evidence(frame_b64=thumb, captured_at=det.ts or time.time()),
            raw_signals=raw,
            ts=det.ts or time.time(),
        )

    def _events_instant(self, per_frame, vehicle_id, thumb, parsed, vlm, ts) -> list[SafetyEvent]:
        """静态检查：命中即出事件，去重后返回。"""
        events: list[SafetyEvent] = []
        seen: set[tuple[str, str]] = set()
        for out in per_frame:
            for det in out.detections:
                if not det.hit:
                    continue
                key = (det.violation.value, det.seat)
                if key in seen:
                    continue
                seen.add(key)
                if not det.ts:
                    det.ts = ts
                events.append(self._build_event(det, vehicle_id, thumb, parsed, vlm))
        return events

    def _events_temporal(self, per_frame, vehicle_id, thumb, parsed, vlm,
                         perclos: PerclosResult | None) -> list[SafetyEvent]:
        """序列复核：逐帧喂给契约层的确认器，只有确认通过才出事件。"""
        events: list[SafetyEvent] = []
        for out in per_frame:
            for det in out.detections:
                c = self.confirmer.update(det.violation, det.hit, confidence=det.confidence,
                                          key=det.key, now=det.ts or time.time())
                if not c.should_alert:
                    continue
                extra = {"confirm_window_ratio": c.confidence,
                         "confirm_policy": "ViolationConfirmer(滑窗投票+最短持续+冷却)"}
                if det.violation is ViolationType.DRIVER_FATIGUE and perclos:
                    extra["perclos"] = perclos.to_dict()
                events.append(self._build_event(
                    det, vehicle_id, thumb, parsed, vlm,
                    severity=c.severity, duration_s=c.duration_s,
                    confidence=c.confidence, extra_signals=extra))
        return events

    # ------------------------------------------------------------------
    def health(self) -> dict:
        return {
            "provider": self.provider.health(),
            "vehicle_id": self.settings.vehicle_id,
            "cabin_min_severity": self.settings.cabin_min_severity,
            "backend_webhook": self.settings.backend_webhook or "(未配置，打印到 stdout)",
            "blur_faces": self.settings.blur_faces,
            "max_image_edge": self.settings.max_image_edge,
        }

    def close(self) -> None:
        try:
            self.dispatcher.close()
        finally:
            self.provider.close()
