"""模式A 的回归测试 —— 全部走 mock 后端，无需任何 API key 即可运行。

覆盖三件最容易出错、且出错代价最高的事：
  1. 解析层能不能扛住模型的各种不规范输出（这是线上最高频的故障）；
  2. 规则层会不会把「看不清」当成「合规」（安全上最危险的错误）；
  3. 疲劳的时序判定会不会因为样本喂法不对而永远不告警（真实踩过的坑）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm_safety import SafetyPipeline, load_settings                      # noqa: E402
from vlm_safety._common import Severity, VehicleContext, ViolationType    # noqa: E402
from vlm_safety.parser import parse_vlm_output                            # noqa: E402
from vlm_safety.providers.mock import MockProvider, SCENARIOS             # noqa: E402
from vlm_safety.rules import RuleConfig, compute_perclos, evaluate_frame  # noqa: E402


def _pipe(scenario=None, **kw):
    s = load_settings(provider="mock", attach_thumbnail=False, **kw)
    return SafetyPipeline(s, provider=MockProvider(s, scenario=scenario))


def _img(name="all_clear"):
    """用 mock 场景名做种子的假图片字节 —— 解析链路不关心图片内容。"""
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (40, 40, 48)).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------- 解析层
class TestParser:
    def test_markdown_fence(self):
        r = parse_vlm_output('```json\n{"scene":{"view":"cabin_front","image_quality":"good",'
                             '"persons_visible":0},"occupants":[]}\n```')
        assert r.ok and r.frames[0].view == "cabin_front"

    def test_leading_chatter(self):
        r = parse_vlm_output('好的，我来分析这张图片：\n{"scene":{"view":"cabin_front",'
                             '"image_quality":"good","persons_visible":0},"occupants":[]}\n希望有帮助！')
        assert r.ok and r.frames[0].view == "cabin_front"

    def test_trailing_comma_and_comment(self):
        r = parse_vlm_output('{"scene":{"view":"cabin_front","image_quality":"good",'
                             '"persons_visible":0,},  // 注释\n"occupants":[],}')
        assert r.ok and r.repaired

    def test_single_element_array_unwrapped(self):
        """模型把枚举候选清单当数组回填 —— 真实踩过的坑，必须兼容。"""
        r = parse_vlm_output('{"scene":{"view":["cabin_front"],"image_quality":["good"],'
                             '"persons_visible":1},"occupants":[{"seat":["driver"],'
                             '"person_present":true,"seatbelt":{"state":["not_fastened"],'
                             '"evidence":"肩部无织带","confidence":0.9}}]}')
        assert r.ok
        assert r.frames[0].view == "cabin_front"
        assert r.frames[0].occupants[0].seat == "driver"
        assert r.frames[0].occupants[0].attr("seatbelt").state == "not_fastened"

    def test_enum_out_of_whitelist_degraded(self):
        r = parse_vlm_output('{"scene":{"view":"cabin_front","image_quality":"good",'
                             '"persons_visible":1},"occupants":[{"seat":"driver",'
                             '"person_present":true,"seatbelt":{"state":"maybe_fastened",'
                             '"evidence":"看着像","confidence":0.9}}]}')
        assert r.ok
        assert r.frames[0].occupants[0].attr("seatbelt").state == "unknown"
        assert r.coerced_fields

    def test_conclusion_without_evidence_degraded(self):
        """有结论说不出依据 —— 按反幻觉规则降级。"""
        r = parse_vlm_output('{"scene":{"view":"cabin_front","image_quality":"good",'
                             '"persons_visible":1},"occupants":[{"seat":"driver",'
                             '"person_present":true,"seatbelt":{"state":"not_fastened",'
                             '"evidence":"","confidence":0.95}}]}')
        assert r.frames[0].occupants[0].attr("seatbelt").state == "unknown"

    def test_garbage_returns_not_ok(self):
        r = parse_vlm_output("我看不清这张图片。")
        assert not r.ok and r.error

    def test_empty_returns_not_ok(self):
        assert not parse_vlm_output("").ok


# ---------------------------------------------------------------- 规则层
class TestRules:
    def test_occluded_seatbelt_is_undecidable_not_compliant(self):
        """核心安全要求：看不清 ≠ 合规。"""
        r = parse_vlm_output('{"scene":{"view":"cabin_front","image_quality":"good",'
                             '"persons_visible":1},"occupants":[{"seat":"driver",'
                             '"person_present":true,"seatbelt":{"state":"occluded",'
                             '"evidence":"被方向盘挡住","confidence":0.8}}]}')
        out = evaluate_frame(r.frames[0])
        assert not [d for d in out.detections if d.hit]
        assert any(u.violation is ViolationType.DRIVER_NO_SEATBELT for u in out.undecidable)

    def test_low_confidence_not_reported(self):
        r = parse_vlm_output('{"scene":{"view":"cabin_front","image_quality":"good",'
                             '"persons_visible":1},"occupants":[{"seat":"driver",'
                             '"person_present":true,"seatbelt":{"state":"not_fastened",'
                             '"evidence":"好像没有","confidence":0.3}}]}')
        out = evaluate_frame(r.frames[0], cfg=RuleConfig(min_confidence=0.55))
        assert not [d for d in out.detections if d.hit and d.violation is ViolationType.DRIVER_NO_SEATBELT]

    def test_single_frame_never_concludes_fatigue(self):
        r = parse_vlm_output('{"scene":{"view":"cabin_front","image_quality":"good",'
                             '"persons_visible":1},"occupants":[{"seat":"driver",'
                             '"person_present":true,"eyes":{"state":"closed",'
                             '"evidence":"眼睑闭合","confidence":0.95},'
                             '"mouth":{"state":"yawning","evidence":"张大嘴","confidence":0.9}}]}')
        out = evaluate_frame(r.frames[0])
        assert not [d for d in out.detections if d.violation is ViolationType.DRIVER_FATIGUE]
        assert any(u.violation is ViolationType.DRIVER_FATIGUE for u in out.undecidable)

    def test_speeding_requires_obd_never_from_image(self):
        from vlm_safety.rules import evaluate_vehicle_signals
        assert evaluate_vehicle_signals(None) == []
        dets = evaluate_vehicle_signals(VehicleContext(vehicle_id="V", speed_kmh=110,
                                                       speed_limit_kmh=80))
        assert len(dets) == 1
        assert dets[0].violation is ViolationType.VEHICLE_SPEEDING
        assert dets[0].source == "signal"     # 不是 vision

    def test_buckle_spoofing_flagged(self):
        """开关说已扣、画面无织带 —— 视觉相对信号的增量价值。"""
        r = parse_vlm_output('{"scene":{"view":"cabin_front","image_quality":"good",'
                             '"persons_visible":1},"occupants":[{"seat":"driver",'
                             '"person_present":true,"seatbelt":{"state":"not_fastened",'
                             '"evidence":"肩部清晰无织带","confidence":0.9}}]}')
        ctx = VehicleContext(vehicle_id="V", seatbelt_switch={"driver": True})
        out = evaluate_frame(r.frames[0], ctx=ctx)
        hit = [d for d in out.detections if d.hit][0]
        assert "插扣欺骗" in hit.evidence

    def test_perclos_insufficient_samples(self):
        r = parse_vlm_output('{"scene":{"view":"cabin_front","image_quality":"good",'
                             '"persons_visible":1},"occupants":[{"seat":"driver",'
                             '"person_present":true,"eyes":{"state":"closed",'
                             '"evidence":"闭眼","confidence":0.9}}]}')
        res = compute_perclos(r.frames)
        assert not res.sufficient and res.reason


# ---------------------------------------------------------------- 流水线
class TestPipeline:
    def test_instant_seatbelt_event_and_alerts(self):
        p = _pipe("driver_no_seatbelt")
        res = p.analyze([_img()], policy="instant")
        assert res.ok
        v = [e.violation for e in res.events]
        assert ViolationType.DRIVER_NO_SEATBELT in v
        ev = res.events[0]
        assert ev.mode.value == "mode_a_vlm"
        assert ev.severity is Severity.CRITICAL
        assert ev.raw_signals["vlm_simulated"] is True        # 模拟必须被如实标注
        assert ev.raw_signals["evidence_text"]
        # 双通道都必须投递成功
        for a in res.alerts:
            assert all(a["channels"].values()), a
        p.close()

    def test_all_clear_no_events(self):
        p = _pipe("all_clear")
        res = p.analyze([_img()], policy="instant")
        assert res.ok and res.events == []
        p.close()

    def test_not_a_cabin_short_circuits(self):
        p = _pipe("not_a_cabin")
        res = p.analyze([_img()], policy="instant")
        assert res.ok and res.events == []
        assert any("不是车内画面" in n for n in res.notes)
        p.close()

    def test_camera_blocked_reported(self):
        p = _pipe("camera_blocked")
        res = p.analyze([_img()], policy="instant")
        assert ViolationType.SYSTEM_CAMERA_BLOCKED in [e.violation for e in res.events]
        p.close()

    def test_temporal_fatigue_fires_after_min_duration(self):
        """回归：疲劳曾因聚合成一条 Detection 而永不告警。"""
        p = _pipe("fatigue")
        res = p.analyze([_img() for _ in range(8)], policy="temporal", frame_interval_s=1.0)
        assert res.perclos and res.perclos.sufficient
        fat = [e for e in res.events if e.violation is ViolationType.DRIVER_FATIGUE]
        assert fat, "多帧疲劳序列必须产生疲劳事件"
        assert fat[0].duration_s >= 4.0        # 契约层 min_duration_s
        assert "perclos" in fat[0].raw_signals
        p.close()

    def test_temporal_normal_no_fatigue_false_alarm(self):
        p = _pipe("all_clear")
        res = p.analyze([_img() for _ in range(8)], policy="temporal", frame_interval_s=1.0)
        assert not [e for e in res.events if e.violation is ViolationType.DRIVER_FATIGUE]
        p.close()

    def test_speeding_only_with_vehicle_context(self):
        p = _pipe("all_clear")
        res = p.analyze([_img()], policy="instant")
        assert ViolationType.VEHICLE_SPEEDING not in [e.violation for e in res.events]
        assert any("VehicleContext" in n for n in res.notes)

        res2 = p.analyze([_img()], policy="instant",
                         vehicle_ctx=VehicleContext(vehicle_id="V", speed_kmh=120,
                                                    speed_limit_kmh=80, engine_on=True))
        assert ViolationType.VEHICLE_SPEEDING in [e.violation for e in res2.events]
        p.close()

    def test_vlm_failure_produces_no_events(self):
        """模型挂了必须安静地失败，不能瞎报，也不能抛异常。"""
        from vlm_safety.providers.base import VLMProvider, VLMResponse

        class Broken(VLMProvider):
            name = "broken"

            def _invoke(self, images, system, user):
                raise RuntimeError("网络超时")

        s = load_settings(provider="mock")
        p = SafetyPipeline(s, provider=Broken(s))
        res = p.analyze([_img()], policy="instant")
        assert not res.ok and res.events == []
        assert res.vlm.error and "网络超时" in res.vlm.error
        p.close()

    def test_event_json_roundtrip(self):
        from vlm_safety._common import SafetyEvent
        import json
        p = _pipe("phone_use")
        res = p.analyze([_img()], policy="instant")
        for ev in res.events:
            back = SafetyEvent.from_dict(json.loads(ev.to_json()))
            assert back.violation is ev.violation
            assert back.vehicle_id == ev.vehicle_id
        p.close()


# ---------------------------------------------------------------- 图像预处理
class TestImaging:
    def test_downscale_saves_tokens(self):
        from PIL import Image
        import io
        from vlm_safety.imaging import prepare_image
        buf = io.BytesIO()
        Image.new("RGB", (1920, 1080), (10, 10, 10)).save(buf, format="JPEG")
        prep = prepare_image(buf.getvalue(), max_edge=896)
        assert max(prep.width, prep.height) == 896
        assert prep.est_vision_tokens() < 1920 * 1080 / 750

    def test_blur_flag_is_honest(self):
        """脱敏不可用时必须如实标记，不能假装成功。"""
        from PIL import Image
        import io
        from vlm_safety.imaging import prepare_image
        buf = io.BytesIO()
        Image.new("RGB", (320, 240), (10, 10, 10)).save(buf, format="JPEG")
        prep = prepare_image(buf.getvalue(), blur_faces=True)
        assert prep.blur_requested is True
        assert isinstance(prep.blur_available, bool)
        if not prep.blur_available:
            assert prep.faces_blurred == 0


class TestMockProviderHonesty:
    def test_mock_always_marked_simulated(self):
        s = load_settings(provider="mock")
        for name in SCENARIOS:
            resp = MockProvider(s, scenario=name).analyze([], "sys", "user")
            assert resp.simulated is True

    def test_scenarios_are_parseable(self):
        """每个预置场景都必须能被真实解析器解析 —— 保证 mock 与真实链路同构。"""
        s = load_settings(provider="mock")
        from PIL import Image
        import io
        from vlm_safety.imaging import prepare_image
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format="JPEG")
        prep = prepare_image(buf.getvalue())
        for name in SCENARIOS:
            resp = MockProvider(s, scenario=name).analyze([prep], "s", "u")
            assert parse_vlm_output(resp.text).ok, name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
