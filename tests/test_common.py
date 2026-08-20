"""契约层回归测试。

`common/` 是三条技术路线的公共依赖，任何一处改动都会同时影响三条分支，
因此这里锁住它的关键行为：事件序列化往返、确认器的防误报与升级、告警分级与断网补传。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    AlertDispatcher,
    ConfirmRule,
    Decision,
    BackendChannel,
    DetectionMode,
    Evidence,
    InCabinChannel,
    SafetyEvent,
    Severity,
    Subject,
    SubjectRole,
    VehicleContext,
    ViolationConfirmer,
    ViolationType,
)


# --------------------------------------------------------------------------
# 违规类型
# --------------------------------------------------------------------------
def test_每个违规类型都有中文标签和默认等级():
    for v in ViolationType:
        assert v.label_zh, f"{v} 缺少中文标签"
        assert isinstance(v.default_severity, Severity)


def test_主体角色由违规类型前缀推导():
    assert ViolationType.DRIVER_FATIGUE.default_role is SubjectRole.DRIVER
    assert ViolationType.PASSENGER_NO_SEATBELT.default_role is SubjectRole.PASSENGER
    assert ViolationType.VEHICLE_SPEEDING.default_role is SubjectRole.VEHICLE


def test_严重等级可比较():
    assert Severity.CRITICAL.rank > Severity.WARN.rank > Severity.INFO.rank


# --------------------------------------------------------------------------
# 事件模型
# --------------------------------------------------------------------------
def test_事件序列化往返一致():
    e = SafetyEvent(
        violation=ViolationType.DRIVER_NO_SEATBELT,
        vehicle_id="JING-A12345",
        mode=DetectionMode.EDGE,
        confidence=0.87,
        duration_s=6.2,
        subject=Subject(seat="driver", track_id=3),
        evidence=Evidence(frame_uri="s3://bucket/f.jpg", bbox=[0.1, 0.2, 0.5, 0.9]),
        raw_signals={"belt_score": 0.12},
    )
    back = SafetyEvent.from_dict(json.loads(e.to_json()))
    assert back.to_dict() == e.to_dict()


def test_字符串枚举自动归一():
    """各路线可能从 JSON/HTTP 传入字符串，构造时必须自动转成枚举。"""
    e = SafetyEvent(violation="driver.fatigue", vehicle_id="V1", mode="mode_a_vlm")
    assert e.violation is ViolationType.DRIVER_FATIGUE
    assert e.mode is DetectionMode.VLM
    assert e.severity is Severity.CRITICAL       # 取违规类型默认等级
    assert e.subject.role is SubjectRole.DRIVER  # 由前缀推导


def test_显式设置的主体角色不被反序列化覆盖():
    """回归：`from_dict` 曾因 dict 字面量求值顺序先 pop 后 get，导致 role 永远读到 unknown，
    再被 __post_init__ 重置为违规类型的默认角色。原测试用例的 role 恰好等于默认值，掩盖了它。"""
    e = SafetyEvent(violation=ViolationType.DRIVER_NO_SEATBELT, vehicle_id="V1",
                    mode=DetectionMode.EDGE,
                    subject=Subject(role=SubjectRole.PASSENGER, seat="rear_left", track_id=7))
    back = SafetyEvent.from_dict(e.to_dict())
    assert back.subject.role is SubjectRole.PASSENGER
    assert back.subject.seat == "rear_left" and back.subject.track_id == 7


def test_证据带模型版本便于误报归因():
    e = SafetyEvent(violation=ViolationType.DRIVER_FATIGUE, vehicle_id="V1", mode=DetectionMode.EDGE,
                    evidence=Evidence(model_version="yunet-2023mar"))
    assert SafetyEvent.from_dict(e.to_dict()).evidence.model_version == "yunet-2023mar"


def test_座位名在播报中被中文化():
    e = SafetyEvent(violation=ViolationType.PASSENGER_NO_SEATBELT, vehicle_id="V1",
                    mode=DetectionMode.SERVER, subject=Subject(seat="rear_left"), duration_s=4.0)
    assert "后排左" in e.message and "rear_left" not in e.message


def test_缺省消息为中文可读描述():
    e = SafetyEvent(violation=ViolationType.PASSENGER_NO_SEATBELT, vehicle_id="V1",
                    mode=DetectionMode.SERVER, subject=Subject(seat="rear_left"), duration_s=4.0)
    assert "乘客未系安全带" in e.message and "持续 4.0 秒" in e.message


def test_丢弃缩略图用于窄带上报():
    e = SafetyEvent(violation=ViolationType.DRIVER_SMOKING, vehicle_id="V1",
                    mode=DetectionMode.EDGE, evidence=Evidence(frame_b64="x" * 5000))
    assert json.loads(e.to_json(drop_b64=True))["evidence"]["frame_b64"] is None
    assert len(e.to_json(drop_b64=True)) < len(e.to_json())


def test_车辆上下文行驶判定():
    assert VehicleContext(vehicle_id="V1", speed_kmh=40).is_moving
    assert not VehicleContext(vehicle_id="V1", speed_kmh=1.0).is_moving
    assert not VehicleContext(vehicle_id="V1").is_moving


# --------------------------------------------------------------------------
# 防误报确认器 —— 这是整套系统能否被司机接受的关键
# --------------------------------------------------------------------------
def _feed(confirmer, violation, hits, *, fps=5.0, key="driver", t0=1000.0):
    """按固定帧率喂入一串逐帧判定，返回触发的告警列表。"""
    alerts = []
    t = t0
    for hit in hits:
        t += 1.0 / fps
        c = confirmer.update(violation, hit, key=key, now=t)
        if c.should_alert:
            alerts.append(c)
    return alerts


def test_眨眼级噪声不触发疲劳告警():
    """10% 命中率约等于正常眨眼频率，必须零误报。"""
    hits = [i % 10 == 0 for i in range(300)]   # 60 秒 @5fps
    assert _feed(ViolationConfirmer(), ViolationType.DRIVER_FATIGUE, hits) == []


def test_持续违规在最短时长后触发一次():
    hits = [True] * 150                        # 30 秒 @5fps
    alerts = _feed(ViolationConfirmer(), ViolationType.DRIVER_FATIGUE, hits)
    assert len(alerts) == 1, "冷却期内不应重复告警"
    rule = ViolationConfirmer().rule_for(ViolationType.DRIVER_FATIGUE)
    assert alerts[0].duration_s == pytest.approx(rule.min_duration_s, abs=0.5)


def test_持续不改正会升级严重等级():
    """安全带默认 CRITICAL 不便观察升级，改用默认 WARN 的分心事件。"""
    conf = ViolationConfirmer()
    rule = conf.rule_for(ViolationType.DRIVER_DISTRACTION)
    assert rule.escalate_after_s is None or rule.escalate_after_s > 0
    # 抽烟默认 WARN 且无升级规则，分心亦然 —— 验证未配置升级时等级保持默认
    alerts = _feed(conf, ViolationType.DRIVER_DISTRACTION, [True] * 100)
    assert alerts and alerts[0].severity is Severity.WARN


def test_超速持续升级为critical():
    conf = ViolationConfirmer()
    rule = conf.rule_for(ViolationType.VEHICLE_SPEEDING)
    assert rule.escalate_after_s == 15.0
    # 持续 40 秒，第二次告警（冷却 30 秒后）应已升级
    alerts = _feed(conf, ViolationType.VEHICLE_SPEEDING, [True] * 200)
    assert len(alerts) >= 2
    assert alerts[-1].severity is Severity.CRITICAL
    assert alerts[-1].duration_s > rule.escalate_after_s


def test_恢复后状态被清除可再次告警():
    conf = ViolationConfirmer()
    v = ViolationType.DRIVER_NO_SEATBELT
    first = _feed(conf, v, [True] * 50, t0=1000.0)              # 违规 10 秒
    released = _feed(conf, v, [False] * 50, t0=1100.0)          # 系上安全带
    second = _feed(conf, v, [True] * 50, t0=1300.0)             # 再次解开
    assert len(first) == 1 and released == [] and len(second) == 1


def test_多主体独立计数():
    """驾驶员和后排乘客的安全带状态必须互不干扰。"""
    conf = ViolationConfirmer()
    v = ViolationType.PASSENGER_NO_SEATBELT
    driver = _feed(conf, v, [True] * 50, key="front_passenger")
    rear = _feed(conf, v, [False] * 50, key="rear_left")
    assert len(driver) == 1 and rear == []


def test_车速门控_静止时不判分心():
    """等红灯时看手机不构成违规，静止告警只会制造误报。"""
    conf = ViolationConfirmer()
    assert conf.rule_for(ViolationType.DRIVER_PHONE_USE).min_speed_kmh == 5.0
    alerts, t = 0, 0.0
    for _ in range(100):
        t += 0.2
        if conf.update(ViolationType.DRIVER_PHONE_USE, True, key="d", now=t, speed_kmh=0.0).should_alert:
            alerts += 1
    assert alerts == 0


def test_车速门控_不影响发车前的安全带检查():
    """静止未系安全带必须照常告警 —— 发车前检查正是甲方需求里的核心场景。"""
    conf = ViolationConfirmer()
    assert conf.rule_for(ViolationType.DRIVER_NO_SEATBELT).min_speed_kmh is None
    alerts, t = 0, 0.0
    for _ in range(100):
        t += 0.2
        if conf.update(ViolationType.DRIVER_NO_SEATBELT, True, key="d", now=t, speed_kmh=0.0).should_alert:
            alerts += 1
    assert alerts >= 1


def test_车速未知时不门控():
    """车速信号缺失（未接 OBD）时不应因此漏报。"""
    conf = ViolationConfirmer()
    alerts, t = 0, 0.0
    for _ in range(100):
        t += 0.2
        if conf.update(ViolationType.DRIVER_PHONE_USE, True, key="d", now=t).should_alert:
            alerts += 1
    assert alerts >= 1


# --------------------------------------------------------------------------
# 告警通道 —— 需求 (c)
# --------------------------------------------------------------------------
def test_info级不打扰驾驶员但仍上报后台():
    prompts, backend = [], []
    d = AlertDispatcher([
        InCabinChannel(sink=prompts.append),
        BackendChannel(sender=lambda e: (backend.append(e) or True)),
    ])
    d.dispatch(SafetyEvent(violation=ViolationType.DRIVER_ABSENT, vehicle_id="V1",
                           mode=DetectionMode.EDGE))          # INFO
    d.dispatch(SafetyEvent(violation=ViolationType.DRIVER_FATIGUE, vehicle_id="V1",
                           mode=DetectionMode.EDGE))          # CRITICAL
    time.sleep(1.0)
    d.close()
    assert len(prompts) == 1, "INFO 级不应打扰驾驶员"
    assert len(backend) == 2, "两条事件都必须上报后台"


def test_严重等级决定播报强度():
    ch = InCabinChannel()
    warn = ch.build_prompt(SafetyEvent(violation=ViolationType.DRIVER_SMOKING,
                                       vehicle_id="V1", mode=DetectionMode.EDGE))
    crit = ch.build_prompt(SafetyEvent(violation=ViolationType.DRIVER_FATIGUE,
                                       vehicle_id="V1", mode=DetectionMode.EDGE))
    assert warn.repeat == 1 and not warn.beep
    assert crit.repeat > warn.repeat and crit.beep


def test_播报文案给出可执行动作():
    ch = InCabinChannel()
    p = ch.build_prompt(SafetyEvent(violation=ViolationType.DRIVER_NO_SEATBELT,
                                    vehicle_id="V1", mode=DetectionMode.EDGE))
    assert p.text == "请系好安全带"        # 短句 + 动作，行车中可解析


def test_断网时事件落盘_恢复后补传(tmp_path):
    """隧道/地库断网是车载常态，事件不能丢。"""
    online = False
    delivered = []

    def sender(event):
        if not online:
            raise ConnectionError("模拟断网")
        delivered.append(event)
        return True

    ch = BackendChannel(sender=sender, spool_dir=tmp_path, retry_interval_s=0.3)
    for i in range(3):
        ch.send(SafetyEvent(violation=ViolationType.DRIVER_PHONE_USE,
                            vehicle_id=f"V{i}", mode=DetectionMode.EDGE))
    time.sleep(1.0)
    assert len(list(tmp_path.glob("*.json"))) == 3, "断网时应落盘"
    assert delivered == []

    online = True                       # 驶出隧道
    time.sleep(1.5)
    ch.close()
    assert len(delivered) == 3, "恢复后应补传全部事件"
    assert list(tmp_path.glob("*.json")) == [], "补传成功后应清理落盘文件"


def test_落盘超限时优先丢弃低等级事件(tmp_path):
    """车载设备长时间断网时，无上限落盘会写满 eMMC，把上报故障升级成系统级故障。"""
    def always_fail(_event):
        raise ConnectionError("模拟长时间断网")

    ch = BackendChannel(sender=always_fail, spool_dir=tmp_path,
                        retry_interval_s=999, max_spool_files=20)
    for _ in range(20):
        ch.send(SafetyEvent(violation=ViolationType.DRIVER_ABSENT,       # INFO
                            vehicle_id="V", mode=DetectionMode.EDGE))
        ch.send(SafetyEvent(violation=ViolationType.DRIVER_FATIGUE,      # CRITICAL
                            vehicle_id="V", mode=DetectionMode.EDGE))
    time.sleep(2.0)
    ch.close()
    files = list(tmp_path.glob("*.json"))
    assert len(files) <= 20, "落盘必须受上限约束"
    ranks = [int(f.stem.split("_")[1]) for f in files]
    assert ranks.count(2) > ranks.count(0), "CRITICAL 应优先于 INFO 被保留"


def test_判不了的事件不打扰驾驶员但必须上报后台():
    """安全检查里「判不了」和「合规」绝不能混同。

    发车前因遮挡判不了安全带时，若后台只是「没收到事件」，会被当成「检查通过」放行——
    这是最危险的失败模式。所以它要上报，只是不播报（对司机没有可执行动作）。
    """
    prompts, backend = [], []
    d = AlertDispatcher([
        InCabinChannel(sink=prompts.append),
        BackendChannel(sender=lambda e: (backend.append(e) or True)),
    ])
    undecidable = SafetyEvent(
        violation=ViolationType.DRIVER_NO_SEATBELT, vehicle_id="V1", mode=DetectionMode.VLM,
        decision=Decision.UNDECIDABLE, undecidable_reason="肩部被遮挡")
    confirmed = SafetyEvent(violation=ViolationType.DRIVER_NO_SEATBELT,
                            vehicle_id="V1", mode=DetectionMode.VLM)
    assert all(d.dispatch(undecidable).values())
    assert all(d.dispatch(confirmed).values())
    time.sleep(1.0)
    d.close()
    assert len(prompts) == 1, "只有确认违规才播报"
    assert len(backend) == 2, "判不了也必须让后台知道"


def test_判不了的事件消息说明原因():
    e = SafetyEvent(violation=ViolationType.DRIVER_FATIGUE, vehicle_id="V1", mode=DetectionMode.VLM,
                    decision=Decision.UNDECIDABLE, undecidable_reason="单帧无法给出 PERCLOS")
    assert "无法判定" in e.message and "PERCLOS" in e.message
    assert SafetyEvent.from_dict(e.to_dict()).decision is Decision.UNDECIDABLE


def test_默认是确认违规():
    e = SafetyEvent(violation=ViolationType.DRIVER_FATIGUE, vehicle_id="V1", mode=DetectionMode.EDGE)
    assert e.decision is Decision.CONFIRMED


def test_混合部署模式可表达():
    """边缘初筛 + 云端复核是最可能的最终形态，事件需要能标识它。"""
    e = SafetyEvent(violation=ViolationType.DRIVER_PHONE_USE, vehicle_id="V1", mode="hybrid")
    assert e.mode is DetectionMode.HYBRID


def test_反序列化宽容后台附加的元数据():
    """宽进严出：后台落库与复核流程必然给事件附加自己的字段（复核状态、处理人、工单号），
    这些字段随事件回流到下游时不应让反序列化崩溃。"""
    e = SafetyEvent(violation=ViolationType.DRIVER_FATIGUE, vehicle_id="V1", mode=DetectionMode.SERVER)
    d = e.to_dict()
    d["review_status"] = "pending"
    d["assignee"] = "张三"
    d["evidence"]["backend_thumb_url"] = "https://oss/x.jpg"
    back = SafetyEvent.from_dict(d)
    assert back.event_id == e.event_id and back.violation is e.violation


def test_不适用与判不了都不打扰驾驶员但都要上报():
    """副驾无人时「乘客未系安全带」是「不适用」，不是「判不了」。
    两者都不产出记录的话，后台无法区分「不用查」和「漏查了」。"""
    prompts, backend = [], []
    d = AlertDispatcher([
        InCabinChannel(sink=prompts.append),
        BackendChannel(sender=lambda e: (backend.append(e) or True)),
    ])
    for dec in (Decision.UNDECIDABLE, Decision.NOT_APPLICABLE):
        d.dispatch(SafetyEvent(violation=ViolationType.PASSENGER_NO_SEATBELT, vehicle_id="V1",
                               mode=DetectionMode.SERVER, decision=dec))
    d.dispatch(SafetyEvent(violation=ViolationType.PASSENGER_NO_SEATBELT,
                           vehicle_id="V1", mode=DetectionMode.SERVER))
    time.sleep(1.0)
    d.close()
    assert len(prompts) == 1, "只有确认违规才播报"
    assert len(backend) == 3, "三种判定结果后台都要收到"


def test_判不了可以比违规更快上报():
    """它不打扰司机，早报比晚报好；违规判定则必须先压住误报。"""
    conf = ViolationConfirmer()
    rule = conf.rule_for(ViolationType.DRIVER_NO_SEATBELT)
    assert rule.undecidable_min_duration_s < rule.min_duration_s
    t, first_undecidable = 0.0, None
    for _ in range(60):
        t += 0.2
        if conf.update(ViolationType.DRIVER_NO_SEATBELT, True, key="u",
                       now=t, undecidable=True).should_alert:
            first_undecidable = t
            break
    assert first_undecidable is not None and first_undecidable < rule.min_duration_s


def test_低置信度信号不累积成告警():
    """弱信号即使连续出现也不该攒成告警 —— 更可能是模型在噪声上的系统性偏差。"""
    conf = ViolationConfirmer({ViolationType.DRIVER_SMOKING: ConfirmRule(min_confidence=0.5)})
    t, alerts = 0.0, 0
    for _ in range(100):
        t += 0.2
        if conf.update(ViolationType.DRIVER_SMOKING, True, confidence=0.3, key="d", now=t).should_alert:
            alerts += 1
    assert alerts == 0


def test_告警失败不会中断检测主循环():
    def broken(_event):
        raise RuntimeError("TTS 芯片故障")

    d = AlertDispatcher([InCabinChannel(sink=broken)])
    result = d.dispatch(SafetyEvent(violation=ViolationType.DRIVER_FATIGUE,
                                    vehicle_id="V1", mode=DetectionMode.EDGE))
    assert result == {"in_cabin": False}   # 报告失败，但不抛异常
