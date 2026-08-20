"""模式B 回归测试。

覆盖的是**容易悄悄坏掉、坏了又不报错**的地方：
  - 契约层序列化往返（subject.role 曾经在这里丢过，看板统计会算错）
  - 防误报确认（眨眼不该告警、持续闭眼该告警）
  - 安全带判据在合成素材上的判别方向
  - 事件落库 → 查询 → 统计 → 复核
  - 端到端：视频源 → 推理 → 事件 → WebSocket 推到看板与车机

运行：  python3 -m pytest tests/ -q
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parents[1]
_ROOT = _HERE.parent
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (AlertDispatcher, Decision, DetectionMode, Evidence,  # noqa: E402
                    InCabinChannel, SafetyEvent, Subject, SubjectRole,
                    VehicleContext, ViolationType)

from modeb.config import Config  # noqa: E402
from modeb.engine.pipeline import VehiclePipeline  # noqa: E402
from modeb.perception import CartoonCockpitDetector, ScriptedMockDetector, assign_seats  # noqa: E402
from modeb.perception import analyzers as A  # noqa: E402
from modeb.perception.base import BBox, IoUTracker, KP_INDEX, PersonObs  # noqa: E402
from modeb.server.db import EventStore, strip_backend_fields  # noqa: E402
from modeb.sources.capture import PushSource, SyntheticSource, ThreadedSource  # noqa: E402

BENCH_CLIP = _ROOT / "bench" / "clips" / "scenario_a.mp4"


class _RecordingBackend:
    """记录后台通道收到的事件，用来断言 dispatch 真的成功了。"""

    name = "backend"

    def __init__(self, sink):
        self.sink = sink

    def send(self, event):
        self.sink.append(event)
        return True


# ---------------------------------------------------------------- 契约层往返
@pytest.mark.parametrize("violation,role,seat", [
    (ViolationType.PASSENGER_NO_SEATBELT, SubjectRole.PASSENGER, "front_passenger"),
    (ViolationType.DRIVER_FATIGUE, SubjectRole.DRIVER, "driver"),
    # 刻意让 role 与违规类型的默认角色不同：曾经这里会被静默改写成默认值
    (ViolationType.SYSTEM_CAMERA_BLOCKED, SubjectRole.PASSENGER, "rear_left"),
])
def test_事件落库读回后主体角色不丢(tmp_path, violation, role, seat):
    store = EventStore(tmp_path / "t.db")
    ev = SafetyEvent(violation=violation, vehicle_id="V1", mode=DetectionMode.SERVER,
                     subject=Subject(role=role, seat=seat, track_id=7),
                     evidence=Evidence(model_version="yolo11n-pose+face478"))
    eid = store.insert_event(ev.to_dict())
    back = store.get_event(eid)

    assert back["subject"]["role"] == role.value
    assert back["subject"]["seat"] == seat
    ev2 = SafetyEvent.from_dict(strip_backend_fields(back))
    assert ev2.subject.role is role
    assert ev2.subject.track_id == 7
    assert ev2.evidence.model_version == "yolo11n-pose+face478"
    store.close()


def test_后台附加字段必须能被剥掉(tmp_path):
    """/api/v1/events 返回值多带了复核字段，下游 from_dict 会崩 —— 必须有剥离入口。"""
    store = EventStore(tmp_path / "t.db")
    ev = SafetyEvent(violation=ViolationType.DRIVER_PHONE_USE, vehicle_id="V2",
                     mode=DetectionMode.SERVER)
    store.insert_event(ev.to_dict())
    row = store.query_events(limit=1)[0]
    assert "review_status" in row
    with pytest.raises(TypeError):
        SafetyEvent.from_dict(row)          # 不剥就是会崩，这是契约层的已知限制
    assert SafetyEvent.from_dict(strip_backend_fields(row)).vehicle_id == "V2"
    store.close()


# ---------------------------------------------------------------- 汇总统计
def test_违规排行按加权扣分而非事件条数(tmp_path):
    """20 次乘客未系带不该排在 6 次疲劳前面 —— 管理者要看的是风险不是计数。"""
    store = EventStore(tmp_path / "t.db")
    store.register_vehicle("多次轻微", driver_name="甲", driver_id="D1")
    store.register_vehicle("少次严重", driver_name="乙", driver_id="D2")
    now = time.time()
    for i in range(20):
        store.insert_event(SafetyEvent(violation=ViolationType.PASSENGER_NO_SEATBELT,
                                       vehicle_id="多次轻微", mode=DetectionMode.SERVER,
                                       ts=now - i).to_dict())
    for i in range(6):
        store.insert_event(SafetyEvent(violation=ViolationType.DRIVER_FATIGUE,
                                       vehicle_id="少次严重", mode=DetectionMode.SERVER,
                                       ts=now - i).to_dict())
    rank = store.vehicle_ranking()
    assert rank[0]["vehicle_id"] == "少次严重", "排行应按加权扣分，不按事件条数"
    assert rank[0]["penalty"] == 6 * 12 and rank[1]["penalty"] == 20 * 3
    drivers = store.driver_scores()
    assert drivers[0]["driver_name"] == "乙"
    assert drivers[0]["grade"] in ("需关注", "高风险")
    store.close()


def test_复核工作流(tmp_path):
    store = EventStore(tmp_path / "t.db")
    ev = SafetyEvent(violation=ViolationType.DRIVER_SMOKING, vehicle_id="V3",
                     mode=DetectionMode.SERVER)
    eid = store.insert_event(ev.to_dict())
    assert store.get_event(eid)["review_status"] == "pending"
    assert store.review_event(eid, "dismissed", "人工判为误报")
    assert store.get_event(eid)["review_status"] == "dismissed"
    assert store.query_events(review_status="dismissed")
    with pytest.raises(ValueError):
        store.review_event(eid, "不存在的状态")
    store.close()


# ---------------------------------------------------------------- 感知与分析
def test_摄像头遮挡是真实判定():
    assert A.camera_blocked({"mean": 8.0, "std": 3.0, "lap_var": 0.4})[0]
    assert A.camera_blocked({"mean": 120.0, "std": 4.0, "lap_var": 1.0})[0]
    assert not A.camera_blocked({"mean": 96.0, "std": 42.0, "lap_var": 320.0})[0]


def test_头部姿态_正脸接近零度():
    kp = np.zeros((17, 3), dtype=np.float32)
    for name, (x, y) in {"nose": (320, 250), "left_eye": (350, 220), "right_eye": (290, 220),
                         "left_ear": (390, 240), "right_ear": (250, 240)}.items():
        kp[KP_INDEX[name]] = (x, y, 9.0)
    p = PersonObs(box=BBox(230, 150, 410, 460), score=0.9, keypoints=kp)
    yaw, pitch, roll = A.head_pose(p, 640, 480, 1.0)
    assert abs(yaw) < 12, f"正脸的 yaw 应接近 0，实际 {yaw}"


def test_安全带判据方向正确_合成素材():
    """有带的帧分数必须显著高于无带的帧。素材缺失时跳过，不伪造结论。"""
    if not BENCH_CLIP.exists():
        pytest.skip("缺少 bench 素材，先跑 python3 bench/make_clip.py")
    import cv2
    det = CartoonCockpitDetector()
    cap = cv2.VideoCapture(str(BENCH_CLIP))

    def belt_at(t: float, seat: str) -> float:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * 15))
        ok, img = cap.read()
        assert ok
        r = det.infer(img)
        assign_seats(r.persons, r.width, r.height)
        for p in r.persons:
            if p.seat == seat:
                return A.seatbelt_score(img, p, 1.0)[0]
        return -1.0

    with_belt = belt_at(18.0, "driver")   # 司机未系带区间是 2-14s，18s 时已系上
    without = belt_at(8.0, "driver")
    cap.release()
    assert with_belt > 0.15 > without, f"有带={with_belt} 无带={without}"


def test_眼睛开合能区分睁眼与闭眼():
    import cv2

    def make(closed: bool):
        img = np.full((60, 120, 3), 170, np.uint8)
        kp = np.zeros((17, 3), dtype=np.float32)
        for name, x in (("left_eye", 84), ("right_eye", 36)):
            kp[KP_INDEX[name]] = (x, 30, 9.0)
            if closed:
                img[29:32, x - 11:x + 11] = 35
            else:
                cv2.circle(img, (x, 30), 6, (250, 250, 250), -1)
                cv2.circle(img, (x, 30), 4, (30, 26, 22), -1)
        return img, PersonObs(box=BBox(0, 0, 120, 60), score=.9, keypoints=kp)

    open_img, open_p = make(False)
    closed_img, closed_p = make(True)
    o = A.eye_openness(open_img, open_p, 1.0)
    c = A.eye_openness(closed_img, closed_p, 1.0)
    assert o is not None and c is not None
    assert o > c, f"睁眼开合度({o}) 应大于闭眼({c})"


def test_跟踪器保持id稳定():
    t = IoUTracker()
    a = [PersonObs(BBox(10, 10, 100, 300), 0.9)]
    t.update(a)
    tid = a[0].track_id
    b = [PersonObs(BBox(14, 12, 104, 302), 0.9)]
    t.update(b)
    assert b[0].track_id == tid


# ---------------------------------------------------------------- 视频源
def test_推流源满了丢最旧的帧():
    src = PushSource("V", maxlen=2)
    for i in range(5):
        src.offer(np.zeros((8, 8, 3), np.uint8), meta={"i": i})
    assert src.stats["dropped"] == 3
    got = [src.read(), src.read()]
    assert [f.meta["i"] for f in got] == [3, 4], "应保留最新的两帧"
    assert src.read() is None


def test_独立解码线程能取到帧():
    src = ThreadedSource(SyntheticSource("V", fps=30))
    try:
        deadline = time.time() + 4
        got = None
        while time.time() < deadline and got is None:
            got = src.read()
            time.sleep(0.01)
        assert got is not None and got.image.shape[2] == 3
    finally:
        src.close()


# ---------------------------------------------------------------- 防误报
def test_眨眼不告警而持续闭眼告警():
    """这是整套系统能不能活下来的关键性质：误报比漏报更致命。"""
    cfg = Config()
    conf = VehiclePipeline("V", cfg, dispatcher=None).confirmer
    t = 1000.0
    blinks = [(5.0, 5.2), (9.5, 9.7), (12.0, 12.2), (15.5, 15.7), (17.0, 17.2), (19.0, 19.2)]
    alerts = 0
    for i in range(20 * 15):
        rel = i / 15.0
        hit = any(a <= rel < b for a, b in blinks)
        if conf.update(ViolationType.DRIVER_FATIGUE, hit, key="driver", now=t + rel).should_alert:
            alerts += 1
    assert alerts == 0, f"眨眼不应触发疲劳告警，实际触发 {alerts} 次"

    fired_at = None
    for i in range(20 * 15):
        if conf.update(ViolationType.DRIVER_FATIGUE, True, key="driver",
                       now=t + 20 + i / 15.0).should_alert:
            fired_at = i / 15.0
            break
    assert fired_at is not None, "持续闭眼必须告警"
    assert fired_at <= 12, f"持续闭眼 {fired_at}s 才告警，太慢"


def test_车速门控_静止时不判分心():
    """契约层 1.1：等红灯低头看手机不算违规，但安全带类不门控（发车前静止检查）。"""
    cfg = Config()
    conf = VehiclePipeline("V", cfg, dispatcher=None).confirmer
    t = 2000.0
    for i in range(15 * 15):
        conf.update(ViolationType.DRIVER_DISTRACTION, True, key="driver",
                    now=t + i / 15.0, speed_kmh=0.0)
    assert not conf.update(ViolationType.DRIVER_DISTRACTION, True, key="driver",
                           now=t + 15, speed_kmh=0.0).active

    conf2 = VehiclePipeline("V2", cfg, dispatcher=None).confirmer
    fired = any(conf2.update(ViolationType.DRIVER_NO_SEATBELT, True, key="driver",
                             now=t + i / 15.0, speed_kmh=0.0).should_alert
                for i in range(15 * 15))
    assert fired, "安全带类必须在静止时也能判定 —— 发车前检查是核心场景"


# ---------------------------------------------------------------- 双通道告警
def test_双通道分发都要返回成功():
    """不能只看没抛异常，要断言 dispatch 返回 True —— 失败会被静默吞掉。"""
    prompts, sent = [], []
    d = AlertDispatcher(channels=[InCabinChannel(sink=prompts.append), _RecordingBackend(sent)])
    ev = SafetyEvent(violation=ViolationType.DRIVER_NO_SEATBELT, vehicle_id="V",
                     mode=DetectionMode.SERVER)
    res = d.dispatch(ev)
    assert res["in_cabin"] is True and res["backend"] is True
    assert prompts and prompts[0].beep is True and prompts[0].repeat == 3
    assert "安全带" in prompts[0].text
    assert sent and sent[0] is ev


# ---------------------------------------------------------------- 端到端
def test_端到端_合成素材产出事件并带证据():
    if not BENCH_CLIP.exists():
        pytest.skip("缺少 bench 素材")
    import cv2
    from modeb.sources.base import Frame
    cfg = Config()
    det = CartoonCockpitDetector()
    got: list = []
    prompts: list = []
    dispatcher = AlertDispatcher(channels=[InCabinChannel(sink=prompts.append),
                                           _RecordingBackend([])])
    pipe = VehiclePipeline("TEST-001", cfg, dispatcher=dispatcher,
                           on_event=got.append, model_version="cartoon/classical-cv")
    cap = cv2.VideoCapture(str(BENCH_CLIP))
    for i in range(15 * 15):              # 只跑前 15 秒，足够触发安全带类事件
        ok, img = cap.read()
        if not ok:
            break
        clip_t = i / 15.0
        res = det.infer(img)
        res.ts = clip_t
        pipe.process(Frame(vehicle_id="TEST-001", image=img, seq=i,
                           meta={"clip_t": clip_t}), res)
    cap.release()

    assert got, "15 秒素材内应至少产出一条事件"
    assert ViolationType.DRIVER_NO_SEATBELT in {e.violation for e in got}
    e = got[0]
    assert e.mode is DetectionMode.SERVER
    assert e.evidence.frame_b64 and e.evidence.frame_b64.startswith("data:image/jpeg;base64,")
    assert e.evidence.model_version == "cartoon/classical-cv"
    assert "clip_t" in e.raw_signals
    assert e.raw_signals["alert_channels"] == {"in_cabin": True, "backend": True}
    assert prompts, "车内通道必须真的收到播报"


def test_端到端_websocket推到看板与车机(tmp_path):
    """起真实服务，断言看板和车机 WebSocket 都收到消息 —— 不是只看服务起来了。"""
    pytest.importorskip("fastapi")
    import json as _json
    from fastapi.testclient import TestClient

    cfg = Config()
    cfg.perception.backend = "mock"
    cfg.server.db_path = str(tmp_path / "ws.db")
    cfg.server.evidence_dir = str(tmp_path / "ev")
    from modeb.server.app import create_app
    app = create_app(cfg)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/dashboard") as dash, \
             client.websocket_connect("/ws/cabin/WS-001") as cabin:
            assert _json.loads(dash.receive_text())["type"] == "hello"
            assert _json.loads(cabin.receive_text())["type"] == "hello"

            ev = SafetyEvent(violation=ViolationType.DRIVER_FATIGUE, vehicle_id="WS-001",
                             mode=DetectionMode.SERVER, duration_s=5.0)
            r = client.post("/api/v1/events", json=ev.to_dict())
            assert r.status_code == 200 and r.json()["ok"]

            msg = _json.loads(dash.receive_text())
            assert msg["type"] == "event"
            assert msg["event"]["violation"] == "driver.fatigue"
            assert msg["event"]["subject"]["role"] == "driver"

            prompt = _json.loads(cabin.receive_text())
            assert prompt["type"] == "cabin_prompt"
            assert prompt["beep"] is True and "休息" in prompt["text"]

        assert client.get("/api/v1/stats/overview").json()["events"] == 1
        assert client.get("/api/v1/stats/violations").json()["items"][0]["violation"] == "driver.fatigue"
        assert client.get("/healthz").json()["ok"]
        assert client.get("/api/v1/system").json()["detector"]["name"] == "mock"


def test_看板与车机页面自包含():
    """两个页面都不能引外部 CDN —— 部署环境的 CSP 会拦掉。"""
    for name in ("dashboard.html", "cabin.html"):
        html = (_HERE / "modeb" / "server" / "static" / name).read_text(encoding="utf-8")
        assert "https://" not in html
        assert "http://" not in html.replace("http://127.0.0.1", "")
        assert "<script" in html


def test_手机演示页自包含():
    p = _ROOT / "mobile-demo" / "mode-b.html"
    if not p.exists():
        pytest.skip("演示页不在本 worktree")
    html = p.read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html, "手机演示页不得引用任何外部资源"
    assert "backdrop-filter" in html and "navigator.vibrate" in html


def test_脚本mock不读画面也能产出人体():
    det = ScriptedMockDetector()
    r = det.infer(np.zeros((480, 640, 3), np.uint8))
    assert len(r.persons) == 2
    assert r.backend == "mock"


# ---------------------------------------------------------------- 可判定性
def test_遮挡时其余检查是判不了而不是合规():
    """安全检查里最危险的失败模式：把「没看清」当成「检查通过」。"""
    cfg = Config()
    det = CartoonCockpitDetector()
    events: list = []
    prompts: list = []
    from modeb.sources.base import Frame
    pipe = VehiclePipeline("BLK", cfg,
                           dispatcher=AlertDispatcher(channels=[InCabinChannel(sink=prompts.append),
                                                                _RecordingBackend([])]),
                           on_event=events.append, model_version="cartoon")
    black = np.zeros((480, 640, 3), np.uint8)          # 摄像头被贴住
    for i in range(15 * 12):
        res = det.infer(black)
        res.ts = i / 15.0
        pipe.process(Frame(vehicle_id="BLK", image=black, seq=i), res)

    confirmed = [e for e in events if e.decision is Decision.CONFIRMED]
    undecidable = [e for e in events if e.decision is Decision.UNDECIDABLE]

    assert any(e.violation is ViolationType.SYSTEM_CAMERA_BLOCKED for e in confirmed), \
        "遮挡本身是确认违规"
    assert {ViolationType.DRIVER_NO_SEATBELT, ViolationType.DRIVER_FATIGUE,
            ViolationType.PASSENGER_NO_SEATBELT} <= {e.violation for e in undecidable}, \
        "遮挡时安全带/疲劳必须是「判不了」，不能是「没违规」"
    # 判不了的记录不打扰驾驶员：他已经收到遮挡告警，其余的对他没有可执行动作
    assert len(prompts) == len([e for e in confirmed if e.severity.rank >= 1])
    for e in undecidable:
        assert e.severity.value == "info"
        assert e.evidence.evidence_text and "未能完成" in e.evidence.evidence_text
        assert e.raw_signals.get("undecidable_reason")
    # 看板要能拿到「这台车哪几项没检查完」
    assert len(pipe.unchecked) >= 3


def test_判不了不会污染违规判定的滑窗():
    """喂 hit=False 会把「看不见」当成合规证据，甚至让已进入违规态的项目错误恢复。"""
    cfg = Config()
    det = CartoonCockpitDetector()
    from modeb.sources.base import Frame
    events: list = []
    pipe = VehiclePipeline("MIX", cfg, dispatcher=None, on_event=events.append)
    black = np.zeros((480, 640, 3), np.uint8)
    for i in range(15 * 8):
        res = det.infer(black)
        res.ts = i / 15.0
        pipe.process(Frame(vehicle_id="MIX", image=black, seq=i), res)
    # 主确认器里安全带一项应当**完全没有样本**（不是「样本全是 False」）
    state = pipe.confirmer._states.get((ViolationType.DRIVER_NO_SEATBELT, "driver"))  # noqa: SLF001
    assert state is None or len(state.samples) == 0, "判不了的帧不应进入违规判定的滑窗"


def test_统计口径把判不了单独拿出来(tmp_path):
    store = EventStore(tmp_path / "t.db")
    store.register_vehicle("京A9", driver_name="丙", driver_id="D9")
    store.insert_event(SafetyEvent(violation=ViolationType.DRIVER_FATIGUE, vehicle_id="京A9",
                                   mode=DetectionMode.SERVER).to_dict())
    for _ in range(4):
        store.insert_event(SafetyEvent(
            violation=ViolationType.DRIVER_NO_SEATBELT, vehicle_id="京A9",
            mode=DetectionMode.SERVER, decision=Decision.UNDECIDABLE,
            raw_signals={"undecidable_reason": "摄像头被遮挡"}).to_dict())

    ov = store.overview()
    assert ov["events"] == 1, "违规统计不能把「判不了」算进去，否则误报率虚高"
    assert ov["undecidable"] == 4 and ov["vehicles_unchecked"] == 1
    # 评分只看确认违规：4 条未完成检查不该扣分
    assert store.vehicle_ranking()[0]["penalty"] == 12
    assert [v["violation"] for v in store.violation_ranking()] == ["driver.fatigue"]
    q = store.data_quality()
    assert q["by_violation"][0]["n"] == 4
    assert q["by_reason"][0]["reason"] == "摄像头被遮挡"
    assert len(store.query_events(decision="undecidable")) == 4
    assert len(store.query_events(decision="confirmed")) == 1
    store.close()


def test_超员用行驶证核定载客数():
    """核定载客数来自行驶证，不是视觉推断 —— 契约层 1.2 的 seat_capacity。"""
    from modeb.engine.rules import ViolationRuleEngine
    from modeb.perception.base import PerceptionResult
    eng = ViolationRuleEngine(Config().rules)
    res = PerceptionResult(ts=0.0, width=640, height=480, backend="t")
    res.persons = [PersonObs(BBox(i * 60, 100, i * 60 + 50, 400), 0.9) for i in range(4)]
    ctx = VehicleContext(vehicle_id="V", seat_capacity=2)
    hits = {h.violation: h for h in eng._occupancy(res, ctx)}          # noqa: SLF001
    over = hits[ViolationType.PASSENGER_OVERLOAD]
    assert over.hit and over.signals["seat_capacity"] == 2
    assert over.signals["capacity_source"] == "行驶证"

    hits2 = {h.violation: h for h in eng._occupancy(res, None)}        # noqa: SLF001
    assert not hits2[ViolationType.PASSENGER_OVERLOAD].hit             # 默认核定 5 人
    assert hits2[ViolationType.PASSENGER_OVERLOAD].signals["capacity_source"] == "配置默认值"


def test_未接车速信号时超速是判不了():
    from modeb.engine.rules import ViolationRuleEngine
    eng = ViolationRuleEngine(Config().rules)
    h = eng._vehicle_level(VehicleContext(vehicle_id="V"))[0]          # noqa: SLF001
    assert not h.decidable and h.decision is Decision.UNDECIDABLE
    h2 = eng._vehicle_level(VehicleContext(vehicle_id="V", speed_kmh=80,   # noqa: SLF001
                                           speed_limit_kmh=60))[0]
    assert h2.decidable and h2.hit and "80" in h2.evidence_text


def test_事件带文字依据便于隐私友好的复核(tmp_path):
    """管理者读一行字就能复核，不必调阅车内录像 —— 契约层 1.2 的 evidence_text。"""
    if not BENCH_CLIP.exists():
        pytest.skip("缺少 bench 素材")
    import cv2
    from modeb.sources.base import Frame
    cfg = Config()
    det = CartoonCockpitDetector()
    got: list = []
    pipe = VehiclePipeline("TXT", cfg, dispatcher=None, on_event=got.append)
    cap = cv2.VideoCapture(str(BENCH_CLIP))
    for i in range(15 * 12):
        ok, img = cap.read()
        if not ok:
            break
        res = det.infer(img)
        res.ts = i / 15.0
        pipe.process(Frame(vehicle_id="TXT", image=img, seq=i), res)
    cap.release()
    assert got
    assert any(e.evidence.evidence_text for e in got), "确认违规也要给出文字依据"
