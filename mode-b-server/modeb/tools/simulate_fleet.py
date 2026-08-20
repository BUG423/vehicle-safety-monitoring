"""模拟车队 —— 让看板有数据可看，也用来压测。

三种模式，用途不同，**汇报时不要混淆**：

  --mode sources   在后台注册 N 路视频源（合成画面或指定视频文件），
                   后台真的跑推理、真的走确认与告警链路。
                   这是**真链路压测**，测出来的吞吐、延迟是真的。
  --mode events    直接构造 SafetyEvent POST 给后台。
                   不经过任何推理，只为快速把看板灌满历史数据做演示。
                   这**不是**性能测试，产生的数字不能当作检测能力。
  --mode agents    起 N 个车端 agent 子进程真实推流（最接近生产形态，最吃 CPU 编码）。

用法::

    python3 -m modeb.tools.simulate_fleet --mode sources --n 8 \\
        --video ../bench/clips/scenario_a.mp4
    python3 -m modeb.tools.simulate_fleet --mode events --n 12 --history-hours 24
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import DetectionMode, Evidence, SafetyEvent, Subject, SubjectRole, ViolationType  # noqa: E402

PLATES = ["京A", "沪B", "粤B", "川A", "浙A", "苏E", "鲁B", "冀A", "湘A", "闽D"]
NAMES = ["张建国", "李伟", "王强", "刘洋", "陈磊", "杨帆", "赵鹏", "孙涛", "周敏", "吴迪",
         "郑凯", "冯超", "何军", "许静", "邓刚", "曹阳"]
FLEETS = ["华北一队", "华东二队", "华南三队", "西南四队"]

# 各违规在真实车队里的相对频次（经验分布：安全带类最常见，身份不符极少）
WEIGHTS = {
    ViolationType.PASSENGER_NO_SEATBELT: 22,
    ViolationType.DRIVER_NO_SEATBELT: 18,
    ViolationType.DRIVER_DISTRACTION: 16,
    ViolationType.DRIVER_PHONE_USE: 12,
    ViolationType.DRIVER_FATIGUE: 9,
    ViolationType.DRIVER_SMOKING: 7,
    ViolationType.VEHICLE_SPEEDING: 6,
    ViolationType.SYSTEM_CAMERA_BLOCKED: 4,
    ViolationType.DRIVER_HANDS_OFF_WHEEL: 3,
    ViolationType.PASSENGER_OVERLOAD: 2,
    ViolationType.DRIVER_IDENTITY_MISMATCH: 1,
}


def _post(url: str, payload: dict, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def make_fleet(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    fleet = []
    for i in range(n):
        plate = f"{PLATES[i % len(PLATES)]}{rng.randint(10000, 99999)}"
        fleet.append({"vehicle_id": plate, "plate": plate,
                      "driver_name": NAMES[i % len(NAMES)],
                      "driver_id": f"D{1000 + i}",
                      "fleet": FLEETS[i % len(FLEETS)]})
    return fleet


def mode_events(args) -> None:
    """构造历史事件灌库 —— 只为演示看板，不代表任何检测能力。"""
    base = args.server.rstrip("/")
    fleet = make_fleet(args.n, args.seed)
    rng = random.Random(args.seed)
    types = list(WEIGHTS)
    weights = [WEIGHTS[t] for t in types]

    for v in fleet:
        _post(f"{base}/api/v1/vehicles/register", v)
    print(f"[fleet] 已注册 {len(fleet)} 台车")

    now = time.time()
    total = 0
    for _ in range(args.events):
        v = rng.choice(fleet)
        vt = rng.choices(types, weights)[0]
        # 疲劳事件在凌晨聚集 —— 让时段趋势图呈现真实的行业规律
        if vt is ViolationType.DRIVER_FATIGUE:
            hours_ago = rng.choice([1, 2, 3, 4, 22, 23, 24]) * rng.uniform(0.8, 1.2)
        else:
            hours_ago = rng.uniform(0, args.history_hours)
        ts = now - hours_ago * 3600
        ev = SafetyEvent(violation=vt, vehicle_id=v["vehicle_id"], mode=DetectionMode.SERVER,
                         ts=ts, duration_s=round(rng.uniform(2, 25), 1),
                         confidence=round(rng.uniform(0.62, 0.98), 3),
                         subject=Subject(role=vt.default_role,
                                         seat="driver" if vt.default_role is SubjectRole.DRIVER
                                         else "front_passenger"),
                         evidence=Evidence(captured_at=ts),
                         raw_signals={"synthetic": True,
                                      "note": "模拟数据，非真实检测结果"})
        _post(f"{base}/api/v1/events", ev.to_dict())
        total += 1
        if total % 50 == 0:
            print(f"[fleet] 已灌入 {total} 条模拟事件")
    print(f"[fleet] 完成，共 {total} 条模拟事件（全部带 raw_signals.synthetic=true 标记）")


def mode_sources(args) -> None:
    """在后台注册 N 路真实视频源 —— 后台会真的推理，这是真链路。"""
    base = args.server.rstrip("/")
    fleet = make_fleet(args.n, args.seed)
    for i, v in enumerate(fleet):
        body = dict(v)
        if args.video and args.video != "synthetic":
            body.update({"uri": args.video, "realtime": not args.stress,
                         "fps": args.fps, "loop": True})
        else:
            body.update({"kind": "synthetic", "fps": args.fps})
        if args.short_side:
            body["resize_short_side"] = args.short_side
        try:
            r = _post(f"{base}/api/v1/sources", body)
            print(f"[fleet] {i+1}/{args.n} 接入 {v['vehicle_id']} -> {r.get('source',{}).get('kind')}")
        except Exception as exc:  # noqa: BLE001
            print(f"[fleet] 接入 {v['vehicle_id']} 失败: {exc}")
    print(f"[fleet] {args.n} 路视频源已接入，后台正在实时推理。打开 {base}/ 查看看板。")


def mode_agents(args) -> None:
    """起 N 个车端 agent 子进程真实推流。"""
    fleet = make_fleet(args.n, args.seed)
    procs = []
    for v in fleet:
        cmd = [sys.executable, "-m", "modeb.tools.vehicle_agent",
               "--server", args.server, "--vehicle", v["vehicle_id"],
               "--video", args.video or "synthetic", "--fps", str(args.fps),
               "--plate", v["plate"], "--driver", v["driver_name"],
               "--transport", args.transport, "--no-cabin"]
        if args.duration:
            cmd += ["--duration", str(args.duration)]
        procs.append(subprocess.Popen(cmd, cwd=str(_HERE)))
        time.sleep(0.25)
    print(f"[fleet] 已启动 {len(procs)} 个车端 agent。Ctrl-C 结束。")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


def main() -> int:
    ap = argparse.ArgumentParser(description="模拟车队：灌数据 / 压测 / 真实推流")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--mode", default="events", choices=["events", "sources", "agents"])
    ap.add_argument("--n", type=int, default=10, help="车辆数")
    ap.add_argument("--events", type=int, default=240, help="events 模式：灌入多少条事件")
    ap.add_argument("--history-hours", type=float, default=24.0)
    ap.add_argument("--video", default="synthetic")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--short-side", type=int, default=None)
    ap.add_argument("--transport", default="ws", choices=["ws", "http"])
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--stress", action="store_true", help="sources 模式：关掉帧率节流，打满 GPU")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    {"events": mode_events, "sources": mode_sources, "agents": mode_agents}[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
