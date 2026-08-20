"""OBD-II / CAN 车辆信号源（模拟）。

**超速无法从图像判断** —— 这是三条路线共同的能力边界。车速必须来自
OBD-II（PID 0x0D 车速）、整车 CAN 总线，或 GPS 测速；限速来自电子地图或路牌识别。
安全带扣合同理：车身总线上本来就有 `seatbelt_switch` 开关量，比任何视觉算法都准。

实车接法（DESIGN.md 有接线图说明）：
  * OBD-II 诊断口取电 + 取信号：ELM327 类芯片或直接 CAN 收发器（SN65HVD230）；
  * 商用车走 J1939，乘用车走 ISO 15765-4 (CAN 11bit/500kbps)；
  * 部分车型安全带信号不在诊断口暴露，需要接车身控制器 BCM 的 B-CAN，
    或退回视觉判定 —— 这是实施阶段必须逐车型确认的事项。

本模块只提供**可重放的模拟源**，让 Demo 不依赖真车即可跑通超速与安全带链路。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from common import VehicleContext

from .perception import DRIVER, FRONT_PASSENGER


@dataclass
class TripPoint:
    """行程脚本上的一个关键点，点之间线性插值。"""

    t: float
    speed_kmh: float
    speed_limit_kmh: float


# 演示行程：起步 → 巡航 → 超速 → 急刹 → 恢复
DEMO_TRIP = [
    TripPoint(0.0, 0.0, 60.0),
    TripPoint(4.0, 30.0, 60.0),
    TripPoint(10.0, 55.0, 60.0),
    TripPoint(20.0, 58.0, 60.0),
    TripPoint(26.0, 82.0, 60.0),      # 超速段开始
    TripPoint(38.0, 86.0, 60.0),
    TripPoint(40.0, 40.0, 60.0),      # 急刹：2 秒从 86 降到 40，约 -6.4 m/s^2
    TripPoint(46.0, 50.0, 60.0),
    TripPoint(60.0, 52.0, 60.0),
]

# 巡航行程：全程不超速，用于跑 bench 时不污染打分
CRUISE_TRIP = [
    TripPoint(0.0, 50.0, 60.0),
    TripPoint(60.0, 54.0, 60.0),
]

# 安全带扣合脚本：(座位, 起始秒, 结束秒, 是否扣合)
DEMO_BELTS = [
    (DRIVER, 0.0, 12.0, False),
    (DRIVER, 12.0, 1e9, True),
    (FRONT_PASSENGER, 0.0, 1e9, False),
]


def _interp(trip: list[TripPoint], t: float) -> tuple[float, float]:
    if t <= trip[0].t:
        return trip[0].speed_kmh, trip[0].speed_limit_kmh
    for a, b in zip(trip, trip[1:]):
        if a.t <= t < b.t:
            r = (t - a.t) / (b.t - a.t)
            return a.speed_kmh + r * (b.speed_kmh - a.speed_kmh), a.speed_limit_kmh
    return trip[-1].speed_kmh, trip[-1].speed_limit_kmh


class SimulatedObd:
    """按行程脚本产出 `VehicleContext`。

    `belts=None` 表示这辆车拿不到总线安全带信号 —— 此时判定完全依赖视觉，
    正是 `RuleEngine._seatbelt` 的兜底分支。
    """

    def __init__(self, vehicle_id: str, *, trip: list[TripPoint] | None = None,
                 belts: list[tuple[str, float, float, bool]] | None = None,
                 engine_on: bool = True) -> None:
        self.vehicle_id = vehicle_id
        self.trip = trip if trip is not None else DEMO_TRIP
        self.belts = belts
        self.engine_on = engine_on

    def read(self, t: float, now: float) -> VehicleContext:
        speed, limit = _interp(self.trip, t)
        switches: dict[str, bool] = {}
        if self.belts is not None:
            for seat, a, b, latched in self.belts:
                if a <= t < b:
                    switches[seat] = latched
        return VehicleContext(vehicle_id=self.vehicle_id, speed_kmh=speed,
                              speed_limit_kmh=limit, gear="D" if speed > 1 else "P",
                              engine_on=self.engine_on, gps=None,
                              seatbelt_switch=switches, ts=now)


PROFILES: dict[str, Callable[[str], "SimulatedObd | None"]] = {
    # 无车辆信号：只跑视觉链路
    "off": lambda vid: None,
    # 全程合规巡航，不产生车辆类事件（跑 bench 用）
    "cruise": lambda vid: SimulatedObd(vid, trip=CRUISE_TRIP, belts=None),
    # 完整演示：超速 + 急刹 + 总线安全带信号
    "demo": lambda vid: SimulatedObd(vid, trip=DEMO_TRIP, belts=DEMO_BELTS),
    # 有车速但拿不到安全带总线信号（大量在用车型的真实情况）
    "no_belt_bus": lambda vid: SimulatedObd(vid, trip=DEMO_TRIP, belts=None),
}


def build_obd(profile: str, vehicle_id: str) -> SimulatedObd | None:
    if profile not in PROFILES:
        raise ValueError(f"未知 OBD 档案: {profile}（可选 {'/'.join(PROFILES)}）")
    return PROFILES[profile](vehicle_id)
