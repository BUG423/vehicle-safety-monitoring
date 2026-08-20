"""车辆安全监测 —— 统一违规类型定义。

三种技术模式（VLM / 后台服务器 / 车载嵌入式）共用本枚举，
保证无论最终选择哪条技术路线，事件语义、告警策略、后台存储结构完全一致。
"""
from __future__ import annotations

from enum import Enum


class SubjectRole(str, Enum):
    """事件主体角色。"""

    DRIVER = "driver"          # 驾驶员
    PASSENGER = "passenger"    # 乘坐人员
    VEHICLE = "vehicle"        # 车辆本身（超速等非人体事件）
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """告警严重等级 —— 决定提醒方式与后台处置流程。"""

    INFO = "info"          # 仅记录，不打扰驾驶员
    WARN = "warn"          # 车内柔性提醒（语音一次 + 屏幕）
    CRITICAL = "critical"  # 车内强提醒（循环语音 + 蜂鸣）+ 后台实时推送

    @property
    def rank(self) -> int:
        return {"info": 0, "warn": 1, "critical": 2}[self.value]


class Decision(str, Enum):
    """一条记录表达的是「确认违规」还是「判不了」。

    安全检查里这两者绝不能混同：发车前因遮挡判不了安全带时，如果后台只是「没收到事件」，
    会被当成「检查通过」放行——这是最危险的失败模式。
    因此「判不了」也要作为一条记录上报，只是不打扰驾驶员（对司机没有可执行动作）。
    """

    CONFIRMED = "confirmed"        # 确认违规
    UNDECIDABLE = "undecidable"    # 无法判定：遮挡、光线不足、目标未入镜、需时序而样本不足


class ViolationType(str, Enum):
    """违规类型。

    命名规则 `<主体>.<行为>`，便于后台按前缀聚合统计。
    """

    # ---- (a) 驾驶员：危险驾驶行为 ----
    DRIVER_NO_SEATBELT = "driver.no_seatbelt"            # 未系安全带
    DRIVER_FATIGUE = "driver.fatigue"                    # 疲劳驾驶（闭眼/哈欠/点头）
    DRIVER_DISTRACTION = "driver.distraction"            # 分心（长时间视线偏离前方）
    DRIVER_PHONE_USE = "driver.phone_use"                # 使用手机
    DRIVER_SMOKING = "driver.smoking"                    # 抽烟
    DRIVER_HANDS_OFF_WHEEL = "driver.hands_off_wheel"    # 双手脱离方向盘
    DRIVER_ABSENT = "driver.absent"                      # 驾驶位无人（车辆将行驶时）
    DRIVER_IDENTITY_MISMATCH = "driver.identity_mismatch"  # 驾驶员身份与派车单不符

    # ---- (b) 乘坐人员 ----
    PASSENGER_NO_SEATBELT = "passenger.no_seatbelt"      # 乘客未系安全带
    PASSENGER_OVERLOAD = "passenger.overload"            # 超员
    PASSENGER_CHILD_FRONT_SEAT = "passenger.child_front_seat"  # 儿童坐副驾
    PASSENGER_SMOKING = "passenger.smoking"              # 乘客抽烟

    # ---- 车辆状态类（依赖 OBD/GPS 信号，非纯视觉） ----
    VEHICLE_SPEEDING = "vehicle.speeding"                # 超速
    VEHICLE_HARSH_DRIVING = "vehicle.harsh_driving"      # 急加速/急刹/急转弯

    # ---- 系统自身状态 ----
    SYSTEM_CAMERA_BLOCKED = "system.camera_blocked"      # 摄像头被遮挡/失效

    @property
    def label_zh(self) -> str:
        return _LABELS_ZH[self]

    @property
    def default_severity(self) -> Severity:
        return _DEFAULT_SEVERITY[self]

    @property
    def default_role(self) -> SubjectRole:
        prefix = self.value.split(".", 1)[0]
        return {
            "driver": SubjectRole.DRIVER,
            "passenger": SubjectRole.PASSENGER,
            "vehicle": SubjectRole.VEHICLE,
        }.get(prefix, SubjectRole.UNKNOWN)


_LABELS_ZH: dict[ViolationType, str] = {
    ViolationType.DRIVER_NO_SEATBELT: "驾驶员未系安全带",
    ViolationType.DRIVER_FATIGUE: "疲劳驾驶",
    ViolationType.DRIVER_DISTRACTION: "驾驶员分心",
    ViolationType.DRIVER_PHONE_USE: "驾驶中使用手机",
    ViolationType.DRIVER_SMOKING: "驾驶中抽烟",
    ViolationType.DRIVER_HANDS_OFF_WHEEL: "双手脱离方向盘",
    ViolationType.DRIVER_ABSENT: "驾驶位无人",
    ViolationType.DRIVER_IDENTITY_MISMATCH: "驾驶员身份不符",
    ViolationType.PASSENGER_NO_SEATBELT: "乘客未系安全带",
    ViolationType.PASSENGER_OVERLOAD: "车辆超员",
    ViolationType.PASSENGER_CHILD_FRONT_SEAT: "儿童乘坐副驾",
    ViolationType.PASSENGER_SMOKING: "乘客吸烟",
    ViolationType.VEHICLE_SPEEDING: "车辆超速",
    ViolationType.VEHICLE_HARSH_DRIVING: "急加速/急刹车",
    ViolationType.SYSTEM_CAMERA_BLOCKED: "摄像头异常",
}

_DEFAULT_SEVERITY: dict[ViolationType, Severity] = {
    ViolationType.DRIVER_NO_SEATBELT: Severity.CRITICAL,
    ViolationType.DRIVER_FATIGUE: Severity.CRITICAL,
    ViolationType.DRIVER_DISTRACTION: Severity.WARN,
    ViolationType.DRIVER_PHONE_USE: Severity.CRITICAL,
    ViolationType.DRIVER_SMOKING: Severity.WARN,
    ViolationType.DRIVER_HANDS_OFF_WHEEL: Severity.WARN,
    ViolationType.DRIVER_ABSENT: Severity.INFO,
    ViolationType.DRIVER_IDENTITY_MISMATCH: Severity.CRITICAL,
    ViolationType.PASSENGER_NO_SEATBELT: Severity.WARN,
    ViolationType.PASSENGER_OVERLOAD: Severity.WARN,
    ViolationType.PASSENGER_CHILD_FRONT_SEAT: Severity.WARN,
    ViolationType.PASSENGER_SMOKING: Severity.INFO,
    ViolationType.VEHICLE_SPEEDING: Severity.CRITICAL,
    ViolationType.VEHICLE_HARSH_DRIVING: Severity.WARN,
    ViolationType.SYSTEM_CAMERA_BLOCKED: Severity.WARN,
}


class DetectionMode(str, Enum):
    """产生该事件的技术模式 —— 用于三条路线的横向对比评估。"""

    VLM = "mode_a_vlm"          # 模式A：大模型直接理解图片
    SERVER = "mode_b_server"    # 模式B：后台服务器实时监测汇总
    EDGE = "mode_c_edge"        # 模式C：车载嵌入式设备
    HYBRID = "hybrid"           # 混合部署：边缘初筛 + 云端复核（最可能的最终形态）
