# common —— 三种技术模式的共享契约层

无论最终选择哪条技术路线，**事件定义、确认逻辑、告警协议都不应该重写**。
本目录固化这三件事，使三条路线的产出可横向对比、可平滑切换、可混合部署。

## 设计约束

- **只依赖 Python 标准库**（dataclasses / enum / json / queue / threading）。
  这是硬性要求：模式 C 的车载嵌入式设备上不适合引入 pydantic 这类重依赖，
  而三条路线必须共用同一份定义，因此就近取最严格的约束。
- **事件结构与来源无关**：后台收到事件时不需要知道它来自 VLM、云端 GPU 还是车载 NPU，
  `mode` 字段仅用于横向评估和灰度切换。

## 模块

| 文件 | 内容 |
|---|---|
| `schema/violation_types.py` | `ViolationType`（14 类违规）、`Severity`（3 级）、`SubjectRole`、`DetectionMode` |
| `schema/safety_event.py` | `SafetyEvent`、`Subject`、`Evidence`、`VehicleContext` |
| `schema/confirmation.py` | `ViolationConfirmer` 防误报确认状态机、`ConfirmRule` 分类型规则 |
| `alerting/channels.py` | `AlertDispatcher`、`InCabinChannel`、`BackendChannel` |

## 用法

```python
from common import (
    ViolationType, DetectionMode, SafetyEvent, Subject,
    ViolationConfirmer, AlertDispatcher,
)

confirmer = ViolationConfirmer()
dispatcher = AlertDispatcher()          # 默认车内 + 后台双通道

for frame in stream:
    raw = detector.predict(frame)       # ← 三条路线唯一的差异点
    c = confirmer.update(ViolationType.DRIVER_FATIGUE, raw.hit,
                         confidence=raw.score, key="driver")
    if c.should_alert:
        dispatcher.dispatch(SafetyEvent(
            violation=c.violation,
            vehicle_id="JING-A12345",
            mode=DetectionMode.EDGE,     # ← 各路线填自己的模式
            severity=c.severity,
            confidence=c.confidence,
            duration_s=c.duration_s,
            subject=Subject(seat="driver"),
        ))
```

## 违规类型速查

| 枚举 | 中文 | 默认等级 | 主要信号源 |
|---|---|---|---|
| `driver.no_seatbelt` | 驾驶员未系安全带 | CRITICAL | 摄像头 / 车身总线 |
| `driver.fatigue` | 疲劳驾驶 | CRITICAL | 摄像头（需时序） |
| `driver.distraction` | 驾驶员分心 | WARN | 摄像头（头部姿态） |
| `driver.phone_use` | 驾驶中使用手机 | CRITICAL | 摄像头 |
| `driver.smoking` | 驾驶中抽烟 | WARN | 摄像头 |
| `driver.hands_off_wheel` | 双手脱离方向盘 | WARN | 摄像头 |
| `driver.absent` | 驾驶位无人 | INFO | 摄像头 |
| `driver.identity_mismatch` | 驾驶员身份不符 | CRITICAL | 人脸比对 |
| `passenger.no_seatbelt` | 乘客未系安全带 | WARN | 摄像头 / 车身总线 |
| `passenger.overload` | 车辆超员 | WARN | 摄像头 |
| `passenger.child_front_seat` | 儿童乘坐副驾 | WARN | 摄像头 |
| `vehicle.speeding` | 车辆超速 | CRITICAL | **OBD/CAN/GPS** |
| `vehicle.harsh_driving` | 急加速/急刹车 | WARN | **IMU/OBD** |
| `system.camera_blocked` | 摄像头异常 | WARN | 自检 |

## 修改约定

契约层是三条路线的公共依赖，**各路线不得自行修改**。
需要新增字段或违规类型时，在各自的 `DESIGN.md` 中提出「对契约层的修改建议」，由汇总方统一合并，
避免三条分支产生互不兼容的事件定义。
