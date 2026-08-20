"""模式B：后台服务器实时监测与汇总。

整条链路：

    视频源(sources) → 感知(perception) → 逐帧规则(engine.rules)
        → 防误报确认(common.ViolationConfirmer) → SafetyEvent
        → 双通道告警(common.AlertDispatcher) → 后台(server) → 车队看板

与另外两条路线的唯一差异在 `perception/`：模式B 把推理放在后台 GPU 上，
因此可以用大模型、可以多路并发、可以做车队级汇总。
"""

__version__ = "0.1.0"
