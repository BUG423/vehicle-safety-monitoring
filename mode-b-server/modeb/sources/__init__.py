"""视频源：拉流（文件/摄像头/RTSP）、合成源、推流（HTTP/WebSocket 帧上传）。"""
from .base import Frame, FrameSource
from .capture import OpenCVSource, PushSource, SyntheticSource

__all__ = ["Frame", "FrameSource", "OpenCVSource", "PushSource", "SyntheticSource"]
