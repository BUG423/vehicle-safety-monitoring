"""FastAPI 服务：上传图片 -> 检测 -> 返回事件 + 告警，并托管手机可访问的 H5 页面。

接口一览：

  GET  /                    手机演示页（响应式，可调摄像头拍照上传）
  GET  /api/health          当前后端、模型、是否模拟、告警配置
  GET  /api/providers       可用的 VLM 后端列表
  GET  /api/scenarios       mock 后端的预置场景（供演示页切换）
  POST /api/detect          单帧检测（JSON: {image, ...}）
  POST /api/detect_sequence 多帧序列检测（JSON: {images: [...], interval_s}）
  GET  /api/alerts          最近的告警流水（车内 + 后台两条通道）
  POST /api/alerts/clear    清空告警流水（演示用）
  GET  /api/cost            成本测算

图片一律走 JSON + base64（``data:image/jpeg;base64,...`` 或裸 base64），
不用 multipart —— 少一个 python-multipart 依赖，且手机端 canvas.toDataURL 直接就是这个格式。
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from ._common import VehicleContext
from .config import Settings, load_settings
from .cost import fleet_report
from .imaging import decode_data_uri
from .pipeline import SafetyPipeline
from .providers import list_providers
from .providers.mock import SCENARIOS

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class DetectRequest(BaseModel):
    image: str = Field(..., description="data URI 或裸 base64 的 JPEG/PNG")
    vehicle_id: str | None = None
    scenario: str | None = Field(None, description="仅 mock 后端有效：强制指定预置场景")
    speed_kmh: float | None = Field(None, description="OBD 车速；超速判定只能来自它")
    speed_limit_kmh: float | None = None
    engine_on: bool | None = None
    dispatch: bool = True


class DetectSequenceRequest(BaseModel):
    images: list[str]
    interval_s: float = 1.0
    vehicle_id: str | None = None
    scenario: str | None = None
    speed_kmh: float | None = None
    speed_limit_kmh: float | None = None
    engine_on: bool | None = None
    dispatch: bool = True


def _decode(value: str) -> bytes:
    try:
        raw = decode_data_uri(value)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise HTTPException(status_code=400, detail=f"图片解码失败: {exc}") from exc
    if len(raw) < 100:
        raise HTTPException(status_code=400, detail="图片过小或为空")
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图片超过 12MB")
    return raw


def _ctx(req: Any, settings: Settings) -> VehicleContext | None:
    if req.speed_kmh is None and req.engine_on is None:
        return None
    return VehicleContext(vehicle_id=req.vehicle_id or settings.vehicle_id,
                          speed_kmh=req.speed_kmh, speed_limit_kmh=req.speed_limit_kmh,
                          engine_on=req.engine_on)


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="车辆安全监测 · 模式A（VLM 图片理解）", version="1.0")
    pipeline = SafetyPipeline(settings)
    app.state.pipeline = pipeline
    app.state.settings = settings

    def _pipeline_for(scenario: str | None) -> SafetyPipeline:
        """mock 后端支持按请求切换场景；真实后端忽略该参数。"""
        if scenario and pipeline.provider.name == "mock":
            from .providers.mock import MockProvider
            p = SafetyPipeline(settings, provider=MockProvider(settings, scenario=scenario),
                               dispatcher=pipeline.dispatcher, recorder=pipeline.recorder,
                               confirmer=pipeline.confirmer)
            return p
        return pipeline

    # ---------------- 页面 ----------------
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            return "<h1>模式A</h1><p>缺少 static/index.html</p>"
        return page.read_text(encoding="utf-8")

    # ---------------- 元信息 ----------------
    @app.get("/api/health")
    def health() -> dict:
        h = pipeline.health()
        h["simulated"] = pipeline.provider.simulated
        h["note"] = ("当前为 Mock 后端：VLM 推理是**模拟**的，其余环节（解析/规则/确认/告警）均为真实代码"
                     if pipeline.provider.simulated else
                     "当前为真实 VLM 后端，全链路真实调用")
        h["schema_version"] = "1.0"
        return h

    @app.get("/api/providers")
    def providers() -> dict:
        return {"available": list_providers(),
                "current": pipeline.provider.name,
                "model": pipeline.provider.model,
                "resolved_from": settings.provider}

    @app.get("/api/scenarios")
    def scenarios() -> dict:
        return {"scenarios": [{"name": k, "desc": v.get("_desc", "")} for k, v in SCENARIOS.items()],
                "applies_to": "mock",
                "note": "仅 mock 后端生效；真实后端由模型看图决定结果"}

    @app.get("/api/cost")
    def cost(fleet_size: int = 100, model: str = "qwen-vl-max-latest") -> dict:
        return {"fleet_size": fleet_size, "model": model,
                "rows": fleet_report(fleet_size=fleet_size, model=model),
                "免责": "价格为公开标价量级参考，非报价依据"}

    # ---------------- 检测 ----------------
    @app.post("/api/detect")
    def detect(req: DetectRequest) -> JSONResponse:
        raw = _decode(req.image)
        p = _pipeline_for(req.scenario)
        t0 = time.perf_counter()
        res = p.analyze([raw], vehicle_ctx=_ctx(req, settings), policy="instant",
                        dispatch=req.dispatch, vehicle_id=req.vehicle_id)
        body = res.to_dict()
        body["server_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return JSONResponse(body)

    @app.post("/api/detect_sequence")
    def detect_sequence(req: DetectSequenceRequest) -> JSONResponse:
        if not req.images:
            raise HTTPException(status_code=400, detail="images 不能为空")
        if len(req.images) > 8:
            raise HTTPException(status_code=400, detail="一次最多 8 帧（token 成本与延迟考虑）")
        blobs = [_decode(v) for v in req.images]
        p = _pipeline_for(req.scenario)
        t0 = time.perf_counter()
        res = p.analyze(blobs, vehicle_ctx=_ctx(req, settings), policy="temporal",
                        frame_interval_s=req.interval_s, dispatch=req.dispatch,
                        vehicle_id=req.vehicle_id)
        body = res.to_dict()
        body["server_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return JSONResponse(body)

    # ---------------- 告警流水 ----------------
    @app.get("/api/alerts")
    def alerts(limit: int = 50) -> dict:
        return {"records": pipeline.recorder.recent(limit),
                "backend_pending": getattr(
                    next((c for c in pipeline.dispatcher.channels if c.name == "backend"), None),
                    "pending", 0)}

    @app.post("/api/alerts/clear")
    def clear_alerts() -> dict:
        pipeline.recorder.clear()
        return {"ok": True}

    @app.on_event("shutdown")
    def _shutdown() -> None:
        pipeline.close()

    return app


app = build_app()
