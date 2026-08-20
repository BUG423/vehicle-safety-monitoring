"""模式B 后台服务 —— FastAPI + WebSocket。

它同时承担四个角色：

  1. **接入端**：车辆注册/心跳；两种帧上传通道（HTTP POST 与 WebSocket 二进制）；
     也可以由后台主动去拉一路 RTSP / 本地文件（`POST /api/v1/sources`）
  2. **推理端**：所有接入的视频流共享一块 GPU，由 InferenceScheduler 跨车攒批
  3. **告警端**：事件同时推到车队看板（`/ws/dashboard`）和对应车机（`/ws/cabin/{id}`）
  4. **汇总端**：落库 + 查询 + 统计 + 复核工作流

启动::

    python3 -m modeb.server.app --backend yolo --port 8080
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from common import (AlertDispatcher, BackendChannel, CabinPrompt, Decision,  # noqa: E402
                    InCabinChannel, SafetyEvent, Severity)

from modeb.config import Config  # noqa: E402
from modeb.engine.pipeline import VehiclePipeline  # noqa: E402
from modeb.engine.scheduler import InferenceScheduler  # noqa: E402
from modeb.perception import build_detector, build_face_module, fallback_log  # noqa: E402
from modeb.server.db import EventStore  # noqa: E402
from modeb.server.hub import WSHub  # noqa: E402
from modeb.sources.capture import OpenCVSource, PushSource, SyntheticSource  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ModeBServer:
    """把存储、推送、调度、告警粘起来的应用容器。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.started_at = time.time()
        self.store = EventStore(cfg.server.db_path, cfg.server.evidence_dir)
        self.hub = WSHub(cfg.server.max_ws_queue)
        self.loop: asyncio.AbstractEventLoop | None = None

        self.detector = build_detector(cfg.perception)
        self.face_module, self.face_error = build_face_module()
        self.fallbacks = fallback_log()

        self.scheduler = InferenceScheduler(self.detector, cfg,
                                            pipeline_factory=self._make_pipeline)
        self.ingest_stats = {"frames": 0, "bytes": 0, "decode_errors": 0}

    # -- 每辆车一条流水线 ----------------------------------------------------
    def _make_pipeline(self, vehicle_id: str) -> VehiclePipeline:
        dispatcher = AlertDispatcher(channels=[
            InCabinChannel(sink=lambda p, v=vehicle_id: self._push_cabin(v, p)),
            BackendChannel(sender=self._persist_and_broadcast,
                           spool_dir=str(Path(self.cfg.server.db_path).parent / ".spool")),
        ])
        return VehiclePipeline(vehicle_id, self.cfg, dispatcher=dispatcher,
                               face_module=self.face_module,
                               model_version=_model_version(self.detector, self.face_module))

    def _push_cabin(self, vehicle_id: str, prompt: CabinPrompt) -> None:
        """车内提醒 —— 由后台推回车机。模式B 的车内提醒依赖网络，这是它的固有短板。"""
        self.hub.broadcast(f"cabin:{vehicle_id}",
                           {"type": "cabin_prompt", "vehicle_id": vehicle_id,
                            "ts": time.time(), **prompt.to_dict()})

    def _persist_and_broadcast(self, event: SafetyEvent) -> bool:
        """后台通道的实际投递动作：落库 + 推看板。

        返回 False 会让 common 的 BackendChannel 把事件落盘等待补传 ——
        所以这里的异常必须被吞掉并返回 False，不能抛出去。
        """
        try:
            d = event.to_dict()
            self.store.insert_event(d)
            self.store.touch_vehicle(event.vehicle_id)
            payload = self.store.get_event(event.event_id) or d
            self.hub.broadcast("dashboard", {"type": "event", "event": payload})
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[server] 事件落库失败，将落盘补传: {exc}")
            return False

    # -- 接入 ---------------------------------------------------------------
    def ensure_push_source(self, vehicle_id: str) -> PushSource:
        src = self.scheduler.get_source(vehicle_id)
        if isinstance(src, PushSource) and src.is_open:
            return src
        src = PushSource(vehicle_id)
        self.scheduler.add_source(src)
        return src

    def offer_jpeg(self, vehicle_id: str, blob: bytes, meta: dict[str, Any] | None = None) -> bool:
        img = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.ingest_stats["decode_errors"] += 1
            return False
        self.ingest_stats["frames"] += 1
        self.ingest_stats["bytes"] += len(blob)
        src = self.ensure_push_source(vehicle_id)
        ts = None
        if meta and "ts" in meta:
            try:
                ts = float(meta["ts"])
            except (TypeError, ValueError):
                ts = None
        return src.offer(img, ts=ts, meta=meta or {})

    def system_info(self) -> dict[str, Any]:
        pipes = {vid: p.stats.to_dict() for vid, p in self.scheduler._pipelines.items()}  # noqa: SLF001
        face_hits = sum(p.rules.face_hit for p in self.scheduler._pipelines.values())  # noqa: SLF001
        face_miss = sum(p.rules.face_miss for p in self.scheduler._pipelines.values())  # noqa: SLF001
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "detector": self.detector.describe(),
            "face_module": (self.face_module.describe() if self.face_module
                            else {"enabled": False, "reason": self.face_error}),
            "face_frames": {"hit": face_hits, "miss": face_miss},
            "fallbacks": self.fallbacks,
            "scheduler": self.scheduler.stats.to_dict(),
            "sources": [s.describe() for s in self.scheduler._sources.values()],  # noqa: SLF001
            "pipelines": pipes,
            "ws": self.hub.stats(),
            "ingest": self.ingest_stats,
            "config": {"backend": self.cfg.perception.backend,
                       "device": self.cfg.perception.device,
                       "max_batch": self.cfg.perception.max_batch,
                       "infer_short_side": self.cfg.perception.infer_short_side},
        }

    def shutdown(self) -> None:
        self.scheduler.stop()
        self.store.close()


def _model_version(detector, face_module) -> str:
    """模型版本串 —— 写进每条事件的 Evidence，灰度期用于把误报归因到具体版本。"""
    d = detector.describe()
    base = f"{d.get('name')}/{d.get('pose_model') or d.get('kind') or '-'}"
    return base + ("+face478" if face_module else "")


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config()
    srv = ModeBServer(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        srv.loop = asyncio.get_running_loop()
        srv.scheduler.start()
        print(f"[server] 感知后端: {srv.detector.describe()}")
        print(f"[server] 人脸模块: {'已启用' if srv.face_module else srv.face_error}")
        try:
            yield
        finally:
            srv.shutdown()

    app = FastAPI(title="车辆安全监测 · 模式B 后台服务", version="0.1.0",
                  description="后台服务器实时监测与车队汇总", lifespan=lifespan)
    app.state.srv = srv

    # ---------------- 车辆接入 ----------------
    @app.post("/api/v1/vehicles/register")
    async def register(body: dict = Body(...)) -> dict:
        vid = body.get("vehicle_id")
        if not vid:
            raise HTTPException(400, "缺少 vehicle_id")
        srv.store.register_vehicle(
            vid, plate=body.get("plate"), fleet=body.get("fleet"),
            driver_name=body.get("driver_name"), driver_id=body.get("driver_id"),
            source_kind=body.get("source_kind", "push"), meta=body.get("meta"))
        srv.hub.broadcast("dashboard", {"type": "vehicle", "vehicle_id": vid, "action": "register"})
        return {"ok": True, "vehicle_id": vid}

    @app.post("/api/v1/vehicles/{vehicle_id}/heartbeat")
    async def heartbeat(vehicle_id: str, body: dict = Body(default={})) -> dict:
        srv.store.touch_vehicle(vehicle_id)
        pipe = srv.scheduler.pipeline(vehicle_id)
        if pipe is not None and body:
            from common import VehicleContext
            pipe.set_context(VehicleContext(
                vehicle_id=vehicle_id, speed_kmh=body.get("speed_kmh"),
                speed_limit_kmh=body.get("speed_limit_kmh"), gear=body.get("gear"),
                engine_on=body.get("engine_on"),
                seatbelt_switch=body.get("seatbelt_switch") or {}))
        srv.hub.broadcast("dashboard", {"type": "heartbeat", "vehicle_id": vehicle_id,
                                        "ts": time.time(), **body})
        return {"ok": True, "server_ts": time.time()}

    @app.post("/api/v1/vehicles/{vehicle_id}/frame")
    async def upload_frame(vehicle_id: str, request: Request) -> dict:
        """HTTP 帧上传。body 为原始 JPEG 字节，车身信号放 query 或 X-Meta 头。

        这是最容易穿透企业网络与 NAT 的接入方式（就是普通 HTTPS POST），
        代价是每帧一次请求，开销高于 WebSocket / WebRTC。
        """
        blob = await request.body()
        meta: dict[str, Any] = dict(request.query_params)
        raw_meta = request.headers.get("x-meta")
        if raw_meta:
            try:
                meta.update(json.loads(raw_meta))
            except json.JSONDecodeError:
                pass
        for k in ("speed_kmh", "speed_limit_kmh", "clip_t", "ts"):
            if k in meta:
                try:
                    meta[k] = float(meta[k])
                except (TypeError, ValueError):
                    meta.pop(k)
        ok = srv.offer_jpeg(vehicle_id, blob, meta)
        srv.store.touch_vehicle(vehicle_id)
        return {"ok": ok, "bytes": len(blob), "server_ts": time.time()}

    @app.post("/api/v1/sources")
    async def add_source(body: dict = Body(...)) -> dict:
        """让后台主动去拉一路视频（RTSP / 本地文件 / 合成源）。"""
        vid = body.get("vehicle_id")
        uri = body.get("uri")
        if not vid:
            raise HTTPException(400, "缺少 vehicle_id")
        try:
            if body.get("kind") == "synthetic" or uri in (None, "synthetic"):
                src = SyntheticSource(vid, fps=float(body.get("fps", 12)))
            else:
                src = OpenCVSource(vid, uri, realtime=bool(body.get("realtime", True)),
                                   loop=bool(body.get("loop", True)),
                                   target_fps=body.get("fps"),
                                   resize_short_side=body.get("resize_short_side"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"无法打开视频源: {exc}") from exc
        srv.scheduler.add_source(src)
        srv.store.register_vehicle(vid, plate=body.get("plate"), fleet=body.get("fleet"),
                                   driver_name=body.get("driver_name"),
                                   driver_id=body.get("driver_id"),
                                   source_kind=src.kind)
        return {"ok": True, "vehicle_id": vid, "source": src.describe()}

    @app.delete("/api/v1/sources/{vehicle_id}")
    async def del_source(vehicle_id: str) -> dict:
        srv.scheduler.remove_source(vehicle_id)
        return {"ok": True}

    # ---------------- 事件上报与查询 ----------------
    @app.post("/api/v1/events")
    async def post_event(body: dict = Body(...)) -> dict:
        """外部（模式C 车载设备 / 模式A 复核结果）直接上报已确认的事件。

        这是三条路线混合部署时的汇合点：边缘设备初筛出的事件走这个接口进同一个库，
        看板与统计不需要区分它来自哪条路线。
        """
        try:
            ev = SafetyEvent.from_dict(body)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"事件格式不合法: {exc}") from exc
        d = ev.to_dict()
        srv.store.insert_event(d)
        srv.store.touch_vehicle(ev.vehicle_id)
        payload = srv.store.get_event(ev.event_id) or d
        srv.hub.broadcast("dashboard", {"type": "event", "event": payload})
        # 「判不了」照常入库上报后台，但不打扰驾驶员——对他没有可执行动作
        if ev.decision is Decision.CONFIRMED and ev.severity.rank >= Severity.WARN.rank:
            srv._push_cabin(ev.vehicle_id, InCabinChannel().build_prompt(ev))  # noqa: SLF001
        return {"ok": True, "event_id": ev.event_id}

    @app.get("/api/v1/events")
    async def list_events(vehicle_id: str | None = None, violation: str | None = None,
                          severity: str | None = None, review_status: str | None = None,
                          decision: str | None = Query(None, description="confirmed | undecidable"),
                          since_s: float | None = Query(None, description="最近 N 秒内"),
                          limit: int = 100, offset: int = 0) -> dict:
        since = time.time() - since_s if since_s else None
        rows = srv.store.query_events(vehicle_id=vehicle_id, violation=violation,
                                      severity=severity, review_status=review_status,
                                      decision=decision, since=since, limit=limit, offset=offset)
        return {"count": len(rows), "events": rows}

    @app.get("/api/v1/events/{event_id}")
    async def get_event(event_id: str) -> dict:
        ev = srv.store.get_event(event_id)
        if ev is None:
            raise HTTPException(404, "事件不存在")
        return ev

    @app.get("/api/v1/events/{event_id}/evidence")
    async def get_evidence(event_id: str):
        p = srv.store.evidence_path(event_id)
        if p is None:
            raise HTTPException(404, "该事件没有证据帧")
        return FileResponse(str(p), media_type="image/jpeg")

    @app.post("/api/v1/events/{event_id}/review")
    async def review(event_id: str, body: dict = Body(...)) -> dict:
        """事件复核与申诉 —— 没有这条工作流，误报就没有出口，系统最终会被关掉。"""
        status = body.get("status", "confirmed")
        try:
            ok = srv.store.review_event(event_id, status, body.get("note", ""))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not ok:
            raise HTTPException(404, "事件不存在")
        srv.hub.broadcast("dashboard", {"type": "review", "event_id": event_id, "status": status})
        return {"ok": True, "event_id": event_id, "status": status}

    # ---------------- 汇总统计 ----------------
    @app.get("/api/v1/vehicles")
    async def vehicles() -> dict:
        rows = srv.store.list_vehicles(srv.cfg.server.heartbeat_timeout_s)
        live = {vid: p.active for vid, p in srv.scheduler._pipelines.items()}  # noqa: SLF001
        unchk = {vid: p.unchecked for vid, p in srv.scheduler._pipelines.items()}  # noqa: SLF001
        stats = {vid: p.stats.to_dict() for vid, p in srv.scheduler._pipelines.items()}  # noqa: SLF001
        for r in rows:
            r["active_violations"] = list(live.get(r["vehicle_id"], {}).values())
            r["unchecked"] = list(unchk.get(r["vehicle_id"], {}).values())
            r["pipeline"] = stats.get(r["vehicle_id"])
        return {"count": len(rows), "vehicles": rows}

    @app.get("/api/v1/stats/overview")
    async def overview(window_s: float = 86400.0) -> dict:
        return srv.store.overview(window_s)

    @app.get("/api/v1/stats/violations")
    async def violations(window_s: float = 86400.0) -> dict:
        return {"items": srv.store.violation_ranking(window_s)}

    @app.get("/api/v1/stats/vehicles")
    async def vstats(window_s: float = 86400.0, limit: int = 20) -> dict:
        return {"items": srv.store.vehicle_ranking(window_s, limit)}

    @app.get("/api/v1/stats/drivers")
    async def dstats(window_s: float = 604800.0, limit: int = 50) -> dict:
        return {"items": srv.store.driver_scores(window_s, limit)}

    @app.get("/api/v1/stats/data_quality")
    async def data_quality(window_s: float = 86400.0, limit: int = 20) -> dict:
        """检查完成度 —— 「这车没问题」与「这车没看清」必须分开回答。"""
        return srv.store.data_quality(window_s, limit)

    @app.get("/api/v1/stats/timeline")
    async def tl(window_s: float = 86400.0, buckets: int = 24) -> dict:
        return {"items": srv.store.timeline(window_s, buckets)}

    @app.get("/api/v1/system")
    async def system() -> dict:
        return srv.system_info()

    # ---------------- WebSocket ----------------
    @app.websocket("/ws/dashboard")
    async def ws_dashboard(ws: WebSocket) -> None:
        client = await srv.hub.connect(ws, "dashboard")
        try:
            await ws.send_text(json.dumps({"type": "hello", "role": "dashboard",
                                           "system": srv.system_info()}, ensure_ascii=False))
            pump = asyncio.create_task(srv.hub.pump(client))
            try:
                while True:
                    await ws.receive_text()      # 心跳/订阅指令，当前只用来保活
            finally:
                pump.cancel()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            await srv.hub.disconnect(client)

    @app.websocket("/ws/cabin/{vehicle_id}")
    async def ws_cabin(ws: WebSocket, vehicle_id: str) -> None:
        """车机端连这里，接收后台下发的车内提醒（语音文本 / 蜂鸣 / 横幅颜色）。"""
        client = await srv.hub.connect(ws, f"cabin:{vehicle_id}")
        try:
            await ws.send_text(json.dumps({"type": "hello", "role": "cabin",
                                           "vehicle_id": vehicle_id}, ensure_ascii=False))
            pump = asyncio.create_task(srv.hub.pump(client))
            try:
                while True:
                    await ws.receive_text()
            finally:
                pump.cancel()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            await srv.hub.disconnect(client)

    @app.websocket("/ws/ingest/{vehicle_id}")
    async def ws_ingest(ws: WebSocket, vehicle_id: str) -> None:
        """车端推帧通道：二进制消息 = JPEG 帧；文本消息 = JSON 车身信号。

        相比逐帧 HTTP POST，长连接省掉了每帧的 TCP/TLS 与 HTTP 头开销，
        1080p@10fps 下大约能省 8~12% 的上行流量，且抖动更小。
        """
        await ws.accept()
        srv.store.register_vehicle(vehicle_id, source_kind="ws_push")
        src = srv.ensure_push_source(vehicle_id)
        meta: dict[str, Any] = {}
        n = 0
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if (blob := msg.get("bytes")) is not None:
                    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        srv.ingest_stats["decode_errors"] += 1
                        continue
                    srv.ingest_stats["frames"] += 1
                    srv.ingest_stats["bytes"] += len(blob)
                    src.offer(img, ts=meta.get("ts"), meta=dict(meta))
                    n += 1
                    if n % 30 == 0:
                        srv.store.touch_vehicle(vehicle_id)
                elif (text := msg.get("text")) is not None:
                    try:
                        meta.update(json.loads(text))
                    except json.JSONDecodeError:
                        pass
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            srv.store.touch_vehicle(vehicle_id, "offline")

    # ---------------- 静态页面 ----------------
    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        p = STATIC_DIR / "dashboard.html"
        if not p.exists():
            return "<h1>看板文件缺失</h1>"
        return p.read_text(encoding="utf-8")

    @app.get("/cabin", response_class=HTMLResponse)
    async def cabin_page() -> str:
        p = STATIC_DIR / "cabin.html"
        if not p.exists():
            return "<h1>车机页面缺失</h1>"
        return p.read_text(encoding="utf-8")

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"ok": True, "uptime_s": round(time.time() - srv.started_at, 1)})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="模式B 后台服务")
    ap.add_argument("--host", default=os.environ.get("MODEB_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MODEB_PORT", 8080)))
    ap.add_argument("--backend", default=os.environ.get("MODEB_BACKEND", "auto"))
    ap.add_argument("--device", default=os.environ.get("MODEB_DEVICE", "cuda:0"))
    ap.add_argument("--max-batch", type=int, default=None)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    cfg = Config()
    cfg.perception.backend = args.backend
    cfg.perception.device = args.device
    if args.max_batch:
        cfg.perception.max_batch = args.max_batch
    if args.db:
        cfg.server.db_path = args.db
    cfg.server.host, cfg.server.port = args.host, args.port

    import uvicorn
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
