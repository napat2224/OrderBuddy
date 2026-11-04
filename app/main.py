"""M17 - Serving: FastAPI app that LINE calls through ngrok.

Run (from the OrderBuddy folder):
    uvicorn app.main:app --port 8000
    ngrok http 8000
    -> LINE Developers console: Webhook URL = https://<ngrok-domain>/webhook, Use webhook = on

Endpoints
    POST /webhook                 LINE events (signature checked, answered with 200 at once)
    GET  /orders                  the shop: list saved orders   (header X-Staff-Key)
    GET  /health
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request

from .line_adapter import ConsoleLineClient, Incoming, LineClient, parse_events, verify_signature
from .state_machine import CONFIGS, OrderAgent
from .trace import setup_file_log

try:                                  # .env is optional; real env vars also work
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("orderbuddy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

MSG_UNSUPPORTED = "ตอนนี้รับเฉพาะข้อความและรูปภาพครับ"
MSG_FAILED = "ขออภัยครับ ระบบขัดข้องชั่วคราว รบกวนส่งข้อความอีกครั้งครับ"


@dataclass
class Settings:
    channel_secret: str = ""
    access_token: str = ""
    staff_key: str = ""
    agent_config: str = "C3"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(channel_secret=os.getenv("LINE_CHANNEL_SECRET", ""),
                   access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN", ""),
                   staff_key=os.getenv("STAFF_API_KEY", ""),
                   agent_config=os.getenv("AGENT_CONFIG", "C3"))


class SeenEvents:
    """LINE may resend an event; remember the last N webhookEventIds."""

    def __init__(self, size: int = 2000):
        self.size, self._ids, self._lock = size, OrderedDict(), threading.Lock()

    def first_time(self, event_id: str) -> bool:
        if not event_id:
            return True
        with self._lock:
            if event_id in self._ids:
                return False
            self._ids[event_id] = None
            if len(self._ids) > self.size:
                self._ids.popitem(last=False)
            return True


def create_app(settings: Optional[Settings] = None, agent: Optional[OrderAgent] = None,
               line=None) -> FastAPI:
    s = settings or Settings.from_env()
    if agent is None:
        agent = OrderAgent(config=CONFIGS[s.agent_config])
    if line is None:
        line = LineClient(s.access_token) if s.access_token else ConsoleLineClient()
    seen = SeenEvents()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # at server start, not import, so offline tests never write to the real log
        setup_file_log(agent.h.log.path.parent, "uvicorn.access", "uvicorn.error")
        log.info("server started: config=%s line=%s", agent.cfg.name, type(line).__name__)
        yield

    app = FastAPI(title="OrderBuddy", lifespan=lifespan)
    app.state.agent, app.state.line = agent, line

    # ------------------------------------------------------------- customer
    def process(ev: Incoming) -> None:
        """Runs after the 200 response, in FastAPI's thread pool."""
        try:
            if ev.kind == "other":
                line.push(ev.user_id, [MSG_UNSUPPORTED])
                return
            line.show_loading(ev.user_id)
            image = line.get_image_b64(ev.message_id) if ev.kind == "image" else None
            replies = agent.handle(ev.user_id, text=ev.text, image_b64=image)
            line.push(ev.user_id, replies)
        except Exception as e:  # noqa: BLE001 - never crash the worker
            log.exception("event %s failed: %s", ev.event_id, e)
            try:
                line.push(ev.user_id, [MSG_FAILED])
            except Exception:  # noqa: BLE001
                log.exception("could not tell the user")

    @app.post("/webhook")
    async def webhook(request: Request, tasks: BackgroundTasks,
                      x_line_signature: str = Header(default="")):
        body = await request.body()
        if not verify_signature(body, x_line_signature, s.channel_secret):
            log.warning("webhook: bad signature, check LINE_CHANNEL_SECRET in .env")
            raise HTTPException(status_code=403, detail="bad signature")
        for ev in parse_events(body):            # LINE "Verify" sends no events: just 200
            log.info("webhook: %s event from %s", ev.kind, ev.user_id)
            if seen.first_time(ev.event_id):
                tasks.add_task(process, ev)
        return {"ok": True}

    # ----------------------------------------------------------------- shop
    def staff(x_staff_key: str = Header(default="")):
        if not s.staff_key or x_staff_key != s.staff_key:
            raise HTTPException(status_code=401, detail="staff key required")

    @app.get("/orders", dependencies=[Depends(staff)])
    def list_orders():
        return agent.store.list_orders()

    @app.get("/health")
    def health():
        return {"ok": True, "config": agent.cfg.name, "line": type(line).__name__}

    return app


app = create_app()
