"""M17 - Serving: FastAPI app that LINE calls through ngrok.

Run (from the OrderBuddy folder):
    uvicorn app.main:app --port 8000
    ngrok http 8000
    -> LINE Developers console: Webhook URL = https://<ngrok-domain>/webhook, Use webhook = on

Endpoints
    POST /webhook                 LINE events (signature checked, answered with 200 at once)
    GET  /orders?status=...       staff: list orders            (header X-Staff-Key)
    POST /orders/{id}/approve     staff: approve, cut stock, tell the customer
    POST /orders/{id}/reject      staff: reject, tell the customer
    GET  /health
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request

from . import catalog
from .line_adapter import ConsoleLineClient, Incoming, LineClient, parse_events, verify_signature
from .state_machine import CONFIGS, OrderAgent

try:                                  # .env is optional; real env vars also work
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("orderbuddy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

MSG_UNSUPPORTED = "ตอนนี้รับเฉพาะข้อความและรูปภาพครับ"
MSG_FAILED = "ขออภัยครับ ระบบขัดข้องชั่วคราว รบกวนส่งข้อความอีกครั้งครับ"
MSG_APPROVED = "ร้านยืนยันออร์เดอร์ {order_id} แล้วครับ ยอดรวม {total} บาท ขอบคุณครับ"
MSG_REJECTED = "ขออภัยครับ ร้านไม่สามารถรับออร์เดอร์ {order_id} ได้ รบกวนติดต่อร้านอีกครั้งครับ"


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
    app = FastAPI(title="OrderBuddy")
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
            raise HTTPException(status_code=403, detail="bad signature")
        for ev in parse_events(body):            # LINE "Verify" sends no events: just 200
            if seen.first_time(ev.event_id):
                tasks.add_task(process, ev)
        return {"ok": True}

    # ---------------------------------------------------------------- staff
    def staff(x_staff_key: str = Header(default="")):
        if not s.staff_key or x_staff_key != s.staff_key:
            raise HTTPException(status_code=401, detail="staff key required")

    @app.get("/orders", dependencies=[Depends(staff)])
    def list_orders(status: Optional[str] = "pending_staff"):
        store = agent.store
        if not store.orders_dir.exists():
            return []
        orders = [store.get_order(p.stem) for p in sorted(store.orders_dir.glob("*.json"))]
        return [o.model_dump(mode="json") for o in orders
                if o and (status is None or o.status == status)]

    def review(order_id: str, approve: bool):
        try:
            order = agent.store.review_order(order_id, approve)
        except KeyError:
            raise HTTPException(status_code=404, detail="order not found")
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        if approve:
            qty: dict[str, int] = {}
            for l in order.items:
                qty[l.sku] = qty.get(l.sku, 0) + l.qty
            catalog.deduct_stock(qty)
        msg = MSG_APPROVED if approve else MSG_REJECTED
        try:
            line.push(order.line_user_id, [msg.format(order_id=order.order_id, total=order.total)])
        except Exception:  # noqa: BLE001 - the decision is saved even if LINE is down
            log.exception("could not notify %s", order.line_user_id)
        return order.model_dump(mode="json")

    @app.post("/orders/{order_id}/approve", dependencies=[Depends(staff)])
    def approve(order_id: str):
        return review(order_id, True)

    @app.post("/orders/{order_id}/reject", dependencies=[Depends(staff)])
    def reject(order_id: str):
        return review(order_id, False)

    @app.get("/health")
    def health():
        return {"ok": True, "config": agent.cfg.name, "line": type(line).__name__}

    return app


app = create_app()
