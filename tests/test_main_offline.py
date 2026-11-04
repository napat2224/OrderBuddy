"""Offline tests for the LINE webhook and the orders endpoint: no Ollama, no LINE.

The real OrderAgent runs with the FakeClient models from test_agent_offline,
and a FakeLine records what would be pushed to LINE.

    python -m unittest discover -s tests -v
"""
import base64
import hashlib
import hmac
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

from test_agent_offline import GOOD_ORDER, MSG_ORDER, UID, Base  # noqa: E402

from app import catalog  # noqa: E402
from app.main import Settings, create_app  # noqa: E402

SECRET, STAFF = "test-secret", "staff-key"


class FakeLine:
    def __init__(self):
        self.pushed, self.images = [], {}

    def push(self, user_id, texts):
        self.pushed.append((user_id, list(texts)))

    def show_loading(self, user_id, seconds=20):
        pass

    def get_image_b64(self, message_id):
        return self.images[message_id]


def line_body(*events):
    return json.dumps({"destination": "x", "events": list(events)}, ensure_ascii=False).encode()


def text_event(text, event_id):
    return {"type": "message", "webhookEventId": event_id, "source": {"type": "user", "userId": UID},
            "message": {"type": "text", "id": "m-" + event_id, "text": text},
            "deliveryContext": {"isRedelivery": False}}


def sign(body):
    return base64.b64encode(hmac.new(SECRET.encode(), body, hashlib.sha256).digest()).decode()


class TestWebhook(Base):
    def setUp(self):
        super().setUp()
        self.agent = self.make({MSG_ORDER: GOOD_ORDER})
        self.line = FakeLine()
        app = create_app(Settings(channel_secret=SECRET, staff_key=STAFF), self.agent, self.line)
        self.http = TestClient(app)

    def post(self, body, signature=None):
        return self.http.post("/webhook", content=body,
                              headers={"X-Line-Signature": signature or sign(body)})

    def test_bad_signature_is_rejected(self):
        r = self.post(line_body(text_event(MSG_ORDER, "e1")), signature="wrong")
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.line.pushed, [])

    def test_line_verify_button(self):
        self.assertEqual(self.post(line_body()).status_code, 200)

    def test_full_order_through_line(self):
        self.assertEqual(self.post(line_body(text_event(MSG_ORDER, "e1"))).status_code, 200)
        self.assertIn("รวม 170 บาท", self.line.pushed[-1][1][0])
        self.post(line_body(text_event("ยืนยัน", "e2")))
        self.assertIn("รับออร์เดอร์", self.line.pushed[-1][1][0])
        self.assertIn("ยอดรวม 170 บาท", self.line.pushed[-1][1][0])
        self.assertEqual(catalog.stock()["F001"], 18)          # 20 - 2

        self.assertEqual(self.http.get("/orders").status_code, 401)
        orders = self.http.get("/orders", headers={"X-Staff-Key": STAFF}).json()
        self.assertEqual([o["status"] for o in orders], ["confirmed"])

    def test_review_endpoints_are_gone(self):
        headers = {"X-Staff-Key": STAFF}
        for action in ("approve", "reject"):
            r = self.http.post(f"/orders/ORD-1/{action}", headers=headers)
            self.assertEqual(r.status_code, 404)

    def test_orders_from_an_older_version_still_list(self):
        self.agent.store.orders_dir.mkdir(parents=True)
        old = self.agent.store.orders_dir / "ORD-OLD.json"
        old.write_text(json.dumps({"order_id": "ORD-OLD", "status": "pending_staff"}), encoding="utf-8")
        orders = self.http.get("/orders", headers={"X-Staff-Key": STAFF}).json()
        self.assertEqual([o["order_id"] for o in orders], ["ORD-OLD"])

    def test_duplicate_event_runs_once(self):
        body = line_body(text_event(MSG_ORDER, "same"))
        self.post(body)
        self.post(body)
        self.assertEqual(len(self.line.pushed), 1)

    def test_image_goes_to_vision(self):
        self.line.images["img1"] = base64.b64encode(b"fake-jpeg").decode()
        self.client.extract["Read the order in this image."] = GOOD_ORDER
        ev = {"type": "message", "webhookEventId": "e9", "source": {"type": "user", "userId": UID},
              "message": {"type": "image", "id": "img1"}}
        self.post(line_body(ev))
        models = [m for m, title, _ in self.client.calls if title == "ExtractedOrder"]
        self.assertEqual(models, [self.http.app.state.agent.h.config.profile("vision").model])
        self.assertIn("รวม 170 บาท", self.line.pushed[-1][1][0])

    def test_sticker_gets_polite_answer(self):
        ev = {"type": "message", "webhookEventId": "e5", "source": {"type": "user", "userId": UID},
              "message": {"type": "sticker", "id": "s1"}}
        self.post(line_body(ev))
        self.assertIn("ข้อความและรูปภาพ", self.line.pushed[-1][1][0])


if __name__ == "__main__":
    unittest.main()
