"""Offline tests for the LINE webhook and staff endpoints: no Ollama, no LINE.

The real OrderAgent runs with the FakeClient models from test_agent_offline,
and a FakeLine records what would be pushed to LINE.

    python -m unittest discover -s tests -v
"""
import base64
import hashlib
import hmac
import json
import shutil
import sys
import tempfile
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
        # stock.json is written on approve: work on a copy of data/
        self.data = Path(tempfile.mkdtemp())
        shutil.copytree(catalog.DATA_DIR, self.data, dirs_exist_ok=True)
        self._old_data, catalog.DATA_DIR = catalog.DATA_DIR, self.data

        self.agent = self.make({MSG_ORDER: GOOD_ORDER})
        self.line = FakeLine()
        app = create_app(Settings(channel_secret=SECRET, staff_key=STAFF), self.agent, self.line)
        self.http = TestClient(app)

    def tearDown(self):
        catalog.DATA_DIR = self._old_data

    def post(self, body, signature=None):
        return self.http.post("/webhook", content=body,
                              headers={"X-Line-Signature": signature or sign(body)})

    def test_bad_signature_is_rejected(self):
        r = self.post(line_body(text_event(MSG_ORDER, "e1")), signature="wrong")
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.line.pushed, [])

    def test_line_verify_button(self):
        self.assertEqual(self.post(line_body()).status_code, 200)

    def test_full_order_through_line_and_staff(self):
        self.assertEqual(self.post(line_body(text_event(MSG_ORDER, "e1"))).status_code, 200)
        self.assertIn("รวม 170 บาท", self.line.pushed[-1][1][0])
        self.post(line_body(text_event("ยืนยัน", "e2")))
        self.assertIn("รับออร์เดอร์", self.line.pushed[-1][1][0])

        self.assertEqual(self.http.get("/orders").status_code, 401)
        pending = self.http.get("/orders", headers={"X-Staff-Key": STAFF}).json()
        self.assertEqual(len(pending), 1)
        oid = pending[0]["order_id"]

        r = self.http.post(f"/orders/{oid}/approve", headers={"X-Staff-Key": STAFF})
        self.assertEqual(r.json()["status"], "approved")
        self.assertIn("ร้านยืนยันออร์เดอร์", self.line.pushed[-1][1][0])
        self.assertEqual(catalog.stock()["F001"], 18)          # 20 - 2
        again = self.http.post(f"/orders/{oid}/approve", headers={"X-Staff-Key": STAFF})
        self.assertEqual(again.status_code, 409)               # no double approve

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
        self.assertEqual(models, ["gemma3:4b-it-q4_K_M"])
        self.assertIn("รวม 170 บาท", self.line.pushed[-1][1][0])

    def test_sticker_gets_polite_answer(self):
        ev = {"type": "message", "webhookEventId": "e5", "source": {"type": "user", "userId": UID},
              "message": {"type": "sticker", "id": "s1"}}
        self.post(line_body(ev))
        self.assertIn("ข้อความและรูปภาพ", self.line.pushed[-1][1][0])


if __name__ == "__main__":
    unittest.main()
