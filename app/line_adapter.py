"""LINE Messaging API: verify, parse, push, download images.

Only what OrderBuddy needs, using `requests` (already a harness dependency).
Docs: https://developers.line.biz/en/reference/messaging-api/
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from typing import Optional

import requests

API = "https://api.line.me/v2/bot"
DATA_API = "https://api-data.line.me/v2/bot"
MAX_TEXT = 5000            # LINE limit per text message
MAX_MESSAGES = 5           # LINE limit per push request
MAX_IMAGE_BYTES = 10 * 1024 * 1024

log = logging.getLogger("orderbuddy.line")


def verify_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    """X-Line-Signature = base64(HMAC-SHA256(channel secret, raw body))."""
    if not signature or not channel_secret:
        return False
    digest = hmac.new(channel_secret.encode(), body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature)


@dataclass
class Incoming:
    event_id: str                  # webhookEventId, used for dedupe and as run label
    user_id: str
    kind: str                      # "text" | "image" | "other"
    text: str = ""
    message_id: str = ""
    redelivery: bool = False


def parse_events(body: bytes) -> list[Incoming]:
    """Keep 1:1 chat message events; ignore follow, unfollow, group chats etc."""
    data = json.loads(body or b"{}")
    out = []
    for ev in data.get("events", []):
        src = ev.get("source", {})
        if ev.get("type") != "message" or src.get("type") != "user":
            continue
        msg = ev.get("message", {})
        kind = msg.get("type") if msg.get("type") in ("text", "image") else "other"
        out.append(Incoming(
            event_id=ev.get("webhookEventId", ""),
            user_id=src.get("userId", ""),
            kind=kind,
            text=msg.get("text", "") if kind == "text" else "",
            message_id=msg.get("id", ""),
            redelivery=ev.get("deliveryContext", {}).get("isRedelivery", False),
        ))
    return out


class LineClient:
    """Push replies (not reply tokens: the 8B model can be slower than a token lives)."""

    def __init__(self, access_token: str, timeout: float = 15):
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {access_token}"}

    def push(self, user_id: str, texts: list[str]) -> None:
        msgs = [{"type": "text", "text": t[:MAX_TEXT]} for t in texts if t][:MAX_MESSAGES]
        if not msgs:
            return
        r = requests.post(f"{API}/message/push", headers=self.headers, timeout=self.timeout,
                          json={"to": user_id, "messages": msgs})
        if r.status_code >= 400:
            raise RuntimeError(f"LINE push failed {r.status_code}: {r.text[:200]}")

    def show_loading(self, user_id: str, seconds: int = 20) -> None:
        """Typing indicator while the agent works. Best effort: errors are ignored."""
        try:
            requests.post(f"{API}/chat/loading/start", headers=self.headers, timeout=5,
                          json={"chatId": user_id, "loadingSeconds": seconds})
        except requests.RequestException:
            pass

    def get_image_b64(self, message_id: str) -> str:
        r = requests.get(f"{DATA_API}/message/{message_id}/content", headers=self.headers,
                         timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"LINE content failed {r.status_code}")
        if len(r.content) > MAX_IMAGE_BYTES:
            raise ValueError("image too large")
        return base64.b64encode(r.content).decode()


class ConsoleLineClient:
    """Dry run without a LINE token: prints replies instead of sending them."""

    def push(self, user_id: str, texts: list[str]) -> None:
        for t in texts:
            print(f"[LINE -> {user_id[:8]}] {t}")

    def show_loading(self, user_id: str, seconds: int = 20) -> None:
        pass

    def get_image_b64(self, message_id: str) -> Optional[str]:
        raise RuntimeError("images need a real LINE token")
