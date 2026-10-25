"""M12 - Session store: one draft + short chat history per LINE user.

Files (plain JSON, no database):
    drafts/<line_user_id>.json        DraftOrder (state lives in draft.state)
    sessions/<line_user_id>.json      last N chat messages for the model
    orders/<order_id>.json            FinalOrder, written once on submit

This module is the storage layer only. The draft_order_manager tool (Part C)
calls these functions, so every write still goes through the tool executor
and its "write" permission (M7 / M19).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schemas import DraftOrder, FinalOrder

ROOT = Path(__file__).resolve().parent.parent
MAX_HISTORY = 20                      # messages kept for the model (10 turns)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _check_id(value: str) -> str:
    """LINE ids are 'U' + 32 hex chars; reject anything unsafe as a file name."""
    if not _SAFE_ID.match(value or ""):
        raise ValueError(f"unsafe id: {value!r}")
    return value


def _write_json(path: Path, data: dict | list) -> None:
    """Atomic write: a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class SessionStore:
    def __init__(self, base_dir: str | Path = ROOT):
        base = Path(base_dir)
        self.drafts_dir = base / "drafts"
        self.sessions_dir = base / "sessions"
        self.orders_dir = base / "orders"
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._id_lock = threading.Lock()

    # ------------------------------------------------------------ locking
    def lock(self, line_user_id: str) -> threading.Lock:
        """One message at a time per user: `with store.lock(uid): ...`"""
        return self._locks[_check_id(line_user_id)]

    # ------------------------------------------------------------- drafts
    def _draft_path(self, uid: str) -> Path:
        return self.drafts_dir / f"{_check_id(uid)}.json"

    def get_draft(self, line_user_id: str) -> Optional[DraftOrder]:
        data = _read_json(self._draft_path(line_user_id))
        return DraftOrder.model_validate(data) if data else None

    def get_or_create_draft(self, line_user_id: str) -> DraftOrder:
        return self.get_draft(line_user_id) or DraftOrder(
            order_id=self.new_order_id(), line_user_id=line_user_id)

    def save_draft(self, draft: DraftOrder) -> DraftOrder:
        draft.updated_at = datetime.now()
        _write_json(self._draft_path(draft.line_user_id), draft.model_dump(mode="json"))
        return draft

    def delete_draft(self, line_user_id: str) -> bool:
        path = self._draft_path(line_user_id)
        if path.exists():
            path.unlink()
            return True
        return False

    # ------------------------------------------------------------- orders
    def _order_path(self, order_id: str) -> Path:
        return self.orders_dir / f"{_check_id(order_id)}.json"

    def submit(self, draft: DraftOrder) -> FinalOrder:
        """Draft -> FinalOrder (pending_staff). Removes the draft.
        The caller (state machine / tool) must already have checked that the
        customer confirmed and that draft.missing is empty."""
        order = FinalOrder.from_draft(draft)
        path = self._order_path(order.order_id)
        if path.exists():
            raise FileExistsError(f"order {order.order_id} already submitted")
        _write_json(path, order.model_dump(mode="json"))
        self.delete_draft(draft.line_user_id)
        return order

    def get_order(self, order_id: str) -> Optional[FinalOrder]:
        data = _read_json(self._order_path(order_id))
        return FinalOrder.model_validate(data) if data else None

    def review_order(self, order_id: str, approve: bool) -> FinalOrder:
        """Staff decision. Only a pending order can be reviewed."""
        order = self.get_order(order_id)
        if order is None:
            raise KeyError(f"no order {order_id}")
        if order.status != "pending_staff":
            raise ValueError(f"order {order_id} is already {order.status}")
        order.status = "approved" if approve else "rejected"
        order.reviewed_at = datetime.now()
        _write_json(self._order_path(order_id), order.model_dump(mode="json"))
        return order

    def new_order_id(self) -> str:
        """ORD-YYYYMMDD-NNNN, counting today's drafts and orders."""
        with self._id_lock:
            day = datetime.now().strftime("%Y%m%d")
            prefix = f"ORD-{day}-"
            used = set()
            for d in (self.orders_dir,):
                if d.exists():
                    used |= {p.stem for p in d.glob(f"{prefix}*.json")}
            if self.drafts_dir.exists():
                for p in self.drafts_dir.glob("*.json"):
                    oid = (_read_json(p) or {}).get("order_id", "")
                    if oid.startswith(prefix):
                        used.add(oid)
            n = max((int(x.rsplit("-", 1)[1]) for x in used), default=0) + 1
            return f"{prefix}{n:04d}"

    # ------------------------------------------------------------ history
    def _history_path(self, uid: str) -> Path:
        return self.sessions_dir / f"{_check_id(uid)}.json"

    def get_history(self, line_user_id: str) -> list[dict]:
        """Messages in Ollama format: [{"role": "user"|"assistant", "content": ...}]"""
        return _read_json(self._history_path(line_user_id)) or []

    def append_history(self, line_user_id: str, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError("role must be user or assistant")
        msgs = self.get_history(line_user_id) + [{"role": role, "content": content}]
        _write_json(self._history_path(line_user_id), msgs[-MAX_HISTORY:])

    def clear_history(self, line_user_id: str) -> None:
        path = self._history_path(line_user_id)
        if path.exists():
            path.unlink()
