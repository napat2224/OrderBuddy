"""OrderBuddy app: LINE chat-to-order agent built on the course Harness.

The course harness lives in ../Harness/harness and is used unchanged.
Importing `app` puts ../Harness on sys.path so `import harness` works.
"""
import sys
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parent.parent / "Harness"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))
