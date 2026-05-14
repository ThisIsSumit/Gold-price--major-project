from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gold_app.snapshot_builder import build_snapshot


if __name__ == "__main__":
    snapshot = build_snapshot(force_retrain=False)
    print(snapshot["created_at"])
