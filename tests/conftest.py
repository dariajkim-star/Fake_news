"""pytest가 repo 루트를 import 경로로 인식하게 한다 (`src.*` 절대 import 사용)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
