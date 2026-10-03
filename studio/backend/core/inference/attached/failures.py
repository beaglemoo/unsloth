# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read the crash-loop marker the engine launch wrappers leave behind.

The bundled wrappers (``omlx-launch``, ``ds4-ondemand-launch``, ``engines_launch.py``) write
``<engines home>/<name>.fail`` when a precondition stops an engine from starting (missing venv,
config or model, port in use) and remove it right before the engine is exec'd. The file is JSON:
``{"reason": str, "ts": epoch seconds, "count": consecutive failures}``. Stdlib only.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


def engines_home() -> Path:
    override = os.environ.get("UNSLOTH_ENGINES_HOME")
    return Path(override).expanduser() if override else Path.home() / ".unsloth" / "engines"


def read_failure(name: str) -> Optional[dict[str, Any]]:
    """The engine's launch failure as ``{reason, ts, count}``, or None when it is not failing.

    A missing, unreadable or malformed file counts as no failure: this only decorates the
    status, it must never break it.
    """
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        return None
    try:
        raw = json.loads((engines_home() / f"{name}.fail").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    reason, ts, count = raw.get("reason"), raw.get("ts"), raw.get("count")
    if not isinstance(reason, str) or not reason.strip():
        return None
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return None
    return {"reason": reason, "ts": float(ts), "count": count}
