# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read the desktop shell's engine intent from ``<engines home>/desktop.json``. Stdlib only.

The Tauri shell owns the file (``engines_enabled``, ``engine_lifetime`` and the per-engine
``helpers`` map); the backend only reads it. A helper is wanted when the master switch is on and
its own choice is on; a missing choice means oMLX on and ComfyUI off, the same defaults as
``engine_lifetime.rs``. Reading never raises: this only decorates status and relaxes a refusal.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from core.inference.attached.failures import engines_home

_DEFAULT_CHOICE = {"omlx": True, "comfyui": False}


def _read() -> Optional[dict[str, Any]]:
    try:
        body = json.loads((engines_home() / "desktop.json").read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def helper_wanted(name: str) -> Optional[bool]:
    """Whether the shell wants helper ``name`` running, or None when that is unknown.

    Unknown means no readable file, or no ``engines_enabled`` in it (an app from before the
    setting existed: the shell then derives it from the registration, which the backend cannot see).
    """
    if name not in _DEFAULT_CHOICE:
        return None
    body = _read()
    if body is None:
        return None
    master = body.get("engines_enabled")
    if not isinstance(master, bool):
        return None
    if not master:
        return False
    helpers = body.get("helpers")
    choice = helpers.get(name) if isinstance(helpers, dict) else None
    return choice if isinstance(choice, bool) else _DEFAULT_CHOICE[name]
