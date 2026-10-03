# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Supervisor clients for engines attached to Studio (oMLX, DwarfStar/ds4). They talk to the engines' own loopback HTTP APIs to read status and load or unload models; chat traffic still goes through the saved-provider path. Nothing here imports ``routes`` or touches Studio state, so the modules stay unit-testable with ``httpx.MockTransport``."""

from __future__ import annotations


class AttachedEngineError(RuntimeError):
    """An attached engine refused or failed a management request."""


# Saved provider rows the attached-engines sync owns. Fixed ids so a sync is an upsert and chat requests naming them can be recognised without a database read.
ATTACHED_OMLX_ID = "attachedomlx0001"
ATTACHED_DS4_ID = "attachedds400001"
ATTACHED_PROVIDER_IDS = frozenset({ATTACHED_OMLX_ID, ATTACHED_DS4_ID})
