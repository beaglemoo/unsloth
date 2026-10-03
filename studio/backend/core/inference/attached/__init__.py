# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Supervisor clients for engines attached to Studio (oMLX, DwarfStar/ds4). They talk to the engines' own loopback HTTP APIs to read status and load or unload models; chat traffic still goes through the saved-provider path. Nothing here imports ``routes`` or touches Studio state, so the modules stay unit-testable with ``httpx.MockTransport``."""

from __future__ import annotations


class AttachedEngineError(RuntimeError):
    """An attached engine refused or failed a management request."""
