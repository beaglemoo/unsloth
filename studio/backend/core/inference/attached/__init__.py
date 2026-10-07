# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Supervisor clients for engines attached to Studio."""

from __future__ import annotations


class AttachedEngineError(RuntimeError):
    """An attached engine refused or failed a management request."""


class AttachedEngineBusy(AttachedEngineError):
    """The engine refused a graceful unload because a client is still being served."""


# Saved provider rows the attached-engines sync owns. Fixed ids so a sync is an upsert and chat requests naming them can be recognised without a database read.
ATTACHED_OMLX_ID = "attachedomlx0001"
ATTACHED_PROVIDER_IDS = frozenset({ATTACHED_OMLX_ID})
