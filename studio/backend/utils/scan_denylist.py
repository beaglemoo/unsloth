# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model-scan deny-list: the owner's entries, plus the attached oMLX engine's own model folders while attached engines are enabled."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Iterable, TypeVar, Union

from loggers import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def _comparable(path: str) -> str:
    system = platform.system()
    if system == "Windows":
        return os.path.normcase(path)
    if system == "Darwin":
        from utils.paths.path_utils import macos_volume_ignores_case
        return path.casefold() if macos_volume_ignores_case(path) else path
    return path


def _existing_entries() -> list[str]:
    from utils.attached_engines_settings import get_config

    config = get_config()
    raws = list(config.scan_denylist)
    if config.enabled:
        # The attached engine's own folders are served by the engine; listing them again would offer a second, untuned copy.
        from utils.attached_engine_dirs import auto_denied_dirs

        raws.extend(auto_denied_dirs())
    entries: list[str] = []
    for raw in raws:
        expanded = os.path.expanduser(raw)
        # An entry that does not exist holds nothing, and no symlink can lead into it.
        if expanded and os.path.lexists(expanded):
            real = os.path.realpath(expanded)
            if real not in entries:
                entries.append(real)
    return entries


def _matches(entries: list[str], path: Union[str, "os.PathLike[str]"]) -> bool:
    resolved = os.path.realpath(os.path.expanduser(os.fspath(path)))
    check = _comparable(resolved)
    for entry in entries:
        prefix = _comparable(entry).rstrip(os.sep) or os.sep
        if check == prefix or check.startswith(
            prefix if prefix.endswith(os.sep) else prefix + os.sep
        ):
            return True
    return False


def is_denied(path: Union[str, "os.PathLike[str]"]) -> bool:
    """True when ``path`` is, or lies inside, a deny-listed directory."""
    try:
        entries = _existing_entries()
        return bool(entries) and _matches(entries, path)
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("Scan deny-list check failed for %r: %s", path, exc)
    return False


def filter_denied(rows: Iterable[T]) -> list[T]:
    """``rows`` without those whose ``.path`` is deny-listed; the one filter every inventory and index applies to its combined scan results. Rows with no path are kept."""
    rows = list(rows)
    try:
        entries = _existing_entries()
    except (OSError, ValueError, TypeError):
        return rows
    if not entries:
        return rows
    kept: list[T] = []
    for row in rows:
        path = getattr(row, "path", None)
        try:
            if path and _matches(entries, path):
                continue
        except (OSError, ValueError, TypeError):
            pass
        kept.append(row)
    return kept
