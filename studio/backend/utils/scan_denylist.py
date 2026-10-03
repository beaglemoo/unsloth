# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-configured model-scan deny-list. A directory named here (the default is the DwarfStar GGUF folder, which belongs to the ds4 launcher and must not appear as a loadable local model) is never listed, never accepted as a scan folder, and never walked as part of one. Unlike every other attached-engines feature this is active even with the ``attached_engines`` flag off. Matching is by resolved path prefix, so a symlink into a denied folder is denied too."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Union

from loggers import get_logger

logger = get_logger(__name__)


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

    entries: list[str] = []
    for raw in get_config().scan_denylist:
        expanded = os.path.expanduser(raw)
        # An entry that does not exist holds nothing, and no symlink can lead into it.
        if expanded and os.path.lexists(expanded):
            entries.append(os.path.realpath(expanded))
    return entries


def is_denied(path: Union[str, "os.PathLike[str]"]) -> bool:
    """True when ``path`` is, or lies inside, a deny-listed directory."""
    try:
        entries = _existing_entries()
        if not entries:
            return False
        resolved = os.path.realpath(os.path.expanduser(os.fspath(path)))
        check = _comparable(resolved)
        for entry in entries:
            prefix = _comparable(entry).rstrip(os.sep) or os.sep
            if check == prefix or check.startswith(
                prefix if prefix.endswith(os.sep) else prefix + os.sep
            ):
                return True
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("Scan deny-list check failed for %r: %s", path, exc)
    return False
