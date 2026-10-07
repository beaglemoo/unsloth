# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The model directories the attached oMLX engine serves, derived from its real config.

Studio's local-model scan has an ``omlx`` source that reads the oMLX macOS app's settings
(``~/.omlx/settings.json``). When that points at the folder the attached engine serves, the same
models would be listed twice and loading one would put a second, untuned copy in Studio. The scan
deny-list calls :func:`auto_denied_dirs` to hide the engine's own folders while attached engines are enabled.

Sources, in the order the engine itself resolves them:

* ``OMLX_MODEL_DIR`` (comma separated) in ``[omlx.env]`` of engines.toml, which oMLX applies over settings.json;
* ``model.model_dirs``, else the legacy ``model.model_dir``, in ``<base_path>/settings.json``;
* ``<base_path>/models`` when neither names a folder.

Folders listed in engines.toml itself (``[omlx] model_dirs`` / ``model_dir`` / ``model_paths``, or a ``--model-dir`` in ``extra_args``) are added on top.
Only the engine's folders are denied. The oMLX app's own dirs (``~/.omlx``) are never added: when they are the same folder the
deny-list already hides the app source's rows (paths are compared as realpaths), and any other folder stays visible.
Everything is best effort: a missing or malformed file contributes nothing and is logged at debug.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from loggers import get_logger

logger = get_logger(__name__)

_DEFAULT_BASE_PATH = "~/.omlx-tuned"
_TOML_DIR_KEYS = ("model_dirs", "model_dir", "model_paths")

_lock = threading.Lock()
_cache: tuple[Any, Path, tuple[str, ...]] | None = None


def _stamp(path: Path) -> tuple[str, int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size)


def _engines_toml_path() -> Path:
    override = os.environ.get("UNSLOTH_ENGINES_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    home = os.environ.get("UNSLOTH_ENGINES_HOME", "").strip()
    root = Path(home).expanduser() if home else Path.home() / ".unsloth" / "engines"
    return root / "engines.toml"


def _read_toml(path: Path) -> dict:
    try:
        try:
            import tomllib  # novermin -- 3.11; the tomli fallback below is the guard
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        logger.debug("Ignoring unreadable engines config %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def _extra_arg_dirs(extra_args: Any) -> list[str]:
    args = [a for a in extra_args if isinstance(a, str)] if isinstance(extra_args, list) else []
    out: list[str] = []
    for i, arg in enumerate(args):
        if arg == "--model-dir" and i + 1 < len(args):
            out.extend(p.strip() for p in args[i + 1].split(",") if p.strip())
        elif arg.startswith("--model-dir="):
            out.extend(p.strip() for p in arg.split("=", 1)[1].split(",") if p.strip())
    return out


def _settings_dirs(settings_path: Path) -> list[str]:
    """``model.model_dirs``, else ``model.model_dir``, from an oMLX settings.json."""
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8-sig"))
        model = data.get("model") or {}
        dirs = _as_list(model.get("model_dirs"))
        if not dirs:
            dirs = _as_list(model.get("model_dir"))
        return dirs
    except Exception as exc:
        logger.debug("Ignoring unreadable oMLX settings %s: %s", settings_path, exc)
        return []


def _real(path: str, base: Path | None = None) -> str | None:
    try:
        expanded = Path(os.path.expanduser(path))
        if base is not None and not expanded.is_absolute():
            expanded = base / expanded
        return os.path.realpath(expanded)
    except (OSError, ValueError, RuntimeError):
        return None


def _omlx_section(toml_path: Path) -> dict:
    section = _read_toml(toml_path).get("omlx")
    return section if isinstance(section, dict) else {}


def _base_path(toml_path: Path) -> Path:
    base_raw = _omlx_section(toml_path).get("base_path")
    return Path(os.path.expanduser(base_raw if isinstance(base_raw, str) and base_raw.strip() else _DEFAULT_BASE_PATH))


def _engine_dirs(toml_path: Path) -> list[str]:
    """The engine's model folders as realpaths (existing or not)."""
    section = _omlx_section(toml_path)
    base = _base_path(toml_path)

    env_section = section.get("env")
    env_dirs = _as_list(
        [p for p in str(env_section.get("OMLX_MODEL_DIR", "")).split(",")]
        if isinstance(env_section, dict)
        else []
    )
    configured = env_dirs or _settings_dirs(base / "settings.json")
    toml_dirs: list[str] = []
    for key in _TOML_DIR_KEYS:
        toml_dirs.extend(_as_list(section.get(key)))
    toml_dirs.extend(_extra_arg_dirs(section.get("extra_args")))
    candidates = configured + toml_dirs
    if not candidates:
        candidates = [str(base / "models")]
    real = [r for r in (_real(c, base) for c in candidates) if r]
    return list(dict.fromkeys(real))


def auto_denied_dirs() -> tuple[str, ...]:
    """Realpaths of the attached engine's own model folders. Never raises.

    Cached until engines.toml, the engine's settings.json or ``HOME`` changes, so a per-path
    check costs two ``stat`` calls.
    """
    global _cache
    try:
        toml_path = _engines_toml_path()
        with _lock:
            cached = _cache
        if cached is not None:
            key, settings_path, result = cached
            if key == (str(toml_path), os.environ.get("HOME", ""), _stamp(toml_path), _stamp(settings_path)):
                return result
        settings_path = _base_path(toml_path) / "settings.json"
        # Stamp before reading: a write that lands in between is picked up on the next call.
        key = (str(toml_path), os.environ.get("HOME", ""), _stamp(toml_path), _stamp(settings_path))
        result = tuple(_engine_dirs(toml_path))
        with _lock:
            _cache = (key, settings_path, result)
        return result
    except Exception as exc:
        logger.debug("Could not derive the attached engine's model dirs: %s", exc)
        return ()
