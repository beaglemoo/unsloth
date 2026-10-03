# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted settings for attached external engines (oMLX, DwarfStar/ds4), all off by default so Studio behaves exactly as upstream unless the owner opts in. ``enabled`` gates every attached-engine code path: the supervisor routes, the memory arbiter hooks and the provider sync. ``omlx_url`` / ``ds4_url`` name the engines' local servers and must be loopback, stored without a trailing ``/v1``. ``scan_denylist`` is a list of directories the model scanner must never surface or accept as a scan folder, and stays active even with ``enabled`` off. ``arbitrate_local_loads`` frees both engines' memory before a local GGUF load or a training run, and ``prewarm_ds4_on_select`` starts ds4 when its provider is picked in the chat. Environment overrides ``UNSLOTH_ATTACHED_ENGINES``, ``UNSLOTH_OMLX_URL``, ``UNSLOTH_DS4_URL`` and ``UNSLOTH_SCAN_DENYLIST`` (``os.pathsep`` separated) win over stored values, for headless deploys. Reads are cached for a short window because the deny-list and flag are on hot paths; writes invalidate the cache."""

from __future__ import annotations

import ipaddress
import os
import threading
import time
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit

from loggers import get_logger
from utils.account_context import OWNER, run_as

logger = get_logger(__name__)

ATTACHED_ENGINES_SETTING_KEY = "attached_engines"

ENABLED_ENV_VAR = "UNSLOTH_ATTACHED_ENGINES"
OMLX_URL_ENV_VAR = "UNSLOTH_OMLX_URL"
DS4_URL_ENV_VAR = "UNSLOTH_DS4_URL"
SCAN_DENYLIST_ENV_VAR = "UNSLOTH_SCAN_DENYLIST"

DEFAULT_ENABLED = False
DEFAULT_OMLX_URL = "http://127.0.0.1:8843"
DEFAULT_DS4_URL = "http://127.0.0.1:8001"
DEFAULT_SCAN_DENYLIST = ("~/Homelab/dwarfstar/gguf",)
DEFAULT_ARBITRATE_LOCAL_LOADS = True
DEFAULT_PREWARM_DS4_ON_SELECT = True

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, "AttachedEnginesConfig"]] = {}


@dataclass(frozen = True)
class AttachedEnginesConfig:
    enabled: bool
    omlx_url: str
    ds4_url: str
    scan_denylist: tuple[str, ...]
    arbitrate_local_loads: bool
    prewarm_ds4_on_select: bool


DEFAULT_CONFIG = AttachedEnginesConfig(
    enabled = DEFAULT_ENABLED,
    omlx_url = DEFAULT_OMLX_URL,
    ds4_url = DEFAULT_DS4_URL,
    scan_denylist = DEFAULT_SCAN_DENYLIST,
    arbitrate_local_loads = DEFAULT_ARBITRATE_LOCAL_LOADS,
    prewarm_ds4_on_select = DEFAULT_PREWARM_DS4_ON_SELECT,
)


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


def normalize_loopback_url(value: Any) -> str:
    """``http(s)://<loopback>[:port]`` without path or trailing ``/v1``; ValueError otherwise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Engine URL must be a non-empty string.")
    raw = value.strip().rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[: -len("/v1")].rstrip("/")
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError as exc:
        raise ValueError("Engine URL is not a valid URL.") from exc
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Engine URL must start with http:// or https://.")
    if parts.path not in {"", "/"} or parts.query or parts.fragment or parts.username:
        raise ValueError("Engine URL must be a bare origin such as http://127.0.0.1:8843.")
    host = parts.hostname.lower()
    if host not in _LOOPBACK_HOSTS:
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError("Engine URL must point at a loopback address.")
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return f"{parts.scheme}://{netloc}"


def _clean_denylist(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = value.split(os.pathsep)
    if not isinstance(value, (list, tuple)):
        raise ValueError("Scan deny-list must be a list of directory paths.")
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("Scan deny-list entries must be strings.")
        item = item.strip()
        if item and item not in cleaned:
            cleaned.append(item)
    return tuple(cleaned)


def _stored_config() -> AttachedEnginesConfig:
    try:
        from storage.studio_db import get_app_setting
        stored = run_as(OWNER, get_app_setting, ATTACHED_ENGINES_SETTING_KEY, None)
    except Exception:
        stored = None
    config = DEFAULT_CONFIG
    if not isinstance(stored, dict):
        return config
    # Field by field: one hand-edited bad value must not discard the rest.
    for key in ("enabled", "arbitrate_local_loads", "prewarm_ds4_on_select"):
        if key in stored:
            parsed = _coerce_bool(stored[key])
            if parsed is not None:
                config = replace(config, **{key: parsed})
    for key in ("omlx_url", "ds4_url"):
        if key in stored:
            try:
                config = replace(config, **{key: normalize_loopback_url(stored[key])})
            except ValueError:
                logger.warning("Ignoring invalid stored attached-engines %s", key)
    if "scan_denylist" in stored:
        try:
            config = replace(config, scan_denylist = _clean_denylist(stored["scan_denylist"]))
        except ValueError:
            logger.warning("Ignoring invalid stored attached-engines scan_denylist")
    return config


_env_warned: set[str] = set()


def _env_warn_once(var: str) -> None:
    if var not in _env_warned:
        _env_warned.add(var)
        logger.warning("Ignoring invalid %s", var)


def _apply_env(config: AttachedEnginesConfig) -> AttachedEnginesConfig:
    raw = os.environ.get(ENABLED_ENV_VAR)
    if raw is not None and raw.strip():
        parsed = _coerce_bool(raw)
        if parsed is None:
            _env_warn_once(ENABLED_ENV_VAR)
        else:
            config = replace(config, enabled = parsed)
    for var, field in ((OMLX_URL_ENV_VAR, "omlx_url"), (DS4_URL_ENV_VAR, "ds4_url")):
        raw = os.environ.get(var)
        if raw is not None and raw.strip():
            try:
                config = replace(config, **{field: normalize_loopback_url(raw)})
            except ValueError:
                _env_warn_once(var)
    raw = os.environ.get(SCAN_DENYLIST_ENV_VAR)
    if raw is not None and raw.strip():
        config = replace(config, scan_denylist = _clean_denylist(raw))
    return config


def get_config() -> AttachedEnginesConfig:
    """Effective config: defaults, then stored values, then environment overrides."""
    key = OWNER.account_id
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and now - hit[0] < _CACHE_TTL_S:
            return hit[1]
    config = _apply_env(_stored_config())
    with _cache_lock:
        _cache[key] = (now, config)
    return config


def _invalidate() -> None:
    with _cache_lock:
        _cache.pop(OWNER.account_id, None)


def set_config(**changes: Any) -> AttachedEnginesConfig:
    """Persist the given fields, leaving the others as stored; ValueError on a bad value. Returns the effective config (an env override still wins over what was just stored)."""
    unknown = set(changes) - set(DEFAULT_CONFIG.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown attached-engines setting: {sorted(unknown)[0]}")
    updates: dict[str, Any] = {}
    for key, value in changes.items():
        if value is None:
            continue
        if key in ("enabled", "arbitrate_local_loads", "prewarm_ds4_on_select"):
            parsed = _coerce_bool(value)
            if parsed is None or isinstance(value, str):
                raise ValueError(f"{key} must be true or false.")
            updates[key] = parsed
        elif key in ("omlx_url", "ds4_url"):
            updates[key] = normalize_loopback_url(value)
        elif key == "scan_denylist":
            updates[key] = list(_clean_denylist(value))
    if updates:
        from storage.studio_db import get_app_setting, upsert_app_settings

        stored = run_as(OWNER, get_app_setting, ATTACHED_ENGINES_SETTING_KEY, None)
        merged = dict(stored) if isinstance(stored, dict) else {}
        merged.update(updates)
        run_as(OWNER, upsert_app_settings, {ATTACHED_ENGINES_SETTING_KEY: merged})
        _invalidate()
    return get_config()
