# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted settings for the attached oMLX engine."""

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
COMFYUI_URL_ENV_VAR = "UNSLOTH_COMFYUI_URL"
COMFYUI_PEER_URLS_ENV_VAR = "UNSLOTH_COMFYUI_PEER_URLS"
SCAN_DENYLIST_ENV_VAR = "UNSLOTH_SCAN_DENYLIST"

DEFAULT_ENABLED = False
DEFAULT_OMLX_URL = "http://127.0.0.1:8843"
DEFAULT_SCAN_DENYLIST: tuple[str, ...] = ()
DEFAULT_ARBITRATE_LOCAL_LOADS = True
DEFAULT_COMFYUI_URL = "http://127.0.0.1:8844"
# StoryPress's ComfyUI: freed when idle, never started, stopped or interrupted.
DEFAULT_COMFYUI_PEER_URLS: tuple[str, ...] = ("http://127.0.0.1:8188",)
DEFAULT_ARBITRATE_COMFYUI = True
DEFAULT_COMFYUI_IDLE_FREE_S = 300
MAX_COMFYUI_IDLE_FREE_S = 86400
MAX_COMFYUI_PEER_URLS = 8

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, "AttachedEnginesConfig"]] = {}


@dataclass(frozen = True)
class AttachedEnginesConfig:
    enabled: bool
    omlx_url: str
    scan_denylist: tuple[str, ...]
    arbitrate_local_loads: bool
    comfyui_url: str = DEFAULT_COMFYUI_URL
    comfyui_peer_urls: tuple[str, ...] = DEFAULT_COMFYUI_PEER_URLS
    arbitrate_comfyui: bool = DEFAULT_ARBITRATE_COMFYUI
    # Seconds of ComfyUI idleness before Studio frees its memory; 0 turns it off.
    comfyui_idle_free_s: int = DEFAULT_COMFYUI_IDLE_FREE_S


DEFAULT_CONFIG = AttachedEnginesConfig(
    enabled = DEFAULT_ENABLED,
    omlx_url = DEFAULT_OMLX_URL,
    scan_denylist = DEFAULT_SCAN_DENYLIST,
    arbitrate_local_loads = DEFAULT_ARBITRATE_LOCAL_LOADS,
    comfyui_url = DEFAULT_COMFYUI_URL,
    comfyui_peer_urls = DEFAULT_COMFYUI_PEER_URLS,
    arbitrate_comfyui = DEFAULT_ARBITRATE_COMFYUI,
    comfyui_idle_free_s = DEFAULT_COMFYUI_IDLE_FREE_S,
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


def _clean_peer_urls(value: Any) -> tuple[str, ...]:
    """Loopback-validated, de-duplicated peer ComfyUI origins; ValueError when any entry is invalid. A string is split on commas."""
    if isinstance(value, str):
        value = [part for part in value.split(",") if part.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError("ComfyUI peer URLs must be a list of loopback URLs.")
    if len(value) > MAX_COMFYUI_PEER_URLS:
        raise ValueError(f"At most {MAX_COMFYUI_PEER_URLS} ComfyUI peer URLs are allowed.")
    cleaned: list[str] = []
    for item in value:
        url = normalize_loopback_url(item)
        if url not in cleaned:
            cleaned.append(url)
    return tuple(cleaned)


def _clean_idle_free_s(value: Any) -> int:
    """Whole seconds clamped to 0..86400; ValueError for anything that is not a whole number."""
    if isinstance(value, bool):
        raise ValueError("comfyui_idle_free_s must be a whole number of seconds.")
    if isinstance(value, str):
        value = value.strip()
        if not value.lstrip("-").isdigit():
            raise ValueError("comfyui_idle_free_s must be a whole number of seconds.")
        value = int(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ValueError("comfyui_idle_free_s must be a whole number of seconds.")
    return max(0, min(MAX_COMFYUI_IDLE_FREE_S, value))


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
    for key in ("enabled", "arbitrate_local_loads", "arbitrate_comfyui"):
        if key in stored:
            parsed = _coerce_bool(stored[key])
            if parsed is not None:
                config = replace(config, **{key: parsed})
    for key in ("omlx_url", "comfyui_url"):
        if key in stored:
            try:
                config = replace(config, **{key: normalize_loopback_url(stored[key])})
            except ValueError:
                logger.warning("Ignoring invalid stored attached-engines %s", key)
    for key, clean in (
        ("comfyui_peer_urls", _clean_peer_urls),
        ("comfyui_idle_free_s", _clean_idle_free_s),
    ):
        if key in stored:
            try:
                config = replace(config, **{key: clean(stored[key])})
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
    for var, field in ((OMLX_URL_ENV_VAR, "omlx_url"), (COMFYUI_URL_ENV_VAR, "comfyui_url")):
        raw = os.environ.get(var)
        if raw is not None and raw.strip():
            try:
                config = replace(config, **{field: normalize_loopback_url(raw)})
            except ValueError:
                _env_warn_once(var)
    raw = os.environ.get(SCAN_DENYLIST_ENV_VAR)
    if raw is not None and raw.strip():
        config = replace(config, scan_denylist = _clean_denylist(raw))
    # Set but empty means "no peers", unlike the other variables, where empty means unset.
    raw = os.environ.get(COMFYUI_PEER_URLS_ENV_VAR)
    if raw is not None:
        try:
            config = replace(config, comfyui_peer_urls = _clean_peer_urls(raw))
        except ValueError:
            _env_warn_once(COMFYUI_PEER_URLS_ENV_VAR)
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
        if key in ("enabled", "arbitrate_local_loads", "arbitrate_comfyui"):
            parsed = _coerce_bool(value)
            if parsed is None or isinstance(value, str):
                raise ValueError(f"{key} must be true or false.")
            updates[key] = parsed
        elif key in ("omlx_url", "comfyui_url"):
            updates[key] = normalize_loopback_url(value)
        elif key == "comfyui_peer_urls":
            if not isinstance(value, (list, tuple)):
                raise ValueError("ComfyUI peer URLs must be a list of loopback URLs.")
            updates[key] = list(_clean_peer_urls(value))
        elif key == "comfyui_idle_free_s":
            updates[key] = _clean_idle_free_s(value)
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
