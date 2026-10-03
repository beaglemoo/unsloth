# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Async client for the ds4 on-demand launcher (``/admin/status``, ``/admin/start``, ``/admin/stop``). ``status`` never raises: a launcher that is down reports ``reachable = False``. ``start`` and ``stop`` also never raise, they report any failure through ``Ds4Status.error``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import httpx

from loggers import get_logger

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(5.0, connect = 2.0)
DEFAULT_START_TIMEOUT_S = 120.0
_ADMIN_SLACK_S = 15.0


@dataclass(frozen = True)
class Ds4Status:
    reachable: bool
    loaded: bool = False
    # The launcher exposes no "starting" flag; a spawned process that is not serving yet is starting.
    starting: bool = False
    pid: Optional[int] = None
    uptime_s: Optional[float] = None
    in_flight: int = 0
    idle_remaining_s: Optional[float] = None
    live_tps: Optional[float] = None
    last_gen_tps: Optional[float] = None
    last_ttft_ms: Optional[float] = None
    totals: Optional[dict[str, int]] = None
    start_timeout_s: float = DEFAULT_START_TIMEOUT_S
    error: Optional[str] = None


def _float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


class Ds4Client:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self._start_timeout_s = DEFAULT_START_TIMEOUT_S

    def _client(self, timeout: httpx.Timeout = _TIMEOUT) -> httpx.AsyncClient:
        # trust_env off: an environment proxy must never see loopback engine traffic.
        return httpx.AsyncClient(
            base_url = self.base_url,
            timeout = timeout,
            transport = self._transport,
            trust_env = False,
        )

    async def status(self) -> Ds4Status:
        try:
            async with self._client() as client:
                response = await client.get("/admin/status")
                response.raise_for_status()
                body = response.json()
            if not isinstance(body, dict):
                raise ValueError("unexpected /admin/status payload")
        except (httpx.HTTPError, ValueError) as exc:
            return Ds4Status(
                reachable = False,
                start_timeout_s = self._start_timeout_s,
                error = type(exc).__name__,
            )
        config = body.get("config") if isinstance(body.get("config"), dict) else {}
        timeout = _float(config.get("ds4_start_timeout"))
        if timeout is not None and timeout > 0:
            self._start_timeout_s = timeout
        stats = body.get("stats") if isinstance(body.get("stats"), dict) else {}
        live = stats.get("live") if isinstance(stats.get("live"), dict) else {}
        last = stats.get("last") if isinstance(stats.get("last"), dict) else {}
        totals = stats.get("totals") if isinstance(stats.get("totals"), dict) else None
        loaded = bool(body.get("loaded"))
        pid = _int(body.get("pid"))
        return Ds4Status(
            reachable = True,
            loaded = loaded,
            starting = pid is not None and not loaded,
            pid = pid,
            uptime_s = _float(body.get("uptime_seconds")),
            in_flight = _int(body.get("in_flight")) or 0,
            idle_remaining_s = _float(body.get("idle_seconds_remaining")),
            live_tps = _float(live.get("gen_tps")) if live else None,
            last_gen_tps = _float(last.get("gen_tps")),
            last_ttft_ms = _float(last.get("ttft_ms")),
            totals = {k: v for k, v in totals.items() if _int(v) is not None} if totals else None,
            start_timeout_s = self._start_timeout_s,
        )

    async def start(self) -> Ds4Status:
        """Start ds4 and wait until it serves (the launcher blocks up to its start timeout)."""
        return await self._admin("/admin/start")

    async def stop(self) -> Ds4Status:
        """Stop ds4. The launcher takes its lock, so this waits out an in-progress start."""
        return await self._admin("/admin/stop")

    async def model_ids(self) -> Optional[list[str]]:
        """Ids on ``/v1/models``, or None when the listing could not be fetched (timeout, HTTP error, malformed body), which is not the same as an empty catalog."""
        try:
            async with self._client() as client:
                response = await client.get("/v1/models")
                response.raise_for_status()
                data = response.json().get("data")
                if not isinstance(data, list):
                    raise ValueError("unexpected /v1/models payload")
            return [
                row["id"]
                for row in data
                if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]
            ]
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            return None

    async def _admin(self, path: str) -> Ds4Status:
        error: Optional[str] = None
        timeout = httpx.Timeout(self._start_timeout_s + _ADMIN_SLACK_S, connect = 2.0)
        try:
            async with self._client(timeout) as client:
                response = await client.post(path)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error = f"HTTP {exc.response.status_code}"
        except httpx.HTTPError as exc:
            error = type(exc).__name__
        if error is not None:
            logger.warning("ds4 %s failed: %s", path.rsplit("/", 1)[-1], error)
        status = await self.status()
        return replace(status, error = error) if error is not None else status
