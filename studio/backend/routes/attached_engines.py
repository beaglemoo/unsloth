# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-only management routes for attached engines (oMLX, DwarfStar/ds4): live status, model residency, and seeding the saved provider rows that put them in the chat picker. Every route except ``/sync`` answers 404 while the ``attached_engines`` flag is off; ``/sync`` stays reachable so turning the flag off can remove the rows it seeded. Model ids travel in JSON bodies only, never in the path, because oMLX ids carry a colon (``swift-1.5-27b:fast``)."""

import asyncio
import hashlib
from dataclasses import asdict
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import policy
from auth.authentication import get_current_subject
from core.inference.attached import (
    ATTACHED_DS4_ID,
    ATTACHED_OMLX_ID,
    AttachedEngineError,
)
from core.inference.attached import arbiter
from core.inference.attached.ds4_client import Ds4Client
from core.inference.attached.omlx_client import OmlxClient
from loggers import get_logger
from storage import providers_db
from utils.attached_engines_settings import AttachedEnginesConfig, get_config

logger = get_logger(__name__)


async def _require_enabled() -> None:
    if not get_config().enabled:
        raise HTTPException(status_code = 404, detail = "Not found")


router = APIRouter(dependencies = [Depends(get_current_subject), Depends(policy.require_owner)])
_gated = APIRouter(dependencies = [Depends(_require_enabled)])

# Strong references: the loop keeps only weak ones to running tasks.
_background: set[asyncio.Task] = set()


class OmlxModelRequest(BaseModel):
    model_id: str = Field(min_length = 1, max_length = 512)


class PrepareRequest(BaseModel):
    provider: Literal["omlx", "dwarfstar"]


def _omlx(config: AttachedEnginesConfig) -> OmlxClient:
    return OmlxClient(config.omlx_url)


def _ds4(config: AttachedEnginesConfig) -> Ds4Client:
    return Ds4Client(config.ds4_url)


def _models_hash(omlx_ids: list[str], ds4_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for engine, ids in (("omlx", omlx_ids), ("ds4", ds4_ids)):
        digest.update(engine.encode())
        for model_id in sorted(ids):
            digest.update(b"\0" + model_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _start_ds4_in_background(client: Ds4Client) -> None:
    async def run() -> None:
        try:
            status = await client.start()
            if status.error:
                logger.warning("DwarfStar start failed: %s", status.error)
        except Exception:  # noqa: BLE001 -- a detached task has no caller to raise to
            logger.exception("DwarfStar start crashed")

    task = asyncio.get_running_loop().create_task(run())
    _background.add(task)
    task.add_done_callback(_background.discard)


async def _kick_ds4_start(config: AttachedEnginesConfig) -> dict:
    client = _ds4(config)
    status = await client.status()
    if not status.reachable:
        return {"starting": False, "loaded": False, "reachable": False}
    if status.loaded or status.starting:
        return {"starting": status.starting, "loaded": status.loaded, "reachable": True}
    _start_ds4_in_background(client)
    return {"starting": True, "loaded": False, "reachable": True}


@_gated.get("/status")
async def attached_status(since: float = 0):
    config = get_config()
    omlx_client, ds4_client = _omlx(config), _ds4(config)
    omlx_status, ds4_status, ds4_ids = await asyncio.gather(
        omlx_client.status(), ds4_client.status(), ds4_client.model_ids()
    )
    omlx_ids = (
        await omlx_client.chat_model_ids(status = omlx_status) if omlx_status.reachable else None
    ) or []
    ds4_ids = ds4_ids or []
    omlx = asdict(omlx_status)
    omlx["chat_model_ids"] = omlx_ids
    return {
        "enabled": True,
        "omlx": omlx,
        "ds4": asdict(ds4_status),
        "models_hash": _models_hash(omlx_ids, ds4_ids),
        "notices": arbiter.recent_notices(since),
    }


def _merge_models(existing: Optional[dict], live: list[str]) -> list[str]:
    """Enabled models after a sync: the user's choices stay, models new since the last sync join."""
    if existing is None:
        return list(live)
    previous = set(existing.get("available_models") or []) | set(existing.get("models") or [])
    enabled = set(existing.get("models") or [])
    return [model for model in live if model in enabled or model not in previous]


def _upsert_row(
    row_id: str,
    provider_type: str,
    display_name: str,
    base_url: str,
    live_models: Optional[list[str]],
) -> str:
    existing = providers_db.get_provider(row_id)
    if existing is None:
        models = live_models or []
        providers_db.create_provider(
            id = row_id,
            provider_type = provider_type,
            display_name = display_name,
            base_url = base_url,
            models = models,
            available_models = models,
        )
        return "created"
    # is_enabled is never touched: an engine being down is not the user turning the provider off.
    if live_models is None:
        providers_db.update_provider(row_id, base_url = base_url)
        return "kept_models"
    providers_db.update_provider(
        row_id,
        base_url = base_url,
        models = _merge_models(existing, live_models),
        available_models = live_models,
    )
    return "updated"


def _delete_rows() -> list[str]:
    return [
        row_id
        for row_id in (ATTACHED_OMLX_ID, ATTACHED_DS4_ID)
        if providers_db.delete_provider(row_id)
    ]


@router.post("/sync")
async def attached_sync():
    config = get_config()
    if not config.enabled:
        deleted = await asyncio.to_thread(_delete_rows)
        return {"enabled": False, "deleted": deleted}
    omlx_client, ds4_client = _omlx(config), _ds4(config)
    omlx_status, ds4_status = await asyncio.gather(omlx_client.status(), ds4_client.status())
    omlx_ids = (
        await omlx_client.chat_model_ids(status = omlx_status) if omlx_status.reachable else None
    )
    ds4_ids = await ds4_client.model_ids() if ds4_status.reachable else None
    result = {}
    for row_id, ptype, name, url, ids in (
        (ATTACHED_OMLX_ID, "omlx", "oMLX", config.omlx_url, omlx_ids),
        (ATTACHED_DS4_ID, "dwarfstar", "DwarfStar", config.ds4_url, ds4_ids),
    ):
        result[ptype] = await asyncio.to_thread(
            _upsert_row,
            row_id,
            ptype,
            name,
            f"{url}/v1",
            # None means "could not ask"; an empty list from a reachable engine is real.
            ids,
        )
    return {
        "enabled": True,
        "rows": result,
        "models_hash": _models_hash(omlx_ids or [], ds4_ids or []),
    }


async def _resolve_omlx_dir(client: OmlxClient, model_id: str) -> str:
    status = await client.status()
    if not status.reachable:
        raise HTTPException(status_code = 502, detail = "oMLX is not reachable.")
    dir_id = client.resolve_dir(list(status.models), model_id)
    if dir_id is None:
        raise HTTPException(status_code = 404, detail = "Model not found on oMLX.")
    return dir_id


@_gated.post("/omlx/load")
async def omlx_load(body: OmlxModelRequest):
    config = get_config()
    client = _omlx(config)
    dir_id = await _resolve_omlx_dir(client, body.model_id)
    notice = await arbiter.before_omlx_use()
    try:
        await client.load(dir_id)
    except AttachedEngineError as exc:
        raise HTTPException(status_code = 502, detail = str(exc)) from exc
    return {"loaded": dir_id, "actions": list(notice.actions)}


@_gated.post("/omlx/unload")
async def omlx_unload(body: OmlxModelRequest):
    client = _omlx(get_config())
    dir_id = await _resolve_omlx_dir(client, body.model_id)
    try:
        await client.unload(dir_id)
    except AttachedEngineError as exc:
        raise HTTPException(status_code = 502, detail = str(exc)) from exc
    return {"unloaded": dir_id}


@_gated.post("/omlx/unload-all")
async def omlx_unload_all():
    return {"unloaded": await _omlx(get_config()).unload_all()}


@_gated.post("/ds4/start")
async def ds4_start():
    return await _kick_ds4_start(get_config())


@_gated.post("/ds4/stop")
async def ds4_stop():
    return asdict(await _ds4(get_config()).stop())


@_gated.post("/prepare")
async def prepare(body: PrepareRequest):
    config = get_config()
    if body.provider == "omlx":
        result = await arbiter.before_omlx_use()
        return {
            "provider": "omlx",
            "actions": list(result.actions),
            "in_flight_killed": result.in_flight_killed,
        }
    if not config.prewarm_ds4_on_select:
        return {"provider": "dwarfstar", "starting": False, "prewarm": False}
    return {"provider": "dwarfstar", "prewarm": True, **await _kick_ds4_start(config)}


router.include_router(_gated)
