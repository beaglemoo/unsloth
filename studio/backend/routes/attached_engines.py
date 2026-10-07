# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-only management routes for the attached oMLX and ComfyUI engines."""

import asyncio
import hashlib
from dataclasses import asdict
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import policy
from auth.authentication import get_current_subject
from core.inference.attached import (
    ATTACHED_OMLX_ID,
    AttachedEngineBusy,
    AttachedEngineError,
)
from core.inference.attached import arbiter, desktop_settings
from core.inference.attached.comfyui_client import ComfyuiClient, ComfyuiError, system_version
from core.inference.attached.omlx_client import OmlxClient, OmlxContext
from loggers import get_logger
from routes.provider_credentials import provider_config_guard
from storage import providers_db
from utils.attached_engines_settings import AttachedEnginesConfig, get_config

logger = get_logger(__name__)


async def _require_enabled() -> None:
    if not get_config().enabled:
        raise HTTPException(status_code = 404, detail = "Not found")


router = APIRouter(dependencies = [Depends(get_current_subject), Depends(policy.require_owner)])
_gated = APIRouter(dependencies = [Depends(_require_enabled)])

MIN_CONTEXT = 4096
# The model folders the ComfyUI panel lists.
COMFYUI_MODEL_FOLDERS = ("diffusion_models", "checkpoints", "text_encoders", "vae", "loras")

class OmlxModelRequest(BaseModel):
    model_id: str = Field(min_length = 1, max_length = 512)


class OmlxContextRequest(BaseModel):
    model_id: str = Field(min_length = 1, max_length = 512)
    # None (JSON null) resets the model to its default cap.
    max_context_window: Optional[int] = Field(default = None, ge = MIN_CONTEXT)


class PrepareRequest(BaseModel):
    provider: Literal["omlx"]


class ComfyuiCancelRequest(BaseModel):
    prompt_id: str = Field(min_length = 1, max_length = 128)


def _omlx(config: AttachedEnginesConfig) -> OmlxClient:
    return OmlxClient(config.omlx_url)


def _comfyui(config: AttachedEnginesConfig) -> ComfyuiClient:
    return ComfyuiClient(config.comfyui_url)


def _models_hash(omlx_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for engine, ids in (("omlx", omlx_ids),):
        digest.update(engine.encode())
        for model_id in sorted(ids):
            digest.update(b"\0" + model_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _engine_failure(name: str):
    try:
        from core.inference.attached.failures import read_failure
    except ModuleNotFoundError as exc:
        if exc.name != "core.inference.attached.failures":
            raise
        return None
    return read_failure(name)


async def _probe_comfyui(config: AttachedEnginesConfig) -> dict:
    """The ``comfyui`` status block. Never raises: a ComfyUI that is down is just ``reachable: false``."""
    own = _comfyui(config)
    peer_urls = [url for url in config.comfyui_peer_urls if url != config.comfyui_url]

    async def peer(url: str) -> dict:
        queue = await ComfyuiClient(url).queue()
        return {"url": url, "reachable": queue.reachable, "busy": queue.busy}

    queue, stats, peers = await asyncio.gather(
        own.queue(), own.system_stats(), asyncio.gather(*(peer(url) for url in peer_urls))
    )
    system = stats.get("system") if isinstance(stats, dict) else None
    system = system if isinstance(system, dict) else {}
    devices = stats.get("devices") if isinstance(stats, dict) else None
    return {
        "url": config.comfyui_url,
        "reachable": queue.reachable,
        "version": system_version(stats),
        "queue_running": len(queue.running_ids),
        "queue_pending": len(queue.pending_ids),
        "devices": devices if isinstance(devices, list) else [],
        "ram_total": system.get("ram_total"),
        "ram_free": system.get("ram_free"),
        "failure": _engine_failure("comfyui"),
        "helper_wanted": desktop_settings.helper_wanted("comfyui"),
        "peers": list(peers),
    }


async def _probe_omlx(config: AttachedEnginesConfig) -> tuple[dict, list[str]]:
    omlx_client = _omlx(config)
    omlx_status = await omlx_client.status()
    omlx_ids = (
        await omlx_client.chat_model_ids(status = omlx_status) if omlx_status.reachable else None
    ) or []
    omlx = asdict(omlx_status)
    omlx["chat_model_ids"] = omlx_ids
    omlx["failure"] = _engine_failure("omlx")
    return omlx, omlx_ids


@_gated.get("/status")
async def attached_status(since: float = 0):
    config = get_config()
    (omlx, omlx_ids), comfyui = await asyncio.gather(_probe_omlx(config), _probe_comfyui(config))
    return {
        "enabled": True,
        "omlx": omlx,
        "comfyui": comfyui,
        "models_hash": _models_hash(omlx_ids),
        "notices": arbiter.recent_notices(since),
    }


def _merge_models(existing: Optional[dict], live: list[str]) -> list[str]:
    """Enabled models after a sync: the user's choices stay, models new since the last sync join."""
    if existing is None:
        return list(live)
    previous = set(existing.get("available_models") or []) | set(existing.get("models") or [])
    enabled = set(existing.get("models") or [])
    return [model for model in live if model in enabled or model not in previous]


async def _upsert_row(
    row_id: str,
    provider_type: str,
    display_name: str,
    base_url: str,
    live_models: Optional[list[str]],
) -> str:
    # Keep this read-merge-write sequence beside normal provider edits. Without
    # the guard, a sync can overwrite a model choice saved between its read and
    # its write.
    async with provider_config_guard(row_id):
        existing = await asyncio.to_thread(providers_db.get_provider, row_id)
        if existing is None:
            models = live_models or []
            await asyncio.to_thread(
                providers_db.create_provider,
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
            await asyncio.to_thread(providers_db.update_provider, row_id, base_url = base_url)
            return "kept_models"
        await asyncio.to_thread(
            providers_db.update_provider,
            row_id,
            base_url = base_url,
            models = _merge_models(existing, live_models),
            available_models = live_models,
        )
        return "updated"


LEGACY_DS4_ROW_ID = "attachedds400001"


async def _delete_row(row_id: str) -> bool:
    # Same guard as _upsert_row and the provider edit routes, so a delete cannot
    # interleave with another write to the same row.
    async with provider_config_guard(row_id):
        return bool(await asyncio.to_thread(providers_db.delete_provider, row_id))


async def _delete_rows() -> list[str]:
    return [
        row_id
        for row_id in (ATTACHED_OMLX_ID, LEGACY_DS4_ROW_ID)
        if await _delete_row(row_id)
    ]


@router.post("/sync")
async def attached_sync():
    config = get_config()
    if not config.enabled:
        deleted = await _delete_rows()
        return {"enabled": False, "deleted": deleted}
    await _delete_row(LEGACY_DS4_ROW_ID)
    omlx_client = _omlx(config)
    omlx_status = await omlx_client.status()
    omlx_ids = (
        await omlx_client.chat_model_ids(status = omlx_status) if omlx_status.reachable else None
    )
    result = {}
    for row_id, ptype, name, url, ids in (
        (ATTACHED_OMLX_ID, "omlx", "oMLX", config.omlx_url, omlx_ids),
    ):
        result[ptype] = await _upsert_row(
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
        "models_hash": _models_hash(omlx_ids or []),
    }


async def _resolve_omlx_dir(client: OmlxClient, model_id: str) -> str:
    status = await client.status()
    if not status.reachable:
        raise HTTPException(status_code = 502, detail = "oMLX is not reachable.")
    dir_id = client.resolve_dir(list(status.models), model_id)
    if dir_id is None:
        raise HTTPException(status_code = 404, detail = "Model not found on oMLX.")
    return dir_id


async def _admit_omlx_use(admit = None):
    result = await (admit or arbiter.before_omlx_use)()
    try:
        result.require_clear()
    except AttachedEngineError as exc:
        raise HTTPException(
            status_code = 503,
            detail = str(exc),
            headers = {"Retry-After": "15"},
        ) from exc
    return result



@_gated.post("/omlx/load")
async def omlx_load(body: OmlxModelRequest):
    config = get_config()
    client = _omlx(config)
    dir_id = await _resolve_omlx_dir(client, body.model_id)
    notice = await _admit_omlx_use(arbiter.before_omlx_load)
    try:
        await client.load(dir_id)
    except AttachedEngineError as exc:
        raise HTTPException(status_code = 502, detail = str(exc)) from exc
    return {"loaded": dir_id, "actions": list(notice.actions)}


def _context_body(model_id: str, context: OmlxContext) -> dict:
    return {
        "model_id": model_id,
        "dir": context.dir,
        "max_context_window": context.max_context_window,
        "native_max": context.native_max,
        "effective": context.effective,
    }


# POST, not GET: model ids carry colons and travel in JSON bodies only.
@_gated.post("/omlx/context/get")
async def omlx_context_get(body: OmlxModelRequest):
    client = _omlx(get_config())
    dir_id = await _resolve_omlx_dir(client, body.model_id)
    notice = await _admit_omlx_use()
    try:
        return _context_body(body.model_id, await client.context(dir_id))
    except AttachedEngineError as exc:
        raise HTTPException(status_code = 502, detail = str(exc)) from exc


@_gated.post("/omlx/context")
async def omlx_context_set(body: OmlxContextRequest):
    client = _omlx(get_config())
    dir_id = await _resolve_omlx_dir(client, body.model_id)
    notice = await _admit_omlx_use()
    try:
        current = await client.context(dir_id)
        if (
            body.max_context_window is not None
            and current.native_max is not None
            and body.max_context_window > current.native_max
        ):
            raise HTTPException(
                status_code = 422,
                detail = f"max_context_window exceeds the model's native length ({current.native_max}).",
            )
        requires_reload = await client.set_context(dir_id, body.max_context_window)
        updated = await client.context(dir_id)
    except AttachedEngineError as exc:
        raise HTTPException(status_code = 502, detail = str(exc)) from exc
    return {**_context_body(body.model_id, updated), "requires_reload": requires_reload}


@_gated.post("/omlx/unload")
async def omlx_unload(body: OmlxModelRequest):
    client = _omlx(get_config())
    dir_id = await _resolve_omlx_dir(client, body.model_id)
    try:
        await client.unload(dir_id)
    except AttachedEngineBusy as exc:
        raise HTTPException(status_code = 503, detail = str(exc), headers = {"Retry-After": "15"}) from exc
    except AttachedEngineError as exc:
        raise HTTPException(status_code = 502, detail = str(exc)) from exc
    return {"unloaded": dir_id}


@_gated.post("/omlx/unload-all")
async def omlx_unload_all():
    try:
        return {"unloaded": await _omlx(get_config()).unload_all()}
    except AttachedEngineBusy as exc:
        raise HTTPException(status_code = 503, detail = str(exc), headers = {"Retry-After": "15"}) from exc


@_gated.post("/prepare")
async def prepare(body: PrepareRequest):
    result = await _admit_omlx_use(arbiter.before_omlx_load)
    return {
        "provider": body.provider,
        "actions": list(result.actions),
        "in_flight_killed": result.in_flight_killed,
    }



def _comfyui_unreachable(exc: Exception) -> HTTPException:
    return HTTPException(status_code = 502, detail = "ComfyUI is not reachable.")


async def _comfyui_queue(client: ComfyuiClient):
    queue = await client.queue()
    if not queue.reachable:
        raise HTTPException(status_code = 502, detail = "ComfyUI is not reachable.")
    return queue


@_gated.post("/comfyui/free")
async def comfyui_free():
    """Unload ComfyUI's models and free its memory. While a job runs ComfyUI applies it once the job ends (``deferred``)."""
    client = _comfyui(get_config())
    queue = await _comfyui_queue(client)
    try:
        await client.free()
    except ComfyuiError as exc:
        raise _comfyui_unreachable(exc) from exc
    return {"freed": True, "deferred": queue.busy}


@_gated.get("/comfyui/queue")
async def comfyui_queue():
    queue = await _comfyui_queue(_comfyui(get_config()))

    def rows(ids: tuple[str, ...], state: str) -> list[dict]:
        return [
            {
                "prompt_id": prompt_id,
                "number": queue.numbers.get(prompt_id),
                "state": state,
                "studio": queue.studio.get(prompt_id),
            }
            for prompt_id in ids
        ]

    return {"running": rows(queue.running_ids, "running"), "pending": rows(queue.pending_ids, "pending")}


@_gated.post("/comfyui/queue/cancel")
async def comfyui_cancel(body: ComfyuiCancelRequest):
    """Interrupt a running job or drop a pending one on Studio's own ComfyUI (never a peer)."""
    client = _comfyui(get_config())
    queue = await _comfyui_queue(client)
    try:
        if body.prompt_id in queue.running_ids:
            await client.interrupt(body.prompt_id)
            return {"cancelled": body.prompt_id, "state": "running"}
        if body.prompt_id in queue.pending_ids:
            await client.delete_pending([body.prompt_id])
            return {"cancelled": body.prompt_id, "state": "pending"}
    except ComfyuiError as exc:
        raise _comfyui_unreachable(exc) from exc
    raise HTTPException(status_code = 404, detail = "Job not found in the ComfyUI queue.")


@_gated.get("/comfyui/models")
async def comfyui_models():
    client = _comfyui(get_config())
    try:
        lists = await asyncio.gather(*(client.models(folder) for folder in COMFYUI_MODEL_FOLDERS))
    except ComfyuiError as exc:
        raise _comfyui_unreachable(exc) from exc
    return {"folders": dict(zip(COMFYUI_MODEL_FOLDERS, lists))}


router.include_router(_gated)
