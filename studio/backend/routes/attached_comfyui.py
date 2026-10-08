# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image generation through Studio's own ComfyUI: templates, jobs, progress, cancel and graph import.

Mounted by ``routes.attached_engines`` inside its gated router, so every route is owner-only and
404s while the attached-engines flag is off. Errors carry a machine-readable ``X-Comfy-Error``
header next to the human ``detail``: ``busy``, ``off``, ``not_running``, ``admission``,
``missing_models``, ``params``, ``graph``, ``execution``, ``cancelled``, ``timeout``,
``unreachable`` or ``template``.

Image templates (``kind`` ``img2img`` or ``edit``, listed with their ``image_slots``) take their inputs in
``POST /comfyui/generate`` as ``input_images: {slot: {data} | {gallery_id}}`` (a data URL or base64, or the
id of a gallery image), plus ``denoise`` (img2img) and ``reference_resolution`` (edit). The inputs travel
inline and are uploaded to ComfyUI's temp folder only once the job is admitted; Studio deletes them when
the job ends. A bad image answers 422 ``params`` before anything is unloaded.
"""

import asyncio
import copy
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from core.inference.attached import AttachedEngineError
from core.inference.attached import comfyui_jobs as jobs
from core.inference.attached.arbiter import AttachedAdmissionError
from core.inference.attached.comfyui_client import ComfyGraphError, ComfyuiClient, ComfyuiError
from core.inference.attached.comfyui_graphs import (
    ComfyParamError,
    ComfyTemplateError,
    delete_user_template,
    detect_slots,
    graph_class_types,
    load_templates,
    save_user_template,
    validate_api_graph,
)
from loggers import get_logger
from models.inference import DiffusionGenerateProgressResponse, GalleryImage
from utils.attached_engines_settings import get_config

logger = get_logger(__name__)
router = APIRouter()


# One inline image is capped at 32 MiB of base64 text, and the whole request at 128 MiB (as the native Images page).
MAX_INPUT_DATA_CHARS = 32 * 1024 * 1024
MAX_TOTAL_INPUT_DATA_CHARS = 128 * 1024 * 1024


class ComfyInputImage(BaseModel):
    """One input image: inline ``data`` (a data URL or raw base64) or the ``gallery_id`` of a gallery image."""

    data: Optional[str] = Field(default = None, max_length = MAX_INPUT_DATA_CHARS)
    # The gallery's own id syntax (image_gallery._ID_RE); an unknown id is a 422 from the job, not here.
    gallery_id: Optional[str] = Field(default = None, pattern = r"^[A-Za-z0-9_-]{1,128}$")

    @model_validator(mode = "after")
    def _exactly_one(self):
        if (self.data is None) == (self.gallery_id is None):
            raise ValueError("Give either data or gallery_id for an input image, not both and not neither.")
        return self


class ComfyLora(BaseModel):
    name: str = Field(min_length = 1, max_length = 256)
    strength: float = Field(default = 1.0, ge = -4.0, le = 4.0)


class ComfyGenerateRequest(BaseModel):
    template_id: str = Field(min_length = 1, max_length = 128)
    prompt: str = Field(min_length = 1, max_length = 8000)
    negative_prompt: Optional[str] = Field(default = None, max_length = 8000)
    width: Optional[int] = Field(default = None, ge = 64, le = 8192)
    height: Optional[int] = Field(default = None, ge = 64, le = 8192)
    # None or -1 picks a random seed; the seed used comes back in the response and in each image.
    seed: Optional[int] = Field(default = None, ge = -1, le = 2**53 - 1)
    steps: Optional[int] = Field(default = None, ge = 1, le = 200)
    cfg: Optional[float] = Field(default = None, ge = 0.0, le = 100.0)
    sampler: Optional[str] = Field(default = None, max_length = 64)
    scheduler: Optional[str] = Field(default = None, max_length = 64)
    batch_size: int = Field(default = 1, ge = 1, le = 4)
    loras: list[ComfyLora] = Field(default_factory = list, max_length = 4)
    # Image templates: one entry per template image slot (see ``image_slots`` in GET /comfyui/templates).
    input_images: dict[str, ComfyInputImage] = Field(default_factory = dict, max_length = 4)
    # img2img: how far from the input image the result may drift (1 = ignore it). Clamped by the template.
    denoise: Optional[float] = Field(default = None, gt = 0.0, le = 1.0)
    # edit: references are resized to about this many pixels per side; the template snaps and clamps it.
    reference_resolution: Optional[int] = Field(default = None, ge = 0, le = 4096)

    @model_validator(mode = "after")
    def _inputs_fit(self):
        if sum(len(image.data or "") for image in self.input_images.values()) > MAX_TOTAL_INPUT_DATA_CHARS:
            raise ValueError("The input images are too large (128 MiB in total).")
        return self


class ComfyGenerateResponse(BaseModel):
    images: list[GalleryImage]
    actions: list[str] = Field(default_factory = list, description = "What admission did first, e.g. 'Unloaded oMLX: ...'")
    prompt_id: str
    seed: int


class ComfyProgressResponse(DiffusionGenerateProgressResponse):
    """The Images page progress shape plus the ComfyUI job's queue state."""

    queue_position: Optional[int] = Field(None, description = "0 = running, N = N-th in ComfyUI's pending queue")
    prompt_id: Optional[str] = None
    engine: str = "comfyui"


class ComfyImportRequest(BaseModel):
    name: str = Field(min_length = 1, max_length = 80)
    graph: dict[str, Any]


def _fail(status: int, detail: str, code: str, headers: Optional[dict] = None) -> HTTPException:
    return HTTPException(status_code = status, detail = detail, headers = {"X-Comfy-Error": code, **(headers or {})})


def _map_error(exc: Exception) -> HTTPException:
    if isinstance(exc, jobs.ComfyBusyError):
        return _fail(409, str(exc), "busy")
    if isinstance(exc, jobs.ComfyOffError):
        return _fail(503, str(exc), "off")
    if isinstance(exc, jobs.ComfyNotRunningError):
        return _fail(502, str(exc), "not_running")
    if isinstance(exc, jobs.ComfyTemplateNotFound):
        return _fail(404, str(exc), "template")
    if isinstance(exc, jobs.ComfyModelsMissing):
        return _fail(422, str(exc), "missing_models")
    if isinstance(exc, (ComfyParamError, ComfyTemplateError)):
        return _fail(422, str(exc), "params")
    if isinstance(exc, ComfyGraphError):
        return _fail(422, str(exc), "graph")
    if isinstance(exc, jobs.ComfyJobError):
        return _fail(502, str(exc), "execution")
    if isinstance(exc, jobs.ComfyCancelled):
        return _fail(409, str(exc), "cancelled")
    if isinstance(exc, jobs.ComfyTimeout):
        return _fail(504, str(exc), "timeout")
    if isinstance(exc, AttachedAdmissionError):
        return _fail(503, str(exc), "admission", {"Retry-After": "15"})
    if isinstance(exc, (ComfyuiError, AttachedEngineError)):
        return _fail(502, "ComfyUI is not reachable.", "unreachable")
    raise exc


def _client() -> ComfyuiClient:
    return ComfyuiClient(get_config().comfyui_url)


async def _missing_for(client: ComfyuiClient, required: dict) -> Optional[dict]:
    try:
        return await jobs.missing_models(client, required)
    except ComfyuiError:
        return None


@router.get("/comfyui/templates")
async def comfyui_templates():
    """Shipped and imported templates. ``missing_models`` is null while ComfyUI cannot be asked."""
    templates = await asyncio.to_thread(load_templates)
    client = _client()
    reachable = (await client.queue()).reachable
    missing = (
        await asyncio.gather(*(_missing_for(client, t.required_models) for t in templates))
        if reachable
        else [None] * len(templates)
    )
    return {
        "comfyui_reachable": reachable,
        "templates": [{**t.summary(), "missing_models": m} for t, m in zip(templates, missing)],
    }


# What ComfyUI offers changes only when it is updated or custom nodes are added, so a short cache is plenty.
SAMPLERS_TTL_S = 60.0
# Used while ComfyUI cannot be asked; the frontend keeps the same list for a backend that predates the route.
FALLBACK_SAMPLERS = [
    "euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "dpmpp_3m_sde",
    "uni_pc", "lcm", "ddim",
]
FALLBACK_SCHEDULERS = [
    "simple", "normal", "karras", "exponential", "sgm_uniform", "beta", "ddim_uniform", "linear_quadratic",
]
# (monotonic expiry, ComfyUI URL, samplers, schedulers) of the last live answer.
_samplers_cache: Optional[tuple[float, str, list[str], list[str]]] = None


@router.get("/comfyui/samplers")
async def comfyui_samplers():
    """The sampler and scheduler names ComfyUI's KSampler accepts (cached briefly), or the built-in lists
    with ``source: "fallback"`` while ComfyUI cannot be asked."""
    global _samplers_cache
    url = get_config().comfyui_url
    cached = _samplers_cache
    if cached is not None and cached[0] > time.monotonic() and cached[1] == url:
        return {"samplers": cached[2], "schedulers": cached[3], "source": "comfyui"}
    try:
        samplers, schedulers = await _client().sampler_options()
    except ComfyuiError:
        return {"samplers": FALLBACK_SAMPLERS, "schedulers": FALLBACK_SCHEDULERS, "source": "fallback"}
    _samplers_cache = (time.monotonic() + SAMPLERS_TTL_S, url, samplers, schedulers)
    return {"samplers": samplers, "schedulers": schedulers, "source": "comfyui"}


@router.post("/comfyui/templates/import")
async def comfyui_import(body: ComfyImportRequest):
    """Validate an API-format graph, detect its parameter inputs, check its nodes against ComfyUI when it
    is up, and store it as a user template (``user:<slug>``)."""
    try:
        graph = validate_api_graph(copy.deepcopy(body.graph))
        detect_slots(copy.deepcopy(graph))
    except ComfyTemplateError as exc:
        raise _map_error(exc) from exc
    client = _client()
    checked = False
    if (await client.queue()).reachable:
        try:
            known = await client.object_info()
        except ComfyuiError:
            known = None
        if known is not None:
            checked = True
            unknown = [c for c in graph_class_types(graph) if c not in known]
            if unknown:
                raise _fail(
                    422,
                    f"ComfyUI does not have these nodes: {', '.join(unknown)}. Install them or remove them from the graph.",
                    "graph",
                )
    try:
        template = await asyncio.to_thread(save_user_template, body.name, graph)
    except ComfyTemplateError as exc:
        raise _map_error(exc) from exc
    return {"template": template.summary(), "slot_targets": template.slots, "checked_nodes": checked}


@router.delete("/comfyui/templates/{template_id}")
async def comfyui_delete_template(template_id: str):
    if not template_id.startswith("user:"):
        raise _fail(400, "Shipped templates cannot be deleted.", "template")
    if not await asyncio.to_thread(delete_user_template, template_id):
        raise _fail(404, "Template not found.", "template")
    return {"deleted": template_id}


@router.post("/comfyui/generate", response_model = ComfyGenerateResponse)
async def comfyui_generate(body: ComfyGenerateRequest):
    params = body.model_dump(exclude = {"template_id"})
    try:
        result = await jobs.get_runner().run(body.template_id, params)
    except Exception as exc:
        raise _map_error(exc) from exc
    return ComfyGenerateResponse(
        images = [GalleryImage(**record) for record in result.images],
        actions = result.actions,
        prompt_id = result.prompt_id,
        seed = result.resolved["seed"],
    )


@router.get("/comfyui/progress", response_model = ComfyProgressResponse)
async def comfyui_progress():
    return ComfyProgressResponse(**jobs.get_runner().progress())


@router.post("/comfyui/generate/cancel")
async def comfyui_generate_cancel():
    try:
        return {"cancelled": await jobs.get_runner().cancel()}
    except Exception as exc:
        raise _map_error(exc) from exc
