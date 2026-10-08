# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/engines/attached/comfyui/{templates,generate,progress,generate/cancel}: validation, error mapping, gating."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.attached_comfyui as gen_routes  # noqa: E402
import routes.attached_engines as routes  # noqa: E402
from auth import policy  # noqa: E402
from auth.authentication import get_current_subject  # noqa: E402
from core.inference.attached import arbiter, comfyui_jobs as jobs  # noqa: E402
from core.inference.attached.arbiter import ArbiterResult, AttachedAdmissionError  # noqa: E402
from core.inference.attached.comfyui_client import ComfyGraphError, ComfyuiClient, ComfyuiError  # noqa: E402
from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG  # noqa: E402
from core.inference.attached.omlx_client import OmlxClient  # noqa: E402
from dataclasses import replace  # noqa: E402

from utils.attached_engines_settings import DEFAULT_CONFIG  # noqa: E402

from .test_attached_comfyui_jobs import PID, FakeComfy, FakeConnect, FakeWS, _no_sleep, msg  # noqa: E402
from .test_attached_comfyui_graphs import (  # noqa: E402
    IMG2IMG_GRAPH,
    PICTUREPRESS_EDIT,
    PLACEHOLDER_GRAPH,
    SD_GRAPH,
    STORYPRESS_REFERENCE,
)

BASE = "/api/engines/attached/comfyui"
TEMPLATE = "qwen-image-2.1-t2i"
BODY = {"template_id": TEMPLATE, "prompt": "a red fox", "seed": 7, "steps": 4, "width": 512, "height": 512}


# The shape of ComfyUI's /object_info/KSampler: input.required.<name> = [choices, options].
KSAMPLER_INFO = {"KSampler": {"input": {"required": {
    "model": ["MODEL", {}],
    "sampler_name": [["euler", "res_multistep", "er_sde"], {}],
    "scheduler": [["simple", "kl_optimal"], {}],
}}}}


class RouteComfy(FakeComfy):
    def __init__(self):
        super().__init__()
        self.object_info = {c: {} for c in (
            "UNETLoader", "CLIPLoader", "VAELoader", "TextEncodeQwenImage21", "EmptyLatentImage", "KSampler",
            "VAEDecode", "SaveImage", "PreviewImage", "CheckpointLoaderSimple", "CLIPTextEncode",
            "LoadImage", "ImageScale", "VAEEncode")}

    ksampler_info: dict | None = None

    def handle(self, request):
        if request.url.path == "/object_info/KSampler":
            self.calls.append((request.method, request.url.path))
            if not self.up:
                raise httpx.ConnectError("refused", request = request)
            return httpx.Response(200, json = self.ksampler_info if self.ksampler_info is not None else KSAMPLER_INFO)
        if request.url.path == "/object_info":
            if not self.up:
                raise httpx.ConnectError("refused", request = request)
            return httpx.Response(200, json = self.object_info)
        return super().handle(request)


@pytest.fixture
def world(monkeypatch, tmp_path):
    fake = RouteComfy()
    config = replace(DEFAULT_CONFIG, enabled = True, comfyui_url = "http://127.0.0.1:18845")
    state = type("State", (), {})()
    state.fake, state.config, state.admissions, state.scheduled = fake, {"value": config}, 0, 0
    state.admission = ArbiterResult(acted = True, actions = ("Unloaded oMLX: swift",))
    state.ws = lambda: FakeWS([msg("execution_start"), msg("executing", node = "6"),
                               msg("progress", value = 2, max = 4, node = "6"), msg("execution_success")])

    for module in (routes, gen_routes, jobs):
        monkeypatch.setattr(module, "get_config", lambda: state.config["value"])
    for module in (gen_routes, jobs):
        monkeypatch.setattr(module, "ComfyuiClient", lambda url, transport = None: ComfyuiClient(url, transport = fake.transport))
    monkeypatch.setattr(routes, "ComfyuiClient", lambda url, transport = None: ComfyuiClient(url, transport = fake.transport))
    def refuse(request):
        raise httpx.ConnectError("no oMLX in tests", request = request)

    monkeypatch.setattr(routes, "OmlxClient", lambda url: OmlxClient(url, transport = httpx.MockTransport(refuse)))
    monkeypatch.setattr(gen_routes, "_samplers_cache", None)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_COMFYUI_TEMPLATES_DIR", str(tmp_path / "templates"))
    monkeypatch.setenv("UNSLOTH_ENGINES_HOME", str(tmp_path / "engines"))

    async def admit(hold = None):
        state.admissions += 1
        return state.admission

    monkeypatch.setattr(arbiter, "before_comfyui_job", admit)
    monkeypatch.setattr(arbiter, "schedule_comfyui_idle_free", lambda: setattr(state, "scheduled", state.scheduled + 1) or True)

    def fresh_runner():
        runner = jobs.ComfyJobRunner(connect = FakeConnect(state.ws()), sleep = _no_sleep)
        runner.poll_interval = 0.0
        monkeypatch.setattr(jobs, "_runner", runner)
        return runner

    state.fresh_runner = fresh_runner
    fresh_runner()
    return state


@pytest.fixture
def client(world):
    app = FastAPI()
    app.include_router(routes.router, prefix = "/api/engines/attached")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    with TestClient(app) as test_client:
        yield test_client


def url(path):
    return f"{BASE}{path}"


def finish_on_submit(world):
    original = world.fake.handle

    def handle(request):
        response = original(request)
        if request.url.path == "/prompt":
            world.fake.finish()
        return response

    world.fake.handle = handle


# ------------------------------------------------------------------- gating


def test_new_routes_are_404_when_the_flag_is_off(client, world):
    world.config["value"] = replace(DEFAULT_CONFIG, enabled = False)
    for method, path, body in (
        ("get", "/templates", None), ("post", "/templates/import", {"name": "x", "graph": {}}),
        ("delete", "/templates/user:x", None), ("post", "/generate", BODY),
        ("get", "/progress", None), ("post", "/generate/cancel", None), ("get", "/samplers", None),
    ):
        assert client.request(method.upper(), url(path), json = body).status_code == 404, path
    assert world.fake.calls == []


def test_new_routes_require_owner(world):
    app = FastAPI()
    app.include_router(routes.router, prefix = "/api/engines/attached")

    async def not_owner():
        raise HTTPException(status_code = 403, detail = "Only the installation owner can do this")

    app.dependency_overrides[get_current_subject] = lambda: "member"
    app.dependency_overrides[policy.require_owner] = not_owner
    with TestClient(app) as test_client:
        for method, path in (("get", "/templates"), ("post", "/generate"), ("get", "/progress"),
                             ("post", "/generate/cancel"), ("post", "/templates/import"), ("delete", "/templates/user:x"),
                             ("get", "/samplers")):
            assert test_client.request(method.upper(), url(path)).status_code == 403, path
    assert world.fake.calls == [] and world.admissions == 0


# ----------------------------------------------------------------- generate


def test_generate_success_returns_gallery_images_and_the_seed(client, world):
    finish_on_submit(world)
    response = client.post(url("/generate"), json = BODY)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prompt_id"] == PID and body["seed"] == 7 and body["actions"] == ["Unloaded oMLX: swift"]
    [image] = body["images"]
    assert image["engine"] == "comfyui" and image["comfyui_template"] == TEMPLATE
    assert image["prompt"] == "a red fox" and image["url"].startswith("/api/inference/images/gallery/")
    sent = world.fake.submitted[0]["prompt"]
    assert sent["6"]["inputs"]["steps"] == 4 and sent["5"]["inputs"]["width"] == 512
    listing = client.get("/api/engines/attached/status")  # the status block reports the job is over
    assert listing.status_code == 200 and listing.json()["comfyui"]["job_active"] is False


def test_generate_forwards_loras_and_random_seed(client, world):
    finish_on_submit(world)
    response = client.post(url("/generate"), json = {
        **BODY, "seed": None, "loras": [{"name": "a.safetensors", "strength": 0.6}], "negative_prompt": "blur"})
    assert response.status_code == 200, response.text
    assert 0 <= response.json()["seed"] <= 2**53 - 1
    sent = world.fake.submitted[0]["prompt"]
    assert sent["unsloth_lora_0"]["inputs"]["strength_model"] == 0.6
    assert sent["4"]["inputs"]["negative_prompt"] == "blur"


@pytest.mark.parametrize(
    "patch",
    [{"prompt": ""}, {"prompt": "x" * 8001}, {"batch_size": 5}, {"batch_size": 0}, {"seed": -2}, {"steps": 0},
     {"cfg": -1}, {"width": 10}, {"loras": [{"name": "a", "strength": 1}] * 5},
     {"loras": [{"name": "a", "strength": 9}]}, {"loras": [{"name": "", "strength": 1}]}, {"template_id": ""}],
)
def test_generate_validates_the_request(client, world, patch):
    assert client.post(url("/generate"), json = {**BODY, **patch}).status_code == 422
    assert world.fake.calls == []


def test_generate_unknown_field_values_that_the_template_rejects_are_422(client, world):
    response = client.post(url("/generate"), json = {**BODY, "sampler": "   "})
    assert response.status_code == 422 and response.headers["X-Comfy-Error"] == "params"
    assert world.fake.calls == []


class Raises:
    def __init__(self, exc):
        self.exc = exc

    def is_running(self):
        return False

    async def run(self, *args, **kwargs):
        raise self.exc


@pytest.mark.parametrize(
    "exc, status, code",
    [
        (jobs.ComfyBusyError("A ComfyUI job is already running."), 409, "busy"),
        (jobs.ComfyOffError(jobs.OFF_MESSAGE), 503, "off"),
        (jobs.ComfyNotRunningError("ComfyUI is not running yet"), 502, "not_running"),
        (jobs.ComfyTemplateNotFound("Unknown ComfyUI template 'x'."), 404, "template"),
        (jobs.ComfyModelsMissing({"vae": ["v.safetensors"]}, ["/models"]), 422, "missing_models"),
        (ComfyGraphError("Prompt outputs failed validation", {}), 422, "graph"),
        (jobs.ComfyJobError("out of memory", "KSampler"), 502, "execution"),
        (jobs.ComfyCancelled(DIFFUSION_CANCELLED_MSG), 409, "cancelled"),
        (jobs.ComfyTimeout("too slow"), 504, "timeout"),
        (AttachedAdmissionError("Studio has a local model loaded; unload it."), 503, "admission"),
        (ComfyuiError("ComfyUI is unreachable (ReadTimeout)"), 502, "unreachable"),
    ],
)
def test_errors_map_to_statuses_and_codes(client, monkeypatch, exc, status, code):
    monkeypatch.setattr(jobs, "_runner", Raises(exc))
    response = client.post(url("/generate"), json = BODY)
    assert response.status_code == status and response.headers["X-Comfy-Error"] == code
    if code == "admission":
        assert response.headers["Retry-After"] == "15" and "local model" in response.json()["detail"]
    else:
        assert "Retry-After" not in response.headers
    if code == "cancelled":
        assert response.json()["detail"] == DIFFUSION_CANCELLED_MSG
    if code == "unreachable":
        assert response.json()["detail"] == "ComfyUI is not reachable."


def test_off_through_the_real_runner_names_the_setting(client, world):
    world.fake.up = False
    response = client.post(url("/generate"), json = BODY)
    assert response.status_code == 503 and response.headers["X-Comfy-Error"] == "off"
    assert response.json()["detail"] == "ComfyUI is off, enable it in Settings > Engines."
    assert world.admissions == 0


def test_admission_refusal_through_the_real_runner(client, world):
    world.admission = ArbiterResult(skipped = "error", error = "Cannot free shared memory for a ComfyUI job: x")
    response = client.post(url("/generate"), json = BODY)
    assert response.status_code == 503 and response.headers["Retry-After"] == "15"
    assert world.fake.submitted == []


def test_missing_models_through_the_real_runner(client, world):
    world.fake.models["vae"] = []
    response = client.post(url("/generate"), json = BODY)
    assert response.status_code == 422 and response.headers["X-Comfy-Error"] == "missing_models"
    assert "vae/qwen_image_2.1_vae_bf16.safetensors" in response.json()["detail"]


def test_execution_error_through_the_real_runner(client, world):
    world.ws = lambda: FakeWS([msg("execution_error", exception_message = "boom", node_type = "KSampler")])
    world.fresh_runner()
    response = client.post(url("/generate"), json = BODY)
    assert response.status_code == 502 and "KSampler: boom" in response.json()["detail"]


def test_interrupt_through_the_real_runner_is_the_cancelled_conflict(client, world):
    world.ws = lambda: FakeWS([msg("execution_interrupted")])
    world.fresh_runner()
    response = client.post(url("/generate"), json = BODY)
    assert response.status_code == 409 and response.json()["detail"] == DIFFUSION_CANCELLED_MSG


# -------------------------------------------------------- progress and cancel


def test_progress_when_idle_has_the_diffusion_shape_plus_comfyui_fields(client):
    body = client.get(url("/progress")).json()
    assert body == {
        "active": False, "step": 0, "total_steps": 0, "fraction": 0.0, "eta_seconds": None, "phase": None,
        "preview": None, "preview_seq": 0, "queue_position": None, "prompt_id": None, "engine": "comfyui",
    }


def test_progress_reports_a_running_job(client, monkeypatch):
    class Running:
        def progress(self):
            return {"active": True, "step": 5, "total_steps": 25, "fraction": 0.2, "eta_seconds": 40.0,
                    "phase": "denoise", "preview": None, "preview_seq": 0, "queue_position": 0,
                    "prompt_id": PID, "engine": "comfyui"}

    monkeypatch.setattr(jobs, "_runner", Running())
    body = client.get(url("/progress")).json()
    assert (body["active"], body["step"], body["total_steps"], body["phase"], body["queue_position"]) == (True, 5, 25, "denoise", 0)


def test_cancel_route(client, monkeypatch):
    assert client.post(url("/generate/cancel")).json() == {"cancelled": False}

    class Active:
        async def cancel(self):
            return True

    monkeypatch.setattr(jobs, "_runner", Active())
    assert client.post(url("/generate/cancel")).json() == {"cancelled": True}


# ------------------------------------------------------------------ samplers


def test_samplers_come_from_comfyui_object_info(client, world):
    body = client.get(url("/samplers")).json()
    assert body == {"samplers": ["euler", "res_multistep", "er_sde"], "schedulers": ["simple", "kl_optimal"], "source": "comfyui"}
    assert world.fake.count("GET", "/object_info/KSampler") == 1


def test_samplers_are_cached_briefly_then_refetched(client, world):
    client.get(url("/samplers"))
    world.fake.ksampler_info = {"KSampler": {"input": {"required": {
        "sampler_name": [["euler", "new_one"], {}], "scheduler": [["simple"], {}]}}}}
    assert client.get(url("/samplers")).json()["samplers"] == ["euler", "res_multistep", "er_sde"]
    assert world.fake.count("GET", "/object_info/KSampler") == 1
    expiry, *rest = gen_routes._samplers_cache
    gen_routes._samplers_cache = (0.0, *rest)
    assert client.get(url("/samplers")).json()["samplers"] == ["euler", "new_one"]
    assert world.fake.count("GET", "/object_info/KSampler") == 2


def test_samplers_fall_back_to_the_static_lists_when_comfyui_is_down(client, world):
    world.fake.up = False
    body = client.get(url("/samplers")).json()
    assert body["source"] == "fallback"
    assert body["samplers"] == gen_routes.FALLBACK_SAMPLERS and "euler" in body["samplers"]
    assert body["schedulers"] == gen_routes.FALLBACK_SCHEDULERS and "simple" in body["schedulers"]
    # Not cached: the live list is used as soon as ComfyUI is back.
    world.fake.up = True
    assert client.get(url("/samplers")).json()["source"] == "comfyui"


@pytest.mark.parametrize("payload", [{}, {"KSampler": {"input": {"required": {}}}},
                                     {"KSampler": {"input": {"required": {"sampler_name": [[], {}], "scheduler": [["a"], {}]}}}},
                                     {"KSampler": {"input": {"required": {"sampler_name": ["COMBO", {}], "scheduler": [["a"], {}]}}}}])
def test_samplers_fall_back_on_an_unexpected_payload(client, world, payload):
    world.fake.ksampler_info = payload
    assert client.get(url("/samplers")).json()["source"] == "fallback"
    assert gen_routes._samplers_cache is None


# ---------------------------------------------------------------- templates


def test_templates_list_shipped_with_missing_models(client, world):
    body = client.get(url("/templates")).json()
    assert body["comfyui_reachable"] is True
    template, uncensored, img2img, edit = body["templates"]
    assert (img2img["id"], img2img["kind"]) == ("qwen-image-2.1-img2img", "img2img")
    assert (edit["id"], edit["kind"]) == ("qwen-image-2.1-edit", "edit")
    assert uncensored["id"] == "qwen-image-2.1-t2i-uncensored" and uncensored["name"] == "Qwen-Image 2.1 (uncensored)"
    assert uncensored["supports_lora"] is True
    assert uncensored["missing_models"] == {"loras": ["qwen-image-2.1-uncensored-lora.safetensors"]}
    assert template["id"] == TEMPLATE and template["source"] == "shipped" and template["kind"] == "t2i"
    assert template["missing_models"] == {} and template["supports_lora"] is True
    assert template["defaults"]["steps"] == 25 and "prompt" in template["slots"] and "seed" in template["slots"]
    assert template["limits"]["multiple"] == 16
    assert template["image_slots"] == [] and img2img["missing_models"] == {} and edit["missing_models"] == {}
    world.fake.models["vae"] = []
    missing = client.get(url("/templates")).json()["templates"][0]["missing_models"]
    assert missing == {"vae": ["qwen_image_2.1_vae_bf16.safetensors"]}


def test_templates_list_while_comfyui_is_down(client, world):
    world.fake.up = False
    body = client.get(url("/templates")).json()
    assert body["comfyui_reachable"] is False and body["templates"][0]["missing_models"] is None


def test_import_list_and_delete_a_user_graph(client, world):
    response = client.post(url("/templates/import"), json = {"name": "My SD", "graph": SD_GRAPH})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["template"]["id"] == "user:my-sd" and body["checked_nodes"] is True
    assert body["slot_targets"]["prompt"] == [{"node": "6", "input": "text"}]
    ids = [t["id"] for t in client.get(url("/templates")).json()["templates"]]
    assert ids == [TEMPLATE, "qwen-image-2.1-t2i-uncensored", "qwen-image-2.1-img2img", "qwen-image-2.1-edit", "user:my-sd"]
    assert client.delete(url("/templates/user:my-sd")).json() == {"deleted": "user:my-sd"}
    assert client.delete(url("/templates/user:my-sd")).status_code == 404
    shipped = client.delete(url(f"/templates/{TEMPLATE}"))
    assert shipped.status_code == 400 and "Shipped" in shipped.json()["detail"]


def test_imported_graph_generates(client, world):
    client.post(url("/templates/import"), json = {"name": "My SD", "graph": SD_GRAPH})
    world.fake.models["checkpoints"] = ["sd.safetensors"]
    finish_on_submit(world)
    response = client.post(url("/generate"), json = {"template_id": "user:my-sd", "prompt": "a dog", "seed": 1})
    assert response.status_code == 200, response.text
    sent = world.fake.submitted[0]["prompt"]
    assert sent["6"]["inputs"]["text"] == "a dog" and sent["9"]["class_type"] == "PreviewImage"


def test_import_accepts_placeholder_graphs(client):
    response = client.post(url("/templates/import"), json = {"name": "SP", "graph": PLACEHOLDER_GRAPH})
    assert response.status_code == 200 and response.json()["template"]["supports_lora"] is True


def test_import_rejections(client, world):
    ui_format = {"nodes": [], "links": [], "version": 0.4}
    no_prompt = copy.deepcopy(SD_GRAPH)
    no_prompt["3"]["inputs"]["positive"] = ["4", 0]
    for graph, text in ((ui_format, "Export (API)"), ({"1": {"class_type": "X"}}, "no inputs"),
                        ({"1": {"class_type": "X", "inputs": {"a": ["9", 0]}}}, "missing node")):
        response = client.post(url("/templates/import"), json = {"name": "bad", "graph": graph})
        assert response.status_code == 422 and text in response.json()["detail"]
    assert client.post(url("/templates/import"), json = {"name": "bad", "graph": {}}).status_code == 422
    assert client.post(url("/templates/import"), json = {"name": "", "graph": SD_GRAPH}).status_code == 422
    assert world.fake.calls == []  # a graph that is invalid on its face never reaches ComfyUI


def test_import_checks_node_types_against_comfyui(client, world):
    del world.fake.object_info["CLIPTextEncode"]
    response = client.post(url("/templates/import"), json = {"name": "SD", "graph": SD_GRAPH})
    assert response.status_code == 422 and "CLIPTextEncode" in response.json()["detail"]
    assert client.get(url("/templates")).json()["templates"][-1]["id"] == "qwen-image-2.1-edit"  # nothing stored


def test_import_without_comfyui_skips_the_node_check(client, world):
    world.fake.up = False
    response = client.post(url("/templates/import"), json = {"name": "SD", "graph": SD_GRAPH})
    assert response.status_code == 200 and response.json()["checked_nodes"] is False


def test_status_block_reports_job_active(client, world):
    assert client.get("/api/engines/attached/status").json()["comfyui"]["job_active"] is False


def test_status_block_carries_the_idle_free_deadline(client, world, monkeypatch):
    idle = client.get("/api/engines/attached/status").json()["comfyui"]
    assert idle["idle_free_at"] is None and idle["idle_free_in_s"] is None
    monkeypatch.setattr(arbiter, "comfyui_idle_free_at", lambda: 1_700_000_300.0)
    monkeypatch.setattr("routes.attached_engines.time.time", lambda: 1_700_000_048.0)
    block = client.get("/api/engines/attached/status").json()["comfyui"]
    assert block["idle_free_at"] == 1_700_000_300.0 and block["idle_free_in_s"] == 252.0


# ------------------------------------------------------------- image inputs

IMG2IMG = "qwen-image-2.1-img2img"
EDIT = "qwen-image-2.1-edit"


def data_url(size = (40, 30)) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (30, 120, 220)).save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image_body(template_id = IMG2IMG, **extra) -> dict:
    return {**BODY, "template_id": template_id, "input_images": {"image": {"data": data_url()}}, **extra}


@pytest.mark.parametrize(
    "patch",
    [
        {"input_images": {"image": {}}},
        {"input_images": {"image": {"data": "abc", "gallery_id": "a" * 32}}},
        {"input_images": {"image": {"gallery_id": "../etc/passwd"}}},
        {"input_images": {"image": {"gallery_id": ""}}},
        {"input_images": {"image": {"gallery_id": "x" * 129}}},
        {"input_images": {f"image_{i}": {"data": "abc"} for i in range(5)}},
        {"input_images": {"image": "not an object"}},
        {"denoise": 0}, {"denoise": 1.5}, {"denoise": -0.1},
        {"reference_resolution": -1}, {"reference_resolution": 4097},
    ],
)
def test_generate_validates_image_inputs(client, world, patch):
    assert client.post(url("/generate"), json = {**image_body(), **patch}).status_code == 422
    assert world.fake.calls == []


def test_oversized_inputs_are_refused(client, world, monkeypatch):
    one_too_big = {"image": {"data": "A" * (gen_routes.MAX_INPUT_DATA_CHARS + 1)}}
    assert client.post(url("/generate"), json = {**image_body(), "input_images": one_too_big}).status_code == 422
    monkeypatch.setattr(gen_routes, "MAX_TOTAL_INPUT_DATA_CHARS", 100)
    two = {"image": {"data": "A" * 60}, "image_2": {"data": "A" * 60}}
    assert client.post(url("/generate"), json = {**image_body(), "input_images": two}).status_code == 422
    assert world.fake.calls == []


def test_generate_with_an_image_returns_the_recipe_with_inputs(client, world):
    finish_on_submit(world)
    response = client.post(url("/generate"), json = image_body(denoise = 0.5))
    assert response.status_code == 200, response.text
    [image] = response.json()["images"]
    assert (image["workflow"], image["strength"], image["comfyui_inputs"]) == ("img2img", 0.5, {"image": None})
    assert image["comfyui_template"] == IMG2IMG and image["reference_resolution"] is None
    sent = world.fake.submitted[0]["prompt"]
    assert sent["5"]["inputs"]["image"].startswith("unsloth-inputs/") and sent["8"]["inputs"]["denoise"] == 0.5
    assert len(world.fake.uploads) == 1


def test_generate_an_edit_with_a_gallery_image(client, world):
    from core.inference import image_gallery
    from PIL import Image

    made = image_gallery.save(
        Image.new("RGB", (64, 48)), {"prompt": "fox", "width": 64, "height": 48, "steps": 1, "guidance": 1.0, "seed": 1, "created_at": 1.0}
    )
    finish_on_submit(world)
    body = {**BODY, "template_id": EDIT, "input_images": {"image": {"gallery_id": made["id"]}}, "reference_resolution": 1000}
    response = client.post(url("/generate"), json = body)
    assert response.status_code == 200, response.text
    [image] = response.json()["images"]
    assert (image["workflow"], image["reference_resolution"], image["strength"]) == ("edit", 992, None)
    assert image["comfyui_inputs"] == {"image": made["id"]}


def test_generate_image_errors_are_params_errors_before_admission(client, world):
    missing = client.post(url("/generate"), json = {**BODY, "template_id": IMG2IMG})
    assert missing.status_code == 422 and missing.headers["X-Comfy-Error"] == "params"
    assert "needs an input image" in missing.json()["detail"]
    unknown = client.post(url("/generate"), json = {**BODY, "input_images": {"image": {"data": data_url()}}})
    assert unknown.status_code == 422 and unknown.headers["X-Comfy-Error"] == "params"
    gone = client.post(url("/generate"), json = {**BODY, "template_id": IMG2IMG, "input_images": {"image": {"gallery_id": "nothere"}}})
    assert gone.status_code == 422 and gone.headers["X-Comfy-Error"] == "params" and "no longer exists" in gone.json()["detail"]
    junk = client.post(url("/generate"), json = image_body() | {"input_images": {"image": {"data": "%%%"}}})
    assert junk.status_code == 422 and junk.headers["X-Comfy-Error"] == "params"
    assert world.admissions == 0 and world.fake.uploads == []


def test_templates_list_the_image_templates_with_kinds_and_slots(client, world):
    listed = {t["id"]: t for t in client.get(url("/templates")).json()["templates"]}
    assert [t["kind"] for t in listed.values()] == ["t2i", "t2i", "img2img", "edit"]
    img2img, edit, t2i = listed[IMG2IMG], listed[EDIT], listed[TEMPLATE]
    assert img2img["image_slots"] == edit["image_slots"] == [{"name": "image", "label": "Input image", "required": True}]
    assert t2i["image_slots"] == []
    assert "denoise" in img2img["slots"] and "width" in img2img["slots"] and img2img["defaults"]["denoise"] == 0.6
    assert "reference_resolution" in edit["slots"] and "width" not in edit["slots"]
    assert edit["defaults"]["reference_resolution"] == 1024 and edit["limits"]["batch_size"] == [1, 1]
    assert img2img["limits"]["denoise"] == [0.01, 1.0] and edit["limits"]["reference_multiple"] == 32


def test_import_accepts_image_graphs_and_reports_their_slots(client, world):
    storypress = client.post(url("/templates/import"), json = {"name": "StoryPress ref", "graph": json.loads(STORYPRESS_REFERENCE)})
    assert storypress.status_code == 200, storypress.text
    body = storypress.json()
    assert body["template"]["kind"] == "edit" and body["checked_nodes"] is True
    assert body["template"]["image_slots"] == [{"name": "image", "label": "Input image 1", "required": True}]
    assert body["template"]["defaults"]["reference_resolution"] == 0
    picturepress = client.post(url("/templates/import"), json = {"name": "PicturePress edit", "graph": json.loads(PICTUREPRESS_EDIT)})
    assert picturepress.status_code == 200, picturepress.text
    template = picturepress.json()["template"]
    assert template["kind"] == "edit" and template["defaults"]["reference_resolution"] == 1024
    assert "reference_resolution" in template["slots"] and len(template["image_slots"]) == 1
    heuristic = client.post(url("/templates/import"), json = {"name": "Photo", "graph": IMG2IMG_GRAPH})
    assert heuristic.status_code == 200, heuristic.text
    assert heuristic.json()["template"]["kind"] == "img2img" and "denoise" in heuristic.json()["template"]["slots"]


def test_import_refuses_masks_and_too_many_images(client, world):
    masked = {**SD_GRAPH, "20": {"class_type": "LoadImageMask", "inputs": {"image": "m.png", "channel": "alpha"}}}
    response = client.post(url("/templates/import"), json = {"name": "masked", "graph": masked})
    assert response.status_code == 422 and "Mask" in response.json()["detail"]
    many = {**SD_GRAPH, **{str(20 + i): {"class_type": "LoadImage", "inputs": {"image": "x.png"}} for i in range(5)}}
    response = client.post(url("/templates/import"), json = {"name": "many", "graph": many})
    assert response.status_code == 422 and "at most 4" in response.json()["detail"]


def test_an_imported_image_graph_then_generates(client, world):
    imported = client.post(url("/templates/import"), json = {"name": "PicturePress edit", "graph": json.loads(PICTUREPRESS_EDIT)})
    template_id = imported.json()["template"]["id"]
    assert template_id == "user:picturepress-edit"
    finish_on_submit(world)
    response = client.post(url("/generate"), json = image_body(template_id, reference_resolution = 768))
    assert response.status_code == 200, response.text
    [image] = response.json()["images"]
    assert (image["workflow"], image["reference_resolution"], image["comfyui_template"]) == ("edit", 768, template_id)
    graph = world.fake.submitted[0]["prompt"]
    assert graph["4"]["inputs"]["image"].startswith("unsloth-inputs/") and graph["5"]["inputs"]["resolution"] == 768
