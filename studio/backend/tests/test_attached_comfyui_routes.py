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
from .test_attached_comfyui_graphs import PLACEHOLDER_GRAPH, SD_GRAPH  # noqa: E402

BASE = "/api/engines/attached/comfyui"
TEMPLATE = "qwen-image-2.1-t2i"
BODY = {"template_id": TEMPLATE, "prompt": "a red fox", "seed": 7, "steps": 4, "width": 512, "height": 512}


class RouteComfy(FakeComfy):
    def __init__(self):
        super().__init__()
        self.object_info = {c: {} for c in (
            "UNETLoader", "CLIPLoader", "VAELoader", "TextEncodeQwenImage21", "EmptyLatentImage", "KSampler",
            "VAEDecode", "SaveImage", "PreviewImage", "CheckpointLoaderSimple", "CLIPTextEncode")}

    def handle(self, request):
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
        ("get", "/progress", None), ("post", "/generate/cancel", None),
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
                             ("post", "/generate/cancel"), ("post", "/templates/import"), ("delete", "/templates/user:x")):
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


# ---------------------------------------------------------------- templates


def test_templates_list_shipped_with_missing_models(client, world):
    body = client.get(url("/templates")).json()
    assert body["comfyui_reachable"] is True
    template, uncensored = body["templates"]
    assert uncensored["id"] == "qwen-image-2.1-t2i-uncensored" and uncensored["name"] == "Qwen-Image 2.1 (uncensored)"
    assert uncensored["supports_lora"] is True
    assert uncensored["missing_models"] == {"loras": ["qwen-image-2.1-uncensored-lora.safetensors"]}
    assert template["id"] == TEMPLATE and template["source"] == "shipped" and template["kind"] == "t2i"
    assert template["missing_models"] == {} and template["supports_lora"] is True
    assert template["defaults"]["steps"] == 25 and "prompt" in template["slots"] and "seed" in template["slots"]
    assert template["limits"]["multiple"] == 16
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
    assert ids == [TEMPLATE, "qwen-image-2.1-t2i-uncensored", "user:my-sd"]
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
    assert client.get(url("/templates")).json()["templates"][-1]["id"] == "qwen-image-2.1-t2i-uncensored"  # nothing stored


def test_import_without_comfyui_skips_the_node_check(client, world):
    world.fake.up = False
    response = client.post(url("/templates/import"), json = {"name": "SD", "graph": SD_GRAPH})
    assert response.status_code == 200 and response.json()["checked_nodes"] is False


def test_status_block_reports_job_active(client, world):
    assert client.get("/api/engines/attached/status").json()["comfyui"]["job_active"] is False
