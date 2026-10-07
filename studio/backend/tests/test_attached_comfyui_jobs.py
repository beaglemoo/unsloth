# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ComfyUI job runner against a fake ComfyUI (httpx.MockTransport) and a fake websocket."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached import arbiter, comfyui_jobs as jobs
from core.inference.attached.arbiter import AttachedAdmissionError, ArbiterResult
from core.inference.attached.comfyui_client import ComfyGraphError, ComfyuiClient, ComfyuiError
from core.inference.attached.comfyui_graphs import ComfyParamError
from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
from models.inference import GalleryImage
from utils.attached_engines_settings import DEFAULT_CONFIG

PID = "pid1"
REQUIRED = {
    "diffusion_models": ["qwen_image_2.1_bf16.safetensors"],
    "text_encoders": ["qwen3vl_8b_bf16.safetensors"],
    "vae": ["qwen_image_2.1_vae_bf16.safetensors"],
}


def png(size = (64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format = "PNG")
    return buf.getvalue()


class FakeComfy:
    """A ComfyUI that answers the endpoints the runner uses."""

    def __init__(self):
        self.up = True
        self.models = {**{k: list(v) for k, v in REQUIRED.items()}, "loras": ["a.safetensors"]}
        self.running: list[str] = []
        self.pending: list[str] = []
        self.history: dict[str, dict] = {}
        self.images = [{"filename": "ComfyUI_temp_00001_.png", "subfolder": "", "type": "temp"}]
        self.submit_status = 200
        self.submit_body: dict | None = None
        self.submitted: list[dict] = []
        self.calls: list[tuple[str, str]] = []
        self.history_polls = 0
        self.finish_after_polls: int | None = None
        self.queue_gone = False

    def finish(self, pid = PID, *, status = None):
        self.running = [i for i in self.running if i != pid]
        self.history[pid] = {
            "outputs": {"8": {"images": self.images}},
            "status": status or {"status_str": "success", "completed": True, "messages": []},
        }

    @property
    def transport(self):
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        self.calls.append((method, path))
        if not self.up:
            raise httpx.ConnectError("refused", request = request)
        if path == "/queue" and method == "GET":
            item = lambda i, p: [i, p, {}, {}, []]  # noqa: E731
            return httpx.Response(200, json = {
                "queue_running": [item(i, p) for i, p in enumerate(self.running)],
                "queue_pending": [item(10 + i, p) for i, p in enumerate(self.pending)],
            })
        if path == "/queue" and method == "POST":
            body = json.loads(request.content)
            self.pending = [p for p in self.pending if p not in body.get("delete", [])]
            return httpx.Response(200)
        if path.startswith("/models/"):
            return httpx.Response(200, json = self.models.get(path.split("/")[-1], []))
        if path == "/prompt":
            body = json.loads(request.content)
            self.submitted.append(body)
            if self.submit_status != 200:
                return httpx.Response(self.submit_status, json = self.submit_body or {})
            self.running.append(PID)
            return httpx.Response(200, json = {"prompt_id": PID, "number": 1})
        if path == f"/history/{PID}":
            self.history_polls += 1
            if self.finish_after_polls is not None and self.history_polls >= self.finish_after_polls:
                self.finish()
            return httpx.Response(200, json = {PID: self.history[PID]} if PID in self.history else {})
        if path == "/view":
            return httpx.Response(200, content = png())
        if path == "/interrupt":
            return httpx.Response(200)
        return httpx.Response(404)

    def client(self) -> ComfyuiClient:
        return ComfyuiClient("http://127.0.0.1:18845", transport = self.transport)

    def count(self, method, path) -> int:
        return sum(1 for c in self.calls if c == (method, path))


def msg(kind, **data):
    return {"type": kind, "data": {"prompt_id": PID, **data}}


class FakeWS:
    """Scripted frames first (callables run inline), then whatever ``push`` adds; blocks when empty."""

    def __init__(self, frames, *, close_after = False):
        self.frames = list(frames)
        self.close_after = close_after
        self.extra: asyncio.Queue | None = None

    def push(self, frame):
        if self.extra is None:
            self.extra = asyncio.Queue()
        self.extra.put_nowait(frame)

    async def recv(self):
        while self.frames:
            frame = self.frames.pop(0)
            if callable(frame):
                frame()
                continue
            return frame if isinstance(frame, (bytes, str)) else json.dumps(frame)
        if self.close_after:
            raise ConnectionError("closed")
        if self.extra is None:
            self.extra = asyncio.Queue()
        frame = await self.extra.get()
        return frame if isinstance(frame, (bytes, str)) else json.dumps(frame)


class FakeConnect:
    def __init__(self, ws = None, error = None):
        self.ws, self.error, self.calls = ws, error, []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outer = self

        class Ctx:
            async def __aenter__(self):
                if outer.error:
                    raise outer.error
                return outer.ws

            async def __aexit__(self, *exc):
                return False

        return Ctx()


async def _no_sleep(_seconds = 0):
    await asyncio.sleep(0)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_COMFYUI_TEMPLATES_DIR", str(tmp_path / "templates"))
    fake = FakeComfy()
    spy = type("Spy", (), {})()
    spy.admissions, spy.scheduled, spy.result = 0, 0, ArbiterResult(acted = True, actions = ("Unloaded oMLX: swift",))

    async def admit():
        spy.admissions += 1
        return spy.result

    monkeypatch.setattr(arbiter, "before_comfyui_job", admit)
    monkeypatch.setattr(arbiter, "schedule_comfyui_idle_free", lambda: setattr(spy, "scheduled", spy.scheduled + 1) or True)
    spy.fake = fake
    return spy


def runner(connect = None) -> jobs.ComfyJobRunner:
    r = jobs.ComfyJobRunner(connect = connect or FakeConnect(error = OSError("no ws")), sleep = _no_sleep)
    r.poll_interval = 0.0
    return r


def go(coro):
    return asyncio.run(coro)


PARAMS = {"prompt": "a red fox", "negative_prompt": "blur", "seed": 7, "steps": 4, "width": 512, "height": 512}


def success_frames(extra = ()):
    return [
        msg("execution_start"),
        msg("execution_cached", nodes = []),
        msg("executing", node = "1"),
        msg("executing", node = "4"),
        *extra,
        msg("executing", node = "6"),
        msg("progress", value = 1, max = 4, node = "6"),
        msg("progress", value = 2, max = 4, node = "6"),
        msg("progress", value = 4, max = 4, node = "6"),
        msg("executing", node = "7"),
        msg("execution_success"),
    ]


# ---------------------------------------------------------------- success


def test_websocket_job_maps_progress_phases_and_saves_to_the_gallery(world):
    snapshots = []
    snap = lambda: snapshots.append(r.progress())  # noqa: E731
    frames = [
        msg("execution_start"), snap,
        msg("executing", node = "4"), snap,
        msg("executing", node = "6"), snap,
        msg("progress", value = 2, max = 4, node = "6"), snap,
        msg("progress", value = 3, max = 4, node = "6"), snap,
        msg("executing", node = "7"), snap,
        msg("execution_success"),
    ]
    connect = FakeConnect(FakeWS(frames))
    r = runner(connect)
    world.fake.images = [{"filename": "a.png", "subfolder": "", "type": "temp"}]
    world.fake.finish()
    world.fake.history.pop(PID)  # history only appears once the job is over

    original = world.fake.handle

    def handle(request):
        if request.url.path == "/prompt":
            response = original(request)
            world.fake.finish()  # the fake job completes right after submit; frames drive the progress
            return response
        return original(request)

    world.fake.handle = handle
    result = go(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))

    assert [s["phase"] for s in snapshots] == ["encode", "encode", "denoise", "denoise", "denoise", "decode"]
    assert snapshots[0]["active"] and snapshots[0]["total_steps"] == 4 and snapshots[0]["step"] == 0
    assert (snapshots[3]["step"], snapshots[3]["total_steps"]) == (2, 4) and snapshots[3]["fraction"] == 0.5
    assert snapshots[4]["eta_seconds"] is not None
    assert snapshots[3]["prompt_id"] == PID and snapshots[3]["engine"] == "comfyui"
    assert not r.progress()["active"] and not r.is_running()

    url, kwargs = connect.calls[0]
    assert url.startswith("ws://127.0.0.1:18845/ws?clientId=") and kwargs["open_timeout"] == 5
    client_id = url.split("clientId=")[1]
    body = world.fake.submitted[0]
    assert body["client_id"] == client_id and body["extra_data"] == {"unsloth": {"template": "qwen-image-2.1-t2i"}}
    assert body["prompt"]["8"]["class_type"] == "PreviewImage"
    assert body["prompt"]["6"]["inputs"]["seed"] == 7 and body["prompt"]["5"]["inputs"]["width"] == 512

    assert result.prompt_id == PID and result.actions == ["Unloaded oMLX: swift"]
    assert world.admissions == 1 and world.scheduled == 1
    [record] = result.images
    image = GalleryImage(**record)
    assert (image.width, image.height) == (64, 48)  # the real output size, not the request
    assert (image.prompt, image.negative_prompt, image.steps, image.guidance, image.seed) == ("a red fox", "blur", 4, 1.0, 7)
    assert record["engine"] == "comfyui" and record["comfyui_template"] == "qwen-image-2.1-t2i"
    assert (record["model"], record["workflow"], record["batch_seed"], record["batch_index"]) == (
        "comfyui:qwen-image-2.1-t2i", "create", 7, 0)
    assert record["sampler"] == "euler" and record["scheduler"] == "simple" and record["loras"] == []
    from core.inference import image_gallery

    assert image_gallery.image_path(record["id"]) is not None
    assert [x["id"] for x in image_gallery.list_images()] == [record["id"]]


def test_several_outputs_become_several_records_with_batch_indexes(world):
    world.fake.images = [
        {"filename": "a.png", "subfolder": "", "type": "temp"},
        {"filename": "b.png", "subfolder": "sub", "type": "temp"},
    ]
    r = runner(FakeConnect(FakeWS([msg("execution_success")])))
    world.fake.running = []
    original = world.fake.handle
    world.fake.handle = lambda req: (original(req), world.fake.finish())[0] if req.url.path == "/prompt" else original(req)
    result = go(r.run("qwen-image-2.1-t2i", {**PARAMS, "loras": [{"name": "a.safetensors", "strength": 0.8}]}, client = world.fake.client()))
    assert [(x["batch_index"], x["batch_size"]) for x in result.images] == [(0, 2), (1, 2)]
    assert result.images[0]["loras"] == ["a.safetensors:0.8"]
    sent = world.fake.submitted[0]["prompt"]
    assert sent["unsloth_lora_0"]["inputs"]["lora_name"] == "a.safetensors" and sent["6"]["inputs"]["model"] == ["unsloth_lora_0", 0]
    views = [c for c in world.fake.calls if c[1] == "/view"]
    assert len(views) == 2


def test_polling_fallback_when_the_websocket_cannot_connect(world):
    world.fake.finish_after_polls = 3
    r = runner(FakeConnect(error = OSError("refused")))
    result = go(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
    assert len(result.images) == 1 and world.fake.history_polls >= 3
    assert world.fake.count("GET", "/queue") >= 3


def test_polling_reports_queue_position_and_survives_a_closed_websocket(world):
    positions = []
    r = runner(FakeConnect(FakeWS([msg("execution_start")], close_after = True)))
    original = world.fake.handle

    def handle(request):
        response = original(request)
        if request.url.path == "/prompt":  # queued behind a foreign job
            world.fake.running, world.fake.pending = ["other"], [PID]
        elif request.url.path == f"/history/{PID}" and PID not in world.fake.history:
            positions.append(r.progress()["queue_position"])
            if len(positions) == 1:
                world.fake.running, world.fake.pending = [PID], []
            elif len(positions) == 3:
                world.fake.finish()
        return response

    world.fake.handle = handle
    go(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
    # Each value is what the previous poll left: the ws start (0), then queued behind one job (1), then running (0).
    assert positions == [0, 1, 0]


# --------------------------------------------------------------- failures


def test_execution_error_raises_a_job_error_and_saves_nothing(world):
    frames = [msg("execution_start"), msg("execution_error", exception_message = "out of memory", node_type = "KSampler")]
    r = runner(FakeConnect(FakeWS(frames)))
    with pytest.raises(jobs.ComfyJobError, match = "KSampler: out of memory"):
        go(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
    assert world.fake.count("GET", "/view") == 0 and world.scheduled == 1 and not r.is_running()


def test_polled_history_error_and_interrupt_are_mapped(world):
    world.fake.finish(status = {"status_str": "error", "completed": False, "messages": [
        ["execution_error", {"exception_message": "boom", "node_type": "VAEDecode"}]]})
    with pytest.raises(jobs.ComfyJobError, match = "VAEDecode: boom"):
        go(runner().run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
    world.fake.finish(status = {"status_str": "error", "completed": False, "messages": [
        ["execution_interrupted", {"node_id": "6"}]]})
    with pytest.raises(jobs.ComfyCancelled, match = DIFFUSION_CANCELLED_MSG):
        go(runner().run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))


def test_interrupted_websocket_message_is_a_cancellation(world):
    r = runner(FakeConnect(FakeWS([msg("execution_start"), msg("execution_interrupted", node_id = "6")])))
    with pytest.raises(jobs.ComfyCancelled) as caught:
        go(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
    assert str(caught.value) == DIFFUSION_CANCELLED_MSG
    assert world.fake.count("GET", "/view") == 0


def test_frames_for_other_prompts_are_ignored(world):
    frames = [{"type": "execution_error", "data": {"prompt_id": "someone-else", "exception_message": "x"}},
              {"type": "status", "data": {"status": {}}}, b"\x00\x01binary", "not json", *success_frames()]
    world.fake.finish()
    r = runner(FakeConnect(FakeWS(frames)))
    assert len(go(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client())).images) == 1


def test_a_graph_rejected_by_comfyui_surfaces_node_errors(world):
    world.fake.submit_status = 400
    world.fake.submit_body = {"error": {"message": "Prompt outputs failed validation"},
                              "node_errors": {"6": {"class_type": "KSampler", "errors": [{"message": "Value not in list: sampler_name"}]}}}
    with pytest.raises(ComfyGraphError, match = "node 6 \\(KSampler\\): Value not in list"):
        go(runner().run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
    assert world.scheduled == 1


def test_a_job_that_vanishes_from_the_queue_fails(world):
    r = runner()
    original = world.fake.handle

    def handle(request):
        response = original(request)
        if request.url.path == "/prompt":
            world.fake.running = []
        return response

    world.fake.handle = handle
    with pytest.raises(jobs.ComfyJobError, match = "left the ComfyUI queue"):
        go(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))


def test_overall_timeout_interrupts_the_job(world):
    r = runner(FakeConnect(FakeWS([msg("execution_start")])))
    r.ws_safety_poll = 60
    with pytest.raises(jobs.ComfyTimeout):
        go(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client(), timeout_s = 0.05))
    assert world.fake.count("POST", "/interrupt") == 1


def test_no_images_in_the_history_is_an_error(world):
    world.fake.images = []
    world.fake.finish()
    with pytest.raises(jobs.ComfyJobError, match = "no images"):
        go(runner(FakeConnect(FakeWS([msg("execution_success")]))).run(
            "qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))


# ---------------------------------------------------------------- refusals


def test_admission_error_means_no_submit_and_no_idle_timer(world):
    world.result = ArbiterResult(skipped = "error", error = "Cannot free shared memory for a ComfyUI job: oMLX is busy")
    with pytest.raises(AttachedAdmissionError, match = "oMLX is busy"):
        go(runner().run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
    assert world.fake.submitted == [] and world.scheduled == 0


def test_missing_models_are_reported_before_admission(world):
    world.fake.models["text_encoders"] = []
    world.fake.models["loras"] = []
    with pytest.raises(jobs.ComfyModelsMissing) as caught:
        go(runner().run("qwen-image-2.1-t2i", {**PARAMS, "loras": [{"name": "gone.safetensors", "strength": 1}]}, client = world.fake.client()))
    assert caught.value.missing == {"text_encoders": ["qwen3vl_8b_bf16.safetensors"], "loras": ["gone.safetensors"]}
    assert "text_encoders/qwen3vl_8b_bf16.safetensors" in str(caught.value) and "StoryPressRuntime" in str(caught.value)
    assert world.admissions == 0 and world.fake.submitted == []


def test_comfyui_off_versus_starting(world, tmp_path, monkeypatch):
    world.fake.up = False
    home = tmp_path / "engines-home"
    home.mkdir()
    with pytest.raises(jobs.ComfyOffError, match = "Settings > Engines"):
        go(runner().run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
    (home / "desktop.json").write_text(json.dumps({"engines_enabled": True, "helpers": {"comfyui": True}}))
    with pytest.raises(jobs.ComfyNotRunningError):
        go(runner().run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
    assert world.admissions == 0


def test_unknown_queue_state_is_an_engine_error(world):
    world.fake.handle = lambda request: httpx.Response(500)
    with pytest.raises(ComfyuiError):
        go(runner().run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))


def test_unknown_template_and_bad_params_fail_before_any_request(world):
    with pytest.raises(jobs.ComfyTemplateNotFound):
        go(runner().run("nope", dict(PARAMS), client = world.fake.client()))
    with pytest.raises(ComfyParamError):
        go(runner().run("qwen-image-2.1-t2i", {"prompt": ""}, client = world.fake.client()))
    assert world.fake.calls == []


def test_a_second_concurrent_job_is_busy_and_the_first_still_finishes(world):
    async def scenario():
        gate = asyncio.Event()

        class GateWS(FakeWS):
            async def recv(self):
                await gate.wait()
                return await super().recv()

        r = runner(FakeConnect(GateWS([msg("execution_success")])))
        world.fake.finish()
        first = asyncio.create_task(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if r.progress()["prompt_id"]:
                break
        with pytest.raises(jobs.ComfyBusyError):
            await r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client())
        assert r.is_running()
        gate.set()
        return await first

    assert len(go(scenario()).images) == 1
    assert len(world.fake.submitted) == 1


# ------------------------------------------------------------------ cancel


def test_cancel_interrupts_a_running_job_and_only_with_its_own_prompt_id(world):
    async def scenario():
        r = runner(FakeConnect(FakeWS([msg("execution_start")])))
        r.ws_safety_poll = 60
        assert await r.cancel() is False
        task = asyncio.create_task(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if r.progress()["prompt_id"]:
                break
        assert await r.cancel() is True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    go(scenario())
    interrupts = [c for c in world.fake.calls if c == ("POST", "/interrupt")]
    assert len(interrupts) >= 1


def test_cancel_interrupt_body_names_the_prompt(world):
    bodies = []
    original = world.fake.handle

    def handle(request):
        if request.url.path == "/interrupt":
            bodies.append(json.loads(request.content))
        return original(request)

    world.fake.handle = handle

    async def scenario():
        ws = FakeWS([msg("execution_start")])
        r = runner(FakeConnect(ws))
        task = asyncio.create_task(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if r.progress()["prompt_id"]:
                break
        await r.cancel()
        ws.push(msg("execution_interrupted"))
        with pytest.raises(jobs.ComfyCancelled):
            await task

    go(scenario())
    assert bodies == [{"prompt_id": PID}]


def test_cancel_removes_a_queued_job_from_the_queue(world):
    world.fake.running = ["someone-elses-job"]
    original = world.fake.handle

    def handle(request):
        response = original(request)
        if request.url.path == "/prompt":  # queued behind a foreign job
            world.fake.running = ["someone-elses-job"]
            world.fake.pending = [PID]
        return response

    world.fake.handle = handle

    async def scenario():
        r = runner(FakeConnect(FakeWS([])))
        r.ws_safety_poll = 60
        task = asyncio.create_task(r.run("qwen-image-2.1-t2i", dict(PARAMS), client = world.fake.client()))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if r.progress()["prompt_id"]:
                break
        assert await r.cancel() is True
        with pytest.raises(jobs.ComfyCancelled):
            await task

    go(scenario())
    assert world.fake.pending == [] and world.fake.running == ["someone-elses-job"]
    assert world.fake.count("POST", "/interrupt") == 0


# --------------------------------------------------- idle free (real arbiter)


def test_idle_free_never_fires_mid_job_and_fires_once_after_the_last_job(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    fake = FakeComfy()
    own = "http://127.0.0.1:18845"
    config = replace(DEFAULT_CONFIG, enabled = True, comfyui_url = own, comfyui_idle_free_s = 300)
    monkeypatch.setattr(arbiter, "get_config", lambda: config)
    monkeypatch.setattr(jobs, "get_config", lambda: config)
    monkeypatch.setenv("STUDIO_COMFYUI_URL", own)
    monkeypatch.setattr(arbiter, "ComfyuiClient", lambda url, transport = None: ComfyuiClient(url, transport = fake.transport))
    monkeypatch.setattr(arbiter, "_local_workload", lambda: None)

    async def no_media():
        return None

    async def no_omlx(_config):
        return []

    monkeypatch.setattr(arbiter, "_media_resident_async", no_media)
    monkeypatch.setattr(arbiter, "_unload_omlx", no_omlx)

    async def scenario():
        sleeps = []
        release = asyncio.Event()

        async def gated_sleep(seconds):
            sleeps.append(seconds)
            await release.wait()

        monkeypatch.setattr(arbiter, "_IDLE_SLEEP", gated_sleep)
        fake.finish()
        first = runner(FakeConnect(FakeWS([msg("execution_success")])))
        await first.run("qwen-image-2.1-t2i", dict(PARAMS), client = fake.client())
        await asyncio.sleep(0)
        assert sleeps == [300], "a finished job starts the idle timer"

        gate = asyncio.Event()

        class GateWS(FakeWS):
            async def recv(self):
                await gate.wait()
                return await super().recv()

        fake.history.clear()
        fake.images = [{"filename": "b.png", "subfolder": "", "type": "temp"}]
        second = runner(FakeConnect(GateWS([msg("execution_success")])))
        task = asyncio.create_task(second.run("qwen-image-2.1-t2i", dict(PARAMS), client = fake.client()))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if second.progress()["prompt_id"]:
                break
        release.set()  # the first job's timer would fire now, if admission had not cancelled it
        await asyncio.sleep(0.05)
        assert fake.count("POST", "/free") == 0, "no free while the second job runs"
        fake.finish()
        gate.set()
        await task
        await asyncio.sleep(0)
        release.clear()
        assert sleeps[-1] == 300 and len(sleeps) == 2
        release.set()
        await asyncio.sleep(0.05)
        assert fake.count("POST", "/free") == 1

    go(scenario())
