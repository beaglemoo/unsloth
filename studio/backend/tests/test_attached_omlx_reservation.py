# SPDX-License-Identifier: AGPL-3.0-only
"""An admitted oMLX request holds a reservation until it ends; competing admissions are refused meanwhile."""
from __future__ import annotations

import asyncio
import sys
import threading
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached import arbiter
from core.inference.attached.omlx_client import BUSY_MESSAGE
from models.inference import ChatCompletionRequest
from routes import inference
from utils.attached_engines_settings import DEFAULT_CONFIG

PASSED = 418


class Studio:
    def __init__(self):
        self.training = False
        self.local = False
        self.omlx_unloads = 0


@pytest.fixture
def studio(monkeypatch):
    state = Studio()
    config = replace(DEFAULT_CONFIG, enabled = True)
    monkeypatch.setattr(arbiter, "get_config", lambda: config)
    monkeypatch.setattr(arbiter, "_training_active", lambda: state.training)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: state.local)
    monkeypatch.setattr(arbiter, "_studio_media_resident", lambda: None)
    arbiter._omlx_holds.clear()
    arbiter._media_holds.clear()
    arbiter._comfyui_holds.clear()
    arbiter._media_probe_future = None

    class Omlx:
        def __init__(self, url):
            pass

        async def unload_all(self):
            state.omlx_unloads += 1
            return ["swift"]

    monkeypatch.setattr(arbiter, "OmlxClient", Omlx)
    yield state
    arbiter._omlx_holds.clear()


def run(coro):
    return asyncio.run(coro)


async def competitors():
    """Every admission that would unload oMLX or start heavy work."""
    media = arbiter.new_media_hold("image")
    job = arbiter.new_comfyui_hold()
    results = {
        "local_load": await arbiter.free_for_local("local_load"),
        "training": await arbiter.free_for_local("training"),
    }
    results["media"] = await arbiter.free_for_local("local_load", media_hold = media)
    media_was_active = media.active
    media.release()  # an admitted load must not block the job probe below
    results["job"] = await arbiter.before_comfyui_job(job)
    job_was_active = job.active
    job.release()
    media.active, job.active = media_was_active, job_was_active
    return results, media, job


def refused(results):
    return all(r.error and BUSY_MESSAGE in r.error for r in results.values())


def admitted_hold():
    hold = arbiter.new_omlx_hold()
    assert run(arbiter.before_omlx_use(hold)).error is None
    assert hold.active
    return hold


# --- arbiter ---------------------------------------------------------------------------------


def test_admitted_omlx_request_blocks_competitors_until_released(studio):
    async def scenario():
        hold = arbiter.new_omlx_hold()
        assert (await arbiter.before_omlx_use(hold)).error is None
        results, media, job = await competitors()
        assert refused(results), {k: r.error for k, r in results.items()}
        assert not media.active and not job.active
        assert arbiter._media_held() is None and not arbiter._comfyui_held()
        assert studio.omlx_unloads == 0, "a held oMLX must never be unloaded"
        with pytest.raises(arbiter.AttachedAdmissionError):
            results["job"].require_clear()
        hold.release()
        hold.release()  # idempotent
        results, media, job = await competitors()
        assert all(r.error is None for r in results.values())
        assert media.active and job.active and studio.omlx_unloads >= 1

    run(scenario())


def test_refusal_uses_the_oMLX_busy_message_and_becomes_a_retryable_503(studio):
    admitted_hold()
    with pytest.raises(HTTPException) as caught:
        run(inference._admit_attached_local_load())
    assert caught.value.status_code == 503
    assert caught.value.headers == {"Retry-After": "15"}
    assert BUSY_MESSAGE in caught.value.detail


def test_a_refused_or_skipped_admission_registers_nothing(studio, monkeypatch):
    studio.training = True
    hold = arbiter.new_omlx_hold()
    assert run(arbiter.before_omlx_use(hold)).error and not hold.active and arbiter.omlx_holds() == 0
    studio.training = False
    studio.local = True
    assert run(arbiter.before_omlx_load(hold)).error and not hold.active
    studio.local = False
    monkeypatch.setattr(arbiter, "get_config", lambda: DEFAULT_CONFIG)  # engines disabled
    assert run(arbiter.before_omlx_use(hold)).skipped == "disabled" and not hold.active


def test_reverse_ordering_is_unchanged(studio):
    hold = arbiter.new_omlx_hold()
    studio.local = True
    assert "local model loaded" in run(arbiter.before_omlx_use(hold)).error
    studio.local = False
    job = arbiter.new_comfyui_hold()
    assert run(arbiter.before_comfyui_job(job)).error is None
    assert "ComfyUI job starting" in run(arbiter.before_omlx_use(hold)).error
    assert not hold.active
    job.release()
    assert run(arbiter.before_omlx_use(hold)).error is None


def test_no_claim_means_no_refusal(studio):
    async def scenario():
        results, _, _ = await competitors()
        assert all(r.error is None for r in results.values())

    run(scenario())


def test_concurrent_requests_share_and_the_competitor_waits_for_both(studio):
    async def scenario():
        first, second = arbiter.new_omlx_hold(), arbiter.new_omlx_hold()
        a, b = await asyncio.gather(arbiter.before_omlx_use(first), arbiter.before_omlx_use(second))
        assert a.error is None and b.error is None and arbiter.omlx_holds() == 2
        assert (await arbiter.free_for_local("local_load")).error
        first.release()
        assert arbiter.omlx_holds() == 1
        assert (await arbiter.before_comfyui_job()).error
        second.release()
        assert arbiter.omlx_holds() == 0
        assert (await arbiter.free_for_local("local_load")).error is None

    run(scenario())


def test_a_competitor_and_an_oMLX_admission_racing_never_both_win(studio):
    async def scenario():
        hold = arbiter.new_omlx_hold()
        use, local = await asyncio.gather(arbiter.before_omlx_use(hold), arbiter.free_for_local("local_load"))
        # oMLX ran first (local refused) or the local load ran first (it unloaded the idle oMLX; no claim existed yet).
        assert use.error is None
        assert local.error is None or BUSY_MESSAGE in local.error

    run(scenario())


def clock_at(monkeypatch):
    """A controllable ``monotonic`` for the arbiter only (the event loop keeps the real one)."""
    import time as real_time

    now = [1000.0]
    monkeypatch.setattr(arbiter, "time", NS(monotonic = lambda: now[0], time = real_time.time))
    return now


def test_a_leaked_reservation_is_ignored_and_logged(studio, monkeypatch):
    warnings = []
    monkeypatch.setattr(arbiter.logger, "warning", lambda *a, **k: warnings.append(a))
    now = clock_at(monkeypatch)
    hold = admitted_hold()  # its owner task ended inside run() without a release: leaked
    assert arbiter.omlx_holds() == 1
    now[0] += arbiter._OMLX_HOLD_MAX_AGE_S + 1
    assert run(arbiter.free_for_local("local_load")).error is None, "an expired claim must not wedge the arbiter"
    assert arbiter.omlx_holds() == 0 and not hold.active and warnings


def test_a_reservation_without_an_owner_is_leaked_past_the_age(studio, monkeypatch):
    now = clock_at(monkeypatch)
    hold = admitted_hold()
    hold._owner = None
    now[0] += arbiter._OMLX_HOLD_MAX_AGE_S + 1
    assert arbiter.omlx_holds() == 0 and not hold.active


def test_a_live_owner_keeps_its_reservation_past_the_age(studio, monkeypatch):
    """A tool loop outlasts the age limit; its next provider turn is not re-admitted, so the claim must hold."""
    now = clock_at(monkeypatch)
    turns = []

    async def scenario():
        resume, finished = asyncio.Event(), asyncio.Event()
        hold = arbiter.new_omlx_hold()

        async def tool_loop():
            assert (await arbiter.before_omlx_use(hold)).error is None
            turns.append("first provider turn")
            await resume.wait()  # tool calls running for longer than the age limit
            assert hold.active, "the claim must still be held for the next provider turn"
            turns.append("second provider turn")
            hold.release()
            finished.set()

        task = asyncio.create_task(tool_loop())
        await until(lambda: hold.active)
        now[0] += arbiter._OMLX_HOLD_MAX_AGE_S + 1
        assert arbiter.omlx_holds() == 1 and hold.active
        results, _media, _job = await competitors()
        assert refused(results), results
        assert studio.omlx_unloads == 0
        resume.set()
        await finished.wait()
        await task
        assert arbiter.omlx_holds() == 0
        assert (await arbiter.free_for_local("local_load")).error is None

    run(scenario())
    assert turns == ["first provider turn", "second provider turn"]


def test_a_finished_owner_loses_the_reservation_past_the_age(studio, monkeypatch):
    now = clock_at(monkeypatch)

    async def scenario():
        hold = arbiter.new_omlx_hold()

        async def abandoned():
            await arbiter.before_omlx_use(hold)  # returns without releasing

        await asyncio.create_task(abandoned())
        assert hold.active and arbiter.omlx_holds() == 1
        now[0] += arbiter._OMLX_HOLD_MAX_AGE_S - 1
        assert arbiter.omlx_holds() == 1, "young enough to keep"
        now[0] += 2
        assert arbiter.omlx_holds() == 0 and not hold.active
        assert (await arbiter.free_for_local("local_load")).error is None

    run(scenario())


def test_a_stream_repoints_the_owner_to_the_task_iterating_it(studio, monkeypatch):
    now = clock_at(monkeypatch)

    async def body():
        yield "a"
        yield "b"
        await asyncio.Event().wait()

    async def scenario():
        hold = arbiter.new_omlx_hold()

        async def handler():
            await arbiter.before_omlx_use(hold)
            return inference._HeldStream(body(), hold)

        stream = await asyncio.create_task(handler())  # the handler task is done; the claim is young
        proceed, iterating, ended = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def respond():
            assert await stream.__anext__() == "a"
            iterating.set()
            await proceed.wait()  # between chunks, long past the age limit
            assert await stream.__anext__() == "b"
            ended.set()  # the response task ends without exhausting or closing the stream

        task = asyncio.create_task(respond())
        await iterating.wait()
        now[0] += arbiter._OMLX_HOLD_MAX_AGE_S + 1
        assert arbiter.omlx_holds() == 1 and hold.active, "the iterating task owns the claim"
        assert refused({"local": await arbiter.free_for_local("local_load")})
        proceed.set()
        await task
        assert ended.is_set()
        now[0] += arbiter._OMLX_HOLD_MAX_AGE_S + 1
        assert arbiter.omlx_holds() == 0 and not hold.active, "a finished iterator that never released is leaked"
        stream._hold = arbiter.new_omlx_hold()  # keep __del__ from touching the dropped claim's state
        await stream._body.aclose()

    run(scenario())


def test_a_fresh_reservation_is_kept(studio):
    hold = admitted_hold()
    hold.since -= arbiter._OMLX_HOLD_MAX_AGE_S - 60
    assert arbiter.omlx_holds() == 1


def test_status_exposes_the_count(studio, monkeypatch):
    from routes import attached_engines

    hold = admitted_hold()
    assert arbiter.omlx_holds() == 1

    async def omlx(config):
        return {}, []

    async def comfyui(config):
        return {}

    monkeypatch.setattr(attached_engines, "get_config", lambda: replace(DEFAULT_CONFIG, enabled = True))
    monkeypatch.setattr(attached_engines, "_probe_omlx", omlx)
    monkeypatch.setattr(attached_engines, "_probe_comfyui", comfyui)
    assert run(attached_engines.attached_status())["omlx_request_holds"] == 1
    hold.release()
    assert run(attached_engines.attached_status())["omlx_request_holds"] == 0


# --- the /omlx/load route ---------------------------------------------------------------------


def test_omlx_load_holds_the_claim_while_loading_and_releases_it(studio, monkeypatch):
    from routes import attached_engines

    seen = []

    class Client:
        async def load(self, dir_id):
            seen.append(arbiter.omlx_holds())
            assert (await arbiter.free_for_local("local_load")).error
            if dir_id == "boom":
                raise attached_engines.AttachedEngineError("down")

    async def resolve(client, model_id):
        return model_id

    async def noop_free(*args, **kwargs):
        return []

    monkeypatch.setattr(attached_engines, "get_config", lambda: replace(DEFAULT_CONFIG, enabled = True))
    monkeypatch.setattr(attached_engines, "_omlx", lambda config: Client())
    monkeypatch.setattr(attached_engines, "_resolve_omlx_dir", resolve)
    monkeypatch.setattr(arbiter, "free_comfyui", noop_free)
    assert run(attached_engines.omlx_load(NS(model_id = "swift")))["loaded"] == "swift"
    assert seen == [1] and arbiter.omlx_holds() == 0
    with pytest.raises(HTTPException) as caught:
        run(attached_engines.omlx_load(NS(model_id = "boom")))
    assert caught.value.status_code == 502 and arbiter.omlx_holds() == 0


# --- callers: the claim spans the gap between admission and the upstream request ---------------


def proxy_args(**fields):
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hi"}], external_model = "swift", provider_id = "userconn", **fields
    )
    request = NS(scope = {}, state = NS(skip_api_monitor = True), headers = {},
                 url = NS(path = "/v1/chat/completions"), is_disconnected = _never_disconnected)
    return payload, request


async def _never_disconnected():
    return False


@pytest.fixture
def credentials(monkeypatch, studio):
    """The credential step blocks in a worker thread (``gate``) and then fails or succeeds as told."""
    box = NS(gate = threading.Event(), outcome = "fail", entered = threading.Event())

    def resolve(*args, **kwargs):
        box.entered.set()
        assert box.gate.wait(10)
        if box.outcome == "fail":
            raise HTTPException(status_code = PASSED, detail = "past admission")
        return ""

    monkeypatch.setattr(inference, "resolve_provider_api_key_or_400", resolve)
    monkeypatch.setattr(inference.providers_db, "get_provider", lambda pid: saved_omlx())
    return box


async def until(predicate, timeout = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.005)


def test_chat_proxy_holds_the_claim_across_a_delayed_dispatch_and_releases_on_failure(credentials):
    async def scenario():
        task = asyncio.create_task(inference._proxy_to_external_provider(*proxy_args(), "user"))
        await until(lambda: arbiter.omlx_holds() == 1 and credentials.entered.is_set())
        results, _, _ = await competitors()
        assert refused(results)
        credentials.gate.set()
        with pytest.raises(HTTPException) as caught:
            await task
        assert caught.value.status_code == PASSED
        assert arbiter.omlx_holds() == 0
        results, _, _ = await competitors()
        assert all(r.error is None for r in results.values())

    run(scenario())


def test_chat_proxy_releases_on_cancellation(credentials):
    async def scenario():
        task = asyncio.create_task(inference._proxy_to_external_provider(*proxy_args(), "user"))
        await until(lambda: arbiter.omlx_holds() == 1)
        task.cancel()
        credentials.gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert arbiter.omlx_holds() == 0

    run(scenario())


def test_chat_proxy_releases_on_a_validation_failure_before_dispatch(studio):
    async def scenario():
        payload, request = proxy_args()
        payload.provider_id = None
        payload.provider_type = "no-such-provider-type"
        with pytest.raises(HTTPException) as caught:
            await inference._proxy_to_external_provider(payload, request, "user")
        assert caught.value.status_code == 400
        assert arbiter.omlx_holds() == 0

    run(scenario())


def test_other_destinations_never_take_a_claim(credentials, studio):
    async def scenario():
        payload, request = proxy_args()
        payload.provider_id = None
        payload.provider_type = "custom"
        payload.provider_base_url = "http://192.168.2.5:9000/v1"
        credentials.gate.set()
        with pytest.raises(HTTPException):
            await inference._proxy_to_external_provider(payload, request, "user")
        assert arbiter.omlx_holds() == 0

    run(scenario())


def test_chat_proxy_streaming_keeps_the_claim_until_the_stream_ends(credentials, monkeypatch):
    class Fake:
        def __init__(self, *args, **kwargs):
            pass

        async def stream_chat_completion(self, **kwargs):
            yield 'data: {"choices":[{"delta":{"content":"a"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"b"}}]}'
            yield "data: [DONE]"

        async def close(self):
            pass

    monkeypatch.setattr(inference, "ExternalProviderClient", Fake)
    credentials.outcome = "ok"
    credentials.gate.set()

    async def scenario():
        payload, request = proxy_args(stream = True)
        response = await inference._proxy_to_external_provider(payload, request, "user")
        assert isinstance(response, StreamingResponse)
        assert arbiter.omlx_holds() == 1, "the handler returned but the stream has not run"
        assert (await arbiter.free_for_local("local_load")).error
        chunks = [chunk async for chunk in response.body_iterator]
        assert any("[DONE]" in str(c) for c in chunks)
        assert arbiter.omlx_holds() == 0

        # Closed by a client disconnect part way through.
        response = await inference._proxy_to_external_provider(payload, request, "user")
        body = response.body_iterator
        await body.__anext__()
        assert arbiter.omlx_holds() == 1
        await body.aclose()
        assert arbiter.omlx_holds() == 0


        # The generator's own finally releases too: an abandoned iterator is closed by the asyncgen finalizer,
        # never through the wrapper.
        response = await inference._proxy_to_external_provider(payload, request, "user")
        inner = response.body_iterator._body
        await inner.__anext__()
        assert arbiter.omlx_holds() == 1
        await inner.aclose()
        assert arbiter.omlx_holds() == 0

    run(scenario())


# --- the wrapper itself: stream completion, error, close, drop, non-streaming ---------------------


def held_handler(response_factory, *, admit_first = True):
    async def handler(*, _omlx_hold = None):
        await inference._admit_attached_omlx_provider("omlx", None, _omlx_hold)
        return await response_factory()

    return inference._holds_omlx(handler)


def streaming(chunks, fail_after = None):
    async def factory():
        async def body():
            for index, chunk in enumerate(chunks):
                if fail_after is not None and index == fail_after:
                    raise RuntimeError("upstream broke")
                yield chunk

        return StreamingResponse(body(), media_type = "text/event-stream")

    return factory


def test_wrapper_releases_when_a_stream_completes(studio):
    async def scenario():
        response = await held_handler(streaming(["a", "b"]))()
        assert arbiter.omlx_holds() == 1
        assert [c async for c in response.body_iterator] == ["a", "b"]
        assert arbiter.omlx_holds() == 0

    run(scenario())


def test_wrapper_releases_when_a_stream_errors(studio):
    async def scenario():
        response = await held_handler(streaming(["a", "b"], fail_after = 1))()
        with pytest.raises(RuntimeError):
            async for _ in response.body_iterator:
                assert arbiter.omlx_holds() == 1
        assert arbiter.omlx_holds() == 0

    run(scenario())


def test_wrapper_releases_when_a_stream_is_closed_mid_way(studio):
    async def scenario():
        response = await held_handler(streaming(["a", "b", "c"]))()
        assert await response.body_iterator.__anext__() == "a"
        assert arbiter.omlx_holds() == 1
        await response.body_iterator.aclose()
        assert arbiter.omlx_holds() == 0

    run(scenario())


def test_wrapper_releases_when_a_stream_is_cancelled_mid_way(studio):
    async def scenario():
        started = asyncio.Event()

        async def factory():
            async def body():
                yield "a"
                started.set()
                await asyncio.sleep(60)
                yield "b"

            return StreamingResponse(body())

        response = await held_handler(factory)()

        async def consume():
            async for _ in response.body_iterator:
                pass

        task = asyncio.create_task(consume())
        await started.wait()
        assert arbiter.omlx_holds() == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert arbiter.omlx_holds() == 0

    run(scenario())


def test_wrapper_releases_a_stream_that_is_dropped_unread(studio):
    import gc

    async def scenario():
        response = await held_handler(streaming(["a"]))()
        assert arbiter.omlx_holds() == 1
        del response
        gc.collect()
        assert arbiter.omlx_holds() == 0

    run(scenario())


def test_wrapper_releases_a_non_streaming_response_and_a_raise(studio):
    async def plain():
        return NS(status_code = 200)

    async def boom():
        raise HTTPException(status_code = 400, detail = "bad")

    async def scenario():
        assert (await held_handler(plain)()).status_code == 200
        assert arbiter.omlx_holds() == 0
        with pytest.raises(HTTPException):
            await held_handler(boom)()
        assert arbiter.omlx_holds() == 0

        async def cancelled():
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await held_handler(cancelled)()
        assert arbiter.omlx_holds() == 0

    run(scenario())


def test_wrapper_leaves_other_streams_untouched(studio):
    async def handler(*, _omlx_hold = None):
        await inference._admit_attached_omlx_provider("custom", "https://api.openai.com/v1", _omlx_hold)
        return StreamingResponse(iter(["x"]))

    async def scenario():
        response = await inference._holds_omlx(handler)()
        assert type(response.body_iterator).__name__ != "_HeldStream"
        assert arbiter.omlx_holds() == 0

    run(scenario())


def test_two_overlapping_streams_both_hold(studio):
    async def scenario():
        one = await held_handler(streaming(["a", "b"]))()
        two = await held_handler(streaming(["a", "b"]))()
        assert arbiter.omlx_holds() == 2
        [_ async for _ in one.body_iterator]
        assert arbiter.omlx_holds() == 1 and (await arbiter.free_for_local("local_load")).error
        [_ async for _ in two.body_iterator]
        assert arbiter.omlx_holds() == 0

    run(scenario())


# --- TTS, transcription, image generation, System One ----------------------------------------------


def saved_omlx():
    return {
        "id": "userconn", "display_name": "My oMLX", "provider_type": "custom",
        "base_url": "http://127.0.0.1:8843/v1", "is_enabled": True,
        "api_type": "chat_completions", "reasoning_config": None,
    }


def stub_connection(monkeypatch, gate):
    monkeypatch.setattr(inference.providers_db, "get_provider", lambda pid: saved_omlx())

    def resolve(*args, **kwargs):
        gate.set()
        raise HTTPException(status_code = PASSED, detail = "past admission")

    monkeypatch.setattr(inference, "resolve_provider_api_key_or_400", resolve)


def _speech():
    from models.inference import AudioSpeechRequest

    body = AudioSpeechRequest(input = "hi", model = "kokoro", voice = "af", provider_id = "userconn")
    return inference._external_tts_speech(body, NS(headers = {}, state = NS(), scope = {}))


def _image():
    from models.inference import DiffusionGenerateRequest

    body = DiffusionGenerateRequest(prompt = "a fox", provider_id = "userconn", model = "flux")
    return inference._external_image_generation(body, NS(headers = {}, state = NS(), scope = {}))


def _transcription():
    return inference._external_stt_transcription(
        provider_id = "userconn", raw = b"RIFFxxxx", filename = "a.wav", content_type = "audio/wav",
        model = "whisper", language = None, response_format = "json", encrypted_api_key = None,
        request = NS(headers = {}, state = NS(), scope = {}),
    )


@pytest.mark.parametrize("call", [_speech, _image, _transcription], ids = ["tts", "image", "stt"])
def test_non_chat_callers_release_after_a_credential_failure(studio, monkeypatch, call):
    stub_connection(monkeypatch, threading.Event())

    async def scenario():
        with pytest.raises(HTTPException) as caught:
            await call()
        assert caught.value.status_code == PASSED
        assert arbiter.omlx_holds() == 0

    run(scenario())


@pytest.mark.parametrize("call", [_speech, _image, _transcription], ids = ["tts", "image", "stt"])
def test_non_chat_callers_hold_the_claim_across_the_gap_and_release_on_cancel(studio, monkeypatch, call):
    gate = threading.Event()
    monkeypatch.setattr(inference.providers_db, "get_provider", lambda pid: saved_omlx())

    def resolve(*args, **kwargs):
        assert gate.wait(10)
        raise HTTPException(status_code = PASSED, detail = "past admission")

    monkeypatch.setattr(inference, "resolve_provider_api_key_or_400", resolve)

    async def scenario():
        task = asyncio.create_task(call())
        await until(lambda: arbiter.omlx_holds() == 1)
        assert (await arbiter.free_for_local("training")).error
        task.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert arbiter.omlx_holds() == 0

    run(scenario())


@pytest.mark.parametrize("outcome", ["ok", "upstream_failure", "credential_failure", "bad_shape"])
def test_system_one_decision_holds_the_claim_and_releases(studio, monkeypatch, outcome):
    import httpx

    from routes import systemone

    config = {
        "provider_type": "custom", "base_url": "http://127.0.0.1:8843/v1", "is_enabled": True,
        "api_type": "chat_completions", "display_name": "My oMLX", "models": ["m"],
    }
    seen = []
    monkeypatch.setattr(systemone, "run_as", lambda owner, fn, *args: config)
    monkeypatch.setattr(systemone.catalog, "decision_models", lambda cfg: ["m"])
    monkeypatch.setattr(systemone, "answers_decisions_only", lambda *args: False)

    async def arun_as(owner, awaitable):
        return await awaitable

    async def key(provider_id, cfg):
        seen.append(arbiter.omlx_holds())
        assert (await arbiter.before_comfyui_job()).error, "the gap before dispatch must hold the claim"
        if outcome == "credential_failure":
            raise HTTPException(status_code = 409, detail = "changed")
        return "k"

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def create_decision(self, model, state, questions):
            seen.append(arbiter.omlx_holds())
            if outcome == "upstream_failure":
                raise httpx.ConnectError("down")
            return {"answers": {}} if outcome == "ok" else {"nope": 1}

    monkeypatch.setattr(systemone, "arun_as", arun_as)
    monkeypatch.setattr(systemone, "_connection_key", key)
    monkeypatch.setattr(systemone, "ExternalProviderClient", Client)

    async def decide():
        return await systemone._connection_decide(systemone.catalog.Connection("c", "m"), "state", {})

    if outcome == "ok":
        assert run(decide()) == {"answers": {}}
    else:
        with pytest.raises(HTTPException):
            run(decide())
    assert seen and set(seen) == {1}
    assert arbiter.omlx_holds() == 0


def test_system_one_refusal_registers_no_claim(studio, monkeypatch):
    from routes import systemone

    studio.training = True
    config = {
        "provider_type": "omlx", "base_url": "http://127.0.0.1:8843/v1", "is_enabled": True,
        "api_type": "chat_completions", "display_name": "oMLX", "models": ["m"],
    }
    monkeypatch.setattr(systemone, "run_as", lambda owner, fn, *args: config)
    monkeypatch.setattr(systemone.catalog, "decision_models", lambda cfg: ["m"])
    monkeypatch.setattr(systemone, "answers_decisions_only", lambda *args: False)
    with pytest.raises(HTTPException) as caught:
        run(systemone._connection_decide(systemone.catalog.Connection("c", "m"), "state", {}))
    assert caught.value.status_code == 503 and arbiter.omlx_holds() == 0
