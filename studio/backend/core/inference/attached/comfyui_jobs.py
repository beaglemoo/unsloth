# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run one Studio-submitted ComfyUI job: admit, submit, follow progress, save the outputs to the gallery.

One job at a time (``ComfyBusyError`` for a second). The sequence is: build the graph, make sure ComfyUI
is up and has the models, ask the arbiter for room (``before_comfyui_job``: unloads oMLX), open the
websocket, ``POST /prompt``, follow ``/ws`` (with a periodic ``/history`` check, and polling alone when
the websocket is unavailable), fetch the outputs through ``/history`` and ``/view`` and store them with
``image_gallery.save``. The graph ships ``PreviewImage`` nodes, so ComfyUI keeps no copy: the Studio
gallery holds the only one. ``/interrupt`` and queue deletes only ever name Studio's own ``prompt_id``.
The idle free timer lives in the arbiter: admission cancels it, and a finished job restarts it.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote, urlsplit

from core.inference.attached import arbiter, desktop_settings
from core.inference.attached.comfyui_client import (
    ComfyGraphError,
    ComfyuiClient,
    ComfyuiError,
)
from core.inference.attached.comfyui_graphs import (
    ComfyParamError,
    ComfyTemplateError,
    Template,
    build_graph,
    get_template,
)
from loggers import get_logger
from utils.attached_engines_settings import get_config

logger = get_logger(__name__)

JOB_TIMEOUT_ENV = "STUDIO_COMFYUI_JOB_TIMEOUT_S"
DEFAULT_JOB_TIMEOUT_S = 1800.0
OFF_MESSAGE = "ComfyUI is off, enable it in Settings > Engines."
DEFAULT_MODEL_DIRS = ["~/Library/Application Support/StoryPressRuntime/ComfyUI/models"]


class ComfyJobFailure(RuntimeError):
    """Base of the job outcomes the route maps to HTTP statuses."""


class ComfyBusyError(ComfyJobFailure):
    """A Studio ComfyUI job is already running."""


class ComfyOffError(ComfyJobFailure):
    """ComfyUI's helper is switched off (or not wanted)."""


class ComfyNotRunningError(ComfyJobFailure):
    """ComfyUI should be running but nothing answers (still starting, or failing)."""


class ComfyTemplateNotFound(ComfyJobFailure):
    """No template with that id."""


class ComfyModelsMissing(ComfyJobFailure):
    def __init__(self, missing: dict[str, list[str]], model_dirs: list[str]):
        files = ", ".join(f"{folder}/{name}" for folder, names in missing.items() for name in names)
        super().__init__(
            f"ComfyUI is missing model files: {files}. Put them in one of its model folders ({', '.join(model_dirs)})."
        )
        self.missing = missing
        self.model_dirs = model_dirs


class ComfyJobError(ComfyJobFailure):
    """ComfyUI reported an execution error."""

    def __init__(self, message: str, node_type: Optional[str] = None):
        super().__init__(f"ComfyUI failed{f' in {node_type}' if node_type else ''}: {message}")
        self.node_type = node_type


class ComfyCancelled(ComfyJobFailure):
    """The job was interrupted or removed from the queue."""


class ComfyTimeout(ComfyJobFailure):
    """The job did not finish within the overall timeout."""


@dataclass
class ComfyJobResult:
    images: list[dict]
    actions: list[str]
    prompt_id: str
    resolved: dict


@dataclass
class _State:
    template_id: str
    total_steps: int
    step: int = 0
    phase: str = "encode"
    first_step_at: float = 0.0
    eta_seconds: Optional[float] = None
    prompt_id: Optional[str] = None
    # The id Studio sent in the /prompt body, set before the request goes out.
    submit_id: Optional[str] = None
    queue_position: Optional[int] = None
    cancel_requested: bool = False
    # Set once the job is confirmed gone from ComfyUI's queue after a cancel (ComfyUI sends no message for a removed pending job).
    cancel_confirmed: bool = False
    admitted: bool = False
    # Studio's claim on the shared memory, registered by the arbiter at admission; released with the job.
    hold: Optional[arbiter.ComfyJobHold] = None
    # True once ComfyUI no longer holds the prompt: it finished, failed, was interrupted, or was cancelled and confirmed gone.
    terminal: bool = False
    stage: str = "starting"
    misses: int = 0
    poll_failures: int = 0


def _estimate_eta(total_steps: int, step: int, first_step_at: float, now: float) -> Optional[float]:
    """Seconds remaining, from the average step time after the first step; None until two steps have run."""
    steps_since_first = step - 1
    if not first_step_at or steps_since_first <= 0:
        return None
    per_step = (now - first_step_at) / steps_since_first
    return max(0.0, (total_steps - step) * per_step)


def configured_model_dirs() -> list[str]:
    """``[comfyui] model_dirs`` from engines.toml, for the missing-model message (best effort)."""
    try:
        import tomllib

        from core.inference.attached.failures import engines_home

        path = Path(os.environ.get("UNSLOTH_ENGINES_CONFIG") or engines_home() / "engines.toml")
        dirs = tomllib.loads(path.read_text(encoding = "utf-8")).get("comfyui", {}).get("model_dirs")
        if isinstance(dirs, list) and dirs and all(isinstance(d, str) for d in dirs):
            return dirs
    except Exception:
        pass
    return list(DEFAULT_MODEL_DIRS)


async def missing_models(client: ComfyuiClient, required: dict[str, list[str]]) -> dict[str, list[str]]:
    """``{folder: [names ComfyUI does not list]}``; raises ``ComfyuiError`` when ComfyUI cannot be asked."""
    folders = [folder for folder, names in required.items() if names]
    listings = await asyncio.gather(*(client.models(folder) for folder in folders))
    out: dict[str, list[str]] = {}
    for folder, have in zip(folders, listings):
        absent = [name for name in required[folder] if name not in set(have)]
        if absent:
            out[folder] = absent
    return out


def _ws_url(base_url: str, client_id: str) -> str:
    parts = urlsplit(base_url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return f"{scheme}://{parts.netloc}/ws?clientId={quote(client_id, safe = '')}"


def _default_connect(url: str, **kwargs: Any):
    from websockets.asyncio.client import connect

    return connect(url, **kwargs)


def _is_sampler(class_type: str) -> bool:
    return class_type.startswith("KSampler") or "Sampler" in class_type


class ComfyJobRunner:
    # Seconds between /history checks without a websocket, and the safety net beside one.
    poll_interval = 1.0
    ws_safety_poll = 5.0
    history_retries = 20
    # How long a cancel keeps retrying until ComfyUI no longer lists the job, and the pause between attempts.
    cancel_wait = 20.0
    cancel_poll = 0.5
    # A job that could not be confirmed stopped keeps the runner busy while this retries, up to this many seconds.
    reap_wait = 600.0

    def __init__(self, *, connect: Optional[Callable[..., Any]] = None, sleep: Callable[..., Any] = asyncio.sleep):
        self._connect = connect
        self._sleep = sleep
        self._state: Optional[_State] = None
        self._client: Optional[ComfyuiClient] = None
        self._reaper: Optional[asyncio.Task] = None

    # ------------------------------------------------------------- state

    def is_running(self) -> bool:
        return self._state is not None

    def progress(self) -> dict:
        """The ``DiffusionGenerateProgressResponse`` fields, plus ``queue_position``, ``prompt_id`` and ``engine``."""
        st = self._state
        if st is None:
            return {
                "active": False, "step": 0, "total_steps": 0, "fraction": 0.0, "eta_seconds": None,
                "phase": None, "preview": None, "preview_seq": 0,
                "queue_position": None, "prompt_id": None, "engine": "comfyui",
            }
        total = st.total_steps
        return {
            "active": True,
            "step": st.step,
            "total_steps": total,
            "fraction": max(0.0, min(1.0, st.step / total)) if total > 0 else 0.0,
            "eta_seconds": st.eta_seconds,
            "phase": st.phase,
            "preview": None,
            "preview_seq": 0,
            "queue_position": st.queue_position,
            "prompt_id": st.prompt_id,
            "engine": "comfyui",
        }

    # --------------------------------------------------------------- run

    async def run(
        self,
        template_id: str,
        params: dict,
        *,
        client: Optional[ComfyuiClient] = None,
        timeout_s: Optional[float] = None,
    ) -> ComfyJobResult:
        # No await between the check and the assignment: the event loop makes this atomic.
        if self._state is not None:
            raise ComfyBusyError("A ComfyUI job is already running.")
        template = get_template(template_id)
        if template is None:
            raise ComfyTemplateNotFound(f"Unknown ComfyUI template {template_id!r}.")
        graph, resolved = build_graph(template, params)
        st = self._state = _State(template_id = template.id, total_steps = int(resolved["steps"]))
        client = self._client = client or ComfyuiClient(get_config().comfyui_url)
        try:
            return await self._run(client, template, graph, resolved, st, timeout_s)
        finally:
            if st.prompt_id and not st.terminal:
                # ComfyUI may still hold the prompt (a cancel or a timeout could not be confirmed): keep owning it.
                st.stage = "stopping"
                self._reaper = asyncio.get_running_loop().create_task(self._reap(client, st))
            else:
                self._release(st)

    def _release(self, st: _State) -> None:
        """Let go of the job: the memory claim first, then the runner; the idle timer restarts last."""
        if st.hold is not None:
            st.hold.release()
        if self._state is st:
            self._state = None
            self._client = None
        if st.admitted:
            with contextlib.suppress(Exception):
                arbiter.schedule_comfyui_idle_free()

    async def _reap(self, client: ComfyuiClient, st: _State) -> None:
        """Keep trying to stop a prompt ComfyUI would not let go of; the runner stays busy until it has."""
        try:
            loop = asyncio.get_running_loop()
            give_up = loop.time() + self.reap_wait
            while not await self._cancel_and_confirm(client, st.prompt_id):
                if loop.time() >= give_up:
                    logger.warning("ComfyUI still holds prompt %s after %s s; releasing Studio's claim", st.prompt_id, int(self.reap_wait))
                    break
            else:
                st.terminal = True
        except Exception as exc:
            logger.warning("ComfyUI job reaper failed: %s", exc)
        finally:
            self._reaper = None
            self._release(st)

    async def _run(
        self, client: ComfyuiClient, template: Template, graph: dict, resolved: dict, st: _State, timeout_s: Optional[float]
    ) -> ComfyJobResult:
        queue = await client.queue()
        if queue.state == "down":
            if desktop_settings.helper_wanted("comfyui") is True:
                raise ComfyNotRunningError(
                    "ComfyUI is not running yet; give it a moment or check Settings > Engines."
                )
            raise ComfyOffError(OFF_MESSAGE)
        if queue.state == "unknown":
            raise ComfyuiError("ComfyUI did not report its queue.")

        required = {folder: list(names) for folder, names in template.required_models.items()}
        lora_names = [entry.rsplit(":", 1)[0] for entry in resolved["loras"]]
        if lora_names:
            required["loras"] = required.get("loras", []) + [n for n in lora_names if n not in required.get("loras", [])]
        absent = await missing_models(client, required)
        if absent:
            raise ComfyModelsMissing(absent, configured_model_dirs())

        st.hold = arbiter.new_comfyui_hold()
        admission = await arbiter.before_comfyui_job(st.hold)
        admission.require_clear()
        st.admitted = True
        if st.cancel_requested:
            raise ComfyCancelled(_cancelled_message())

        client_id = uuid.uuid4().hex
        timeout = timeout_s if timeout_s is not None else _job_timeout()
        async with contextlib.AsyncExitStack() as stack:
            ws = await self._open_ws(stack, client, client_id)
            # Studio names the prompt itself and records it before sending, so a /prompt that times out after
            # ComfyUI queued it can still be cancelled by id. st.prompt_id stays unset until the reply: cancel()
            # treats an unset id as "not submitted yet" and run() cancels right after submission.
            sent_id = str(uuid.uuid4())
            st.submit_id = sent_id
            try:
                prompt_id, number = await client.submit(
                    graph, client_id, {"unsloth": {"template": template.id}}, prompt_id = sent_id
                )
            except ComfyGraphError as exc:
                # A 400: ComfyUI rejected the graph and queued nothing.
                st.submit_id = None
                raise ComfyGraphError(_graph_error_text(exc), exc.node_errors) from exc
            except BaseException:
                # A timeout, a transport error, a 5xx or a task cancellation: ComfyUI may have queued the prompt
                # anyway. Own it (the reaper keeps trying when this cannot confirm it gone) and stop it by id.
                st.prompt_id = sent_id
                st.terminal = await asyncio.shield(self._cancel_and_confirm(client, sent_id))
                raise
            if prompt_id != sent_id:
                logger.warning("ComfyUI answered /prompt with id %s, not the %s Studio sent", prompt_id, sent_id)
            st.prompt_id, st.stage = prompt_id, "queued"
            try:
                if st.cancel_requested:
                    st.cancel_confirmed = await self._cancel_and_confirm(client, prompt_id)
                await self._watch(client, graph, st, ws, prompt_id, timeout)
                st.terminal = True
            except (ComfyJobError, ComfyCancelled):
                # An execution error, an interrupt or a vanished job: ComfyUI is done with the prompt.
                st.terminal = True
                raise
            except ComfyTimeout as exc:
                stopped = await self._cancel_and_confirm(client, prompt_id)
                st.terminal = stopped
                raise ComfyTimeout(
                    f"The ComfyUI job did not finish within {int(timeout)} s and was "
                    + ("cancelled." if stopped else "asked to stop, but ComfyUI still lists it; it may still be running.")
                ) from exc
            except BaseException:
                # Task cancellation or a ComfyUI that stopped answering: stop our prompt (pending or running) before letting go.
                st.terminal = await asyncio.shield(self._cancel_and_confirm(client, prompt_id))
                raise

        st.stage, st.phase = "saving", "decode"
        entry = await self._final_history(client, prompt_id)
        records = await self._persist(client, entry, template, resolved)
        return ComfyJobResult(
            images = records, actions = list(admission.actions), prompt_id = prompt_id, resolved = resolved
        )

    # ---------------------------------------------------------- websocket

    async def _open_ws(self, stack: contextlib.AsyncExitStack, client: ComfyuiClient, client_id: str):
        connect = self._connect or _default_connect
        try:
            return await stack.enter_async_context(
                connect(_ws_url(client.base_url, client_id), open_timeout = 5, max_size = 2**24)
            )
        except Exception as exc:
            logger.info("ComfyUI websocket unavailable, polling instead: %s", type(exc).__name__)
            return None

    # -------------------------------------------------------------- watch

    async def _watch(self, client: ComfyuiClient, graph: dict, st: _State, ws: Any, prompt_id: str, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        def next_poll() -> float:
            return loop.time() + (self.poll_interval if ws is None else self.ws_safety_poll)

        due = next_poll()
        while True:
            if st.cancel_confirmed:
                await self._resolve_cancel(client, prompt_id)
                return
            now = loop.time()
            if now >= deadline:
                raise ComfyTimeout(f"The ComfyUI job did not finish within {int(timeout)} s.")
            wait = max(0.0, min(due, deadline) - now)
            if ws is not None:
                try:
                    # Wake at least twice a second so a cancel that removed a queued job (no ws message) lands promptly. A zero timeout would cancel recv before it reads a ready frame.
                    raw = await asyncio.wait_for(ws.recv(), timeout = max(min(wait, 0.5), 0.01))
                except asyncio.TimeoutError:
                    raw = None
                except Exception as exc:
                    logger.info("ComfyUI websocket closed, polling instead: %s", type(exc).__name__)
                    ws, raw = None, None
                    due = next_poll()
                if raw is not None and self._handle(st, graph, prompt_id, raw):
                    return
            else:
                await self._sleep(wait)
            if loop.time() >= due:
                if await self._poll(client, st, prompt_id):
                    return
                due = next_poll()

    def _handle(self, st: _State, graph: dict, prompt_id: str, raw: Any) -> bool:
        """Apply one websocket frame; True when the job finished. Raises on an error or an interrupt."""
        if isinstance(raw, (bytes, bytearray)):
            return False
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return False
        kind = message.get("type") if isinstance(message, dict) else None
        data = message.get("data") if isinstance(message, dict) else None
        if not isinstance(data, dict) or data.get("prompt_id") not in (None, prompt_id):
            return False
        if kind == "execution_start":
            st.stage = "running"
            st.queue_position = 0
        elif kind == "executing":
            node = data.get("node")
            if node is None:
                return True
            class_type = str((graph.get(str(node)) or {}).get("class_type", ""))
            st.stage, st.queue_position = "running", 0
            if _is_sampler(class_type):
                st.phase = "denoise"
            elif class_type.startswith("VAEDecode"):
                st.phase = "decode"
        elif kind == "progress":
            class_type = str((graph.get(str(data.get("node"))) or {}).get("class_type", ""))
            value, maximum = data.get("value"), data.get("max")
            if (
                (_is_sampler(class_type) or (class_type == "" and st.phase == "denoise"))
                and isinstance(value, int) and isinstance(maximum, int) and maximum > 0
            ):
                now = time.monotonic()
                st.phase, st.stage, st.queue_position = "denoise", "running", 0
                st.step, st.total_steps = min(value, maximum), maximum
                if not st.first_step_at:
                    st.first_step_at = now
                st.eta_seconds = _estimate_eta(maximum, st.step, st.first_step_at, now)
        elif kind == "execution_success":
            return True
        elif kind == "execution_error":
            raise ComfyJobError(
                str(data.get("exception_message") or data.get("exception_type") or "execution error").strip(),
                data.get("node_type") if isinstance(data.get("node_type"), str) else None,
            )
        elif kind == "execution_interrupted":
            raise ComfyCancelled(_cancelled_message())
        return False

    async def _poll(self, client: ComfyuiClient, st: _State, prompt_id: str) -> bool:
        """One /queue + /history check. True when the job finished; raises when it failed or vanished."""
        queue = await client.queue()
        try:
            entry = await client.history(prompt_id)
        except ComfyuiError:
            entry = None
            queue_ok = False
        else:
            queue_ok = queue.reachable and not queue.uncertain
        if not queue_ok:
            st.poll_failures += 1
            if st.poll_failures >= 5:
                raise ComfyuiError("ComfyUI stopped answering during the job.")
            return False
        st.poll_failures = 0
        if prompt_id in queue.running_ids:
            st.queue_position = 0
        elif prompt_id in queue.pending_ids:
            st.queue_position = queue.pending_ids.index(prompt_id) + 1
        if entry is not None:
            return _history_outcome(entry)
        if prompt_id in queue.running_ids or prompt_id in queue.pending_ids:
            st.misses = 0
            return False
        st.misses += 1
        if st.misses >= 2:
            if st.cancel_requested:
                raise ComfyCancelled(_cancelled_message())
            raise ComfyJobError("The job left the ComfyUI queue without a result.")
        return False

    async def _final_history(self, client: ComfyuiClient, prompt_id: str) -> dict:
        """The history entry once ComfyUI has written it (it lags the success message by a moment)."""
        for _ in range(self.history_retries):
            entry = await client.history(prompt_id)
            if entry is not None and _history_outcome(entry):
                return entry
            await self._sleep(0.5)
        raise ComfyJobError("ComfyUI finished but did not report its result.")

    # ------------------------------------------------------------ outputs

    async def _persist(self, client: ComfyuiClient, entry: dict, template: Template, resolved: dict) -> list[dict]:
        outputs = entry.get("outputs") if isinstance(entry.get("outputs"), dict) else {}
        files = []
        for node_id in sorted(outputs, key = _natural_key):
            for image in (outputs[node_id] or {}).get("images", []) if isinstance(outputs[node_id], dict) else []:
                if isinstance(image, dict) and isinstance(image.get("filename"), str):
                    files.append(image)
        if not files:
            raise ComfyJobError("The graph produced no images.")
        blobs = [
            await client.view(f["filename"], str(f.get("subfolder") or ""), str(f.get("type") or "temp"))
            for f in files
        ]
        created_at = time.time()
        batch_size = len(files)

        def save() -> list[dict]:
            from PIL import Image

            from core.inference import image_gallery

            records = []
            for index, blob in enumerate(blobs):
                with Image.open(io.BytesIO(blob)) as opened:
                    opened.load()
                    image = opened.copy()
                records.append(
                    image_gallery.save(
                        image,
                        {
                            "prompt": resolved["prompt"],
                            "negative_prompt": resolved.get("negative_prompt") or None,
                            "width": image.width,
                            "height": image.height,
                            "steps": resolved["steps"],
                            "guidance": resolved["cfg"],
                            "seed": resolved["seed"],
                            "batch_seed": resolved["seed"],
                            "batch_index": index,
                            "batch_size": batch_size,
                            "model": f"comfyui:{template.id}",
                            "loras": list(resolved["loras"]),
                            "workflow": "create",
                            "created_at": created_at,
                            "engine": "comfyui",
                            "comfyui_template": template.id,
                            "sampler": resolved.get("sampler"),
                            "scheduler": resolved.get("scheduler"),
                        },
                    )
                )
            return records

        return await asyncio.to_thread(save)

    # ------------------------------------------------------------- cancel

    async def cancel(self) -> bool:
        """Stop Studio's own job and wait (bounded) until ComfyUI no longer has it.

        True when the job is confirmed gone from the queue, or has not been submitted yet (``run`` then cancels
        it right after submission). False when nothing is active, or when ComfyUI still shows the job after
        ``cancel_wait`` seconds: it stays owned by the runner and Cancel can be retried.
        """
        st, client = self._state, self._client
        if st is None or client is None:
            return False
        st.cancel_requested = True
        if st.prompt_id is None:
            return True
        if await self._cancel_and_confirm(client, st.prompt_id):
            st.cancel_confirmed = True
            return True
        return False

    async def _cancel_and_confirm(self, client: ComfyuiClient, prompt_id: str) -> bool:
        """Cancel one prompt of ours until it is absent from ComfyUI's queue; False if it still is after ``cancel_wait``.

        The pinned ComfyUI cancels atomically (``/api/jobs/{id}/cancel``: a pending prompt that started in the
        meantime is interrupted instead). Without that endpoint the fallback is queue delete or ``/interrupt``
        by the state seen in the latest ``/queue``; the loop re-reads the queue after every action, so a delete
        that raced with the start is followed by an interrupt on the next pass. Absent from the queue means
        terminal: ComfyUI writes a prompt to its history before it leaves the running list. A ComfyUI that
        is down holds nothing. Only ever names ``prompt_id``.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.cancel_wait
        use_endpoint = True
        while True:
            queue = await client.queue()
            if queue.state == "down":
                return True
            if queue.state != "unknown":
                if prompt_id not in queue.running_ids and prompt_id not in queue.pending_ids:
                    return True
                try:
                    acted = await client.cancel_job(prompt_id) if use_endpoint else None
                    if acted is None:
                        use_endpoint = False
                        if prompt_id in queue.pending_ids:
                            await client.delete_pending([prompt_id])
                        else:
                            await client.interrupt(prompt_id)
                except ComfyuiError as exc:
                    logger.warning("ComfyUI cancel failed: %s", exc)
            if loop.time() >= deadline:
                return False
            await self._sleep(self.cancel_poll)

    async def _resolve_cancel(self, client: ComfyuiClient, prompt_id: str) -> None:
        """After a confirmed cancel: a job that finished anyway is kept, anything else is a cancellation."""
        try:
            entry = await client.history(prompt_id)
        except ComfyuiError:
            entry = None
        if entry is not None and _history_outcome(entry):
            return
        raise ComfyCancelled(_cancelled_message())


def _cancelled_message() -> str:
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    return DIFFUSION_CANCELLED_MSG


def _job_timeout() -> float:
    try:
        value = float(os.environ.get(JOB_TIMEOUT_ENV, ""))
        return value if value > 0 else DEFAULT_JOB_TIMEOUT_S
    except ValueError:
        return DEFAULT_JOB_TIMEOUT_S


def _natural_key(value: str) -> tuple:
    return (0, int(value), "") if value.isdigit() else (1, 0, value)


def _graph_error_text(exc: ComfyGraphError) -> str:
    detail = []
    for node_id, info in list(exc.node_errors.items())[:3]:
        errors = info.get("errors") if isinstance(info, dict) else None
        first = errors[0] if isinstance(errors, list) and errors and isinstance(errors[0], dict) else {}
        detail.append(f"node {node_id} ({info.get('class_type', '?') if isinstance(info, dict) else '?'}): {first.get('message', 'invalid')}")
    return f"{exc}" + (f" ({'; '.join(detail)})" if detail else "")


def _history_outcome(entry: dict) -> bool:
    """True when a history entry is a finished success; raises for an interrupted or failed one."""
    status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
    for item in status.get("messages") or []:
        if not (isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict)):
            continue
        if item[0] == "execution_interrupted":
            raise ComfyCancelled(_cancelled_message())
        if item[0] == "execution_error":
            raise ComfyJobError(
                str(item[1].get("exception_message") or item[1].get("exception_type") or "execution error").strip(),
                item[1].get("node_type") if isinstance(item[1].get("node_type"), str) else None,
            )
    if status.get("status_str") == "error":
        raise ComfyJobError("ComfyUI reported an error.")
    return bool(status.get("completed")) or (not status and "outputs" in entry)


_runner = ComfyJobRunner()


def get_runner() -> ComfyJobRunner:
    return _runner
