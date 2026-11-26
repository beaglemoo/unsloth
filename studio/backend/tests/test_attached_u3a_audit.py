# SPDX-License-Identifier: AGPL-3.0-only
"""Pins for the attached-engine audit of the 2026-11-26 rebase: every path that reaches oMLX or keeps a local model
resident is either hooked (its own regression test) or listed here as deliberately unhooked, with the reason.

A new call site fails one of these tests on purpose: classify it (admit it and add a test, or list it here)
before changing the expected set."""
from __future__ import annotations

import ast
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def _tree(relative: str) -> ast.Module:
    return ast.parse((BACKEND / relative).read_text(encoding = "utf-8"))


def _is_name(node, name: str) -> bool:
    return (isinstance(node, ast.Name) and node.id == name) or (
        isinstance(node, ast.Attribute) and node.attr == name
    )


def _enclosing_top_level(tree: ast.Module) -> dict[int, str]:
    owner: dict[int, str] = {}
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for node in ast.walk(top):
                owner[id(node)] = top.name
    return owner


def _backend_sources() -> list[Path]:
    skipped = {"tests", ".venv", "node_modules"}
    return [
        path
        for path in sorted(BACKEND.rglob("*.py"))
        if not skipped & set(path.relative_to(BACKEND).parts)
    ]


# --- every ExternalProviderClient construction ---------------------------------------------------------------

# site -> why it is safe. HOOKED sites admit the request (targets_omlx) and hold it; METADATA sites only read the
# provider's model list or window from /v1/models, which never loads an oMLX model.
CONSTRUCTION_SITES = {
    ("core/rag/embed_connection.py", "_client_for"): "HOOKED: _admit_attached_omlx in ConnectionEmbeddingBackend._post (test_attached_rag_embed_admission)",
    ("core/research_runs.py", "_saved_connection_client"): "METADATA: served-window probe only; generation goes through the loopback /v1/chat/completions route",
    ("routes/inference.py", "_external_image_generation"): "HOOKED: @_holds_omlx + _admit_attached_omlx_provider",
    ("routes/inference.py", "_external_stt_transcription"): "HOOKED: @_holds_omlx + _admit_attached_omlx_provider",
    ("routes/inference.py", "_external_tts_speech"): "HOOKED: @_holds_omlx + _admit_attached_omlx_provider",
    ("routes/inference.py", "_proxy_to_external_provider"): "HOOKED: @_holds_omlx + _admit_attached_omlx_provider",
    ("routes/inference.py", "_resolve_openai_cloud_client"): "UNREACHABLE: OpenAI container API, refused unless the host is api.openai.com",
    ("routes/providers.py", "_catalog_window"): "METADATA: hosted catalogs (_CATALOG_WINDOW_PROVIDERS), never oMLX",
    ("routes/providers.py", "_list_model_capabilities"): "METADATA: model list",
    ("routes/providers.py", "list_provider_models"): "METADATA: model list",
    ("routes/providers.py", "load_served_window"): "METADATA: served window",
    ("routes/providers.py", "test_provider"): "HOOKED: _admit_attached_omlx_provider before the probe",
    ("routes/systemone.py", "_connection_decide"): "HOOKED: before_omlx_use(omlx_hold) before _connection_key",
    ("routes/systemone.py", "refresh_listed_decision_models"): "UNREACHABLE: openrouter rows only",
}


def test_every_external_provider_client_construction_site_is_classified():
    found = set()
    for path in _backend_sources():
        relative = str(path.relative_to(BACKEND))
        if relative == "core/inference/external_provider.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        owner = _enclosing_top_level(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # ExternalProviderClient(...) and asyncio.to_thread / functools.partial(ExternalProviderClient, ...)
            if _is_name(node.func, "ExternalProviderClient") or (
                node.args and _is_name(node.args[0], "ExternalProviderClient")
            ):
                found.add((relative, owner.get(id(node), "<module>")))
    assert found == set(CONSTRUCTION_SITES), (
        f"unclassified: {sorted(found - set(CONSTRUCTION_SITES))}; gone: {sorted(set(CONSTRUCTION_SITES) - found)}"
    )


def _referenced_in(relative: str, name: str) -> set[str]:
    """Top-level functions that mention ``name`` (called, or passed along, e.g. to ``asyncio.to_thread``)."""
    tree = _tree(relative)
    owner = _enclosing_top_level(tree)
    return {
        owner[id(node)]
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == name and id(node) in owner
    }


def test_the_research_probe_reads_metadata_and_generation_uses_the_loopback_route():
    source = (BACKEND / "core/research_runs.py").read_text(encoding = "utf-8")
    # The unadmitted client is built for the served-window probe only, and that probe only reads the window.
    assert _referenced_in("core/research_runs.py", "_saved_connection_client") == {"_guarded_connection_client"}
    assert _referenced_in("core/research_runs.py", "_guarded_connection_client") == {"_refresh_served_context_length"}
    probe = ast.unparse(
        next(n for n in _tree("core/research_runs.py").body if getattr(n, "name", "") == "_refresh_served_context_length")
    )
    assert "served_context_window(" in probe and "stream_chat_completion" not in probe
    # Generation is an HTTP call back into Studio's own chat route, which is the admitted proxy.
    assert "/v1/chat/completions" in source


# --- every local media load ----------------------------------------------------------------------------------


def test_the_only_media_begin_load_calls_are_inside_the_gated_loaders():
    sites = set()
    for relative in tuple(
        str(p.relative_to(BACKEND))
        for directory, pattern in (("routes", "*.py"), ("core/inference", "media_*.py"))
        for p in sorted((BACKEND / directory).glob(pattern))
    ):
        try:
            tree = _tree(relative)
        except SyntaxError:
            continue
        owner = _enclosing_top_level(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "begin_load":
                sites.add((relative, owner.get(id(node), "<module>")))
    assert sites == {
        ("routes/inference.py", "load_diffusion_model_gated"),
        ("routes/video.py", "load_video_model_gated"),
    }


def test_media_auto_switch_and_the_chat_swap_load_through_the_gated_loaders():
    switch = (BACKEND / "core/inference/media_auto_switch.py").read_text(encoding = "utf-8")
    assert "await load_diffusion_model_gated(" in switch and "await load_video_model_gated(" in switch
    swap = (BACKEND / "core/inference/chat_media_swap.py").read_text(encoding = "utf-8")
    assert "await _start_load(DIFFUSION" in swap and "_admit_attached_swap" in swap


def test_the_gated_loaders_admit_before_they_begin_the_load():
    for relative, function, begins in (
        ("routes/inference.py", "load_diffusion_model_gated", (".begin_load(", "begin_load_on(")),
        ("routes/video.py", "load_video_model_gated", ("begin_media_load(",)),
    ):
        node = next(n for n in _tree(relative).body if isinstance(n, ast.AsyncFunctionDef) and n.name == function)
        source = ast.unparse(node)
        admitted = source.index("await _admit_attached_local_load(media_hold)")
        assert source.count("await _admit_attached_local_load(media_hold)") == 1
        for begin in begins:
            assert admitted < source.index(begin), (function, begin)


def test_the_image_queue_never_loads_a_model():
    """A queued job runs only on the model that was resident when it was queued (a mismatch fails the job), so the
    residency the media probe sees is the residency the queue uses; nothing is admitted at dispatch because nothing loads."""
    queue = (BACKEND / "core/inference/image_queue.py").read_text(encoding = "utf-8")
    assert "begin_load" not in queue and "load_diffusion_model" not in queue
    tree = _tree("routes/inference.py")
    runner = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_queued_image_job")
    source = ast.unparse(runner)
    assert "begin_load" not in source and "load_diffusion_model" not in source
    assert "_QUEUE_MODEL_CHANGED_MSG" in source


# --- local chat loads ----------------------------------------------------------------------------------------


def test_the_ollama_lease_is_a_file_lock_taken_only_inside_the_load_and_validate_routes():
    # The load path is _load_model_impl (admitted there); validate_model materialises files and loads nothing.
    assert _referenced_in("routes/inference.py", "_lease_ollama_model_ref") == {"_load_model_impl", "validate_model"}


def test_computer_use_loads_nothing_and_never_reaches_a_provider():
    package = BACKEND / "core/inference/computer_use"
    text = "\n".join(path.read_text(encoding = "utf-8") for path in sorted(package.rglob("*.py")))
    for needle in ("external_provider", "ExternalProvider", "load_model", "llama_cpp", "omlx", "8843", "httpx"):
        assert needle not in text, needle


def test_a_worker_that_survives_a_failed_shutdown_is_recorded_in_one_place():
    tree = _tree("core/inference/orchestrator.py")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InferenceOrchestrator")
    sets_true, drops_handle, kills = set(), set(), set()
    for fn in cls.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "_unload_left_worker":
                        if isinstance(node.value, ast.Constant) and node.value.value is True:
                            sets_true.add(fn.name)
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "_proc"
                        and isinstance(node.value, ast.Constant)
                        and node.value.value is None
                        and fn.name != "__init__"
                    ):
                        drops_handle.add(fn.name)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("kill", "terminate")
                and "_proc" in ast.unparse(node.func.value)
            ):
                kills.add(fn.name)
    # Every shutdown (unload, failed load cleanup, narrow-context fallback) goes through this one function, which
    # records the survivor; the other two assignments of the flag (load, unload) re-derive it from liveness.
    assert sets_true == {"_shutdown_subprocess_locked"}
    assert drops_handle == {"_shutdown_subprocess_locked"}
    # _spawn_subprocess only kills the child it just started, before it is registered as the worker.
    assert kills == {"_shutdown_subprocess_locked", "_spawn_subprocess"}


def test_the_mlx_engine_runs_in_the_orchestrators_worker_so_its_survivor_accounting_applies():
    """MLXInferenceBackend is built only in the inference worker subprocess, which the orchestrator owns: a failed
    shutdown of that worker is what _unload_left_worker records (pinned above), so mlx_inference.py and mlx_memory.py
    (sizing and offload planning inside the worker) add no resident state of their own."""
    builders = set()
    for path in _backend_sources():
        relative = str(path.relative_to(BACKEND))
        if relative == "core/inference/mlx_inference.py":
            continue
        if "MLXInferenceBackend(" in path.read_text(encoding = "utf-8", errors = "ignore"):
            builders.add(relative)
    assert builders == {"core/inference/worker.py"}
    for relative in ("core/inference/mlx_inference.py", "core/inference/mlx_memory.py"):
        text = (BACKEND / relative).read_text(encoding = "utf-8")
        assert "worker_holds_unreleased_memory" not in text and "_unload_left_worker" not in text
