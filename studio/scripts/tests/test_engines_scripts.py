# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the engine wrappers and scripts. Every run uses a scratch UNSLOTH_ENGINES_HOME and
log dir; nothing touches ~/.unsloth or any live service.

    studio/scripts/tests: python -m pytest test_engines_scripts.py -q
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
ENGINES = SCRIPTS / "engines"

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="the engine scripts are macOS only")


def run(cmd, env_extra=None, **kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UNSLOTH_", "ENGINES_"))}
    # the legacy ds4 migration probes a launcher URL: never let a test reach the real :8001
    env["LEGACY_DS4_URL"] = "http://127.0.0.1:9"
    env.update(env_extra or {})
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60, **kw)


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "engines"
    h.mkdir()
    (tmp_path / "logs").mkdir()
    return h


def env_for(home, **extra):
    return {
        "UNSLOTH_ENGINES_HOME": str(home),
        "UNSLOTH_ENGINES_LOG_DIR": str(home.parent / "logs"),
        "ENGINES_FAIL_NO_SLEEP": "1",
        **extra,
    }


def fake_python(path: Path, body="echo fake-engine \"$@\"\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def read_fail(home, name):
    return json.loads((home / f"{name}.fail").read_text())


# --- engines_launch.py -----------------------------------------------------------------------


def launch(home, engine, cfg_text, *extra, **env):
    cfg = home / "engines.toml"
    cfg.write_text(cfg_text)
    return run(
        [sys.executable, str(ENGINES / "engines_launch.py"), engine, *extra],
        env_for(home, UNSLOTH_ENGINES_BUNDLE=str(home / "bundle"), **env),
    )


@pytest.mark.skipif(sys.version_info < (3, 11), reason="engines_launch needs tomllib")
class TestEnginesLaunch:
    def test_missing_venv_records_a_failure_and_counts_up(self, home):
        for expected in (1, 2, 3):
            result = launch(home, "omlx", "[omlx]\nport = 1\n")
            assert result.returncode == 78
            fail = read_fail(home, "omlx")
            assert fail["count"] == expected
            assert "oMLX venv missing" in fail["reason"]
            assert fail["ts"] > 1_700_000_000

    def test_missing_config_is_a_failure(self, home):
        result = run(
            [sys.executable, str(ENGINES / "engines_launch.py"), "omlx"],
            env_for(home),
        )
        assert result.returncode == 78
        assert "config not found" in read_fail(home, "omlx")["reason"]

    def test_port_in_use_is_a_failure_and_a_free_port_clears_it(self, home):
        fake_python(home / "omlx" / "bin" / "python")
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            port = busy.getsockname()[1]
            result = launch(home, "omlx", f"[omlx]\nport = {port}\n")
        assert result.returncode == 78
        fail = read_fail(home, "omlx")
        assert f"port {port}" in fail["reason"] and fail["count"] == 1
        # the listener is gone: the engine is exec'd (the fake interpreter) and the marker removed
        result = launch(home, "omlx", f"[omlx]\nport = {port}\n")
        assert result.returncode == 0, result.stderr
        assert "fake-engine" in result.stdout and "-m omlx.cli serve" in result.stdout
        assert not (home / "omlx.fail").exists()

    def test_print_mode_leaves_no_marker_and_ignores_a_busy_port(self, home):
        fake_python(home / "omlx" / "bin" / "python")
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            result = launch(home, "omlx", f"[omlx]\nport = {busy.getsockname()[1]}\n", "--print")
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["argv"][1:4] == ["-m", "omlx.cli", "serve"]
        assert not (home / "omlx.fail").exists()

    def test_backoff_schedule(self):
        sys.path.insert(0, str(ENGINES))
        try:
            import engines_launch as el
        finally:
            sys.path.remove(str(ENGINES))
        assert [el.backoff_seconds(n) for n in (1, 2, 3, 4, 5, 6, 7, 50)] == [10, 20, 40, 80, 160, 300, 300, 300]

    def test_a_malformed_marker_restarts_the_count(self, home):
        (home / "omlx.fail").write_text("garbage")
        launch(home, "omlx", "[omlx]\nport = 1\n")
        assert read_fail(home, "omlx")["count"] == 1


def comfy_layout(home, main=True):
    """A fake ComfyUI runtime: a python that echoes its argv, and src/main.py."""
    fake_python(home / "comfyui" / "bin" / "python")
    if main:
        (home / "comfyui" / "src").mkdir(parents=True, exist_ok=True)
        (home / "comfyui" / "src" / "main.py").write_text("")


def comfy_cfg(home, port=18844, models=None, host=None, extra=""):
    lines = ["[comfyui]", f"port = {port}", f'data_dir = "{home}/data"']
    if host:
        lines.append(f'host = "{host}"')
    if models is not None:
        lines.append("model_dirs = [" + ", ".join(f'"{m}"' for m in models) + "]")
    lines.append('extra_args = ["--lowvram", "--offline"]')
    lines += ["", "[comfyui.env]", 'COMFYUI_TEST = "1"']
    return "\n".join(lines) + "\n" + extra


@contextlib.contextmanager
def _ipv6_socket():
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.bind(("::1", 0))
    except OSError:
        pytest.skip("IPv6 loopback is not usable here")
    sock.listen()
    try:
        yield sock, sock.getsockname()[1]
    finally:
        sock.close()


def ipv6_listener_port():
    """A context manager holding a listener on ::1 (skips the test when ::1 is unusable)."""
    return _ipv6_socket()


def listen_as(marker_arg, tmp_path):
    """A listener process whose argv carries marker_arg (short, so it survives the launcher's
    truncation of the owner's command line). Returns (process, port)."""
    script = tmp_path / "listener.py"
    script.write_text(
        "import socket, time\n"
        "s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen()\n"
        "print(s.getsockname()[1], flush=True); time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "listener.py", marker_arg], cwd=tmp_path, stdout=subprocess.PIPE, text=True)
    return proc, int(proc.stdout.readline())


@pytest.mark.skipif(sys.version_info < (3, 11), reason="engines_launch needs tomllib")
class TestComfyuiLaunch:
    def test_print_shows_the_argv_cwd_and_env_and_writes_nothing(self, home):
        comfy_layout(home)
        result = launch(home, "comfyui", comfy_cfg(home, models=[str(home / "models")]), "--print")
        assert result.returncode == 0, result.stderr
        shown = json.loads(result.stdout)
        src = home / "comfyui" / "src"
        assert shown["argv"] == [
            str(home / "comfyui" / "bin" / "python"), str(src / "main.py"),
            "--listen", "127.0.0.1", "--port", "18844",
            "--base-directory", str(home / "data"),
            "--extra-model-paths-config", str(home / "data" / "extra_model_paths.unsloth.yaml"),
            "--disable-auto-launch", "--lowvram", "--offline",
        ]
        assert shown["cwd"] == str(src)
        assert shown["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "1" and shown["env"]["COMFYUI_TEST"] == "1"
        assert not (home / "data").exists() and not (home / "comfyui.fail").exists()

    def test_the_default_data_dir_is_under_the_engines_home(self, home):
        comfy_layout(home)
        result = launch(home, "comfyui", "[comfyui]\nport = 18844\n", "--print")
        argv = json.loads(result.stdout)["argv"]
        assert argv[argv.index("--base-directory") + 1] == str(home / "comfyui-data")

    def test_yaml_for_zero_dirs_is_an_empty_mapping(self, home):
        comfy_layout(home)
        result = launch(home, "comfyui", comfy_cfg(home, models=[]))
        assert result.returncode == 0, result.stderr
        text = (home / "data" / "extra_model_paths.unsloth.yaml").read_text()
        assert text.splitlines()[1:] == ["{}"]  # an empty file would load as None and crash ComfyUI
        yaml = pytest.importorskip("yaml")
        assert yaml.safe_load(text) == {}

    def test_yaml_for_one_dir_is_exact(self, home):
        comfy_layout(home)
        models = home.parent / "weights"
        models.mkdir()
        assert launch(home, "comfyui", comfy_cfg(home, models=[str(models)])).returncode == 0
        text = (home / "data" / "extra_model_paths.unsloth.yaml").read_text()
        assert text.splitlines()[1:] == [
            "unsloth_shared_0:",
            f'  base_path: "{models}"',
            "  is_default: false",
            '  checkpoints: "checkpoints"',
            '  diffusion_models: "diffusion_models\\nunet"',
            '  text_encoders: "text_encoders\\nclip"',
            '  vae: "vae"',
            '  loras: "loras"',
            '  clip_vision: "clip_vision"',
            '  controlnet: "controlnet"',
            '  upscale_models: "upscale_models"',
            '  embeddings: "embeddings"',
            '  latent_upscale_models: "latent_upscale_models"',
            '  model_patches: "model_patches"',
        ]

    def test_yaml_for_two_dirs_parses_like_comfyuis_loader_expects(self, home):
        yaml = pytest.importorskip("yaml")
        comfy_layout(home)
        a, b = home.parent / "a", home.parent / "b"
        a.mkdir(), b.mkdir()
        assert launch(home, "comfyui", comfy_cfg(home, models=[str(a), str(b)])).returncode == 0
        config = yaml.safe_load((home / "data" / "extra_model_paths.unsloth.yaml").read_text())
        assert list(config) == ["unsloth_shared_0", "unsloth_shared_1"]
        for name, base in zip(config, (a, b)):
            conf = config[name]
            assert conf.pop("base_path") == str(base) and conf.pop("is_default") is False
            # what comfy's extra_config does with each remaining key
            assert all(isinstance(v, str) for v in conf.values())
            assert conf["diffusion_models"].split("\n") == ["diffusion_models", "unet"]
            assert conf["text_encoders"].split("\n") == ["text_encoders", "clip"]

    def test_a_tilde_in_model_dirs_is_expanded_before_it_is_written(self, home):
        comfy_layout(home)
        assert launch(home, "comfyui", comfy_cfg(home, models=["~/nowhere"])).returncode == 0
        text = (home / "data" / "extra_model_paths.unsloth.yaml").read_text()
        assert f'base_path: "{Path.home()}/nowhere"' in text

    def test_a_missing_model_dir_warns_and_still_execs(self, home):
        comfy_layout(home)
        gone = home.parent / "not-there"
        result = launch(home, "comfyui", comfy_cfg(home, models=[str(gone)]))
        assert result.returncode == 0, result.stderr
        assert "model dir not found" in result.stderr and str(gone) in result.stderr
        assert "fake-engine" in result.stdout and "--extra-model-paths-config" in result.stdout
        assert str(gone) in (home / "data" / "extra_model_paths.unsloth.yaml").read_text()
        assert not (home / "comfyui.fail").exists()

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "::"])
    def test_a_non_loopback_host_writes_the_marker(self, home, host):
        comfy_layout(home)
        result = launch(home, "comfyui", comfy_cfg(home, host=host, models=[]))
        assert result.returncode == 78
        assert "not loopback" in read_fail(home, "comfyui")["reason"]
        assert not (home / "data").exists()

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_hosts_are_accepted(self, home, host):
        comfy_layout(home)
        result = launch(home, "comfyui", comfy_cfg(home, host=host, models=[]), "--print")
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)["argv"]
        assert argv[argv.index("--listen") + 1] == host

    def test_a_port_held_by_storypress_names_it(self, home):
        comfy_layout(home)
        holder, port = listen_as("StoryPressRuntime/ComfyUI/main.py", home.parent)
        try:
            result = launch(home, "comfyui", comfy_cfg(home, port=port, models=[]))
        finally:
            holder.kill()
            holder.wait()
        assert result.returncode == 78
        fail = read_fail(home, "comfyui")
        assert f"port {port} (ComfyUI) is held by StoryPress's ComfyUI (pid {holder.pid})" in fail["reason"]
        assert "stop it or change [comfyui] port" in fail["reason"]
        assert not (home / "data").exists()

    def test_a_port_held_by_another_process_names_its_command(self, home):
        comfy_layout(home)
        holder, port = listen_as("some-other-service", home.parent)
        try:
            result = launch(home, "comfyui", comfy_cfg(home, port=port, models=[]))
        finally:
            holder.kill()
            holder.wait()
        assert result.returncode == 78
        reason = read_fail(home, "comfyui")["reason"]
        assert f"port {port} (ComfyUI) is held by " in reason and "some-other-service" in reason
        assert f"(pid {holder.pid})" in reason and "StoryPress" not in reason

    def test_the_owner_is_named_for_omlx_too(self, home):
        fake_python(home / "omlx" / "bin" / "python")
        holder, port = listen_as("some-other-service", home.parent)
        try:
            result = launch(home, "omlx", f"[omlx]\nport = {port}\n")
        finally:
            holder.kill()
            holder.wait()
        assert result.returncode == 78
        assert f"port {port} (oMLX) is held by " in read_fail(home, "omlx")["reason"]

    def test_print_ignores_a_busy_port(self, home):
        comfy_layout(home)
        holder, port = listen_as("some-other-service", home.parent)
        try:
            result = launch(home, "comfyui", comfy_cfg(home, port=port, models=[]), "--print")
        finally:
            holder.kill()
            holder.wait()
        assert result.returncode == 0 and not (home / "comfyui.fail").exists()

    @pytest.mark.parametrize("what", ["python", "main"])
    def test_a_missing_venv_or_source_writes_the_marker(self, home, what):
        comfy_layout(home, main=what != "main")
        if what == "python":
            (home / "comfyui" / "bin" / "python").unlink()
        result = launch(home, "comfyui", comfy_cfg(home, models=[]))
        assert result.returncode == 78
        reason = read_fail(home, "comfyui")["reason"]
        assert "ComfyUI venv missing" in reason and "build-engines-mac.sh" in reason

    def test_a_successful_start_clears_the_marker(self, home):
        comfy_layout(home)
        launch(home, "comfyui", comfy_cfg(home, host="0.0.0.0", models=[]))
        assert read_fail(home, "comfyui")["count"] == 1
        result = launch(home, "comfyui", comfy_cfg(home, models=[]))
        assert result.returncode == 0, result.stderr
        assert not (home / "comfyui.fail").exists()
        assert "fake-engine" in result.stdout

    def test_an_ipv6_loopback_host_launches_when_the_port_is_free(self, home):
        with ipv6_listener_port():  # skips when ::1 is unusable here
            pass
        comfy_layout(home)
        result = launch(home, "comfyui", comfy_cfg(home, host="::1", port=free_port(), models=[]))
        assert result.returncode == 0, result.stderr
        assert "--listen ::1" in result.stdout and "Traceback" not in result.stderr
        assert not (home / "comfyui.fail").exists()

    def test_an_ipv6_loopback_host_with_a_busy_port_writes_the_marker(self, home):
        comfy_layout(home)
        with ipv6_listener_port() as (sock, port):
            result = launch(home, "comfyui", comfy_cfg(home, host="::1", port=port, models=[]))
            assert sock.fileno() != -1
        assert result.returncode == 78 and "Traceback" not in result.stderr
        assert f"port {port} (ComfyUI)" in read_fail(home, "comfyui")["reason"]

    def test_an_unresolvable_host_is_a_failure_marker_not_a_traceback(self, home):
        fake_python(home / "omlx" / "bin" / "python")
        result = launch(home, "omlx", f'[omlx]\nport = {free_port()}\nhost = "no-such-host.invalid"\n')
        assert result.returncode == 78 and "Traceback" not in result.stderr
        assert "cannot check whether port" in read_fail(home, "omlx")["reason"]

    @pytest.mark.parametrize("extra", [
        ["--listen"], ["--listen", "0.0.0.0"], ["--listen=0.0.0.0"], ["--lis", "::"], ["--l"],
        ["--port", "9"], ["--port=9"], ["--po", "9"],
        ["--tls-keyfile", "k"], ["--tls-certfile=c"], ["--tls-k", "k"],
        ["--enable-cors-header"], ["--enable-cors-header=*"], ["--enable-cors", "*"],
        ["--base-directory", "/x"], ["--base-dir=/x"],
        ["--lowvram", "--listen 0.0.0.0"],
    ])
    def test_extra_args_cannot_override_the_binding_flags(self, home, extra):
        comfy_layout(home)
        args = ", ".join(json.dumps(a) for a in extra)
        cfg = comfy_cfg(home, models=[]).replace('extra_args = ["--lowvram", "--offline"]', f"extra_args = [{args}]")
        result = launch(home, "comfyui", cfg)
        assert result.returncode == 78 and "fake-engine" not in result.stdout
        reason = read_fail(home, "comfyui")["reason"]
        assert "extra_args sets" in reason and "[comfyui]" in reason
        assert not (home / "data").exists()

    def test_the_default_extra_args_and_harmless_flags_are_accepted(self, home):
        comfy_layout(home)
        extra = '["--lowvram", "--disable-smart-memory", "--cpu-vae", "--disable-all-custom-nodes", "--offline", "--preview-method", "auto"]'
        cfg = comfy_cfg(home, models=[]).replace('["--lowvram", "--offline"]', extra)
        result = launch(home, "comfyui", cfg, "--print")
        assert result.returncode == 0, result.stderr

    def test_print_also_rejects_an_overriding_flag_without_a_marker(self, home):
        comfy_layout(home)
        cfg = comfy_cfg(home, models=[]).replace('["--lowvram", "--offline"]', '["--listen"]')
        result = launch(home, "comfyui", cfg, "--print")
        assert result.returncode == 78 and "extra_args sets" in result.stderr
        assert not (home / "comfyui.fail").exists()

    @pytest.mark.parametrize("extra", [["--port", "9"], ["--host=0.0.0.0"], ["--base-path", "/x"], ["--por=9"]])
    def test_omlx_extra_args_cannot_override_the_launchers_flags_either(self, home, extra):
        fake_python(home / "omlx" / "bin" / "python")
        args = ", ".join(json.dumps(a) for a in extra)
        result = launch(home, "omlx", f"[omlx]\nport = {free_port()}\nextra_args = [{args}]\n")
        assert result.returncode == 78 and "fake-engine" not in result.stdout
        assert "extra_args sets" in read_fail(home, "omlx")["reason"]

    def test_omlx_keeps_accepting_its_other_extra_args(self, home):
        fake_python(home / "omlx" / "bin" / "python")
        result = launch(home, "omlx", f'[omlx]\nport = {free_port()}\nextra_args = ["--log-level", "debug"]\n')
        assert result.returncode == 0, result.stderr
        assert "--log-level debug" in result.stdout

    def test_the_usage_names_both_engines(self, home):
        result = run([sys.executable, str(ENGINES / "engines_launch.py"), "bogus"], env_for(home))
        assert result.returncode == 78 and "omlx|comfyui" in result.stderr


# --- wrappers --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper,name,log",
    [("omlx-launch", "omlx", "omlx.log"), ("comfyui-launch", "comfyui", "comfyui.log")],
)
class TestWrappers:
    def test_missing_venv_writes_the_marker_and_the_log(self, home, wrapper, name, log):
        for expected in (1, 2):
            result = run(["bash", str(ENGINES / wrapper)], env_for(home))
            assert result.returncode == 78
            fail = read_fail(home, name)
            assert fail["count"] == expected and fail["reason"].startswith("venv missing: ")
        text = (home.parent / "logs" / log).read_text()
        assert "venv missing" in text and "failure 2" not in text  # NO_SLEEP: no backoff message

    def test_reason_is_valid_json_with_hostile_paths(self, home, wrapper, name, log):
        weird = home.parent / 'we"ird\\home'
        weird.mkdir()
        result = run(["bash", str(ENGINES / wrapper)], env_for(weird))
        assert result.returncode == 78
        reason = json.loads((weird / f"{name}.fail").read_text())["reason"]
        assert str(weird) in reason

    def test_rotates_a_log_over_20mb_and_keeps_three(self, home, wrapper, name, log):
        logs = home.parent / "logs"
        (logs / f"{log}.1").write_text("one")
        (logs / f"{log}.2").write_text("two")
        (logs / f"{log}.3").write_text("three")
        with (logs / log).open("wb") as fh:
            fh.truncate(21 * 1024 * 1024)
        run(["bash", str(ENGINES / wrapper)], env_for(home))
        assert (logs / f"{log}.1").stat().st_size == 21 * 1024 * 1024
        assert (logs / f"{log}.2").read_text() == "one"
        assert (logs / f"{log}.3").read_text() == "two"
        assert not (logs / f"{log}.4").exists()
        assert (logs / log).stat().st_size < 4096

    def test_leaves_a_small_log_alone(self, home, wrapper, name, log):
        logs = home.parent / "logs"
        (logs / log).write_text("x" * 1000)
        run(["bash", str(ENGINES / wrapper)], env_for(home))
        assert (logs / log).read_text().startswith("x" * 1000)
        assert not (logs / f"{log}.1").exists()


def test_comfyui_launch_is_executable_and_runs_the_launcher_with_the_comfyui_venv(home):
    wrapper = ENGINES / "comfyui-launch"
    assert wrapper.stat().st_mode & stat.S_IXUSR
    text = wrapper.read_text()
    assert 'PY="$ENGINES_HOME/comfyui/bin/python"' in text
    assert 'exec "$PY" "$HERE/engines_launch.py" comfyui "$@"' in text
    assert "comfyui.log" in text
    # a venv python that echoes its argv is exec'd with the launcher script and the engine name
    fake_python(home / "comfyui" / "bin" / "python")
    result = run(["bash", str(wrapper)], env_for(home))
    assert result.returncode == 0
    log = (home.parent / "logs" / "comfyui.log").read_text()
    assert "engines_launch.py comfyui" in log


def test_build_engines_stages_the_common_helper():
    text = (SCRIPTS / "build-engines-mac.sh").read_text()
    assert "engines-common.sh" in text


def migrate(home_dir, snippet, **env):
    """Source the migrate script (its main is guarded) under a scratch HOME and run a snippet."""
    return run(
        ["bash", "-c", f'. "{SCRIPTS}/migrate-engines-mac.sh"; {snippet}'],
        {"HOME": str(home_dir), "APPS_DIR": str(home_dir / "Apps"), **env},
    )


def make_app(path: Path, marker="x", cli=None):
    """A fake bundle. cli: None = no executable, "cli" = an unsloth-studio that supports
    `--engine-helpers` (fakes/fake-unsloth-studio), "old" = one that does not and would start the
    GUI for any argument (it logs gui-started)."""
    (path / "Contents").mkdir(parents=True)
    (path / "Contents" / "marker").write_text(marker)
    if cli is None:
        return
    binary = path / "Contents" / "MacOS" / "unsloth-studio"
    binary.parent.mkdir()
    if cli == "cli":
        binary.write_text((FAKES / "fake-unsloth-studio").read_text().replace("__FAKES__", str(FAKES)))
    else:
        binary.write_text('#!/bin/bash\necho gui-started >>"${FAKE_LC:-/dev/null}"\n')
    binary.chmod(0o755)


def test_rollback_moves_the_retired_apps_back(tmp_path):
    retired = tmp_path / ".unsloth" / "engines" / "migration" / "retired-apps"
    make_app(retired / "oMLX.app", "omlx")
    make_app(retired / "EngineBar.app", "bar")
    (tmp_path / "Apps").mkdir()
    (tmp_path / "Applications").mkdir()
    result = migrate(tmp_path, "restore_retired_apps")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "Apps" / "oMLX.app" / "Contents" / "marker").read_text() == "omlx"
    assert (tmp_path / "Applications" / "EngineBar.app" / "Contents" / "marker").read_text() == "bar"
    assert not (retired / "oMLX.app").exists() and not (retired / "EngineBar.app").exists()
    # idempotent: a second run keeps what is in place and does not complain about the empty folder
    again = migrate(tmp_path, "restore_retired_apps")
    assert again.returncode == 0 and "already in place" in again.stdout


def test_rollback_never_replaces_an_app_that_is_already_there(tmp_path):
    retired = tmp_path / ".unsloth" / "engines" / "migration" / "retired-apps"
    make_app(retired / "oMLX.app", "retired")
    make_app(tmp_path / "Apps" / "oMLX.app", "current")
    result = migrate(tmp_path, "restore_retired_apps")
    assert result.returncode == 0
    assert (tmp_path / "Apps" / "oMLX.app" / "Contents" / "marker").read_text() == "current"
    assert (retired / "oMLX.app").exists()


def test_rollback_dry_run_changes_nothing_and_warns_when_nothing_is_retired(tmp_path):
    retired = tmp_path / ".unsloth" / "engines" / "migration" / "retired-apps"
    make_app(retired / "oMLX.app")
    result = migrate(tmp_path, "DRY=1; restore_retired_apps")
    assert result.returncode == 0
    assert "[dry-run] mv" in result.stdout
    assert (retired / "oMLX.app").exists() and not (tmp_path / "Apps" / "oMLX.app").exists()
    assert "no" in result.stderr and "EngineBar.app to restore" in result.stderr


def test_the_latest_state_dir_ignores_retired_apps(tmp_path):
    root = tmp_path / ".unsloth" / "engines" / "migration"
    for name in ("20261001-101010", "20261003-120000", "retired-apps"):
        (root / name).mkdir(parents=True)
    result = migrate(tmp_path, "latest_state_dir")
    assert result.stdout.strip().endswith("migration/20261003-120000")


# --- build-engines-mac.sh: staged venv swap and rollback ---------------------------------------


def build_fn(home, snippet, **env):
    """Source build-engines-mac.sh (its main is guarded) against a scratch engines home."""
    return run(
        ["bash", "-c", f'set -euo pipefail; . "{SCRIPTS}/build-engines-mac.sh"; mkdir -p "$ENGINES_HOME"; touch "$MARKER"; {snippet}'],
        {"UNSLOTH_ENGINES_HOME": str(home), **env},
    )


def make_venv(path: Path, version: str, staged=None):
    fake_python(path / "bin" / "python", f'echo {version}\n')
    (path / "VERSION").write_text(version)
    # an entry point whose shebang names the build location, as pip writes it
    script = path / "bin" / "omlx"
    script.write_text(f"#!{path}/bin/python\nprint()\n")
    script.chmod(0o755)
    (path / "pyvenv.cfg").write_text(f"home = /usr/bin\nprompt = {path}\n")
    if staged:
        (path / ".staged").write_text("\n".join(staged) + "\n")


def marker(home):
    return dict(line.split("=", 1) for line in (home / ".provisioned").read_text().splitlines() if "=" in line)


def test_swap_waits_while_an_engine_runs_from_the_venv(home):
    make_venv(home / "omlx", "old")
    (home / ".provisioned").write_text("omlx=old-key\n")
    make_venv(home / "omlx.new", "new", ["omlx", "new-key"])
    result = build_fn(home, 'venv_in_use() { return 0; }; swap_venv "$ENGINES_HOME/omlx"; echo pending=$PENDING_SWAPS')
    assert result.returncode == 0
    assert "pending=1" in result.stdout and "update-engines-mac.sh" in result.stdout
    assert (home / "omlx" / "VERSION").read_text() == "old" and (home / "omlx.new").exists()
    assert marker(home) == {"omlx": "old-key"}  # never written before the swap succeeded
    forced = build_fn(home, 'venv_in_use() { return 0; }; swap_venv "$ENGINES_HOME/omlx"', FORCE_REPLACE_RUNNING="1")
    assert forced.returncode == 0 and (home / "omlx" / "VERSION").read_text() == "new"


def test_swap_only_fails_while_an_engine_runs(home):
    make_venv(home / "omlx", "old")
    make_venv(home / "omlx.new", "new", ["omlx", "k"])
    result = build_fn(home, 'venv_in_use() { return 0; }; MODE=swap; main')
    assert result.returncode != 0 and "not swapped" in result.stderr


def test_swap_onto_a_fresh_install_has_nothing_to_keep(home):
    make_venv(home / "omlx.new", "new", ["omlx", "k1"])
    result = build_fn(home, 'swap_venv "$ENGINES_HOME/omlx"')
    assert result.returncode == 0, result.stderr
    assert (home / "omlx" / "VERSION").read_text() == "new" and not (home / "omlx.old").exists()
    assert marker(home) == {"omlx": "k1"}


def test_a_staged_venv_without_a_record_is_refused(home):
    make_venv(home / "omlx", "old")
    make_venv(home / "omlx.new", "new")
    result = build_fn(home, 'swap_venv "$ENGINES_HOME/omlx"')
    assert result.returncode != 0 and ".staged" in result.stderr
    assert (home / "omlx" / "VERSION").read_text() == "old"


def test_rollback_restores_the_old_venv_and_markers(home):
    make_venv(home / "omlx", "old")
    (home / ".provisioned").write_text("omlx=old-key\nomlx_kernels=1\n")
    make_venv(home / "omlx.new", "new", ["omlx", "new-key", "omlx_kernels=0"])
    assert build_fn(home, 'venv_in_use() { return 1; }; swap_venv "$ENGINES_HOME/omlx"').returncode == 0
    result = build_fn(home, 'venv_in_use() { return 1; }; rollback_venv "$ENGINES_HOME/omlx"')
    assert result.returncode == 0, result.stderr
    assert (home / "omlx" / "VERSION").read_text() == "old"
    assert (home / "omlx.failed" / "VERSION").read_text() == "new"
    assert not (home / "omlx.old").exists() and not (home / "omlx" / ".prev-markers").exists()
    assert marker(home) == {"omlx": "old-key", "omlx_kernels": "1"}
    again = build_fn(home, 'rollback_venv "$ENGINES_HOME/omlx"')
    assert again.returncode == 0 and "no omlx.old" in again.stdout


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd, check=True, capture_output=True,
    )


@pytest.fixture
def repo_with_submodules(tmp_path):
    subs = tmp_path / "subs"
    for name in ("omlx", "comfyui"):
        d = subs / name
        d.mkdir(parents=True)
        git(d, "init", "-q", "-b", "main")
        (d / "f").write_text("1")
        git(d, "add", ".")
        git(d, "commit", "-q", "-m", "one")
    repo = tmp_path / "repo"
    (repo / "studio" / "engines").mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    for name in ("omlx", "comfyui"):
        git(repo, "submodule", "add", "-q", str(subs / name), f"studio/engines/{name}")
    git(repo, "commit", "-q", "-m", "pin")
    return repo


def pins(repo, home, **env):
    return build_fn(home, f'REPO="{repo}"; ENGINES_SRC="{repo}/studio/engines"; check_submodule_pins; echo pins-ok', **env)


def test_pin_check_passes_on_a_clean_pinned_checkout(repo_with_submodules, home):
    result = pins(repo_with_submodules, home)
    assert result.returncode == 0 and "pins-ok" in result.stdout, result.stderr


def test_pin_check_fails_on_a_moved_submodule(repo_with_submodules, home):
    sub = repo_with_submodules / "studio" / "engines" / "omlx"
    (sub / "f").write_text("2")
    git(sub, "commit", "-qam", "two")
    result = pins(repo_with_submodules, home)
    assert result.returncode != 0 and "pin mismatch" in result.stderr


def test_pin_check_fails_on_an_uninitialised_submodule(repo_with_submodules, home):
    git(repo_with_submodules, "submodule", "deinit", "-f", "studio/engines/omlx")
    result = pins(repo_with_submodules, home)
    assert result.returncode != 0 and "pin mismatch" in result.stderr


def test_pin_check_fails_on_a_moved_comfyui_submodule(repo_with_submodules, home):
    sub = repo_with_submodules / "studio" / "engines" / "comfyui"
    (sub / "f").write_text("2")
    git(sub, "commit", "-qam", "two")
    result = pins(repo_with_submodules, home)
    assert result.returncode != 0 and "pin mismatch" in result.stderr and "comfyui" in result.stderr


def test_pin_check_fails_on_a_dirty_comfyui_submodule(repo_with_submodules, home):
    (repo_with_submodules / "studio" / "engines" / "comfyui" / "f").write_text("dirty")
    result = pins(repo_with_submodules, home)
    assert result.returncode != 0 and "studio/engines/comfyui has local changes" in result.stderr


def test_pin_check_fails_on_an_uninitialised_comfyui_submodule(repo_with_submodules, home):
    git(repo_with_submodules, "submodule", "deinit", "-f", "studio/engines/comfyui")
    result = pins(repo_with_submodules, home)
    assert result.returncode != 0 and "pin mismatch" in result.stderr


@pytest.mark.parametrize("how", ["moved", "dirty", "uninitialised"])
def test_unsloth_build_comfyui_0_ignores_the_comfyui_submodule(repo_with_submodules, home, how):
    sub = repo_with_submodules / "studio" / "engines" / "comfyui"
    if how == "moved":
        (sub / "f").write_text("2")
        git(sub, "commit", "-qam", "two")
    elif how == "dirty":
        (sub / "f").write_text("dirty")
    else:
        git(repo_with_submodules, "submodule", "deinit", "-f", "studio/engines/comfyui")
    skipped = pins(repo_with_submodules, home, UNSLOTH_BUILD_COMFYUI="0")
    assert skipped.returncode == 0 and "pins-ok" in skipped.stdout, skipped.stderr
    assert pins(repo_with_submodules, home).returncode != 0


def test_init_submodules_initialises_and_reads_both_commits(repo_with_submodules, home):
    git(repo_with_submodules, "submodule", "deinit", "-f", "studio/engines/comfyui")
    snippet = (
        f'REPO="{repo_with_submodules}"; ENGINES_SRC="{repo_with_submodules}/studio/engines"; '
        'git() { echo "git $*" >&2; command git -c protocol.file.allow=always "$@"; }; '
        'init_submodules; echo "omlx=${OMLX_COMMIT:0:7} comfyui=${COMFYUI_COMMIT:0:7}"'
    )
    result = build_fn(home, snippet)
    assert result.returncode == 0, result.stderr
    assert "initialising submodule studio/engines/comfyui" in result.stdout
    assert "initialising submodule studio/engines/omlx" not in result.stdout
    assert "submodule update --init --depth 1 studio/engines/comfyui" in result.stderr
    omlx, comfyui = result.stdout.split()[-2:]
    assert len(omlx.split("=")[1]) == 7 and len(comfyui.split("=")[1]) == 7


def test_the_submodule_list_follows_unsloth_build_comfyui(home):
    assert build_fn(home, 'echo "${ENGINE_SUBMODULES[*]}"').stdout.strip() == "omlx comfyui"
    assert build_fn(home, 'echo "${ENGINE_SUBMODULES[*]}"', UNSLOTH_BUILD_COMFYUI="0").stdout.strip() == "omlx"


# --- update-engines-mac.sh against fake engines and a fake launchctl ----------------------------

FAKES = Path(__file__).resolve().parent / "fakes"
UPDATE = SCRIPTS / "update-engines-mac.sh"


class Rig:
    def __init__(self, tmp_path, home):
        self.home = home
        self.lc = tmp_path / "lc"
        self.lc.mkdir()
        self.app = tmp_path / "helper-app" / "Unsloth.app"
        make_app(self.app, "helper", cli="cli")
        self.omlx_port = free_port()
        self.comfyui_port = free_port()
        self.ds4_port = free_port()
        self.labels = ("ai.unsloth.studio.omlx",)
        make_venv(home / "omlx", "old")
        (home / ".provisioned").write_text("omlx=old-key\nomlx_kernels=1\n")
        # stage-only is simulated (a real build needs uv, the network and minutes); swap and
        # rollback run the real build-engines-mac.sh.
        self.build = tmp_path / "fake-build"
        self.build.write_text(
            "#!/bin/bash\n"
            'if [ "$1" = --stage-only ]; then\n'
            '  echo "fake stage: $STAGE_VERSION" >>"$FAKE_LC/calls.log"\n'
            '  [ "$STAGE_VERSION" = fail ] && exit 1\n'
            '  [ "${STAGE_COMFYUI_VERSION:-none}" = fail ] && exit 1\n'
            '  if [ "${STAGE_VERSION:-none}" != none ]; then\n'
            '    mkdir -p "$UNSLOTH_ENGINES_HOME/omlx.new/bin"\n'
            f'    cp "{home}/omlx/bin/python" "$UNSLOTH_ENGINES_HOME/omlx.new/bin/python"\n'
            '    printf %s "$STAGE_VERSION" >"$UNSLOTH_ENGINES_HOME/omlx.new/VERSION"\n'
            '    printf "omlx\\nnew-key\\nomlx_kernels=0\\n" >"$UNSLOTH_ENGINES_HOME/omlx.new/.staged"\n'
            "  fi\n"
            '  if [ "${STAGE_COMFYUI_VERSION:-none}" != none ] && [ "${UNSLOTH_BUILD_COMFYUI:-1}" != 0 ]; then\n'
            '    echo "fake stage comfyui: $STAGE_COMFYUI_VERSION" >>"$FAKE_LC/calls.log"\n'
            '    mkdir -p "$UNSLOTH_ENGINES_HOME/comfyui.new/bin" "$UNSLOTH_ENGINES_HOME/comfyui.new/src"\n'
            f'    cp "{home}/omlx/bin/python" "$UNSLOTH_ENGINES_HOME/comfyui.new/bin/python"\n'
            '    printf %s "$STAGE_COMFYUI_VERSION" >"$UNSLOTH_ENGINES_HOME/comfyui.new/VERSION"\n'
            '    printf "# %s\\n" "$STAGE_COMFYUI_VERSION" >"$UNSLOTH_ENGINES_HOME/comfyui.new/src/main.py"\n'
            '    printf "comfyui\\nnew-ckey\\n" >"$UNSLOTH_ENGINES_HOME/comfyui.new/.staged"\n'
            "  fi\n"
            "  exit 0\n"
            "fi\n"
            f'exec "{SCRIPTS}/build-engines-mac.sh" "$@"\n'
        )
        self.build.chmod(0o755)
        self.env = {
            "UNSLOTH_ENGINES_HOME": str(home),
            "OMLX_URL": f"http://127.0.0.1:{self.omlx_port}",
            "COMFYUI_URL": f"http://127.0.0.1:{self.comfyui_port}",
            "ENGINE_PORTS": f"{self.omlx_port} {self.comfyui_port}",
            "LAUNCHCTL": str(FAKES / "fake-launchctl"),
            "HELPER_APP": str(self.app),
            "PGREP": str(FAKES / "fake-pgrep"),
            "HELPER_START_WAIT": "5",
            "BUILD_ENGINES": str(self.build),
            "HEALTH_TIMEOUT": "6",
            "HEALTH_POLL": "0.2",
            "PORT_GATE_TIMEOUT": "10",
            "FAKE_LC": str(self.lc),
            "FAKE_HOME": str(home),
            "FAKE_OMLX_PORT": str(self.omlx_port),
            "FAKE_COMFYUI_PORT": str(self.comfyui_port),
            "FAKE_PY": sys.executable,
            "STAGE_VERSION": "none",
            "STAGE_COMFYUI_VERSION": "none",
            # the legacy ds4 launcher of a pre-removal app: absent unless a test starts it
            "LEGACY_DS4_URL": f"http://127.0.0.1:{self.ds4_port}",
            "LEGACY_DS4_PORTS": str(self.ds4_port),
            "LEGACY_DS4_WAIT": "3",
            "FAKE_DS4_PORT": str(self.ds4_port),
        }
        # the pre-with_app behaviour by default: the helpers outlive the app. The with_app tests
        # call set_desktop.
        self.set_desktop("always")
        # the helpers start registered, as after the Settings toggle
        assert run([str(self.app / "Contents/MacOS/unsloth-studio"), "--engine-helpers", "register"], self.env).returncode == 0
        self.wait_up()
        (self.lc / "calls.log").unlink(missing_ok=True)

    def add_comfyui(self):
        """A live ComfyUI venv (venv + src) with its marker. The updater's smoke test is the fake
        check, which fails while comfyui/VERSION says bad."""
        make_comfy_venv(self.home / "comfyui", "old")
        with (self.home / ".provisioned").open("a") as fh:
            fh.write("comfyui=old-ckey\n")
        self.env["COMFYUI_CHECK"] = str(FAKES / "fake-comfyui-check")

    def start_legacy_ds4(self, busy=False, starting=False):
        """The ds4 helper a pre-removal app registered: a job in launchd answering /admin/status."""
        import time
        import urllib.request

        if busy:
            (self.home / "ds4-busy").write_text("1")
        if starting:
            (self.home / "ds4-starting").write_text("1")
        label = "ai.unsloth.studio.ds4"
        # detached like the fake CLI starts engines, so a kill is reaped by init and `kill -0` ends
        subprocess.run(
            ["bash", "-c", '"$FAKE_PY" "$0/fake_engine.py" ds4 "$FAKE_HOME" "$FAKE_DS4_PORT" >/dev/null 2>&1 & echo $! >"$FAKE_LC/$1.pid"; echo $! >>"$FAKE_LC/spawned.pids"; disown', str(FAKES), label],
            env={**os.environ, **self.env},
            check=True,
        )
        (self.lc / f"registered.{label}").touch()
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.ds4_port}/admin/status", timeout=1).read()
                break
            except OSError:
                time.sleep(0.1)

    def launchctl(self, *args):
        return run([str(FAKES / "fake-launchctl"), *args], self.env)

    def set_desktop(self, lifetime, enabled=True, path=None, helpers=None):
        """desktop.json as the app writes it: engine_lifetime, engines_enabled (None omits it) and the
        per-engine `helpers` choices (None omits them: oMLX on, ComfyUI off)."""
        body = {"engine_lifetime": lifetime}
        if enabled is not None:
            body["engines_enabled"] = enabled
        if helpers is not None:
            body["helpers"] = helpers
        (path or self.home / "desktop.json").write_text(json.dumps(body))

    def enable_comfyui(self):
        """The user turned ComfyUI on (desktop.json helpers.comfyui) and registered it, like the app."""
        self.set_desktop("always", helpers={"omlx": True, "comfyui": True})
        assert run([str(self.app / "Contents/MacOS/unsloth-studio"), "--engine-helpers", "register", "comfyui"], self.env).returncode == 0
        self.wait_up()
        (self.lc / "calls.log").unlink(missing_ok=True)
        self.labels = ("ai.unsloth.studio.omlx", "ai.unsloth.studio.comfyui")

    def set_app_running(self, running):
        flag = self.lc / "app-running"
        flag.touch() if running else flag.unlink(missing_ok=True)

    def fork_env(self, extra=None):
        """The environment for sourcing build-fork-mac.sh: the helpers belong to INSTALLED_APP.
        UNSLOTH_ENGINES_HOME stays, so desktop.json is the rig's and never the real one."""
        env = {k: v for k, v in self.env.items() if k != "HELPER_APP"}
        return {**env, **(extra or {})}

    def cli_calls(self):
        return [c for c in self.calls() if c.startswith("engine-helpers ") and c != "engine-helpers status"]

    def wait_up(self):
        import time
        import urllib.request

        urls = [f"http://127.0.0.1:{self.omlx_port}/api/status"]
        if (self.lc / "registered.ai.unsloth.studio.comfyui").exists():
            urls.append(f"http://127.0.0.1:{self.comfyui_port}/system_stats")
        for url in urls:
            for _ in range(100):
                try:
                    urllib.request.urlopen(url, timeout=1).read()
                    break
                except OSError:
                    time.sleep(0.1)

    def update(self, *args, **env):
        return run(["bash", str(UPDATE), "--yes", *args], {**self.env, **env})

    def calls(self):
        return (self.lc / "calls.log").read_text().splitlines() if (self.lc / "calls.log").exists() else []

    def loaded(self, label):
        return self.launchctl("print", f"gui/501/{label}").returncode == 0

    def close(self):
        """Stop every fake engine this rig started: TERM, then KILL for one that ignored it. The
        fakes are detached, and a label.pid overwritten by a restart hides the earlier instance, so
        spawned.pids (every pid the fake CLI and start_legacy_ds4 started) is read as well."""
        import signal
        import time

        pids = set()
        for pid_file in [*self.lc.glob("*.pid"), self.lc / "spawned.pids"]:
            try:
                pids.update(int(p) for p in pid_file.read_text().split())
            except (OSError, ValueError):
                pass
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in pids:
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
            deadline = time.time() + 2
            while time.time() < deadline and any(_alive(pid) for pid in pids):
                time.sleep(0.05)
        return pids


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@pytest.fixture
def rig(tmp_path, home):
    r = Rig(tmp_path, home)
    yield r
    pids = r.close()
    # the fakes (oMLX, ComfyUI, the legacy launcher) must never outlive the test
    assert not [pid for pid in pids if _alive(pid)], "a fake engine survived the test"


def test_update_swaps_the_venv_with_the_helpers_stopped_for_the_swap_only(rig):
    (rig.home / "omlx-loaded").write_text("1")
    result = rig.update(STAGE_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (rig.home / "omlx" / "VERSION").read_text() == "new"
    assert (rig.home / "omlx.old" / "VERSION").read_text() == "old"
    assert marker(rig.home)["omlx"] == "new-key"
    calls = rig.calls()
    # staged first, while the helpers still serve; then idle unload, stop, restart through the app
    order = [
        "fake stage: new",
        "omlx-unload",
        "engine-helpers unregister omlx",
        "engine-helpers register omlx",
    ]
    positions = [calls.index(item) for item in order]
    assert positions == sorted(positions)
    assert rig.cli_calls() == ["engine-helpers unregister omlx", "engine-helpers register omlx"]
    assert_no_launchd_start(rig)
    assert all(rig.loaded(label) for label in rig.labels)
    assert not (rig.home / "omlx.new").exists()


def assert_no_launchd_start(rig):
    """macOS refuses launchctl bootstrap of an SMAppService helper: nothing may depend on it."""
    assert not any(c.startswith(("launchctl bootstrap", "launchctl kickstart")) for c in rig.calls())
    assert "gui-started" not in rig.calls()


def helpers_touched(rig):
    return [c for c in rig.calls() if c.startswith("launchctl bootout") or c.startswith(("engine-helpers unregister", "engine-helpers register", "engine-helpers restart"))]


def test_update_with_nothing_staged_does_not_touch_the_helpers(rig):
    result = rig.update(STAGE_VERSION="none")
    assert result.returncode == 0 and "nothing to do" in result.stdout
    assert helpers_touched(rig) == []


def test_update_rolls_back_when_the_new_venv_is_unhealthy(rig):
    result = rig.update(STAGE_VERSION="bad")
    assert result.returncode != 0
    assert "rolling back" in result.stdout and "rolled back" in result.stdout, result.stdout + result.stderr
    assert (rig.home / "omlx" / "VERSION").read_text() == "old"
    assert (rig.home / "omlx.failed" / "VERSION").read_text() == "bad"
    assert marker(rig.home) == {"omlx": "old-key", "omlx_kernels": "1"}
    assert all(rig.loaded(label) for label in rig.labels)


def test_update_restarts_the_helpers_when_the_swap_cannot_happen(rig):
    # something outside launchd holds the live venv (cwd inside it, path in its argv): the real
    # swap guard refuses, and the update must still bring the helpers back with the venv untouched
    stray = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", f"{rig.home}/omlx/bin/python"],
        cwd=rig.home / "omlx" / "bin",
    )
    try:
        before = (rig.home / ".provisioned").read_text()
        result = rig.update(STAGE_VERSION="new")
        assert result.returncode != 0 and "not swapped" in result.stderr, result.stdout + result.stderr
        assert (rig.home / "omlx" / "VERSION").read_text() == "old"
        assert (rig.home / "omlx.new").exists() and not (rig.home / "omlx.old").exists()
        assert (rig.home / ".provisioned").read_text() == before
        assert all(rig.loaded(label) for label in rig.labels)
    finally:
        stray.kill()
        stray.wait()


def test_update_fails_cleanly_when_the_stage_fails(rig):
    result = rig.update(STAGE_VERSION="fail")
    assert result.returncode != 0
    assert helpers_touched(rig) == []
    assert (rig.home / "omlx" / "VERSION").read_text() == "old"


def test_update_reports_when_a_helper_cannot_be_restarted(rig):
    result = rig.update(STAGE_VERSION="new", FAKE_REGISTER_FAIL="1", FAKE_RESTART_FAIL="1")
    assert result.returncode != 0
    out = result.stderr + result.stdout
    assert "did not restart" in out and "Engines > Engines enabled" in out
    # register, then one restart; never launchctl bootstrap or kickstart
    assert [c for c in rig.cli_calls() if c != "engine-helpers unregister omlx"][:2] == ["engine-helpers register omlx", "engine-helpers restart omlx"]
    assert_no_launchd_start(rig)


def lib_fn(rig, snippet, extra=None):
    """Run a snippet with engines-lib.sh sourced against the rig's fake engines, launchctl and app."""
    prelude = (
        'log() { echo "[t] $*"; }; warn() { echo "[t] warning: $*" >&2; }; '
        'die() { echo "[t] error: $*" >&2; exit 1; }; '
        'run() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }; DRY=0; '
    )
    return run(
        ["bash", "-c", f'set -euo pipefail; {prelude} . "{SCRIPTS}/engines-lib.sh"; {snippet}'],
        {**rig.env, **(extra or {})},
    )


def test_helpers_stop_and_start_go_through_the_app_cli(rig):
    result = lib_fn(rig, 'helpers_stop; echo "owed=${STOPPED_HELPERS[*]}"; helpers_start; echo "owed=${STOPPED_HELPERS[*]-}"')
    assert result.returncode == 0, result.stdout + result.stderr
    assert "owed=ai.unsloth.studio.omlx" in result.stdout
    assert result.stdout.strip().splitlines()[-1] == "owed="
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx"]
    assert not any(c.startswith("launchctl bootout") for c in rig.calls())
    assert_no_launchd_start(rig)
    assert all(rig.loaded(label) for label in rig.labels)


def test_the_cli_is_detected_by_a_status_that_exits_0_with_json(rig):
    ok = lib_fn(rig, 'helper_cli_supported "$(helper_cli)"')
    assert ok.returncode == 0
    for env in ({"FAKE_STATUS_FAIL": "1"},):
        bad = lib_fn(rig, 'helper_cli_supported "$(helper_cli)"', env)
        assert bad.returncode != 0


def test_an_app_without_the_cli_is_never_run_and_falls_back_to_bootout(rig, tmp_path):
    old = tmp_path / "old-app" / "Unsloth.app"
    make_app(old, "old", cli="old")
    result = lib_fn(rig, "helpers_stop", {"HELPER_APP": str(old)})
    assert result.returncode == 0, result.stdout + result.stderr
    # the old binary would have started its GUI for an unknown flag: it must not even be probed
    assert "gui-started" not in rig.calls()
    assert [c for c in rig.calls() if c.startswith("launchctl bootout")] == [
        "launchctl bootout gui/501/ai.unsloth.studio.omlx",
    ]
    assert rig.cli_calls() == []
    assert not any(rig.loaded(label) for label in rig.labels)
    # and the old app cannot bring them back: say so, with the manual fallback, and do not call
    # launchctl bootstrap or kickstart
    started = lib_fn(rig, 'STOPPED_HELPERS=(ai.unsloth.studio.omlx); helpers_start', {"HELPER_APP": str(old)})
    assert started.returncode != 0
    assert "no --engine-helpers" in started.stderr and "Engines > Engines enabled" in started.stderr
    assert_no_launchd_start(rig)


def test_an_app_with_no_executable_falls_back_to_bootout(rig, tmp_path):
    bare = tmp_path / "bare" / "Unsloth.app"
    make_app(bare, "bare")
    result = lib_fn(rig, "helpers_stop", {"HELPER_APP": str(bare)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert sum(c.startswith("launchctl bootout") for c in rig.calls()) == 1


def test_a_failing_status_falls_back_to_bootout(rig):
    result = lib_fn(rig, "helpers_stop", {"FAKE_STATUS_FAIL": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert sum(c.startswith("launchctl bootout") for c in rig.calls()) == 1
    assert rig.cli_calls() == []


def test_a_booted_out_helper_keeps_its_registration_so_register_is_followed_by_restart(rig, tmp_path):
    # the first install: the old app has no CLI (bootout), the new app starts them again
    old = tmp_path / "old-app" / "Unsloth.app"
    make_app(old, "old", cli="old")
    started = lib_fn(rig, f'HELPER_APP="{old}" helpers_stop; test -e "$FAKE_LC/registered.ai.unsloth.studio.omlx"; helpers_start')
    assert started.returncode == 0, started.stdout + started.stderr
    assert rig.cli_calls() == ["engine-helpers register omlx", "engine-helpers restart omlx"]
    assert "trying restart" in started.stderr
    assert_no_launchd_start(rig)
    assert all(rig.loaded(label) for label in rig.labels)


def test_a_register_that_does_nothing_and_a_restart_that_fails_leave_a_clear_warning(rig):
    started = lib_fn(rig, "helpers_stop; helpers_start", {"FAKE_REGISTER_NOOP": "1", "FAKE_RESTART_FAIL": "1", "HELPER_START_WAIT": "1"})
    assert started.returncode != 0
    assert "did not restart" in started.stderr and "Engines > Engines enabled" in started.stderr
    assert not any(rig.loaded(label) for label in rig.labels)


def test_helpers_start_with_nothing_owed_does_nothing(rig):
    result = lib_fn(rig, "helpers_start")
    assert result.returncode == 0
    assert rig.cli_calls() == []


def test_helpers_that_are_already_loaded_are_left_alone(rig):
    result = lib_fn(rig, 'STOPPED_HELPERS=(ai.unsloth.studio.omlx); helpers_start')
    assert result.returncode == 0 and "already loaded" in result.stdout
    assert rig.cli_calls() == []


def test_helpers_stop_with_no_loaded_helper_cleans_legacy_registration(rig):
    lib_fn(rig, "helpers_stop")
    rig.calls_before = len(rig.calls())
    again = lib_fn(rig, "helpers_stop")
    assert again.returncode == 0 and "not loaded, leaving it alone" in again.stdout
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers unregister"]


def test_helper_dry_run_prints_the_cli_calls_and_changes_nothing(rig):
    result = lib_fn(rig, "DRY=1; helpers_stop; helpers_start")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--engine-helpers unregister" in result.stdout and "--engine-helpers register" in result.stdout
    assert rig.cli_calls() == []
    assert all(rig.loaded(label) for label in rig.labels)


def test_update_dry_run_changes_nothing(rig):
    result = rig.update("--dry-run", STAGE_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[dry-run]" in result.stdout and "--engine-helpers unregister" in result.stdout and "--swap-only" in result.stdout
    assert "--engine-helpers register" in result.stdout
    assert (rig.home / "omlx" / "VERSION").read_text() == "old"
    assert not (rig.home / "omlx.new").exists() and not (rig.home / "omlx.old").exists()
    assert helpers_touched(rig) == []
    assert all(rig.loaded(label) for label in rig.labels)


# --- update-engines-mac.sh and the ComfyUI venv --------------------------------------------------


def test_a_comfyui_only_update_swaps_it_and_never_touches_omlx_or_its_helper(rig):
    rig.add_comfyui()
    (rig.home / "omlx-loaded").write_text("1")
    result = rig.update(STAGE_COMFYUI_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (rig.home / "comfyui" / "VERSION").read_text() == "new"
    assert (rig.home / "comfyui" / "src" / "main.py").read_text() == "# new\n"
    assert (rig.home / "comfyui.old" / "VERSION").read_text() == "old"
    assert not (rig.home / "comfyui.new").exists()
    assert marker(rig.home) == {"omlx": "old-key", "omlx_kernels": "1", "comfyui": "new-ckey"}
    assert (rig.home / "omlx" / "VERSION").read_text() == "old" and not (rig.home / "omlx.old").exists()
    calls = rig.calls()
    assert "comfyui-check" in calls and "omlx-unload" not in calls
    assert helpers_touched(rig) == [] and rig.cli_calls() == []
    assert (rig.home / "omlx-loaded").exists() and all(rig.loaded(label) for label in rig.labels)
    assert "oMLX and its helper were not touched" in result.stdout


def test_an_update_of_both_swaps_both_with_one_helper_bounce(rig):
    rig.add_comfyui()
    result = rig.update(STAGE_VERSION="new", STAGE_COMFYUI_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (rig.home / "omlx" / "VERSION").read_text() == "new"
    assert (rig.home / "comfyui" / "VERSION").read_text() == "new"
    assert marker(rig.home)["comfyui"] == "new-ckey" and marker(rig.home)["omlx"] == "new-key"
    # both venvs swap, so every helper is stopped: the plain unregister, then the loaded one by name
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx"]
    assert "comfyui-check" in rig.calls()


def test_a_failed_comfyui_smoke_test_rolls_both_venvs_back(rig):
    rig.add_comfyui()
    result = rig.update(STAGE_VERSION="new", STAGE_COMFYUI_VERSION="bad")
    assert result.returncode != 0
    assert "rolling back" in result.stdout and "rolled back" in result.stdout, result.stdout + result.stderr
    for name, failed in (("omlx", "new"), ("comfyui", "bad")):
        assert (rig.home / name / "VERSION").read_text() == "old"
        assert (rig.home / f"{name}.failed" / "VERSION").read_text() == failed
        assert not (rig.home / f"{name}.old").exists()
    assert (rig.home / "comfyui" / "src" / "main.py").read_text() == "# old\n"
    assert marker(rig.home) == {"omlx": "old-key", "omlx_kernels": "1", "comfyui": "old-ckey"}
    assert all(rig.loaded(label) for label in rig.labels)


def test_an_unhealthy_omlx_rolls_back_the_comfyui_venv_swapped_with_it(rig):
    rig.add_comfyui()
    result = rig.update(STAGE_VERSION="bad", STAGE_COMFYUI_VERSION="new")
    assert result.returncode != 0 and "rolled back" in result.stdout, result.stdout + result.stderr
    assert (rig.home / "omlx" / "VERSION").read_text() == "old"
    assert (rig.home / "comfyui" / "VERSION").read_text() == "old"
    assert (rig.home / "comfyui.failed" / "VERSION").read_text() == "new"
    assert marker(rig.home) == {"omlx": "old-key", "omlx_kernels": "1", "comfyui": "old-ckey"}


def test_a_failed_comfyui_smoke_test_alone_rolls_back_without_touching_omlx(rig):
    rig.add_comfyui()
    result = rig.update(STAGE_COMFYUI_VERSION="bad")
    assert result.returncode != 0 and "rolled back" in result.stdout, result.stdout + result.stderr
    assert (rig.home / "comfyui" / "VERSION").read_text() == "old"
    assert marker(rig.home)["comfyui"] == "old-ckey"
    assert helpers_touched(rig) == [] and rig.cli_calls() == []


def test_a_running_comfyui_blocks_the_update_before_anything_changes(rig):
    rig.add_comfyui()
    stray = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", f"{rig.home}/comfyui/src/main.py"],
        cwd=rig.home / "comfyui" / "src",
    )
    try:
        result = rig.update(STAGE_VERSION="new", STAGE_COMFYUI_VERSION="new")
        assert result.returncode != 0 and "ComfyUI is running" in result.stderr, result.stdout + result.stderr
        assert helpers_touched(rig) == []
        for name in ("omlx", "comfyui"):
            assert (rig.home / name / "VERSION").read_text() == "old"
            assert (rig.home / f"{name}.new").exists() and not (rig.home / f"{name}.old").exists()
        assert marker(rig.home)["comfyui"] == "old-ckey" and "omlx-unload" not in rig.calls()
    finally:
        stray.kill()
        stray.wait()


def test_unsloth_build_comfyui_0_leaves_comfyui_out_of_the_update(rig):
    rig.add_comfyui()
    make_comfy_venv(rig.home / "comfyui.new", "staged-before", ["comfyui", "new-ckey"])
    result = rig.update(STAGE_COMFYUI_VERSION="new", UNSLOTH_BUILD_COMFYUI="0")
    assert result.returncode == 0 and "nothing to do" in result.stdout, result.stdout + result.stderr
    assert (rig.home / "comfyui" / "VERSION").read_text() == "old" and (rig.home / "comfyui.new").exists()
    assert helpers_touched(rig) == []


def test_the_update_dry_run_covers_comfyui_too(rig):
    rig.add_comfyui()
    result = rig.update("--dry-run", STAGE_VERSION="new", STAGE_COMFYUI_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "would smoke-test the swapped ComfyUI venv" in result.stdout
    assert "comfyui-check" not in rig.calls()
    assert (rig.home / "comfyui" / "VERSION").read_text() == "old" and helpers_touched(rig) == []


# --- build-fork-mac.sh --install pieces -------------------------------------------------------

FORK = SCRIPTS / "build-fork-mac.sh"


def fork_fn(tmp_path, snippet, extra_env=None):
    """Source build-fork-mac.sh (its main is guarded) under a scratch HOME with fake venv/app paths."""
    env = {
        "HOME": str(tmp_path / "home"),
        "UNSLOTH_STUDIO_VENV": str(tmp_path / "studio" / "unsloth_studio"),
        "BACKUP_ROOT": str(tmp_path / "backups"),
        "INSTALLED_APP": str(tmp_path / "Apps" / "Unsloth.app"),
        "APP_PREV": str(tmp_path / "prev" / "Unsloth-prev.app.bak"),
        "DIST_DIR": str(tmp_path / "dist"),
        **(extra_env or {}),
    }
    (tmp_path / "home").mkdir(exist_ok=True)
    return run(["bash", "-c", f'set -euo pipefail; . "{FORK}"; {snippet}'], env)


def make_backend(tmp_path):
    venv = tmp_path / "studio" / "unsloth_studio"
    fake_python(venv / "bin" / "python")
    (venv / "lib").mkdir()
    (venv / "lib" / "mod.py").write_text("original")
    (venv / "bin" / "python3").symlink_to("python")
    return venv


def test_snapshot_and_restore_bring_back_the_original_backend(tmp_path):
    venv = make_backend(tmp_path)
    snippet = (
        'DRY=0; TS=t1; snapshot_backend; '
        'echo changed >"$STUDIO_VENV/lib/mod.py"; echo stray >"$STUDIO_VENV/lib/stray.py"; '
        'restore_backend'
    )
    result = fork_fn(tmp_path, snippet)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (venv / "lib" / "mod.py").read_text() == "original"
    assert not (venv / "lib" / "stray.py").exists()
    assert (venv / "bin" / "python3").is_symlink() and os.readlink(venv / "bin" / "python3") == "python"
    failed = tmp_path / "backups" / "t1" / "failed-unsloth_studio"
    assert (failed / "lib" / "stray.py").exists()


def test_the_snapshot_is_independent_of_later_changes(tmp_path):
    venv = make_backend(tmp_path)
    result = fork_fn(tmp_path, 'DRY=0; TS=t2; snapshot_backend; rm -rf "$STUDIO_VENV/lib"')
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "backups" / "t2" / "unsloth_studio" / "lib" / "mod.py").read_text() == "original"
    assert not (venv / "lib").exists()


def test_restore_without_a_snapshot_refuses(tmp_path):
    make_backend(tmp_path)
    result = fork_fn(tmp_path, "DRY=0; BACKEND_BACKUP=; restore_backend")
    assert result.returncode != 0 and "no backend snapshot" in result.stderr


def test_only_the_newest_two_snapshots_are_kept(tmp_path):
    for name in ("20261001-000000", "20261002-000000", "20261003-000000", "20261004-000000"):
        (tmp_path / "backups" / name).mkdir(parents=True)
    (tmp_path / "backups" / "notes").mkdir()
    result = fork_fn(tmp_path, "prune_backend_backups")
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in (tmp_path / "backups").iterdir()) == ["20261003-000000", "20261004-000000", "notes"]


def test_wheel_gate_fails_before_anything_is_touched(tmp_path):
    result = fork_fn(tmp_path, 'DRY=0; uv() { echo "boom" >&2; return 1; }; build_fork_wheel')
    assert result.returncode != 0 and "does not build" in result.stderr


def test_wheel_gate_requires_the_fork_modules(tmp_path):
    builder = (
        'uv() { python3 - "$4" <<PY\nimport sys, zipfile\n'
        'z = zipfile.ZipFile(sys.argv[1] + "/unsloth-1.0-py3-none-any.whl", "w")\n'
        'z.writestr("unsloth/x.py", "")\nz.close()\nPY\n}; '
    )
    result = fork_fn(tmp_path, f"DRY=0; {builder} build_fork_wheel")
    assert result.returncode != 0 and "attached-engines backend" in result.stderr
    engines = 'z.writestr("studio/backend/routes/attached_engines.py", "")'
    no_marker = builder.replace('z.writestr("unsloth/x.py", "")', engines)
    result = fork_fn(tmp_path, f"DRY=0; {no_marker} build_fork_wheel")
    assert result.returncode != 0 and "studio/_fork.py" in result.stderr
    no_pins = builder.replace('z.writestr("unsloth/x.py", "")', engines + '; z.writestr("studio/_fork.py", "")')
    result = fork_fn(tmp_path, f"DRY=0; {no_pins} build_fork_wheel")
    assert result.returncode != 0 and "fork-pins.toml" in result.stderr
    with_pins = engines + '; z.writestr("studio/_fork.py", ""); z.writestr("studio/fork-pins.toml", "")'
    no_template = builder.replace('z.writestr("unsloth/x.py", "")', with_pins)
    result = fork_fn(tmp_path, f"DRY=0; {no_template} build_fork_wheel")
    assert result.returncode != 0 and "qwen-image-2.1-t2i.json" in result.stderr
    prefix = "studio/backend/core/inference/attached/comfyui_templates"

    def templates(*names):
        return "; ".join(f'z.writestr("{prefix}/qwen-image-2.1-{name}.json", "{{}}")' for name in names)

    only_t2i = builder.replace('z.writestr("unsloth/x.py", "")', f"{with_pins}; {templates('t2i')}")
    result = fork_fn(tmp_path, f"DRY=0; {only_t2i} build_fork_wheel")
    assert result.returncode != 0 and "qwen-image-2.1-img2img.json" in result.stderr
    no_edit = builder.replace('z.writestr("unsloth/x.py", "")', f"{with_pins}; {templates('t2i', 'img2img')}")
    result = fork_fn(tmp_path, f"DRY=0; {no_edit} build_fork_wheel")
    assert result.returncode != 0 and "qwen-image-2.1-edit.json" in result.stderr
    good = builder.replace('z.writestr("unsloth/x.py", "")', f"{with_pins}; {templates('t2i', 'img2img', 'edit')}")
    result = fork_fn(tmp_path, f'DRY=0; {good} build_fork_wheel; echo "wheel=$WHEEL"')
    assert result.returncode == 0, result.stderr
    assert "wheel=" in result.stdout and result.stdout.strip().endswith("unsloth-1.0-py3-none-any.whl")


STUBS = (
    "quit_unsloth() { :; }; engines_require_idle() { :; }; build_fork_wheel() { WHEEL=/none; }; "
    "install_app() { echo app-installed; }; prune_backend_backups() { echo pruned; }; "
    "codesign() { :; }; DRY=0; "
)


def phase(tmp_path, overrides):
    make_backend(tmp_path)
    (tmp_path / "dist" / "Unsloth.app").mkdir(parents=True)
    (tmp_path / "Apps" / "Unsloth.app").mkdir(parents=True)
    return fork_fn(tmp_path, STUBS + overrides + "; phase_install")


def test_install_restores_the_backend_when_the_install_fails(tmp_path):
    result = phase(
        tmp_path,
        'install_backend() { echo half >"$STUDIO_VENV/lib/mod.py"; return 1; }; verify_backend() { :; }; check_installed_version() { :; }',
    )
    assert result.returncode != 0 and "previous backend was restored" in result.stderr, result.stdout + result.stderr
    assert (tmp_path / "studio" / "unsloth_studio" / "lib" / "mod.py").read_text() == "original"
    assert "app-installed" not in result.stdout  # the app was never swapped


def test_install_restores_the_backend_when_verification_fails(tmp_path):
    result = phase(
        tmp_path,
        'install_backend() { echo new >"$STUDIO_VENV/lib/mod.py"; }; check_installed_version() { :; }; verify_backend() { die "bad"; }',
    )
    assert result.returncode != 0 and "did not verify" in result.stderr, result.stdout + result.stderr
    assert (tmp_path / "studio" / "unsloth_studio" / "lib" / "mod.py").read_text() == "original"
    assert "app-installed" not in result.stdout


def test_install_restores_the_backend_when_the_version_is_unacceptable(tmp_path):
    result = phase(
        tmp_path,
        'install_backend() { echo new >"$STUDIO_VENV/lib/mod.py"; }; check_installed_version() { die "too old"; }; verify_backend() { :; }',
    )
    assert result.returncode != 0 and "previous backend was restored" in result.stderr
    assert (tmp_path / "studio" / "unsloth_studio" / "lib" / "mod.py").read_text() == "original"


def test_a_good_install_keeps_the_new_backend_and_swaps_the_app(tmp_path):
    result = phase(
        tmp_path,
        'install_backend() { echo new >"$STUDIO_VENV/lib/mod.py"; }; check_installed_version() { :; }; verify_backend() { :; }; APP_BACKUP="$HOME/app.bak"; ditto() { mkdir -p "$2"; }',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "home" / "app.bak").is_dir() and not (tmp_path / "home" / "app.bak.partial").exists()
    assert (tmp_path / "studio" / "unsloth_studio" / "lib" / "mod.py").read_text().strip() == "new"
    assert "app-installed" in result.stdout and "pruned" in result.stdout
    snapshots = list((tmp_path / "backups").glob("*/unsloth_studio/lib/mod.py"))
    assert len(snapshots) == 1 and snapshots[0].read_text() == "original"


def test_the_app_swap_happens_with_the_helpers_out_of_launchd(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        make_app(tmp_path / "dist" / "Unsloth.app", "new", cli="cli")
        make_app(tmp_path / "Apps" / "Unsloth.app", "old", cli="cli")
        log = rig.lc / "calls.log"
        snippet = (
            "codesign() { :; }; DRY=0; "
            f'mv() {{ echo "mv $1" >>"{log}"; command mv "$@"; }}; '
            "install_app"
        )
        result = fork_fn(tmp_path, snippet, rig.fork_env({"DIST_DIR": str(tmp_path / "dist")}))
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "new"
        assert (tmp_path / "prev" / "Unsloth-prev.app.bak" / "Contents" / "marker").read_text() == "old"
        assert not (tmp_path / "Apps" / "Unsloth.app.new").exists()
        calls = rig.calls()
        # stopped through the OLD app, started through the NEW one: the app moves in between
        stop = calls.index("engine-helpers unregister")
        moves = [i for i, c in enumerate(calls) if c.startswith("mv ")]
        start = calls.index("engine-helpers register omlx")
        assert len(moves) == 2 and stop < min(moves) and max(moves) < start
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx"]
        assert not any(c.startswith("launchctl bootout") for c in calls)
        assert_no_launchd_start(rig)
        assert all(rig.loaded(label) for label in rig.labels)
    finally:
        rig.close()

def install_app_rig(tmp_path, rig, extra_env=None, snippet="install_app", old_cli="cli"):
    """Run install_app against the fake engines and app; returns the result."""
    make_app(tmp_path / "dist" / "Unsloth.app", "new", cli="cli")
    make_app(tmp_path / "Apps" / "Unsloth.app", "old", cli=old_cli)
    return fork_fn(tmp_path, f"codesign() {{ :; }}; DRY=0; {snippet}", rig.fork_env(extra_env))


def test_the_first_install_over_an_app_without_the_cli_boots_out_then_registers_from_the_new_app(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        result = install_app_rig(tmp_path, rig, old_cli="old")
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "new"
        calls = rig.calls()
        # the old binary is never run (it would start its GUI); launchctl takes the helpers down
        assert "gui-started" not in calls
        assert sum(c.startswith("launchctl bootout") for c in calls) == 1
        # the booted-out helpers keep their registration, so register alone does nothing and the
        # new app's restart is what brings them back
        assert rig.cli_calls() == ["engine-helpers register omlx", "engine-helpers restart omlx"]
        assert_no_launchd_start(rig)
        assert all(rig.loaded(label) for label in rig.labels)
    finally:
        rig.close()


TXN_STUBS = (
    "quit_unsloth() { :; }; build_fork_wheel() { WHEEL=/none; }; prune_backend_backups() { echo pruned; }; "
    "codesign() { :; }; DRY=0; install_backend() { echo new >\"$STUDIO_VENV/lib/mod.py\"; }; "
    "check_installed_version() { :; }; verify_backend() { :; }; APP_BACKUP=\"$HOME/app.bak\"; "
)


def assert_rolled_back(tmp_path, rig, result):
    out = result.stdout + result.stderr
    assert result.returncode != 0, out
    # the backend snapshot and the previous app come back together
    assert (tmp_path / "studio" / "unsloth_studio" / "lib" / "mod.py").read_text() == "original", out
    assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "old", out
    assert not (tmp_path / "Apps" / "Unsloth.app.new").exists()
    assert not (tmp_path / "home" / "app.bak.partial").exists()
    assert not (tmp_path / "prev" / "Unsloth-prev.app.bak").exists()  # moved back, not left behind
    assert "pruned" not in result.stdout
    assert "previous backend was restored" in result.stderr
    assert all(rig.loaded(label) for label in rig.labels), out


FAILURES = {
    "app-backup-copy": ('ditto() { case "$2" in *.partial) mkdir -p "$2"; return 1;; esac; command ditto "$@"; }', {}),
    "staging-the-new-app": ('ditto() { case "$2" in *.new) mkdir -p "$2"; return 1;; esac; command ditto "$@"; }', {}),
    "verifying-the-new-bundle": ('codesign() { case "$*" in *Unsloth.app.new*) return 1;; esac; }', {}),
    "quiesce": ('engines_quiesce() { die "oMLX became busy"; }', {}),
    "unregister": ("true", {"FAKE_UNREGISTER_FAIL_AFTER": "omlx"}),
    "swap": ('mv() { [ "$1" = "$INSTALLED_APP.new" ] && return 1; command mv "$@"; }', {}),
    "final-verify": ('codesign() { [ "${!#}" = "$INSTALLED_APP" ] && return 1; return 0; }', {}),
}


def txn(tmp_path, rig, overrides="true", extra_env=None):
    """phase_install against the fake engines and launchctl with every backend step stubbed."""
    make_backend(tmp_path)
    make_app(tmp_path / "dist" / "Unsloth.app", "new", cli="cli")
    make_app(tmp_path / "Apps" / "Unsloth.app", "old", cli="cli")
    return fork_fn(tmp_path, TXN_STUBS + overrides + "; phase_install", rig.fork_env(extra_env))


@pytest.mark.parametrize("phase_name", list(FAILURES))
def test_a_failure_after_the_backend_swap_restores_the_backend_and_the_app(tmp_path, home, phase_name):
    overrides, extra_env = FAILURES[phase_name]
    rig = Rig(tmp_path, home)
    try:
        result = txn(tmp_path, rig, overrides, extra_env)
        assert_rolled_back(tmp_path, rig, result)
        if phase_name == "app-backup-copy":
            assert not (tmp_path / "home" / "app.bak").exists()  # an interrupted copy is not a backup
        if phase_name == "final-verify":
            # the helpers had come up from the new bundle: unregistered again for the restore
            assert rig.cli_calls().count("engine-helpers unregister") == 2
    finally:
        rig.close()


def test_a_successful_transaction_keeps_the_new_backend_and_app_and_disarms(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        result = txn(tmp_path, rig)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "studio" / "unsloth_studio" / "lib" / "mod.py").read_text().strip() == "new"
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "new"
        assert (tmp_path / "prev" / "Unsloth-prev.app.bak" / "Contents" / "marker").read_text() == "old"
        assert (tmp_path / "home" / "app.bak" / "Contents" / "marker").read_text() == "old"
        assert "pruned" in result.stdout and "rolling back" not in result.stderr
        assert all(rig.loaded(label) for label in rig.labels)
    finally:
        rig.close()


def test_a_failure_in_the_prune_after_verification_does_not_roll_back(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        result = txn(tmp_path, rig, "prune_backend_backups() { return 1; }")
        assert result.returncode != 0
        assert (tmp_path / "studio" / "unsloth_studio" / "lib" / "mod.py").read_text().strip() == "new"
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "new"
        assert "rolling back" not in result.stderr
    finally:
        rig.close()


def test_an_up_to_date_omlx_venv_discards_a_stale_staged_one(home):
    make_venv(home / "omlx", "live")
    key = "abc:py3.13:kernels1:extras[]"
    (home / ".provisioned").write_text(f"omlx={key}\n")
    make_venv(home / "omlx.new", "stale", ["omlx", "other-key"])
    snippet = (
        'HAVE_UV=0; OMLX_COMMIT=abc; FORCE=0; OMLX_PYTHON=3.13; OMLX_WITH_CUSTOM_KERNEL=1; OMLX_EXTRAS=; '
        'make_venv() { return 1; }; stage_omlx'
    )
    result = build_fn(home, snippet)
    assert result.returncode == 0, result.stderr
    assert not (home / "omlx.new").exists()
    assert marker(home) == {"omlx": key}


def make_comfy_venv(path: Path, version: str, staged=None):
    """A fake ComfyUI swap unit: the venv and src/ together."""
    make_venv(path, version, staged)
    (path / "src").mkdir()
    (path / "src" / "main.py").write_text(f"# {version}\n")


def test_comfyui_swap_moves_the_venv_and_src_together(home):
    make_comfy_venv(home / "comfyui", "old")
    (home / ".provisioned").write_text("comfyui=old-key\n")
    make_comfy_venv(home / "comfyui.new", "new", ["comfyui", "new-key"])
    result = build_fn(home, 'venv_in_use() { return 1; }; swap_venv "$ENGINES_HOME/comfyui"')
    assert result.returncode == 0, result.stderr
    assert (home / "comfyui" / "src" / "main.py").read_text() == "# new\n"
    assert (home / "comfyui.old" / "src" / "main.py").read_text() == "# old\n"
    assert marker(home) == {"comfyui": "new-key"}


def test_comfyui_swap_refuses_while_it_is_in_use(home):
    make_comfy_venv(home / "comfyui", "old")
    make_comfy_venv(home / "comfyui.new", "new", ["comfyui", "new-key"])
    result = build_fn(home, 'venv_in_use() { return 0; }; swap_venv "$ENGINES_HOME/comfyui"; echo pending=$PENDING_SWAPS')
    assert result.returncode == 0 and "pending=1" in result.stdout
    assert (home / "comfyui" / "src" / "main.py").read_text() == "# old\n" and (home / "comfyui.new").exists()


def test_comfyui_rollback_restores_the_old_unit_and_markers(home):
    make_comfy_venv(home / "comfyui", "old")
    (home / ".provisioned").write_text("comfyui=old-key\nomlx=keep\n")
    make_comfy_venv(home / "comfyui.new", "new", ["comfyui", "new-key"])
    assert build_fn(home, 'venv_in_use() { return 1; }; swap_venv "$ENGINES_HOME/comfyui"').returncode == 0
    result = build_fn(home, 'venv_in_use() { return 1; }; rollback_venv "$ENGINES_HOME/comfyui"')
    assert result.returncode == 0, result.stderr
    assert (home / "comfyui" / "src" / "main.py").read_text() == "# old\n"
    assert (home / "comfyui.failed" / "src" / "main.py").read_text() == "# new\n"
    assert marker(home) == {"comfyui": "old-key", "omlx": "keep"}


def test_a_comfyui_process_running_from_the_unit_is_detected_by_its_source_path(home):
    make_comfy_venv(home / "comfyui", "live")
    # the venv python resolves to its base interpreter in `ps`, so only src/main.py names the venv
    stray = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", f"{home}/comfyui/src/main.py"],
        cwd=home / "comfyui" / "src",
    )
    try:
        import time

        for _ in range(50):
            if build_fn(home, 'venv_in_use "$ENGINES_HOME/comfyui"').returncode == 0:
                break
            time.sleep(0.1)
        assert build_fn(home, 'venv_in_use "$ENGINES_HOME/comfyui"').returncode == 0
        assert build_fn(home, 'venv_in_use "$ENGINES_HOME/omlx"').returncode != 0
    finally:
        stray.kill()
        stray.wait()
    assert build_fn(home, 'venv_in_use "$ENGINES_HOME/comfyui"').returncode != 0


STAGE_STUBS = (
    'HAVE_UV=0; FORCE=0; COMFYUI_PYTHON=3.13; COMFYUI_COMMIT=abc; BUILD_COMFYUI=1; '
    'make_venv() { mkdir -p "$1/bin"; printf "#!/bin/sh\\n" >"$1/bin/python"; chmod +x "$1/bin/python"; echo "make_venv $2" >>"$ENGINES_HOME/calls"; }; '
    'export_tree() { mkdir -p "$2"; : >"$2/requirements.txt"; echo "export_tree $1 $2" >>"$ENGINES_HOME/calls"; }; '
    'pip_install() { shift; echo "pip_install $*" >>"$ENGINES_HOME/calls"; }; '
    'validate_comfyui() { echo "validate $1" >>"$ENGINES_HOME/calls"; }; '
)


def constraints_key(path: Path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def stage_comfyui(home, constraints_text, **env):
    src = home.parent / "src-engines"
    src.mkdir(exist_ok=True)
    (src / "comfyui-constraints.txt").write_text(constraints_text)
    result = build_fn(home, f'ENGINES_SRC="{src}"; mkdir -p "$BUILD_ROOT"; {STAGE_STUBS} stage_comfyui', **env)
    return result, src / "comfyui-constraints.txt"


def test_the_comfyui_stage_key_includes_the_commit_python_and_constraints_hash(home):
    result, constraints = stage_comfyui(home, "torch==1\n")
    assert result.returncode == 0, result.stderr
    staged = (home / "comfyui.new" / ".staged").read_text().splitlines()
    assert staged == ["comfyui", f"abc:py3.13:constraints-{constraints_key(constraints)}"]
    calls = (home / "calls").read_text().splitlines()
    assert calls[0] == "make_venv 3.13" and calls[1].startswith("export_tree ") and calls[1].endswith("/comfyui.new/src")
    assert calls[2] == f"pip_install -r {home}/comfyui.new/src/requirements.txt -c {constraints}"
    assert calls[3] == f"validate {home}/comfyui.new"
    # the log of the install goes to the build root, not the terminal
    assert (home / ".build").is_dir()


def test_changing_the_constraints_restages_but_an_unchanged_file_does_not(home):
    first, constraints = stage_comfyui(home, "torch==1\n")
    assert first.returncode == 0
    (home / "calls").unlink()
    again, _ = stage_comfyui(home, "torch==1\n")
    assert again.returncode == 0 and "already staged" in again.stdout and not (home / "calls").exists()
    changed, constraints = stage_comfyui(home, "torch==2\n")
    assert changed.returncode == 0 and (home / "calls").exists()
    assert (home / "comfyui.new" / ".staged").read_text().splitlines()[1].endswith(constraints_key(constraints))


def test_an_up_to_date_comfyui_venv_is_not_rebuilt_and_a_stale_staged_one_is_discarded(home):
    make_comfy_venv(home / "comfyui", "live")
    src = home.parent / "src-engines"
    src.mkdir()
    (src / "comfyui-constraints.txt").write_text("torch==1\n")
    key = f"abc:py3.13:constraints-{constraints_key(src / 'comfyui-constraints.txt')}"
    (home / ".provisioned").write_text(f"comfyui={key}\n")
    make_comfy_venv(home / "comfyui.new", "stale", ["comfyui", "other-key"])
    result = build_fn(home, f'ENGINES_SRC="{src}"; mkdir -p "$BUILD_ROOT"; {STAGE_STUBS} stage_comfyui')
    assert result.returncode == 0, result.stderr
    assert "up to date" in result.stdout and not (home / "comfyui.new").exists() and not (home / "calls").exists()


def test_a_failed_comfyui_install_stops_the_build_and_swaps_nothing(home):
    src = home.parent / "src-engines"
    src.mkdir()
    (src / "comfyui-constraints.txt").write_text("torch==1\n")
    stubs = STAGE_STUBS + 'pip_install() { echo "resolver said no" >&2; return 1; }; '
    result = build_fn(home, f'ENGINES_SRC="{src}"; mkdir -p "$BUILD_ROOT"; {stubs} stage_comfyui')
    assert result.returncode != 0 and "comfyui install failed" in result.stderr
    assert not (home / "comfyui").exists() and not (home / "comfyui.new" / ".staged").exists()


def full_main(home, build_comfyui, snippet="main", **env):
    """main() with the oMLX and ComfyUI stages stubbed; records what ran."""
    stubs = (
        'stage_omlx() { echo stage_omlx >>"$ENGINES_HOME/order"; }; stage_comfyui() { echo stage_comfyui >>"$ENGINES_HOME/order"; }; '
        'swap_venv() { echo "swap $(basename "$1")" >>"$ENGINES_HOME/order"; }; init_submodules() { :; }; check_submodule_pins() { :; }; '
        'stage_wrappers() { :; }; write_default_config() { :; }; '
    )
    return build_fn(home, f"{stubs} {snippet}", UNSLOTH_BUILD_COMFYUI=build_comfyui, **env)


def test_a_full_run_stages_then_swaps_comfyui_after_omlx(home):
    assert full_main(home, "1").returncode == 0
    assert (home / "order").read_text().split("\n")[:-1] == ["stage_omlx", "stage_comfyui", "swap omlx", "swap comfyui"]


def test_stage_only_stages_both_and_swaps_neither(home):
    assert full_main(home, "1", "MODE=stage; main").returncode == 0
    assert (home / "order").read_text().split("\n")[:-1] == ["stage_omlx", "stage_comfyui"]


def test_swap_only_swaps_both(home):
    assert full_main(home, "1", "MODE=swap; main").returncode == 0
    assert (home / "order").read_text().split("\n")[:-1] == ["swap omlx", "swap comfyui"]


def test_unsloth_build_comfyui_0_skips_the_stage_and_the_swap_and_says_so(home):
    result = full_main(home, "0")
    assert result.returncode == 0 and "comfyui: skipped (UNSLOTH_BUILD_COMFYUI=0)" in result.stdout
    assert (home / "order").read_text().split("\n")[:-1] == ["stage_omlx", "swap omlx"]
    (home / "order").unlink()
    assert full_main(home, "0", "MODE=swap; main").returncode == 0
    assert (home / "order").read_text().split("\n")[:-1] == ["swap omlx"]


def test_the_default_rollback_covers_both_venvs(home):
    for name in ("omlx", "comfyui"):
        make_venv(home / name, "new")
        make_venv(home / f"{name}.old", "old")
    result = build_fn(home, 'venv_in_use() { return 1; }; MODE=rollback; ROLLBACK_NAMES=(); main')
    assert result.returncode == 0, result.stderr
    assert all((home / name / "VERSION").read_text() == "old" for name in ("omlx", "comfyui"))


def test_rollback_names_accept_comfyui_on_the_command_line(home):
    make_venv(home / "comfyui", "new")
    make_venv(home / "comfyui.old", "old")
    make_venv(home / "omlx", "new")
    make_venv(home / "omlx.old", "old")
    result = run(["bash", str(SCRIPTS / "build-engines-mac.sh"), "--rollback-venvs", "comfyui"], {"UNSLOTH_ENGINES_HOME": str(home)})
    assert result.returncode == 0, result.stderr
    assert (home / "comfyui" / "VERSION").read_text() == "old" and (home / "omlx" / "VERSION").read_text() == "new"


# --- engines.toml migration ---------------------------------------------------------------------

def code_lines(path: Path) -> str:
    return "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("#"))


def test_install_and_update_restart_the_helpers_only_through_the_shared_functions():
    for script in (FORK, UPDATE):
        code = code_lines(script)
        assert "helpers_stop" in code and "helpers_start" in code, script.name
        assert "launchctl" not in code.lower(), script.name
        assert "bootstrap" not in code and "kickstart" not in code, script.name
    lib = code_lines(SCRIPTS / "engines-lib.sh")
    # macOS refuses launchctl bootstrap of an SMAppService helper; the lib may only bootout or print
    assert "bootstrap" not in lib and "kickstart" not in lib
    assert "--engine-helpers register" in lib and "--engine-helpers unregister" in lib
    assert "HELPER_PLIST_DIR" not in lib


# --- engine lifetime: with_app (the default) and always -------------------------------------------


def desktop_json(rig):
    return json.loads((rig.home / "desktop.json").read_text())


def test_the_lifetime_is_read_from_desktop_json_and_defaults_to_with_app(rig):
    desktop = rig.home / "desktop.json"
    for text, expected in (
        (None, "with_app"),
        ('{"engine_lifetime": "always"}', "always"),
        ('{"engine_lifetime": "with_app"}', "with_app"),
        ('{"engines_enabled": true}', "with_app"),
        ('{"engine_lifetime": "forever"}', "with_app"),
        ("not json", "with_app"),
        ("[]", "with_app"),
    ):
        desktop.unlink(missing_ok=True)
        if text is not None:
            desktop.write_text(text)
        result = lib_fn(rig, "engine_lifetime")
        assert result.returncode == 0 and result.stdout.strip() == expected, (text, result.stdout, result.stderr)


def test_app_running_asks_pgrep_for_unsloth_studio(rig):
    assert lib_fn(rig, "app_running").returncode != 0
    rig.set_app_running(True)
    assert lib_fn(rig, "app_running").returncode == 0


def test_with_app_and_the_app_closed_leaves_the_helpers_stopped(rig):
    rig.set_desktop("with_app")
    result = lib_fn(rig, 'helpers_stop; helpers_start; echo "owed=${STOPPED_HELPERS[*]-} left=$HELPERS_LEFT_STOPPED"')
    assert result.returncode == 0, result.stdout + result.stderr
    assert "leaving the helpers stopped" in result.stdout
    assert result.stdout.strip().splitlines()[-1] == "owed= left=1"
    # stopped through the CLI, never registered again, and the CLI was not probed a second time
    assert rig.cli_calls() == ["engine-helpers unregister"]
    assert not any(rig.loaded(label) for label in rig.labels)
    assert_no_launchd_start(rig)


def test_with_app_and_the_app_running_registers_the_helpers_again(rig):
    rig.set_desktop("with_app")
    rig.set_app_running(True)
    result = lib_fn(rig, "helpers_stop; helpers_start")
    assert result.returncode == 0, result.stdout + result.stderr
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx"]
    assert all(rig.loaded(label) for label in rig.labels)


def test_a_missing_desktop_json_is_with_app_too(rig):
    (rig.home / "desktop.json").unlink()
    result = lib_fn(rig, 'helpers_stop; helpers_start; echo "left=$HELPERS_LEFT_STOPPED"')
    assert result.returncode == 0 and result.stdout.strip().endswith("left=1"), result.stdout + result.stderr
    assert rig.cli_calls() == ["engine-helpers unregister"]


def test_always_registers_the_helpers_again_with_the_app_closed(rig):
    rig.set_desktop("always")
    result = lib_fn(rig, 'helpers_stop; helpers_start; echo "left=$HELPERS_LEFT_STOPPED"')
    assert result.returncode == 0 and result.stdout.strip().endswith("left=0"), result.stdout + result.stderr
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx"]
    assert all(rig.loaded(label) for label in rig.labels)


def test_a_with_app_start_with_the_app_closed_never_runs_the_cli(rig):
    # the CLI probe spawns the app binary: with the app closed in with_app mode it must not run
    rig.set_desktop("with_app")
    result = lib_fn(rig, 'STOPPED_HELPERS=(ai.unsloth.studio.omlx); helpers_start')
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(c.startswith("engine-helpers") or c == "gui-started" for c in rig.calls())


def test_the_first_stop_records_engines_enabled_when_the_app_has_not(rig):
    (rig.home / "desktop.json").unlink()
    result = lib_fn(rig, "helpers_stop")
    assert result.returncode == 0, result.stdout + result.stderr
    assert desktop_json(rig) == {"engines_enabled": True}
    assert "recorded engines_enabled=true" in result.stdout


def test_a_recorded_choice_is_never_overwritten(rig):
    for enabled in (False, True):
        rig.set_desktop("with_app", enabled=enabled)
        rig.launchctl("print", f"gui/501/{rig.labels[0]}")
        assert lib_fn(rig, "seed_engines_enabled").returncode == 0
        assert desktop_json(rig) == {"engine_lifetime": "with_app", "engines_enabled": enabled}


def test_the_seed_keeps_the_lifetime_already_chosen(rig):
    rig.set_desktop("always", enabled=None)
    assert lib_fn(rig, "seed_engines_enabled").returncode == 0
    assert desktop_json(rig) == {"engine_lifetime": "always", "engines_enabled": True}


def test_an_unreadable_desktop_json_is_left_alone(rig):
    (rig.home / "desktop.json").write_text("not json")
    assert lib_fn(rig, "seed_engines_enabled").returncode == 0
    assert (rig.home / "desktop.json").read_text() == "not json"


def test_a_dry_run_records_nothing(rig):
    (rig.home / "desktop.json").unlink()
    result = lib_fn(rig, "DRY=1; helpers_stop")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (rig.home / "desktop.json").exists()


def test_helpers_that_were_not_loaded_record_nothing(rig):
    (rig.home / "desktop.json").unlink()
    assert run([str(rig.app / "Contents/MacOS/unsloth-studio"), "--engine-helpers", "unregister"], rig.env).returncode == 0
    result = lib_fn(rig, "helpers_stop")
    assert result.returncode == 0 and "not loaded" in result.stdout
    assert not (rig.home / "desktop.json").exists()


def test_update_in_with_app_mode_leaves_the_helpers_stopped_while_the_app_is_closed(rig):
    rig.set_desktop("with_app")
    result = rig.update(STAGE_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (rig.home / "omlx" / "VERSION").read_text() == "new"
    assert rig.cli_calls() == ["engine-helpers unregister omlx"]
    assert not any(rig.loaded(label) for label in rig.labels)
    assert "leaving the helpers stopped" in result.stdout and "next launch" in result.stdout
    assert_no_launchd_start(rig)


def test_update_in_with_app_mode_registers_the_helpers_while_the_app_runs(rig):
    rig.set_desktop("with_app")
    rig.set_app_running(True)
    result = rig.update(STAGE_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert rig.cli_calls() == ["engine-helpers unregister omlx", "engine-helpers register omlx"]
    assert all(rig.loaded(label) for label in rig.labels)


def test_update_rollback_in_with_app_mode_also_leaves_the_helpers_stopped(rig):
    rig.set_desktop("with_app")
    # a venv that is bad would only show once the helpers ran; make the swap itself fail instead
    stray = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", f"{rig.home}/omlx/bin/python"],
        cwd=rig.home / "omlx" / "bin",
    )
    try:
        result = rig.update(STAGE_VERSION="new")
        assert result.returncode != 0 and "not swapped" in result.stderr, result.stdout + result.stderr
        assert (rig.home / "omlx" / "VERSION").read_text() == "old"
        assert not any(rig.loaded(label) for label in rig.labels)
    finally:
        stray.kill()
        stray.wait()


def test_install_app_in_with_app_mode_swaps_the_app_and_leaves_the_helpers_stopped(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.set_desktop("with_app")
        result = install_app_rig(tmp_path, rig)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "new"
        assert rig.cli_calls() == ["engine-helpers unregister"]
        assert not any(rig.loaded(label) for label in rig.labels)
        assert_no_launchd_start(rig)
    finally:
        rig.close()


def test_install_in_with_app_mode_finds_the_helpers_already_down_after_the_app_quit(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.set_desktop("with_app")
        # the with_app quit has unregistered the helpers by the time the install gets to them
        assert run([str(rig.app / "Contents/MacOS/unsloth-studio"), "--engine-helpers", "unregister"], rig.env).returncode == 0
        (rig.lc / "calls.log").unlink(missing_ok=True)
        result = txn(tmp_path, rig)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "new"
        # nothing to stop, nothing owed, nothing registered: the app does that at its next launch
        assert helpers_touched(rig) == ["engine-helpers unregister"]
        assert not any(rig.loaded(label) for label in rig.labels)
    finally:
        rig.close()


def test_install_in_always_mode_still_brings_the_helpers_back(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.set_desktop("always")
        result = txn(tmp_path, rig)
        assert result.returncode == 0, result.stdout + result.stderr
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx"]
        assert all(rig.loaded(label) for label in rig.labels)
    finally:
        rig.close()


def test_the_install_waits_up_to_the_quit_wait_for_the_app_and_never_forces_it(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.set_app_running(True)
        result = fork_fn(
            tmp_path,
            "osascript() { echo \"osascript $*\"; }; DRY=0; quit_unsloth",
            rig.fork_env({"QUIT_WAIT": "2"}),
        )
        assert result.returncode != 0 and "did not quit within 2 s" in result.stderr, result.stdout + result.stderr
        assert "osascript -e" in result.stdout
        # the default is long enough for a with_app quit (45 s of engines, then the backend)
        assert 'QUIT_WAIT:-120' in FORK.read_text()
    finally:
        rig.close()


def test_the_install_does_not_quit_an_app_that_is_not_running(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        result = fork_fn(
            tmp_path,
            "osascript() { echo \"osascript $*\"; }; curl() { return 7; }; DRY=0; quit_unsloth; echo done",
            rig.fork_env(),
        )
        assert result.returncode == 0 and "osascript" not in result.stdout and "done" in result.stdout, result.stdout + result.stderr
    finally:
        rig.close()


def test_bash_backoff_schedule_matches_python(home):
    waits = []
    for _ in range(8):
        result = run(
            ["bash", "-c", 'sleep() { echo "SLEEP $1"; }; . "$1"; engines_fail omlx "boom"', "_", str(ENGINES / "engines-common.sh")],
            {"ENGINES_HOME": str(home)},
        )
        assert result.returncode == 78
        waits.append(int(result.stdout.split("SLEEP ")[1].split()[0]))
    assert waits == [10, 20, 40, 80, 160, 300, 300, 300]
    assert read_fail(home, "omlx")["count"] == 8



def test_swap_replaces_the_venv_keeps_old_and_records_the_marker_after(home):
    make_venv(home / "omlx", "old")
    (home / ".provisioned").write_text("omlx=old-key\nomlx_kernels=1\n")
    make_venv(home / "omlx.new", "new", ["omlx", "new-key", "omlx_kernels=0"])
    result = build_fn(home, 'venv_in_use() { return 1; }; swap_venv "$ENGINES_HOME/omlx"')
    assert result.returncode == 0, result.stderr
    assert (home / "omlx" / "VERSION").read_text() == "new"
    assert (home / "omlx.old" / "VERSION").read_text() == "old"
    assert not (home / "omlx.new").exists() and not (home / "omlx" / ".staged").exists()
    assert marker(home) == {"omlx": "new-key", "omlx_kernels": "0"}
    # the entry point and pyvenv.cfg now name the live path, not omlx.new
    assert (home / "omlx" / "bin" / "omlx").read_text().startswith(f"#!{home}/omlx/bin/python")
    assert f"{home}/omlx.new" not in (home / "omlx" / "pyvenv.cfg").read_text()



def test_pin_check_fails_on_a_dirty_submodule(repo_with_submodules, home):
    (repo_with_submodules / "studio" / "engines" / "omlx" / "f").write_text("dirty")
    result = pins(repo_with_submodules, home)
    assert result.returncode != 0 and "local changes" in result.stderr


# Legacy engines.toml migration and bundle hygiene.
def config_migrate(home, text):
    config = home / "engines.toml"
    config.write_text(text)
    result = build_fn(home, "migrate_config")
    return config, result


def test_generated_config_has_omlx_and_comfyui():
    import tomllib
    config = tomllib.loads((ENGINES / "engines.default.toml").read_text())
    assert set(config) == {"omlx", "comfyui"}
    assert "OMLX_PEER_EVICT_URLS" not in config["omlx"]["env"]
    comfy = config["comfyui"]
    assert comfy["port"] == 8844 and "host" not in comfy
    assert comfy["extra_args"] == ["--lowvram", "--disable-smart-memory", "--cpu-vae", "--disable-all-custom-nodes", "--offline"]
    assert comfy["env"] == {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    assert comfy["model_dirs"] == ["~/Library/Application Support/StoryPressRuntime/ComfyUI/models"]
    # the migration copies from "# ComfyUI (bundled helper" to the end of the file
    assert (ENGINES / "engines.default.toml").read_text().count("# ComfyUI (bundled helper") == 1


DEFAULT_TEXT = (ENGINES / "engines.default.toml").read_text()
COMFYUI_BLOCK = DEFAULT_TEXT[DEFAULT_TEXT.index("# ComfyUI (bundled helper"):]


def test_the_comfyui_section_is_appended_once_after_one_blank_line_with_a_backup(home):
    import tomllib
    original = '# saved\n[omlx]\nport = 9000\n'
    config, result = config_migrate(home, original)
    assert result.returncode == 0, result.stderr
    assert "appended [comfyui] section" in result.stdout
    assert config.read_text() == original + "\n" + COMFYUI_BLOCK
    assert tomllib.loads(config.read_text())["comfyui"]["port"] == 8844
    backups = list(home.glob("engines.toml.bak-*"))
    assert len(backups) == 1 and backups[0].read_text() == original
    before = config.stat().st_mtime_ns
    again = build_fn(home, "migrate_config")
    assert again.returncode == 0 and "nothing to migrate" in again.stdout
    assert config.stat().st_mtime_ns == before and list(home.glob("engines.toml.bak-*")) == backups
    assert config.read_text().count("[comfyui]") == 1


def test_the_appended_block_is_byte_identical_to_the_default_files(home):
    config, result = config_migrate(home, "[omlx]\nport = 1\n")
    assert result.returncode == 0
    assert config.read_text().endswith(COMFYUI_BLOCK)


@pytest.mark.parametrize("original,joined", [
    ("[omlx]\nport = 1", "[omlx]\nport = 1\n\n"),      # no final newline
    ("[omlx]\nport = 1\n\n", "[omlx]\nport = 1\n\n"),  # already a blank line: not doubled
])
def test_the_blank_line_before_the_appended_section_is_exactly_one(home, original, joined):
    config, result = config_migrate(home, original)
    assert result.returncode == 0, result.stderr
    assert config.read_text() == joined + COMFYUI_BLOCK


@pytest.mark.parametrize("text", [
    '[omlx]\nport = 1\n[comfyui]\nport = 9999\n',
    '[omlx]\nport = 1\n[comfyui.env]\nX = "1"\n',
    '[ comfyui ]\nport = 5\n',
])
def test_an_existing_comfyui_section_is_left_untouched(home, text):
    config = home / "engines.toml"
    config.write_text(text)
    before = config.stat().st_mtime_ns
    result = build_fn(home, "migrate_config")
    assert result.returncode == 0 and config.read_text() == text and config.stat().st_mtime_ns == before
    assert not list(home.glob("engines.toml.bak-*"))


def test_both_migrations_apply_in_one_pass_and_one_backup(home):
    peer = 'OMLX_PEER_EVICT_URLS = "http://127.0.0.1:8001"'
    original = f'[omlx.env]\nOMLX_NAX = "1"\n{peer}\n'
    config, result = config_migrate(home, original)
    assert result.returncode == 0, result.stderr
    assert config.read_text() == '[omlx.env]\nOMLX_NAX = "1"\n\n' + COMFYUI_BLOCK
    assert [b.read_text() for b in home.glob("engines.toml.bak-*")] == [original]


def test_a_fresh_install_gets_the_default_file_with_the_comfyui_section(home, tmp_path):
    result = build_fn(home, f'WRAPPERS="{ENGINES}"; write_default_config')
    assert result.returncode == 0, result.stderr
    assert (home / "engines.toml").read_text() == DEFAULT_TEXT


def test_peer_url_removal_also_removes_the_comment_the_old_migration_wrote(home):
    note = "# oMLX asks the ds4 launcher to unload when it needs memory (added by the engines.toml migration)."
    peer = 'OMLX_PEER_EVICT_URLS = "http://127.0.0.1:8001"'
    config, result = config_migrate(home, f'[omlx.env]\nOMLX_NAX = "1"\n{note}\n{peer}\n[ds4]\nhost = "0.0.0.0"\n[comfyui]\n')
    assert result.returncode == 0, result.stderr
    assert config.read_text() == '[omlx.env]\nOMLX_NAX = "1"\n[ds4]\nhost = "0.0.0.0"\n[comfyui]\n'


def test_peer_url_removal_keeps_other_comments_and_a_non_adjacent_note(home):
    note = "# oMLX asks the ds4 launcher to unload."
    peer = 'OMLX_PEER_EVICT_URLS = "http://127.0.0.1:8001"'
    config, result = config_migrate(home, f'[omlx.env]\n{note}\nOMLX_NAX = "1"\n# my own comment\n{peer}\n[comfyui]\n')
    assert result.returncode == 0, result.stderr
    assert config.read_text() == f'[omlx.env]\n{note}\nOMLX_NAX = "1"\n# my own comment\n[comfyui]\n'
    other = 'x = 1\n# oMLX asks something else\n' + f'[omlx.env]\n# oMLX asks something else\n{peer}\n[comfyui]\n'
    config, result = config_migrate(home, other)
    assert config.read_text() == 'x = 1\n# oMLX asks something else\n[omlx.env]\n# oMLX asks something else\n[comfyui]\n'


def test_peer_url_removal_preserves_config_and_backs_up_once(home):
    import tomllib
    original = '# saved\n[omlx]\nport = 9000\n[omlx.env]\nOMLX_NAX = "1"\nOMLX_PEER_EVICT_URLS = "http://127.0.0.1:8001"\n[ds4]\nhost = "0.0.0.0"\n'
    config, result = config_migrate(home, original)
    assert result.returncode == 0, result.stderr
    backups = list(home.glob("engines.toml.bak-*"))
    assert len(backups) == 1 and backups[0].read_text() == original
    data = tomllib.loads(config.read_text())
    assert data["omlx"]["port"] == 9000
    assert data["omlx"]["env"] == {"OMLX_NAX": "1"}
    assert data["ds4"] == {"host": "0.0.0.0"}
    before = config.stat().st_mtime_ns
    assert build_fn(home, "migrate_config").returncode == 0
    assert config.stat().st_mtime_ns == before
    assert list(home.glob("engines.toml.bak-*")) == backups


@pytest.mark.parametrize("text", [
    '[omlx]\nport = 9000\n[comfyui]\n',
    '[omlx.env]\nOMLX_PEER_EVICT_URLS = "http://127.0.0.1:9999"\n[comfyui]\n',
    '[other]\nOMLX_PEER_EVICT_URLS = "http://127.0.0.1:8001"\n[comfyui]\n',
    '[omlx.env\ninvalid TOML',
])
def test_config_noop_never_writes_or_backs_up(home, text):
    config = home / "engines.toml"
    config.write_text(text)
    before = config.stat().st_mtime_ns
    assert build_fn(home, "migrate_config").returncode == 0
    assert config.read_text() == text and config.stat().st_mtime_ns == before
    assert not list(home.glob("engines.toml.bak-*"))


def test_staging_removes_previous_engine_resources(home):
    staging = home / "staging"
    staging.mkdir()
    (staging / "obsolete-resource").write_text("old")
    result = build_fn(home, 'STAGING="$ENGINES_HOME/staging"; stage_wrappers')
    assert result.returncode == 0, result.stderr
    assert {p.name for p in staging.iterdir()} == {"omlx-launch", "comfyui-launch", "engines-common.sh", "engines_launch.py"}
    assert all(os.access(staging / p.name, os.X_OK) for p in staging.iterdir())


def test_install_unregisters_using_old_bundle_before_swap():
    text = (SCRIPTS / "build-fork-mac.sh").read_text()
    stop = text.index('HELPER_APP="$INSTALLED_APP" helpers_stop')
    swap = text.index('run mv "$INSTALLED_APP" "$APP_PREV"')
    assert stop < swap


def test_helper_cleanup_with_no_current_label_calls_old_cli(home, tmp_path):
    # No HTTP server or launchctl: a stand-in old app records the unregister attempt.
    binary = tmp_path / "Old.app" / "Contents" / "MacOS" / "unsloth-studio"
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/sh\n# --engine-helpers\nif [ "$2" = status ]; then echo \'{"helpers":[]}\'; else echo "$2" >>"$UNSLOTH_ENGINES_HOME/cli-calls"; fi\n')
    binary.chmod(0o755)
    result = run(["bash", "-c", f'log() {{ :; }}; warn() {{ :; }}; die() {{ exit 1; }}; run() {{ "$@"; }}; DRY=0; . "{SCRIPTS}/engines-lib.sh"; helper_loaded() {{ return 1; }}; helpers_stop; echo "owed=${{#STOPPED_HELPERS[@]}}"'], env_for(home, HELPER_APP=str(tmp_path / "Old.app")))
    assert result.returncode == 0, result.stderr
    assert (home / "cli-calls").read_text() == "unregister\n"
    assert "owed=0" in result.stdout
    assert not (home / "desktop.json").exists()


def test_update_refuses_while_an_engine_is_busy_and_stages_nothing(rig):
    (rig.home / "omlx-busy").write_text("1")
    result = rig.update(STAGE_VERSION="new")
    assert result.returncode != 0 and "oMLX is busy" in result.stderr
    assert not any(c.startswith("fake stage") for c in rig.calls()) and helpers_touched(rig) == []
    assert (rig.home / "omlx" / "VERSION").read_text() == "old"



def test_update_stops_on_a_model_busy_unload_without_forcing_it(rig):
    # idle by /api/status, but oMLX answers 409 model_busy to the graceful unload (a client raced in)
    (rig.home / "omlx-loaded").write_text("1")
    (rig.home / "omlx-unload-busy").write_text("1")
    result = rig.update(STAGE_VERSION="new")
    assert result.returncode != 0 and "refused to unload" in result.stderr and "busy" in result.stderr
    calls = rig.calls()
    assert "omlx-unload-busy" in calls
    assert "omlx-unload-force" not in calls and "omlx-unload" not in calls
    assert (rig.home / "omlx-loaded").exists()
    assert helpers_touched(rig) == []
    assert (rig.home / "omlx" / "VERSION").read_text() == "old"


def test_the_quiesce_unload_never_passes_force():
    text = (SCRIPTS / "engines-lib.sh").read_text()
    assert "force=1" not in text.replace('Never "?force=1"', "")


def test_a_failed_unregister_still_restarts_the_helper_that_was_stopped(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        # The helper stopped before unregister reported failure; rollback must restart it.
        result = install_app_rig(tmp_path, rig, {"FAKE_UNREGISTER_FAIL_AFTER": "omlx"})
        assert result.returncode != 0, result.stdout + result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "old"
        assert not any(c.startswith("mv ") for c in rig.calls())
        assert rig.cli_calls()[0] == "engine-helpers unregister"
        assert "engine-helpers register omlx" in rig.cli_calls()
        assert_no_launchd_start(rig)
        assert all(rig.loaded(label) for label in rig.labels)
    finally:
        rig.close()



def test_a_port_gate_timeout_restarts_the_stopped_helpers(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.enable_comfyui()  # the gate watches only the ports of helpers that were running
        with socket.socket() as held:
            held.bind(("127.0.0.1", 0))
            held.listen()
            port = held.getsockname()[1]
            result = install_app_rig(
                tmp_path, rig, {"ENGINE_PORTS": f"{rig.omlx_port} {port}", "PORT_GATE_TIMEOUT": "2"}
            )
        assert result.returncode != 0 and "ports still held" in result.stderr, result.stdout + result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "old"
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx", "engine-helpers register comfyui"]
        assert all(rig.loaded(label) for label in rig.labels)
    finally:
        rig.close()


def test_a_busy_engine_blocks_the_app_swap_and_leaves_the_app_alone(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        make_app(tmp_path / "dist" / "Unsloth.app", "new", cli="cli")
        make_app(tmp_path / "Apps" / "Unsloth.app", "old", cli="cli")
        (home / "omlx-busy").write_text("1")
        result = fork_fn(tmp_path, "codesign() { :; }; DRY=0; install_app", rig.fork_env())
        assert result.returncode != 0 and "oMLX is busy" in result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "old"
        assert helpers_touched(rig) == []
    finally:
        rig.close()



def test_a_port_gate_timeout_after_the_backend_swap_restores_everything(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.enable_comfyui()
        with socket.socket() as held:
            held.bind(("127.0.0.1", 0))
            held.listen()
            result = txn(
                tmp_path,
                rig,
                extra_env={
                    "ENGINE_PORTS": f"{rig.omlx_port} {held.getsockname()[1]}",
                    "PORT_GATE_TIMEOUT": "2",
                },
            )
        assert "ports still held" in result.stderr, result.stdout + result.stderr
        assert_rolled_back(tmp_path, rig, result)
    finally:
        rig.close()



# --- migration over a pre-removal app: the legacy ds4 helper ------------------------------------

LEGACY_LABEL = "ai.unsloth.studio.ds4"


def test_a_legacy_ds4_job_left_after_the_first_unregister_is_unregistered_again(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.start_legacy_ds4()
        result = install_app_rig(tmp_path, rig, {"FAKE_LEGACY_DS4_STICKY": "1"})
        assert result.returncode == 0, result.stdout + result.stderr
        assert "legacy ai.unsloth.studio.ds4 job is still loaded" in result.stderr
        assert "the legacy ai.unsloth.studio.ds4 job is gone" in result.stdout
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "new"
        # unregister twice through the OLD app, then register through the new one
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers unregister", "engine-helpers register omlx"]
        assert not rig.loaded(LEGACY_LABEL)
        # the launcher was seen idle, so it was asked to stop while idle
        assert "ds4-stop /admin/stop?if_idle=1" in rig.calls()
        assert all(rig.loaded(label) for label in rig.labels)
    finally:
        rig.close()


def test_a_legacy_ds4_job_that_survives_both_unregisters_fails_the_install_and_rolls_back(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.start_legacy_ds4()
        result = txn(tmp_path, rig, extra_env={"FAKE_LEGACY_DS4_STICKY": "99", "LEGACY_DS4_WAIT": "1"})
        assert "still loaded after 1 s" in result.stderr, result.stdout + result.stderr
        assert_rolled_back(tmp_path, rig, result)
        assert not any(c.startswith("mv ") for c in rig.calls())
    finally:
        rig.close()


@pytest.mark.parametrize("state", [{"busy": True}, {"starting": True}])
def test_a_busy_legacy_ds4_launcher_refuses_the_install_before_anything_stops(tmp_path, home, state):
    rig = Rig(tmp_path, home)
    try:
        rig.start_legacy_ds4(**state)
        result = install_app_rig(tmp_path, rig)
        assert result.returncode != 0 and "legacy DwarfStar launcher" in result.stderr and "busy or starting" in result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "old"
        assert helpers_touched(rig) == []
        assert not any(c.startswith("ds4-stop") for c in rig.calls())
    finally:
        rig.close()


def test_the_idle_check_and_the_port_gate_ignore_a_legacy_launcher_that_is_not_there(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        with socket.socket() as held:
            held.bind(("127.0.0.1", 0))
            held.listen()
            # a stranger on a legacy port with no launcher answering: not watched, not refused
            result = install_app_rig(tmp_path, rig, {"LEGACY_DS4_PORTS": str(held.getsockname()[1])})
        assert result.returncode == 0, result.stdout + result.stderr
        assert not any(c.startswith("ds4-stop") for c in rig.calls())
    finally:
        rig.close()


def test_the_port_gate_watches_the_legacy_ports_only_when_the_launcher_was_seen(home):
    snippet = (
        'log() { :; }; warn() { :; }; die() { exit 1; }; DRY=0; . "%s/engines-lib.sh"; '
        'echo "unseen=$(gate_ports ai.unsloth.studio.omlx)"; LEGACY_DS4_SEEN=1; echo "seen=$(gate_ports ai.unsloth.studio.omlx)"'
    ) % SCRIPTS
    result = run(["bash", "-c", snippet], env_for(home, ENGINE_PORTS="1111", LEGACY_DS4_PORTS="2222 3333"))
    assert result.returncode == 0, result.stderr
    assert "unseen=1111\n" in result.stdout and "seen=1111 2222 3333" in result.stdout


def test_an_idle_legacy_launcher_is_stopped_during_the_quiesce(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.start_legacy_ds4()
        result = fork_fn(tmp_path, "DRY=0; engines_quiesce", rig.fork_env())
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ds4-stop /admin/stop?if_idle=1" in rig.calls()
    finally:
        rig.close()


def test_a_busy_legacy_launcher_is_refused_by_the_update_script_too(rig):
    rig.start_legacy_ds4(busy=True)
    result = rig.update(STAGE_VERSION="new")
    assert result.returncode != 0 and "legacy DwarfStar launcher" in result.stderr
    assert helpers_touched(rig) == []


# --- two helpers: ComfyUI beside oMLX -------------------------------------------------------------


COMFYUI_LABEL = "ai.unsloth.studio.comfyui"
OMLX_LABEL = "ai.unsloth.studio.omlx"


def status_json(rig, **env):
    result = run([str(rig.app / "Contents/MacOS/unsloth-studio"), "--engine-helpers", "status"], {**rig.env, **env})
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_status_lists_both_helpers_with_wanted_and_the_cli_version(rig):
    status = status_json(rig)
    assert status["cli"] == 2
    assert [(h["name"], h["wanted"]) for h in status["helpers"]] == [("omlx", True), ("comfyui", False)]
    # ComfyUI is off until the user turns it on; the master switch gates both
    rig.set_desktop("always", helpers={"comfyui": True})
    assert [h["wanted"] for h in status_json(rig)["helpers"]] == [True, True]
    rig.set_desktop("always", enabled=False, helpers={"comfyui": True})
    assert [h["wanted"] for h in status_json(rig)["helpers"]] == [False, False]


def test_an_old_app_has_no_cli_field_and_one_helper(rig):
    status = status_json(rig, FAKE_CLI_V1="1")
    assert "cli" not in status and [h["name"] for h in status["helpers"]] == ["omlx"]
    assert lib_fn(rig, 'helper_cli_v2 "$(helper_cli)"').returncode == 0
    assert lib_fn(rig, 'helper_cli_v2 "$(helper_cli)"', {"FAKE_CLI_V1": "1"}).returncode != 0


def test_the_untargeted_register_starts_the_chosen_helpers_only(rig):
    cli = str(rig.app / "Contents/MacOS/unsloth-studio")
    assert run([cli, "--engine-helpers", "unregister"], rig.env).returncode == 0
    assert run([cli, "--engine-helpers", "register"], rig.env).returncode == 0
    assert rig.loaded(OMLX_LABEL) and not rig.loaded(COMFYUI_LABEL)
    rig.set_desktop("always", helpers={"omlx": False, "comfyui": True})
    assert run([cli, "--engine-helpers", "unregister"], rig.env).returncode == 0
    assert run([cli, "--engine-helpers", "register"], rig.env).returncode == 0
    assert rig.loaded(COMFYUI_LABEL) and not rig.loaded(OMLX_LABEL)
    # a target registers that helper whether or not it is chosen
    assert run([cli, "--engine-helpers", "register", "omlx"], rig.env).returncode == 0
    assert rig.loaded(OMLX_LABEL)
    assert run([cli, "--engine-helpers", "register", "plex"], rig.env).returncode == 2


def test_an_idle_check_dies_on_a_busy_comfyui(rig):
    rig.enable_comfyui()
    (rig.home / "comfyui-busy").write_text("1")
    result = lib_fn(rig, "engines_require_idle")
    assert result.returncode != 0
    assert "ComfyUI has 1 jobs running or queued" in result.stderr and "Settings > Engines" in result.stderr
    (rig.home / "comfyui-pending").write_text("1")
    assert "ComfyUI has 2 jobs" in lib_fn(rig, "engines_require_idle").stderr
    for name in ("comfyui-busy", "comfyui-pending"):
        (rig.home / name).unlink()
    assert lib_fn(rig, "engines_require_idle").returncode == 0


def test_an_unreadable_comfyui_queue_counts_as_busy(rig):
    result = lib_fn(rig, 'echo "not json" | comfyui_busy_count; echo "{}" | comfyui_busy_count; echo \'{"queue_running":[],"queue_pending":[]}\' | comfyui_busy_count')
    assert result.stdout.split() == ["1", "1", "0"]


def test_the_quiesce_frees_comfyui_and_unloads_omlx(rig):
    rig.enable_comfyui()
    (rig.home / "omlx-loaded").write_text("1")
    result = lib_fn(rig, "engines_quiesce")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = rig.calls()
    assert "omlx-unload" in calls
    frees = [c for c in calls if c.startswith("comfyui-free")]
    assert len(frees) == 1
    assert json.loads(frees[0].split(" ", 1)[1]) == {"unload_models": True, "free_memory": True}


def test_the_quiesce_can_be_limited_to_one_engine(rig):
    rig.enable_comfyui()
    (rig.home / "omlx-loaded").write_text("1")
    assert lib_fn(rig, "engines_quiesce comfyui").returncode == 0
    assert "omlx-unload" not in rig.calls() and any(c.startswith("comfyui-free") for c in rig.calls())
    (rig.lc / "calls.log").unlink()
    assert lib_fn(rig, "engines_quiesce omlx").returncode == 0
    assert "omlx-unload" in rig.calls() and not any(c.startswith("comfyui-free") for c in rig.calls())


def test_a_failed_comfyui_free_is_a_warning_not_a_failure(rig):
    rig.enable_comfyui()
    (rig.home / "comfyui-free-fail").write_text("1")
    result = lib_fn(rig, "engines_quiesce")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ComfyUI /free answered HTTP 500" in result.stderr


def test_an_absent_comfyui_is_nothing_to_free(rig):
    result = lib_fn(rig, "engines_quiesce")
    assert result.returncode == 0 and "ComfyUI: " in result.stdout and "nothing to free" in result.stdout
    assert not any(c.startswith("comfyui-free") for c in rig.calls())


def test_helpers_stop_owes_only_the_loaded_labels(rig):
    # ComfyUI is not loaded: stopping everything must not make the restart bring it up
    result = lib_fn(rig, 'helpers_stop; echo "owed=${STOPPED_HELPERS[*]}"; helpers_start')
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"owed={OMLX_LABEL}\n" in result.stdout
    assert f"{COMFYUI_LABEL} is not loaded" in result.stdout
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx"]
    assert rig.loaded(OMLX_LABEL) and not rig.loaded(COMFYUI_LABEL)


def test_both_loaded_helpers_are_stopped_together_and_registered_by_name(rig):
    rig.enable_comfyui()
    result = lib_fn(rig, 'helpers_stop; echo "owed=${STOPPED_HELPERS[*]}"; helpers_start')
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"owed={OMLX_LABEL} {COMFYUI_LABEL}" in result.stdout
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx", "engine-helpers register comfyui"]
    assert rig.loaded(OMLX_LABEL) and rig.loaded(COMFYUI_LABEL)
    assert_no_launchd_start(rig)


def test_a_subset_is_stopped_by_name_and_leaves_the_other_helper_running(rig):
    rig.enable_comfyui()
    result = lib_fn(rig, f'helpers_stop {COMFYUI_LABEL}; echo "owed=${{STOPPED_HELPERS[*]}}"; helpers_start')
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"owed={COMFYUI_LABEL}" in result.stdout
    assert rig.cli_calls() == ["engine-helpers unregister comfyui", "engine-helpers register comfyui"]
    assert rig.loaded(OMLX_LABEL) and rig.loaded(COMFYUI_LABEL)


def test_a_subset_with_an_old_app_is_booted_out_not_unregistered(rig):
    # an app without the named-target CLI would unregister every helper: use launchctl bootout
    result = lib_fn(rig, f"helpers_stop {OMLX_LABEL}", {"FAKE_CLI_V1": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert rig.cli_calls() == []
    assert any(c.startswith("launchctl bootout") and c.endswith(OMLX_LABEL) for c in rig.calls())


def test_an_old_app_gets_the_plain_register_and_never_a_target(rig):
    result = lib_fn(rig, "helpers_stop; helpers_start", {"FAKE_CLI_V1": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register"]
    assert rig.loaded(OMLX_LABEL)


def test_an_old_app_cannot_bring_comfyui_back_and_says_so(rig):
    result = lib_fn(rig, f"STOPPED_HELPERS=({OMLX_LABEL} {COMFYUI_LABEL}); helpers_start", {"FAKE_CLI_V1": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"cannot register {COMFYUI_LABEL}" in result.stderr
    assert rig.cli_calls() == ["engine-helpers register"]
    assert rig.loaded(OMLX_LABEL) and not rig.loaded(COMFYUI_LABEL)


def test_with_app_and_the_app_closed_leaves_both_helpers_stopped(rig):
    rig.enable_comfyui()
    rig.set_desktop("with_app", helpers={"omlx": True, "comfyui": True})
    result = lib_fn(rig, 'helpers_stop; helpers_start; echo "owed=${STOPPED_HELPERS[*]-} left=$HELPERS_LEFT_STOPPED"')
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "owed= left=1"
    assert rig.cli_calls() == ["engine-helpers unregister"]
    assert not rig.loaded(OMLX_LABEL) and not rig.loaded(COMFYUI_LABEL)


def test_the_port_gate_covers_the_comfyui_port_and_follows_the_labels(rig):
    held = socket.socket()
    try:
        held.bind(("127.0.0.1", rig.comfyui_port))
        held.listen()
        every = lib_fn(rig, f"wait_ports_free {OMLX_LABEL} {COMFYUI_LABEL}", {"PORT_GATE_TIMEOUT": "2"})
        assert every.returncode != 0 and "ports still held" in every.stderr
        # no label, no engine port: nothing is waited for
        none = lib_fn(rig, "wait_ports_free", {"PORT_GATE_TIMEOUT": "2"})
        assert none.returncode == 0 and "no engine port to wait for" in none.stdout
        # the oMLX helper alone is stopped: ComfyUI keeping its port is not a reason to wait
        assert run([str(rig.app / "Contents/MacOS/unsloth-studio"), "--engine-helpers", "unregister", "omlx"], rig.env).returncode == 0
        only_omlx = lib_fn(rig, f"wait_ports_free {OMLX_LABEL}", {"PORT_GATE_TIMEOUT": "2"})
        assert only_omlx.returncode == 0, only_omlx.stdout + only_omlx.stderr
        only_comfyui = lib_fn(rig, f"wait_ports_free {COMFYUI_LABEL}", {"PORT_GATE_TIMEOUT": "2"})
        assert only_comfyui.returncode != 0
    finally:
        held.close()


def test_the_health_wait_checks_each_expected_engine(rig):
    rig.enable_comfyui()
    ok = lib_fn(rig, "wait_engines_healthy 3 1 1")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    (rig.home / "comfyui").mkdir(exist_ok=True)
    (rig.home / "comfyui" / "VERSION").write_text("bad")
    assert lib_fn(rig, "wait_engines_healthy 1 1 1").returncode != 0
    # not expecting ComfyUI: its bad answer does not matter; the two-argument form still works
    assert lib_fn(rig, "wait_engines_healthy 3 1 0").returncode == 0
    assert lib_fn(rig, "wait_engines_healthy 3 1").returncode == 0


def test_the_dry_run_of_the_helper_cycle_lists_both_helpers(rig):
    result = lib_fn(rig, "DRY=1; helpers_stop; helpers_start")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--engine-helpers unregister" in result.stdout
    assert "--engine-helpers register omlx" in result.stdout and "--engine-helpers register comfyui" in result.stdout
    assert rig.cli_calls() == []


def test_the_install_dry_run_lists_both_helpers_and_both_ports(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        make_app(tmp_path / "dist" / "Unsloth.app", "new", cli="cli")
        make_app(tmp_path / "Apps" / "Unsloth.app", "old", cli="cli")
        result = fork_fn(tmp_path, "DRY=1; install_app", rig.fork_env())
        assert result.returncode == 0, result.stdout + result.stderr
        assert "free ComfyUI" in result.stdout and "[dry-run]" in result.stdout
        assert "--engine-helpers register omlx" in result.stdout and "--engine-helpers register comfyui" in result.stdout
        assert f"ports {rig.omlx_port} {rig.comfyui_port} must stop listening" in result.stdout
        assert rig.cli_calls() == []
    finally:
        rig.close()


def test_an_install_brings_back_only_the_helpers_that_were_running(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.enable_comfyui()
        result = install_app_rig(tmp_path, rig)
        assert result.returncode == 0, result.stdout + result.stderr
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx", "engine-helpers register comfyui"]
        assert rig.loaded(OMLX_LABEL) and rig.loaded(COMFYUI_LABEL)
        assert any(c.startswith("comfyui-free") for c in rig.calls())
    finally:
        rig.close()


def test_an_install_without_comfyui_running_does_not_start_it(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        result = install_app_rig(tmp_path, rig)
        assert result.returncode == 0, result.stdout + result.stderr
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx"]
        assert rig.loaded(OMLX_LABEL) and not rig.loaded(COMFYUI_LABEL)
    finally:
        rig.close()


def test_a_failed_install_restarts_both_owed_helpers_by_name(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.enable_comfyui()
        # the unregister stops oMLX and then reports failure: the install dies, and the rollback
        # brings back every helper it owes, each by name
        result = install_app_rig(tmp_path, rig, {"FAKE_UNREGISTER_FAIL_AFTER": "omlx"})
        assert result.returncode != 0
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx", "engine-helpers register comfyui"]
        assert rig.loaded(OMLX_LABEL) and rig.loaded(COMFYUI_LABEL)
    finally:
        rig.close()


def test_a_comfyui_only_update_bounces_only_the_comfyui_helper(rig):
    rig.add_comfyui()
    rig.enable_comfyui()
    (rig.home / "omlx-loaded").write_text("1")
    result = rig.update(STAGE_COMFYUI_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (rig.home / "comfyui" / "VERSION").read_text() == "new"
    assert rig.cli_calls() == ["engine-helpers unregister comfyui", "engine-helpers register comfyui"]
    calls = rig.calls()
    assert "omlx-unload" not in calls and (rig.home / "omlx-loaded").exists()
    assert any(c.startswith("comfyui-free") for c in calls)
    assert rig.loaded(OMLX_LABEL) and rig.loaded(COMFYUI_LABEL)
    assert "oMLX and its helper were not touched" in result.stdout
    assert_no_launchd_start(rig)


def test_an_omlx_only_update_leaves_a_running_comfyui_alone(rig):
    rig.add_comfyui()
    rig.enable_comfyui()
    result = rig.update(STAGE_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert rig.cli_calls() == ["engine-helpers unregister omlx", "engine-helpers register omlx"]
    assert not any(c.startswith("comfyui-free") for c in rig.calls())
    assert rig.loaded(COMFYUI_LABEL) and (rig.home / "comfyui" / "VERSION").read_text() == "old"
    assert "ComfyUI and its helper were not touched" in result.stdout


def test_an_unhealthy_comfyui_helper_rolls_the_swap_back(rig):
    rig.add_comfyui()
    rig.enable_comfyui()
    # the smoke test passes, but /system_stats answers 500 while the venv says bad
    result = rig.update(STAGE_COMFYUI_VERSION="bad", COMFYUI_CHECK="/usr/bin/true")
    assert result.returncode != 0
    assert "rolling back" in result.stdout and "rolled back" in result.stdout, result.stdout + result.stderr
    assert (rig.home / "comfyui" / "VERSION").read_text() == "old"
    assert (rig.home / "comfyui.failed" / "VERSION").read_text() == "bad"
    assert rig.loaded(COMFYUI_LABEL) and rig.loaded(OMLX_LABEL)
    assert rig.cli_calls().count("engine-helpers unregister comfyui") == 2


def test_a_hand_started_comfyui_still_blocks_its_swap_but_the_helper_does_not(rig):
    rig.add_comfyui()
    rig.enable_comfyui()
    # the helper runs from the venv's path in its argv in real life; the fake does not, so only the
    # decision is checked here: with the helper loaded the process guard is skipped
    result = rig.update(STAGE_COMFYUI_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "outside the helper" not in result.stderr


def test_the_update_dry_run_lists_both_helpers(rig):
    rig.add_comfyui()
    result = rig.update("--dry-run", STAGE_VERSION="new", STAGE_COMFYUI_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--engine-helpers unregister" in result.stdout
    assert "--engine-helpers register omlx" in result.stdout and "--engine-helpers register comfyui" in result.stdout
    assert rig.cli_calls() == []


@pytest.mark.parametrize("exit_code", [0, 1])
def test_the_comfyui_smoke_dir_is_removed_after_validation(home, exit_code):
    venv = home / "venv"
    fake_python(venv / "bin" / "python", f'[ "$1" = -c ] && exit 0\nexit {exit_code}\n')
    (venv / "src").mkdir()
    result = build_fn(home, f'rc=0; validate_comfyui "{venv}" || rc=$?; echo "rc=$rc"; ls "$BUILD_ROOT"')
    assert f"rc={exit_code}" in result.stdout, result.stdout + result.stderr
    assert "comfyui-smoke" not in result.stdout


# --- rollback with a stuck engine ----------------------------------------------------------------


def test_a_stuck_comfyui_does_not_strand_omlx_in_an_update_rollback(rig):
    rig.add_comfyui()
    rig.enable_comfyui()
    # step 4 stops both helpers (unregister call 1); in the rollback oMLX stops (2) and ComfyUI's
    # unregister (3) fails: it keeps running and keeps its port
    result = rig.update(
        STAGE_VERSION="bad",
        STAGE_COMFYUI_VERSION="new",
        FAKE_UNREGISTER_FAIL="comfyui",
        FAKE_UNREGISTER_FAIL_SINCE="3",
    )
    assert result.returncode != 0
    out = result.stdout + result.stderr
    assert "rolling back" in out, out
    # oMLX: venv restored, helper registered again
    assert (rig.home / "omlx" / "VERSION").read_text() == "old"
    assert (rig.home / "omlx.failed" / "VERSION").read_text() == "bad"
    assert rig.loaded(OMLX_LABEL) and (rig.lc / f"registered.{OMLX_LABEL}").exists()
    # ComfyUI: its live process is not pulled out from under it, and the warning says what to do
    assert (rig.home / "comfyui" / "VERSION").read_text() == "new"
    assert (rig.home / "comfyui.old" / "VERSION").read_text() == "old"
    assert rig.loaded(COMFYUI_LABEL)
    assert "the comfyui helper is still running" in result.stderr
    assert "--rollback-venvs comfyui" in result.stderr
    assert "keeps its new venv" in result.stderr


def test_a_stuck_omlx_does_not_strand_comfyui_in_an_update_rollback(rig):
    rig.add_comfyui()
    rig.enable_comfyui()
    result = rig.update(
        STAGE_VERSION="new",
        STAGE_COMFYUI_VERSION="bad",
        COMFYUI_CHECK="/usr/bin/true",
        FAKE_UNREGISTER_FAIL="omlx",
        FAKE_UNREGISTER_FAIL_SINCE="2",
    )
    assert result.returncode != 0
    # ComfyUI (the unhealthy one): restored and running again
    assert (rig.home / "comfyui" / "VERSION").read_text() == "old"
    assert rig.loaded(COMFYUI_LABEL)
    # oMLX could not be stopped: its venv stays, with the manual step in the warning
    assert (rig.home / "omlx" / "VERSION").read_text() == "new"
    assert rig.loaded(OMLX_LABEL)
    assert "the omlx helper is still running" in result.stderr and "--rollback-venvs omlx" in result.stderr


def test_a_stuck_comfyui_does_not_strand_the_previous_app_or_omlx_in_an_install_rollback(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.enable_comfyui()
        overrides, _ = FAILURES["final-verify"]
        # install stops everything (unregister 1) and registers from the new app; the rollback's
        # unregister (2) leaves ComfyUI running and holding its port
        result = txn(
            tmp_path,
            rig,
            overrides,
            {"FAKE_UNREGISTER_FAIL": "comfyui", "FAKE_UNREGISTER_FAIL_SINCE": "2"},
        )
        assert result.returncode != 0
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "old"
        assert rig.loaded(OMLX_LABEL) and (rig.lc / f"registered.{OMLX_LABEL}").exists()
        assert rig.loaded(COMFYUI_LABEL)
        assert "the comfyui helper still holds port" in result.stderr
        assert "the previous backend was restored" in result.stderr
    finally:
        rig.close()


def test_an_install_ignores_a_process_on_the_comfyui_port_when_the_helper_was_not_running(tmp_path, home):
    rig = Rig(tmp_path, home)
    held = socket.socket()
    try:
        # ComfyUI is disabled (no helper), but something unrelated listens where it would
        held.bind(("127.0.0.1", rig.comfyui_port))
        held.listen()
        result = install_app_rig(tmp_path, rig, {"PORT_GATE_TIMEOUT": "3"})
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "new"
        assert f"ports {rig.omlx_port} must stop listening" in result.stdout
        assert str(rig.comfyui_port) not in result.stdout.split("port-free gate")[1].split("\n")[0]
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx"]
        assert rig.loaded(OMLX_LABEL)
    finally:
        held.close()
        rig.close()


def test_an_install_without_any_running_helper_gates_no_engine_port(tmp_path, home):
    rig = Rig(tmp_path, home)
    held = socket.socket()
    try:
        rig.set_desktop("with_app")
        assert run([str(rig.app / "Contents/MacOS/unsloth-studio"), "--engine-helpers", "unregister"], rig.env).returncode == 0
        held.bind(("127.0.0.1", rig.comfyui_port))
        held.listen()
        result = install_app_rig(tmp_path, rig, {"PORT_GATE_TIMEOUT": "3"})
        assert result.returncode == 0, result.stdout + result.stderr
        assert "no engine port to wait for" in result.stdout
    finally:
        held.close()
        rig.close()


def test_an_install_stops_and_gates_the_comfyui_helper_when_it_was_running(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.enable_comfyui()
        result = install_app_rig(tmp_path, rig)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"ports {rig.omlx_port} {rig.comfyui_port} must stop listening" in result.stdout
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register omlx", "engine-helpers register comfyui"]
    finally:
        rig.close()


def test_a_comfyui_that_keeps_its_port_after_the_stop_fails_the_install_and_rolls_back(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        rig.enable_comfyui()
        # the first unregister (the install's stop) leaves ComfyUI running and holding its port
        result = install_app_rig(
            tmp_path,
            rig,
            {"FAKE_UNREGISTER_FAIL": "comfyui", "PORT_GATE_TIMEOUT": "3"},
        )
        assert result.returncode != 0
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "old"
        assert rig.loaded(OMLX_LABEL) and rig.loaded(COMFYUI_LABEL)
    finally:
        rig.close()
