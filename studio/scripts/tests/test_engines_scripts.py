# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the engine wrappers and scripts. Every run uses a scratch UNSLOTH_ENGINES_HOME and
log dir; nothing touches ~/.unsloth or any live service.

    studio/scripts/tests: python -m pytest test_engines_scripts.py -q
"""

from __future__ import annotations

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


# --- wrappers --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper,name,log",
    [("omlx-launch", "omlx", "omlx.log")],
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
    for name in ("omlx",):
        d = subs / name
        d.mkdir(parents=True)
        git(d, "init", "-q", "-b", "main")
        (d / "f").write_text("1")
        git(d, "add", ".")
        git(d, "commit", "-q", "-m", "one")
    repo = tmp_path / "repo"
    (repo / "studio" / "engines").mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    for name in ("omlx",):
        git(repo, "submodule", "add", "-q", str(subs / name), f"studio/engines/{name}")
    git(repo, "commit", "-q", "-m", "pin")
    return repo


def pins(repo, home):
    return build_fn(home, f'REPO="{repo}"; ENGINES_SRC="{repo}/studio/engines"; check_submodule_pins; echo pins-ok')


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
            '  [ "${STAGE_VERSION:-none}" = none ] && exit 0\n'
            '  [ "$STAGE_VERSION" = fail ] && exit 1\n'
            '  mkdir -p "$UNSLOTH_ENGINES_HOME/omlx.new/bin"\n'
            f'  cp "{home}/omlx/bin/python" "$UNSLOTH_ENGINES_HOME/omlx.new/bin/python"\n'
            '  printf %s "$STAGE_VERSION" >"$UNSLOTH_ENGINES_HOME/omlx.new/VERSION"\n'
            '  printf "omlx\\nnew-key\\nomlx_kernels=0\\n" >"$UNSLOTH_ENGINES_HOME/omlx.new/.staged"\n'
            "  exit 0\n"
            "fi\n"
            f'exec "{SCRIPTS}/build-engines-mac.sh" "$@"\n'
        )
        self.build.chmod(0o755)
        self.env = {
            "UNSLOTH_ENGINES_HOME": str(home),
            "OMLX_URL": f"http://127.0.0.1:{self.omlx_port}",
            "ENGINE_PORTS": f"{self.omlx_port}",
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
            "FAKE_PY": sys.executable,
            "STAGE_VERSION": "none",
        }
        # the pre-with_app behaviour by default: the helpers outlive the app. The with_app tests
        # call set_desktop.
        self.set_desktop("always")
        # the helpers start registered, as after the Settings toggle
        assert run([str(self.app / "Contents/MacOS/unsloth-studio"), "--engine-helpers", "register"], self.env).returncode == 0
        self.wait_up()
        (self.lc / "calls.log").unlink(missing_ok=True)

    def launchctl(self, *args):
        return run([str(FAKES / "fake-launchctl"), *args], self.env)

    def set_desktop(self, lifetime, enabled=True, path=None):
        """desktop.json as the app writes it: engine_lifetime and engines_enabled (None omits it)."""
        body = {"engine_lifetime": lifetime}
        if enabled is not None:
            body["engines_enabled"] = enabled
        (path or self.home / "desktop.json").write_text(json.dumps(body))

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

        for url in (f"http://127.0.0.1:{self.omlx_port}/api/status",):
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
        import signal

        for pid_file in self.lc.glob("*.pid"):
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except (OSError, ValueError):
                pass


@pytest.fixture
def rig(tmp_path, home):
    r = Rig(tmp_path, home)
    yield r
    r.close()


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
        "engine-helpers unregister",
        "engine-helpers register",
    ]
    positions = [calls.index(item) for item in order]
    assert positions == sorted(positions)
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register"]
    assert_no_launchd_start(rig)
    assert all(rig.loaded(label) for label in rig.labels)
    assert not (rig.home / "omlx.new").exists()


def assert_no_launchd_start(rig):
    """macOS refuses launchctl bootstrap of an SMAppService helper: nothing may depend on it."""
    assert not any(c.startswith(("launchctl bootstrap", "launchctl kickstart")) for c in rig.calls())
    assert "gui-started" not in rig.calls()


def helpers_touched(rig):
    return [c for c in rig.calls() if c.startswith("launchctl bootout") or c in ("engine-helpers unregister", "engine-helpers register", "engine-helpers restart")]


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
    assert "did not restart" in out and "Attached engines > Engines enabled" in out
    # register, then one restart; never launchctl bootstrap or kickstart
    assert [c for c in rig.cli_calls() if c != "engine-helpers unregister"][:2] == ["engine-helpers register", "engine-helpers restart"]
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
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register"]
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
    assert "no --engine-helpers" in started.stderr and "Attached engines > Engines enabled" in started.stderr
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
    assert rig.cli_calls() == ["engine-helpers register", "engine-helpers restart"]
    assert "trying restart" in started.stderr
    assert_no_launchd_start(rig)
    assert all(rig.loaded(label) for label in rig.labels)


def test_a_register_that_does_nothing_and_a_restart_that_fails_leave_a_clear_warning(rig):
    started = lib_fn(rig, "helpers_stop; helpers_start", {"FAKE_REGISTER_NOOP": "1", "FAKE_RESTART_FAIL": "1", "HELPER_START_WAIT": "1"})
    assert started.returncode != 0
    assert "did not restart" in started.stderr and "Attached engines > Engines enabled" in started.stderr
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
    good = builder.replace(
        'z.writestr("unsloth/x.py", "")',
        engines + '; z.writestr("studio/_fork.py", ""); z.writestr("studio/fork-pins.toml", "")',
    )
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
        start = calls.index("engine-helpers register")
        assert len(moves) == 2 and stop < min(moves) and max(moves) < start
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register"]
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
        assert rig.cli_calls() == ["engine-helpers register", "engine-helpers restart"]
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
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register"]
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
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register"]
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
    assert rig.cli_calls() == ["engine-helpers unregister"]
    assert not any(rig.loaded(label) for label in rig.labels)
    assert "leaving the helpers stopped" in result.stdout and "next launch" in result.stdout
    assert_no_launchd_start(rig)


def test_update_in_with_app_mode_registers_the_helpers_while_the_app_runs(rig):
    rig.set_desktop("with_app")
    rig.set_app_running(True)
    result = rig.update(STAGE_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register"]
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
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register"]
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


def test_generated_config_has_only_omlx():
    import tomllib
    config = tomllib.loads((ENGINES / "engines.default.toml").read_text())
    assert set(config) == {"omlx"}
    assert "OMLX_PEER_EVICT_URLS" not in config["omlx"]["env"]


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
    '[omlx]\nport = 9000\n',
    '[omlx.env]\nOMLX_PEER_EVICT_URLS = "http://127.0.0.1:9999"\n',
    '[other]\nOMLX_PEER_EVICT_URLS = "http://127.0.0.1:8001"\n',
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
    assert {p.name for p in staging.iterdir()} == {"omlx-launch", "engines-common.sh", "engines_launch.py"}


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



def test_a_failed_unregister_still_restarts_the_helper_that_was_stopped(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        # The helper stopped before unregister reported failure; rollback must restart it.
        result = install_app_rig(tmp_path, rig, {"FAKE_UNREGISTER_FAIL_AFTER": "omlx"})
        assert result.returncode != 0, result.stdout + result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "old"
        assert not any(c.startswith("mv ") for c in rig.calls())
        assert rig.cli_calls()[0] == "engine-helpers unregister"
        assert "engine-helpers register" in rig.cli_calls()
        assert_no_launchd_start(rig)
        assert all(rig.loaded(label) for label in rig.labels)
    finally:
        rig.close()



def test_a_port_gate_timeout_restarts_the_stopped_helpers(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        with socket.socket() as held:
            held.bind(("127.0.0.1", 0))
            held.listen()
            port = held.getsockname()[1]
            result = install_app_rig(
                tmp_path, rig, {"ENGINE_PORTS": f"{rig.omlx_port} {port}", "PORT_GATE_TIMEOUT": "2"}
            )
        assert result.returncode != 0 and "ports still held" in result.stderr, result.stdout + result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "old"
        assert rig.cli_calls() == ["engine-helpers unregister", "engine-helpers register"]
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
