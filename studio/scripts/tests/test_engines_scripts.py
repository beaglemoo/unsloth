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

    def test_ds4_model_missing_and_ports(self, home):
        fake_python(home / "ds4-ondemand" / "bin" / "python")
        (home / "bundle" / "ds4").mkdir(parents=True)
        (home / "bundle" / "ds4" / "ds4_ondemand.py").write_text("")
        cfg = f'[ds4]\nmodel = "{home}/nope.gguf"\nlauncher_port = {free_port()}\nserver_port = {free_port()}\n'
        result = launch(home, "ds4", cfg)
        assert result.returncode == 78
        assert "model file missing" in read_fail(home, "ds4")["reason"]
        (home / "nope.gguf").write_text("x")
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            result = launch(home, "ds4", cfg.replace(f"server_port = {cfg.split('server_port = ')[1].split()[0]}", f"server_port = {busy.getsockname()[1]}"))
        assert result.returncode == 78
        assert "ds4-server" in read_fail(home, "ds4")["reason"]
        assert read_fail(home, "ds4")["count"] == 2
        result = launch(home, "ds4", cfg)
        assert result.returncode == 0, result.stderr
        assert not (home / "ds4.fail").exists()

    def test_print_mode_leaves_no_marker_and_ignores_a_busy_port(self, home):
        fake_python(home / "omlx" / "bin" / "python")
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            result = launch(home, "omlx", f"[omlx]\nport = {busy.getsockname()[1]}\n", "--print")
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["argv"][1:4] == ["-m", "omlx.cli", "serve"]
        assert not (home / "omlx.fail").exists()
        result = run([sys.executable, str(ENGINES / "engines_launch.py"), "ds4", "--print"], env_for(home))
        assert result.returncode == 78
        assert not (home / "ds4.fail").exists()

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
    [("omlx-launch", "omlx", "omlx.log"), ("ds4-ondemand-launch", "ds4", "ds4-launcher.log")],
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


def test_shipped_defaults():
    import tomllib

    cfg = tomllib.loads((ENGINES / "engines.default.toml").read_text())
    assert cfg["ds4"]["host"] == "127.0.0.1"
    assert cfg["omlx"]["env"]["OMLX_PEER_EVICT_URLS"] == "http://127.0.0.1:8001"
    assert 'section.get("host", "127.0.0.1")' in (ENGINES / "engines_launch.py").read_text()


def test_build_engines_stages_the_common_helper():
    text = (SCRIPTS / "build-engines-mac.sh").read_text()
    assert "engines-common.sh" in text


def test_bash_backoff_schedule_matches_python(home):
    waits = []
    for _ in range(8):
        result = run(
            ["bash", "-c", 'sleep() { echo "SLEEP $1"; }; . "$1"; engines_fail ds4 "boom"', "_", str(ENGINES / "engines-common.sh")],
            {"ENGINES_HOME": str(home)},
        )
        assert result.returncode == 78
        waits.append(int(result.stdout.split("SLEEP ")[1].split()[0]))
    assert waits == [10, 20, 40, 80, 160, 300, 300, 300]
    assert read_fail(home, "ds4")["count"] == 8


# --- migrate-engines-mac.sh rollback -----------------------------------------------------------


def migrate(home_dir, snippet, **env):
    """Source the migrate script (its main is guarded) under a scratch HOME and run a snippet."""
    return run(
        ["bash", "-c", f'. "{SCRIPTS}/migrate-engines-mac.sh"; {snippet}'],
        {"HOME": str(home_dir), "APPS_DIR": str(home_dir / "Apps"), **env},
    )


def make_app(path: Path, marker="x"):
    (path / "Contents").mkdir(parents=True)
    (path / "Contents" / "marker").write_text(marker)


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


def test_swap_replaces_the_venv_keeps_old_and_records_the_marker_after(home):
    make_venv(home / "omlx", "old")
    (home / ".provisioned").write_text("omlx=old-key\nomlx_kernels=1\nds4_build=abc\n")
    make_venv(home / "omlx.new", "new", ["omlx", "new-key", "omlx_kernels=0"])
    result = build_fn(home, 'venv_in_use() { return 1; }; swap_venv "$ENGINES_HOME/omlx"')
    assert result.returncode == 0, result.stderr
    assert (home / "omlx" / "VERSION").read_text() == "new"
    assert (home / "omlx.old" / "VERSION").read_text() == "old"
    assert not (home / "omlx.new").exists() and not (home / "omlx" / ".staged").exists()
    assert marker(home) == {"omlx": "new-key", "omlx_kernels": "0", "ds4_build": "abc"}
    # the entry point and pyvenv.cfg now name the live path, not omlx.new
    assert (home / "omlx" / "bin" / "omlx").read_text().startswith(f"#!{home}/omlx/bin/python")
    assert f"{home}/omlx.new" not in (home / "omlx" / "pyvenv.cfg").read_text()


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


def test_staging_never_touches_the_live_venv_or_the_marker(home):
    make_venv(home / "ds4-ondemand", "old")
    (home / ".provisioned").write_text("ds4_ondemand=old\n")
    snippet = (
        'HAVE_UV=0; DS4_COMMIT=newcommit; FORCE=0; ENGINES_SRC=/nonexistent; '
        'make_venv() { mkdir -p "$1/bin"; printf "#!/bin/sh\\n" >"$1/bin/python"; chmod +x "$1/bin/python"; }; '
        'pip_install() { :; }; validate_ds4_ondemand() { :; }; '
        'stage_ds4_ondemand; stage_ds4_ondemand'
    )
    result = build_fn(home, snippet)
    assert result.returncode == 0, result.stderr
    assert (home / "ds4-ondemand" / "VERSION").read_text() == "old"
    assert (home / "ds4-ondemand.new" / ".staged").read_text().splitlines() == ["ds4_ondemand", "newcommit"]
    assert marker(home) == {"ds4_ondemand": "old"}
    assert result.stdout.count("staging newcommit") == 1 and "already staged" in result.stdout


def test_a_failed_validation_leaves_no_staged_record(home):
    snippet = (
        'HAVE_UV=0; DS4_COMMIT=c; FORCE=0; ENGINES_SRC=/nonexistent; '
        'make_venv() { mkdir -p "$1/bin"; }; pip_install() { :; }; validate_ds4_ondemand() { return 1; }; '
        'stage_ds4_ondemand'
    )
    result = build_fn(home, snippet)
    assert result.returncode != 0
    assert not (home / "ds4-ondemand.new" / ".staged").exists()
    swap = build_fn(home, 'swap_venv "$ENGINES_HOME/ds4-ondemand"')
    assert swap.returncode != 0  # the half-built venv is refused, not swapped


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd, check=True, capture_output=True,
    )


@pytest.fixture
def repo_with_submodules(tmp_path):
    subs = tmp_path / "subs"
    for name in ("omlx", "ds4"):
        d = subs / name
        d.mkdir(parents=True)
        git(d, "init", "-q", "-b", "main")
        (d / "f").write_text("1")
        git(d, "add", ".")
        git(d, "commit", "-q", "-m", "one")
    repo = tmp_path / "repo"
    (repo / "studio" / "engines").mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    for name in ("omlx", "ds4"):
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


def test_pin_check_fails_on_a_dirty_submodule(repo_with_submodules, home):
    (repo_with_submodules / "studio" / "engines" / "ds4" / "f").write_text("dirty")
    result = pins(repo_with_submodules, home)
    assert result.returncode != 0 and "local changes" in result.stderr


def test_pin_check_fails_on_an_uninitialised_submodule(repo_with_submodules, home):
    git(repo_with_submodules, "submodule", "deinit", "-f", "studio/engines/ds4")
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
        self.plists = tmp_path / "plists"
        self.plists.mkdir()
        self.omlx_port, self.ds4_port = free_port(), free_port()
        self.labels = ("ai.unsloth.studio.omlx", "ai.unsloth.studio.ds4")
        make_venv(home / "omlx", "old")
        make_venv(home / "ds4-ondemand", "old")
        (home / ".provisioned").write_text("omlx=old-key\nomlx_kernels=1\nds4_ondemand=old-ds4\n")
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
            "DS4_URL": f"http://127.0.0.1:{self.ds4_port}",
            "ENGINE_PORTS": f"{self.omlx_port} {self.ds4_port}",
            "LAUNCHCTL": str(FAKES / "fake-launchctl"),
            "HELPER_PLIST_DIR": str(self.plists),
            "BUILD_ENGINES": str(self.build),
            "HEALTH_TIMEOUT": "6",
            "HEALTH_POLL": "0.2",
            "PORT_GATE_TIMEOUT": "10",
            "FAKE_LC": str(self.lc),
            "FAKE_HOME": str(home),
            "FAKE_OMLX_PORT": str(self.omlx_port),
            "FAKE_DS4_PORT": str(self.ds4_port),
            "FAKE_PY": sys.executable,
            "STAGE_VERSION": "none",
        }
        for label in self.labels:
            self.launchctl("bootstrap", "gui/501", str(self.plists / f"{label}.plist"))
        self.wait_up()

    def launchctl(self, *args):
        return run([str(FAKES / "fake-launchctl"), *args], self.env)

    def wait_up(self):
        import time
        import urllib.request

        for url in (f"http://127.0.0.1:{self.omlx_port}/api/status", f"http://127.0.0.1:{self.ds4_port}/admin/status"):
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
    (rig.home / "ds4-loaded").write_text("1")
    result = rig.update(STAGE_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (rig.home / "omlx" / "VERSION").read_text() == "new"
    assert (rig.home / "omlx.old" / "VERSION").read_text() == "old"
    assert marker(rig.home)["omlx"] == "new-key"
    calls = rig.calls()
    # staged first, while the helpers still serve; then idle unload, stop, bootout, bootstrap
    order = [
        "fake stage: new",
        "omlx-unload",
        "ds4-stop /admin/stop?if_idle=1",
        "launchctl bootout gui/501/ai.unsloth.studio.omlx",
        "launchctl bootout gui/501/ai.unsloth.studio.ds4",
    ]
    positions = [calls.index(item) for item in order]
    assert positions == sorted(positions)
    after_stop = calls[calls.index("launchctl bootout gui/501/ai.unsloth.studio.ds4"):]
    assert sum(c.startswith("launchctl bootstrap") for c in after_stop) == 2
    assert all(rig.loaded(label) for label in rig.labels)
    assert not (rig.home / "omlx.new").exists()


def test_update_with_nothing_staged_does_not_touch_the_helpers(rig):
    result = rig.update(STAGE_VERSION="none")
    assert result.returncode == 0 and "nothing to do" in result.stdout
    assert not any(c.startswith("launchctl bootout") for c in rig.calls())


def test_update_refuses_while_an_engine_is_busy_and_stages_nothing(rig):
    (rig.home / "ds4-busy").write_text("1")
    result = rig.update(STAGE_VERSION="new")
    assert result.returncode != 0 and "in flight" in result.stderr
    assert not any(c.startswith("fake stage") or c.startswith("launchctl bootout") for c in rig.calls())
    assert (rig.home / "omlx" / "VERSION").read_text() == "old"


def test_update_rolls_back_when_the_new_venv_is_unhealthy(rig):
    result = rig.update(STAGE_VERSION="bad")
    assert result.returncode != 0
    assert "rolling back" in result.stdout and "rolled back" in result.stdout, result.stdout + result.stderr
    assert (rig.home / "omlx" / "VERSION").read_text() == "old"
    assert (rig.home / "omlx.failed" / "VERSION").read_text() == "bad"
    assert marker(rig.home) == {"omlx": "old-key", "omlx_kernels": "1", "ds4_ondemand": "old-ds4"}
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
    assert not any(c.startswith("launchctl bootout") for c in rig.calls())
    assert (rig.home / "omlx" / "VERSION").read_text() == "old"


def test_update_reports_when_a_helper_cannot_be_restarted(rig):
    result = rig.update(STAGE_VERSION="new", FAKE_BOOTSTRAP_FAIL="1")
    assert result.returncode != 0
    assert "did not restart" in result.stderr + result.stdout
    assert any(c.startswith("launchctl kickstart") for c in rig.calls())


def test_update_dry_run_changes_nothing(rig):
    result = rig.update("--dry-run", STAGE_VERSION="new")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[dry-run]" in result.stdout and "bootout" in result.stdout and "--swap-only" in result.stdout
    assert (rig.home / "omlx" / "VERSION").read_text() == "old"
    assert not (rig.home / "omlx.new").exists() and not (rig.home / "omlx.old").exists()
    assert not any(c.startswith("launchctl bootout") for c in rig.calls())
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
    good = builder.replace('z.writestr("unsloth/x.py", "")', 'z.writestr("studio/backend/routes/attached_engines.py", "")')
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
        'install_backend() { echo new >"$STUDIO_VENV/lib/mod.py"; }; check_installed_version() { :; }; verify_backend() { :; }; APP_BACKUP=/nonexistent-ok; ditto() { :; }',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "studio" / "unsloth_studio" / "lib" / "mod.py").read_text().strip() == "new"
    assert "app-installed" in result.stdout and "pruned" in result.stdout
    snapshots = list((tmp_path / "backups").glob("*/unsloth_studio/lib/mod.py"))
    assert len(snapshots) == 1 and snapshots[0].read_text() == "original"


def test_the_app_swap_happens_with_the_helpers_out_of_launchd(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        make_app(tmp_path / "dist" / "Unsloth.app", "new")
        make_app(tmp_path / "Apps" / "Unsloth.app", "old")
        log = rig.lc / "calls.log"
        snippet = (
            "codesign() { :; }; DRY=0; "
            f'mv() {{ echo "mv $1" >>"{log}"; command mv "$@"; }}; '
            "install_app"
        )
        env = {k: v for k, v in rig.env.items() if k not in ("UNSLOTH_ENGINES_HOME",)}
        result = fork_fn(tmp_path, snippet, {**env, "DIST_DIR": str(tmp_path / "dist")})
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "new"
        assert (tmp_path / "prev" / "Unsloth-prev.app.bak" / "Contents" / "marker").read_text() == "old"
        assert not (tmp_path / "Apps" / "Unsloth.app.new").exists()
        calls = rig.calls()
        stops = [i for i, c in enumerate(calls) if c.startswith("launchctl bootout")]
        moves = [i for i, c in enumerate(calls) if c.startswith("mv ")]
        starts = [i for i, c in enumerate(calls) if c.startswith("launchctl bootstrap")][-2:]
        assert len(stops) == 2 and len(moves) == 2 and len(starts) == 2
        assert max(stops) < min(moves) and max(moves) < min(starts)
        assert all(rig.loaded(label) for label in rig.labels)
    finally:
        rig.close()


def test_a_busy_engine_blocks_the_app_swap_and_leaves_the_app_alone(tmp_path, home):
    rig = Rig(tmp_path, home)
    try:
        make_app(tmp_path / "dist" / "Unsloth.app", "new")
        make_app(tmp_path / "Apps" / "Unsloth.app", "old")
        (home / "ds4-busy").write_text("1")
        env = {k: v for k, v in rig.env.items() if k not in ("UNSLOTH_ENGINES_HOME",)}
        result = fork_fn(tmp_path, "codesign() { :; }; DRY=0; install_app", env)
        assert result.returncode != 0 and "in flight" in result.stderr
        assert (tmp_path / "Apps" / "Unsloth.app" / "Contents" / "marker").read_text() == "old"
        assert not any(c.startswith("launchctl bootout") for c in rig.calls())
    finally:
        rig.close()


def test_an_up_to_date_venv_discards_a_stale_staged_one(home):
    make_venv(home / "ds4-ondemand", "live")
    (home / ".provisioned").write_text("ds4_ondemand=pinned\n")
    make_venv(home / "ds4-ondemand.new", "stale", ["ds4_ondemand", "reverted-bump"])
    snippet = (
        'HAVE_UV=0; DS4_COMMIT=pinned; FORCE=0; ENGINES_SRC=/nonexistent; '
        'make_venv() { echo unexpected-build >&2; return 1; }; stage_ds4_ondemand; '
        'swap_venv "$ENGINES_HOME/ds4-ondemand"'
    )
    result = build_fn(home, snippet)
    assert result.returncode == 0, result.stderr
    assert "discarding stale staged venv" in result.stdout
    assert not (home / "ds4-ondemand.new").exists() and not (home / "ds4-ondemand.old").exists()
    assert (home / "ds4-ondemand" / "VERSION").read_text() == "live"
    assert marker(home) == {"ds4_ondemand": "pinned"}


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

# What the first release generated: ds4 on the LAN, no peer eviction.
OLD_CONFIG = """\
# Unsloth attached engines (oMLX and DwarfStar/ds4).
# my own note: keep this file tidy

[omlx]
base_path = "~/.omlx-tuned"
port = 8843
# host = "127.0.0.1"   # unset: oMLX uses the host from <base_path>/settings.json

[omlx.env]
OMLX_QWEN35_SPARSE_BOUNDARIES = "1"
OMLX_NAX = "1"
OMLX_SUPERVISED = "launchd"

[ds4]
model = "~/Homelab/dwarfstar/ds4flash.gguf"
# Launcher (OpenAI-compatible, on demand) and the internal ds4-server port.
host = "0.0.0.0"
launcher_port = 8001
ctx = 65536
"""


def write_config(home, text):
    cfg = home / "engines.toml"
    cfg.write_text(text)
    return cfg


def backups(home):
    return sorted(p.name for p in home.glob("engines.toml.bak-*"))


def migrate_cfg(home):
    return build_fn(home, "migrate_config")


def parsed(cfg):
    import tomllib

    return tomllib.loads(cfg.read_text())


def test_an_old_generated_config_is_migrated_with_a_backup(home):
    cfg = write_config(home, OLD_CONFIG)
    cfg.chmod(0o640)
    result = migrate_cfg(home)
    assert result.returncode == 0, result.stdout + result.stderr
    data = parsed(cfg)
    assert data["ds4"]["host"] == "127.0.0.1"
    assert data["omlx"]["env"]["OMLX_PEER_EVICT_URLS"] == "http://127.0.0.1:8001"
    # the backup is the untouched original, written before the edit
    (backup,) = backups(home)
    assert (home / backup).read_text() == OLD_CONFIG
    # every other key, comment and the file mode survive
    assert data["ds4"]["ctx"] == 65536 and data["omlx"]["env"]["OMLX_NAX"] == "1"
    new = cfg.read_text()
    assert "# my own note: keep this file tidy" in new and "# Launcher (OpenAI-compatible" in new
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o640
    removed = [line for line in OLD_CONFIG.splitlines() if line not in new.splitlines()]
    assert removed == ['host = "0.0.0.0"']
    # the peer URL sits inside [omlx.env], before [ds4]
    assert new.index("OMLX_PEER_EVICT_URLS") < new.index("[ds4]")
    # every change is logged
    assert 'host "0.0.0.0" -> "127.0.0.1"' in result.stdout and "OMLX_PEER_EVICT_URLS" in result.stdout
    assert str(home / backup) in result.stdout


@pytest.mark.parametrize(
    "marker",
    ["# keep-lan", "# KEEP-LAN: the iPad reads this", "lan = true"],
)
def test_a_deliberate_lan_config_is_left_on_the_lan(home, marker):
    text = OLD_CONFIG.replace('host = "0.0.0.0"', f'host = "0.0.0.0"\n{marker}' if marker.startswith("lan") else f'{marker}\nhost = "0.0.0.0"')
    cfg = write_config(home, text)
    result = migrate_cfg(home)
    assert result.returncode == 0, result.stdout + result.stderr
    data = parsed(cfg)
    assert data["ds4"]["host"] == "0.0.0.0"
    assert "opt-out marker" in result.stdout
    # the opt-out only covers the host; the missing peer URL is still added
    assert data["omlx"]["env"]["OMLX_PEER_EVICT_URLS"] == "http://127.0.0.1:8001"
    assert 'host "0.0.0.0" -> ' not in result.stdout
    (backup,) = backups(home)
    assert (home / backup).read_text() == text


def test_a_custom_host_is_never_touched(home):
    text = OLD_CONFIG.replace('host = "0.0.0.0"', 'host = "192.168.3.78"')
    cfg = write_config(home, text)
    assert migrate_cfg(home).returncode == 0
    assert parsed(cfg)["ds4"]["host"] == "192.168.3.78"


def test_a_second_run_changes_nothing(home):
    cfg = write_config(home, OLD_CONFIG)
    assert migrate_cfg(home).returncode == 0
    first, first_backups = cfg.read_bytes(), backups(home)
    assert len(first_backups) == 1
    result = migrate_cfg(home)
    assert result.returncode == 0, result.stderr
    assert cfg.read_bytes() == first and backups(home) == first_backups
    assert "up to date" in result.stdout


def test_a_current_config_gets_no_backup(home):
    cfg = write_config(home, (ENGINES / "engines.default.toml").read_text())
    before = cfg.read_bytes()
    result = migrate_cfg(home)
    assert result.returncode == 0 and "up to date" in result.stdout
    assert cfg.read_bytes() == before and backups(home) == []


def test_a_config_without_omlx_env_gets_the_section_even_without_a_final_newline(home):
    text = '[ds4]\nhost = "0.0.0.0"  # lan\nctx = 1'
    cfg = write_config(home, text)
    assert migrate_cfg(home).returncode == 0
    data = parsed(cfg)
    assert data["ds4"]["host"] == "127.0.0.1" and data["ds4"]["ctx"] == 1
    assert data["omlx"]["env"] == {"OMLX_PEER_EVICT_URLS": "http://127.0.0.1:8001"}
    assert 'host = "127.0.0.1"  # lan' in cfg.read_text()


def test_an_existing_peer_url_is_kept(home):
    text = OLD_CONFIG.replace('OMLX_SUPERVISED = "launchd"', 'OMLX_SUPERVISED = "launchd"\nOMLX_PEER_EVICT_URLS = "http://127.0.0.1:9001"')
    cfg = write_config(home, text)
    assert migrate_cfg(home).returncode == 0
    assert parsed(cfg)["omlx"]["env"]["OMLX_PEER_EVICT_URLS"] == "http://127.0.0.1:9001"
    assert cfg.read_text().count("OMLX_PEER_EVICT_URLS") == 1


def test_an_invalid_config_is_left_alone(home):
    cfg = write_config(home, "[ds4\nhost = oops\n")
    result = migrate_cfg(home)
    assert result.returncode == 0 and "not valid TOML" in result.stdout
    assert cfg.read_text() == "[ds4\nhost = oops\n" and backups(home) == []


def test_a_missing_config_is_not_created_by_the_migration(home):
    result = migrate_cfg(home)
    assert result.returncode == 0 and "nothing to migrate" in result.stdout
    assert not (home / "engines.toml").exists()


def test_the_build_script_migrates_through_its_flag(home):
    cfg = write_config(home, OLD_CONFIG)
    result = run(["bash", str(SCRIPTS / "build-engines-mac.sh"), "--migrate-config"], {"UNSLOTH_ENGINES_HOME": str(home)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert parsed(cfg)["ds4"]["host"] == "127.0.0.1" and len(backups(home)) == 1


def test_update_migrates_the_config_and_dry_run_does_not(rig):
    cfg = write_config(rig.home, OLD_CONFIG)
    dry = rig.update("--dry-run")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert cfg.read_text() == OLD_CONFIG and backups(rig.home) == []
    result = rig.update()
    assert result.returncode == 0, result.stdout + result.stderr
    assert parsed(cfg)["ds4"]["host"] == "127.0.0.1" and len(backups(rig.home)) == 1
    assert "migrated" in result.stdout
