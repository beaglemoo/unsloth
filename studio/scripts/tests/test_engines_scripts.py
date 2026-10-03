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
