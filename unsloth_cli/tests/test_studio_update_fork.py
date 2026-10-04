# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fork mode: a plain `unsloth studio update` must never reach PyPI or upstream.

The beaglemoo fork ships `studio/_fork.py`. With it present `update` (no --local) refuses and
points at update-fork.sh; `update --local` stays the one install path (build-fork-mac.sh and the
Tauri repair run it) and is always non-editable. Without the marker nothing changes.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.fork

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod

    return _studio_mod


class _NoopLauncherUpdate:
    def __enter__(self):
        return self

    def validate_launcher(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _neutered(monkeypatch):
    studio = _studio()
    seen = {"setup_calls": 0}
    monkeypatch.setattr(studio, "_ensure_studio_env_exported", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_WindowsLauncherUpdateTransaction", _NoopLauncherUpdate)
    monkeypatch.setattr(studio, "_refresh_desktop_shortcuts", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_fail_if_install_damaged", lambda *a, **k: None, raising = False)

    def _setup(*a, **k):
        seen["setup_calls"] += 1
        seen["STUDIO_LOCAL_REPO"] = os.environ.get("STUDIO_LOCAL_REPO")
        seen["STUDIO_LOCAL_INSTALL"] = os.environ.get("STUDIO_LOCAL_INSTALL")
        seen["STUDIO_LOCAL_NONEDITABLE"] = os.environ.get("STUDIO_LOCAL_NONEDITABLE")

    monkeypatch.setattr(studio, "_run_setup_script", _setup)
    monkeypatch.delenv("STUDIO_LOCAL_NONEDITABLE", raising = False)
    return studio, seen


def _marker(**overrides):
    marker = types.ModuleType("studio._fork")
    marker.UPDATE_SCRIPT_HINT = "~/Homelab/unsloth/unsloth/studio/scripts/update-fork.sh"
    marker.UPDATE_SCRIPT_REL = "studio/scripts/update-fork.sh"
    for key, value in overrides.items():
        setattr(marker, key, value)
    return marker


def _checkout(tmp_path):
    checkout = tmp_path / "unsloth"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'unsloth'\n")
    return checkout


def test_the_real_marker_module_exists_in_the_fork():
    # Deleting studio/_fork.py would silently turn every guard off.
    import importlib

    marker = importlib.import_module("studio._fork")
    assert marker.FORK_REPO_URL == "https://github.com/beaglemoo/unsloth"
    assert marker.UPDATE_SCRIPT_HINT.endswith(marker.UPDATE_SCRIPT_REL)


def test_plain_update_is_refused_and_never_runs_setup(monkeypatch):
    studio, seen = _neutered(monkeypatch)
    monkeypatch.setattr(studio, "_fork_marker", lambda: _marker())
    result = CliRunner().invoke(studio.studio_app, ["update"])
    assert result.exit_code == 2, result.output
    assert "This is the beaglemoo fork." in result.output
    assert "~/Homelab/unsloth/unsloth/studio/scripts/update-fork.sh" in result.output
    assert seen["setup_calls"] == 0


@pytest.mark.parametrize("extra", [["--package", "unsloth"], ["--no-verify"], ["-v"]])
def test_plain_update_variants_are_refused(monkeypatch, extra):
    studio, seen = _neutered(monkeypatch)
    monkeypatch.setattr(studio, "_fork_marker", lambda: _marker())
    result = CliRunner().invoke(studio.studio_app, ["update", *extra])
    assert result.exit_code == 2, result.output
    assert seen["setup_calls"] == 0


def test_local_update_still_runs_and_is_forced_non_editable(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path)
    studio, seen = _neutered(monkeypatch)
    monkeypatch.setattr(studio, "_fork_marker", lambda: _marker())
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(checkout))
    result = CliRunner().invoke(studio.studio_app, ["update", "--local"])
    assert result.exit_code == 0, result.output
    assert seen["STUDIO_LOCAL_REPO"] == str(checkout)
    assert seen["STUDIO_LOCAL_INSTALL"] == "1"
    assert seen["STUDIO_LOCAL_NONEDITABLE"] == "1"


def test_local_update_defaults_to_the_fork_checkout(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path)
    studio, seen = _neutered(monkeypatch)
    marker = _marker(
        UPDATE_SCRIPT_HINT = f"{checkout}/studio/scripts/update-fork.sh",
    )
    monkeypatch.setattr(studio, "_fork_marker", lambda: marker)
    monkeypatch.delenv("STUDIO_LOCAL_REPO", raising = False)
    result = CliRunner().invoke(studio.studio_app, ["update", "--local"])
    assert result.exit_code == 0, result.output
    assert seen["STUDIO_LOCAL_REPO"] == str(checkout.resolve())


def test_a_bad_local_repo_never_suggests_pypi(monkeypatch, tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    studio, seen = _neutered(monkeypatch)
    monkeypatch.setattr(studio, "_fork_marker", lambda: _marker())
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(site))
    result = CliRunner().invoke(studio.studio_app, ["update", "--local"])
    assert result.exit_code == 2, result.output
    assert "PyPI" not in result.output
    assert "This is the beaglemoo fork." in result.output
    assert seen["setup_calls"] == 0


def test_without_the_marker_upstream_behaviour_is_unchanged(monkeypatch):
    studio, seen = _neutered(monkeypatch)
    monkeypatch.setattr(studio, "_fork_marker", lambda: None)
    monkeypatch.setenv("STUDIO_LOCAL_REPO", "/nonexistent")
    result = CliRunner().invoke(studio.studio_app, ["update"])
    assert result.exit_code == 0, result.output
    assert seen["setup_calls"] == 1
    assert seen["STUDIO_LOCAL_INSTALL"] == "0"
    assert seen["STUDIO_LOCAL_NONEDITABLE"] is None


def test_fork_marker_reads_the_module(monkeypatch):
    studio = _studio()
    monkeypatch.delitem(sys.modules, "studio._fork", raising = False)
    assert studio._fork_marker() is not None
    monkeypatch.setitem(sys.modules, "studio._fork", None)
    assert studio._fork_marker() is None


def test_the_hidden_setup_command_is_refused_without_local(monkeypatch):
    studio, seen = _neutered(monkeypatch)
    monkeypatch.setattr(studio, "_fork_marker", lambda: _marker())
    monkeypatch.delenv("STUDIO_LOCAL_INSTALL", raising = False)
    result = CliRunner().invoke(studio.studio_app, ["setup"])
    assert result.exit_code == 2, result.output
    assert "This is the beaglemoo fork." in result.output
    assert seen["setup_calls"] == 0
