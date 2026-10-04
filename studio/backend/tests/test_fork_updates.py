# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The beaglemoo fork must never check PyPI for a newer Unsloth, whatever `direct_url.json` says."""

from __future__ import annotations

import sys

import pytest

from utils import update_status


@pytest.fixture(autouse = True)
def _fresh_cache():
    update_status.reset_update_status_cache()
    yield
    update_status.reset_update_status_cache()


def _no_network(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("the fork build contacted PyPI")

    monkeypatch.setattr(update_status.urllib.request, "urlopen", _boom)
    monkeypatch.setattr(update_status, "_fetch_latest_pypi_version", _boom)


def test_the_marker_module_ships_in_the_fork():
    monkey = sys.modules.pop("studio._fork", None)
    try:
        assert update_status.is_fork_build() is True
    finally:
        if monkey is not None:
            sys.modules["studio._fork"] = monkey


def test_is_fork_build_is_false_without_the_marker(monkeypatch):
    monkeypatch.setitem(sys.modules, "studio._fork", None)
    assert update_status.is_fork_build() is False


@pytest.mark.parametrize("source", ["pypi", "local_path", "unknown", "editable"])
def test_fork_never_reports_an_update(monkeypatch, source):
    # "pypi" is the dangerous one: a wheel installed without direct_url.json looks like a PyPI install.
    _no_network(monkeypatch)
    monkeypatch.delenv(update_status.DISABLE_ENV_VAR, raising = False)
    monkeypatch.delenv(update_status.FAKE_UPDATE_ENV_VAR, raising = False)
    monkeypatch.setattr(update_status, "is_fork_build", lambda: True)
    monkeypatch.setattr(update_status, "detect_install_source", lambda: source)
    status = update_status.get_studio_update_status("2026.9.14")
    assert status["update_available"] is False
    assert status["can_show_web_notification"] is False
    assert status["latest_version"] is None
    assert status["reason"] == "fork_build"


def test_fork_install_source_status_names_the_fork(monkeypatch):
    monkeypatch.setattr(update_status, "is_fork_build", lambda: True)
    monkeypatch.setattr(update_status, "detect_install_source", lambda: "pypi")
    assert update_status.get_studio_install_source_status("1")["reason"] == "fork_build"


def test_fork_latest_pypi_version_makes_no_request(monkeypatch):
    _no_network(monkeypatch)
    monkeypatch.setattr(update_status, "is_fork_build", lambda: True)
    result = update_status.get_latest_pypi_version()
    assert result.latest_version is None
    assert result.reason == "fork_build"


def test_upstream_behaviour_is_unchanged_without_the_marker(monkeypatch):
    monkeypatch.setattr(update_status, "is_fork_build", lambda: False)
    monkeypatch.setattr(update_status, "detect_install_source", lambda: "pypi")
    monkeypatch.delenv(update_status.DISABLE_ENV_VAR, raising = False)
    monkeypatch.delenv(update_status.FAKE_UPDATE_ENV_VAR, raising = False)
    monkeypatch.setattr(
        update_status,
        "_fetch_latest_pypi_version",
        lambda: update_status.LatestVersionResult(latest_version = "2099.1.1", checked_at = "now"),
    )
    status = update_status.get_studio_update_status("2026.9.14")
    assert status["update_available"] is True
    assert status["latest_version"] == "2099.1.1"
