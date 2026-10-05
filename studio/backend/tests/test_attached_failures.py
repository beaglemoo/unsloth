# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.attached.failures import read_failure  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_ENGINES_HOME", str(tmp_path))
    return tmp_path


def test_reads_a_valid_marker(home):
    (home / "omlx.fail").write_text(json.dumps({"reason": "port 8843 already in use", "ts": 1760000000.5, "count": 3}))
    assert read_failure("omlx") == {"reason": "port 8843 already in use", "ts": 1760000000.5, "count": 3}


def test_ts_may_be_an_integer(home):
    (home / "legacy.fail").write_text('{"reason": "model missing", "ts": 1760000000, "count": 1}')
    assert read_failure("legacy") == {"reason": "model missing", "ts": 1760000000.0, "count": 1}


def test_missing_file_is_no_failure(home):
    assert read_failure("omlx") is None


@pytest.mark.parametrize(
    "body",
    [
        "",
        "not json",
        "[]",
        "null",
        '{"reason": "x", "ts": 1}',
        '{"reason": "", "ts": 1, "count": 1}',
        '{"reason": 5, "ts": 1, "count": 1}',
        '{"reason": "x", "ts": "now", "count": 1}',
        '{"reason": "x", "ts": true, "count": 1}',
        '{"reason": "x", "ts": 1, "count": 0}',
        '{"reason": "x", "ts": 1, "count": 1.5}',
        '{"reason": "x", "ts": 1, "count": "2"}',
    ],
)
def test_malformed_marker_is_no_failure(home, body):
    (home / "omlx.fail").write_text(body)
    assert read_failure("omlx") is None


def test_binary_garbage_is_no_failure(home):
    (home / "omlx.fail").write_bytes(b"\xff\xfe\x00")
    assert read_failure("omlx") is None


@pytest.mark.parametrize("name", ["", "../omlx", "omlx/x", "OMLX", "a b", ".hidden"])
def test_rejects_names_that_could_escape_the_home(home, name):
    (home.parent / "omlx.fail").write_text('{"reason": "x", "ts": 1, "count": 1}')
    assert read_failure(name) is None


def test_default_home_is_under_the_user_home(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_ENGINES_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    target = tmp_path / ".unsloth" / "engines"
    target.mkdir(parents=True)
    (target / "omlx.fail").write_text('{"reason": "venv missing", "ts": 2, "count": 2}')
    assert read_failure("omlx")["reason"] == "venv missing"
