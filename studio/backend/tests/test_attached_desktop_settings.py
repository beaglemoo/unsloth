# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached.desktop_settings import helper_wanted


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_ENGINES_HOME", str(tmp_path))
    return tmp_path


def _write(home, body):
    (home / "desktop.json").write_text(body if isinstance(body, str) else json.dumps(body))


def test_missing_file_is_unknown(home):
    assert helper_wanted("omlx") is None and helper_wanted("comfyui") is None


@pytest.mark.parametrize("text", ["", "not json", "[]", "3", '{"engine_lifetime": "always"}', '{"engines_enabled": "yes"}'])
def test_unreadable_or_masterless_file_is_unknown(home, text):
    _write(home, text)
    assert helper_wanted("omlx") is None


def test_master_off_wants_nothing(home):
    _write(home, {"engines_enabled": False, "helpers": {"omlx": True, "comfyui": True}})
    assert helper_wanted("omlx") is False and helper_wanted("comfyui") is False


def test_defaults_without_helpers_map(home):
    _write(home, {"engines_enabled": True})
    assert helper_wanted("omlx") is True and helper_wanted("comfyui") is False


def test_helpers_map_and_per_field_fallback(home):
    _write(home, {"engines_enabled": True, "helpers": {"omlx": False, "comfyui": "yes"}})
    assert helper_wanted("omlx") is False and helper_wanted("comfyui") is False
    _write(home, {"engines_enabled": True, "helpers": {"comfyui": True}})
    assert helper_wanted("omlx") is True and helper_wanted("comfyui") is True
    _write(home, {"engines_enabled": True, "helpers": "garbage"})
    assert helper_wanted("omlx") is True


def test_unknown_helper_name(home):
    _write(home, {"engines_enabled": True})
    assert helper_wanted("other") is None
