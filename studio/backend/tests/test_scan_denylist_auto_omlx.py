# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""While attached engines are enabled, the oMLX engine's own model folders join the scan deny-list."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.attached_engine_dirs as engine_dirs  # noqa: E402
import utils.attached_engines_settings as settings  # noqa: E402
from utils import scan_denylist  # noqa: E402


def _config(*entries: str, enabled: bool = True) -> settings.AttachedEnginesConfig:
    return settings.AttachedEnginesConfig(
        enabled = enabled,
        omlx_url = settings.DEFAULT_OMLX_URL,
        scan_denylist = tuple(entries),
        arbitrate_local_loads = True,
    )


def _models(root: Path, *names: str) -> Path:
    for name in names:
        (root / name).mkdir(parents = True, exist_ok = True)
        (root / name / "config.json").write_text("{}")
    return root


def _write_settings(base: Path, **model) -> None:
    base.mkdir(parents = True, exist_ok = True)
    (base / "settings.json").write_text(json.dumps({"model": model}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A scratch HOME, engines home and engine base path; attached engines enabled."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("UNSLOTH_ENGINES_HOME", str(tmp_path / "engines"))
    monkeypatch.delenv("UNSLOTH_ENGINES_CONFIG", raising = False)
    monkeypatch.delenv("OMLX_MODEL_DIR", raising = False)
    monkeypatch.delenv("OMLX_BASE_PATH", raising = False)
    monkeypatch.setattr(engine_dirs, "_cache", None)
    state = {"config": _config()}
    monkeypatch.setattr(settings, "get_config", lambda: state["config"])

    class Env:
        pass

    e = Env()
    e.tmp, e.home, e.state = tmp_path, home, state
    e.base = tmp_path / "engine-base"
    e.toml = tmp_path / "engines" / "engines.toml"
    e.toml.parent.mkdir()

    def write_toml(body: str = "") -> None:
        e.toml.write_text(f'[omlx]\nbase_path = "{e.base}"\n{body}')

    e.write_toml = write_toml
    write_toml()
    return e


def test_engine_dir_is_denied_when_enabled(env):
    served = _models(env.tmp / "served", "Swift", "Draft")
    _write_settings(env.base, model_dirs = [str(served)], model_dir = str(served))
    assert scan_denylist.is_denied(served)
    assert scan_denylist.is_denied(served / "Swift")
    assert scan_denylist.is_denied(served / "Swift" / "config.json")
    assert not scan_denylist.is_denied(env.tmp / "other")


def test_nothing_is_added_when_disabled(env):
    served = _models(env.tmp / "served", "Swift")
    _write_settings(env.base, model_dirs = [str(served)])
    env.state["config"] = _config(enabled = False)
    assert not scan_denylist.is_denied(served / "Swift")
    env.state["config"] = _config(enabled = True)
    assert scan_denylist.is_denied(served / "Swift")


def test_legacy_model_dir_is_used_when_model_dirs_is_empty(env):
    served = _models(env.tmp / "served", "Swift")
    _write_settings(env.base, model_dirs = [], model_dir = str(served))
    assert scan_denylist.is_denied(served / "Swift")


def test_default_models_folder_under_the_base_path_is_the_fallback(env):
    _write_settings(env.base)
    served = _models(env.base / "models", "Swift")
    assert scan_denylist.is_denied(served / "Swift")


@pytest.mark.parametrize("settings_text", [None, "", "{not json", "[]", '{"model": 7}', '{"model": {"model_dirs": 5}}'])
def test_missing_or_invalid_settings_never_raise(env, settings_text):
    other = _models(env.tmp / "other", "Model")
    if settings_text is not None:
        env.base.mkdir(parents = True)
        (env.base / "settings.json").write_text(settings_text)
    assert engine_dirs.auto_denied_dirs() == (str(env.base.resolve() / "models"),)
    assert not scan_denylist.is_denied(other / "Model")


def test_missing_engines_toml_and_invalid_toml_never_raise(env):
    env.toml.unlink()
    assert isinstance(engine_dirs.auto_denied_dirs(), tuple)
    env.toml.write_text("[omlx\nbroken")
    assert isinstance(engine_dirs.auto_denied_dirs(), tuple)
    assert scan_denylist.is_denied(env.tmp / "anything") is False


def test_nonexistent_engine_dir_denies_nothing(env):
    _write_settings(env.base, model_dirs = [str(env.tmp / "gone")])
    assert not scan_denylist.is_denied(env.tmp)
    assert not scan_denylist.is_denied(env.tmp / "gone" / "x")


def test_omlx_app_dir_equal_to_the_engine_dir_is_hidden_and_a_different_one_kept(env):
    served = _models(env.tmp / "served", "Swift")
    mine = _models(env.tmp / "app-models", "Mine")
    _write_settings(env.base, model_dirs = [str(served)])
    app_base = env.home / ".omlx"
    _write_settings(app_base, model_dirs = [str(served)])
    assert scan_denylist.is_denied(served / "Swift")

    from utils.paths.storage_roots import omlx_model_dirs

    assert [Path(p).resolve() for p in omlx_model_dirs()] == [served.resolve()]

    # The app's folder differs from the engine's: those rows stay visible.
    _write_settings(app_base, model_dirs = [str(mine)])
    assert [Path(p).resolve() for p in omlx_model_dirs()] == [mine.resolve()]
    assert not scan_denylist.is_denied(mine / "Mine")
    assert scan_denylist.is_denied(served / "Swift")


def test_symlinked_and_real_spellings_are_equivalent(env):
    served = _models(env.tmp / "served", "Swift")
    link = env.tmp / "link-to-served"
    link.symlink_to(served, target_is_directory = True)
    _write_settings(env.base, model_dirs = [str(link)])
    assert engine_dirs.auto_denied_dirs() == (str(served.resolve()),)
    assert scan_denylist.is_denied(served / "Swift")
    assert scan_denylist.is_denied(link / "Swift")
    # A symlink elsewhere that leads into the served folder is denied too.
    alias = env.tmp / "alias"
    alias.symlink_to(served / "Swift", target_is_directory = True)
    assert scan_denylist.is_denied(alias)


def test_tilde_is_expanded_in_settings_and_toml(env):
    served = _models(env.home / "served", "Swift")
    _write_settings(env.base, model_dirs = ["~/served"])
    assert scan_denylist.is_denied(served / "Swift")


def test_user_entries_are_merged_on_top(env):
    served = _models(env.tmp / "served", "Swift")
    user = _models(env.tmp / "user", "Old")
    kept = _models(env.tmp / "kept", "Keep")
    _write_settings(env.base, model_dirs = [str(served)])
    env.state["config"] = _config(str(user))
    assert scan_denylist.is_denied(served / "Swift")
    assert scan_denylist.is_denied(user / "Old")
    assert not scan_denylist.is_denied(kept / "Keep")
    # Disabled: only the user's entries apply.
    env.state["config"] = _config(str(user), enabled = False)
    assert not scan_denylist.is_denied(served / "Swift")
    assert scan_denylist.is_denied(user / "Old")


def test_engines_toml_model_paths_and_env_are_used(env):
    a = _models(env.tmp / "a", "A")
    b = _models(env.tmp / "b", "B")
    c = _models(env.tmp / "c", "C")
    d = _models(env.tmp / "d", "D")
    _write_settings(env.base, model_dirs = [str(env.tmp / "unused")])
    env.write_toml(
        f'model_dirs = ["{a}"]\nextra_args = ["--model-dir", "{b}"]\n'
        f'[omlx.env]\nOMLX_MODEL_DIR = "{c},{d}"\n'
    )
    dirs = set(engine_dirs.auto_denied_dirs())
    assert dirs == {str(p.resolve()) for p in (a, b, c, d)}
    assert scan_denylist.is_denied(c / "C") and scan_denylist.is_denied(a / "A")


def test_engines_config_env_override_wins(env, monkeypatch):
    served = _models(env.tmp / "served", "Swift")
    alt_base = env.tmp / "alt-base"
    _write_settings(alt_base, model_dirs = [str(served)])
    alt = env.tmp / "alt.toml"
    alt.write_text(f'[omlx]\nbase_path = "{alt_base}"\n')
    monkeypatch.setenv("UNSLOTH_ENGINES_CONFIG", str(alt))
    assert scan_denylist.is_denied(served / "Swift")


def test_cache_follows_file_changes(env):
    first = _models(env.tmp / "first", "One")
    second = _models(env.tmp / "second", "Two")
    _write_settings(env.base, model_dirs = [str(first)])
    assert scan_denylist.is_denied(first / "One")
    assert not scan_denylist.is_denied(second / "Two")
    (env.base / "settings.json").write_text(
        json.dumps({"model": {"model_dirs": [str(second)], "pad": "x" * 10}})
    )
    assert scan_denylist.is_denied(second / "Two")
    assert not scan_denylist.is_denied(first / "One")


def test_filter_denied_drops_the_engines_rows_and_keeps_the_rest(env):
    served = _models(env.tmp / "served", "Swift", "Draft")
    hf = _models(env.tmp / "hf", "models--acme--tiny")
    _write_settings(env.base, model_dirs = [str(served)])

    class Row:
        def __init__(self, path):
            self.path = path

    rows = [Row(str(served / "Swift")), Row(str(served / "Draft")), Row(str(hf / "models--acme--tiny")), Row(None)]
    kept = scan_denylist.filter_denied(rows)
    assert [r.path for r in kept] == [str(hf / "models--acme--tiny"), None]
