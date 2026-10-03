# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.attached_engines_settings as settings  # noqa: E402
import utils.scan_denylist as denylist  # noqa: E402
from hub.storage import scan_folders as hub_scan_folders  # noqa: E402
from storage import studio_db  # noqa: E402


def _config(*entries: str, enabled: bool = False) -> settings.AttachedEnginesConfig:
    return settings.AttachedEnginesConfig(
        enabled = enabled,
        omlx_url = settings.DEFAULT_OMLX_URL,
        ds4_url = settings.DEFAULT_DS4_URL,
        scan_denylist = tuple(entries),
        arbitrate_local_loads = True,
        prewarm_ds4_on_select = True,
    )


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """models/ with a denied dwarfstar gguf folder and an ordinary one, both holding fake .gguf."""
    models = tmp_path / "models"
    denied = models / "dwarfstar-gguf"
    allowed = models / "llama-gguf"
    for folder in (denied, allowed):
        folder.mkdir(parents = True)
        (folder / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(settings, "get_config", lambda: _config(str(denied)))
    return models, denied, allowed


def test_is_denied_matches_entry_and_descendants_only(tree, tmp_path):
    models, denied, allowed = tree
    assert denylist.is_denied(denied)
    assert denylist.is_denied(str(denied / "model-Q4_K_M.gguf"))
    assert denylist.is_denied(denied / "nested" / "deeper")
    assert not denylist.is_denied(allowed)
    assert not denylist.is_denied(models)
    sibling = models / "dwarfstar-gguf-extra"
    sibling.mkdir()
    assert not denylist.is_denied(sibling)


def test_active_even_with_flag_off_and_off_by_nothing_when_empty(tree, monkeypatch):
    _, denied, _ = tree
    monkeypatch.setattr(settings, "get_config", lambda: _config(str(denied), enabled = False))
    assert denylist.is_denied(denied)
    monkeypatch.setattr(settings, "get_config", lambda: _config())
    assert not denylist.is_denied(denied)


def test_missing_entry_denies_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "get_config", lambda: _config(str(tmp_path / "nope")))
    assert not denylist.is_denied(tmp_path / "nope" / "x")
    assert not denylist.is_denied(tmp_path)


def test_symlink_into_denied_folder_is_denied(tree, tmp_path):
    models, denied, _ = tree
    link = models / "innocent-name"
    link.symlink_to(denied, target_is_directory = True)
    assert denylist.is_denied(link)


def test_tilde_entries_expand(tmp_path, monkeypatch):
    home = tmp_path / "home"
    target = home / "Homelab" / "dwarfstar" / "gguf"
    target.mkdir(parents = True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(settings, "get_config", lambda: _config("~/Homelab/dwarfstar/gguf"))
    assert denylist.is_denied(target / "mmproj.gguf")
    assert denylist.is_denied("~/Homelab/dwarfstar/gguf")


def test_garbage_input_never_raises(tree):
    assert denylist.is_denied("\0bad") is False


def test_scan_models_dir_skips_denied_children(tree):
    from routes.models import _scan_models_dir

    models, denied, allowed = tree
    names = {Path(row.path).name for row in _scan_models_dir(models)}
    assert names == {"llama-gguf"}


def test_scan_models_dir_returns_nothing_for_a_denied_root(tree):
    from routes.models import _scan_models_dir
    _, denied, _ = tree
    assert _scan_models_dir(denied) == []


def test_scan_models_dir_unaffected_without_a_denylist(tree, monkeypatch):
    from routes.models import _scan_models_dir

    models, _, _ = tree
    monkeypatch.setattr(settings, "get_config", lambda: _config())
    names = {Path(row.path).name for row in _scan_models_dir(models)}
    assert names == {"llama-gguf", "dwarfstar-gguf"}


@pytest.mark.parametrize("module", [studio_db, hub_scan_folders])
def test_registering_a_denied_folder_is_refused(tree, module):
    _, denied, _ = tree
    with pytest.raises(ValueError, match = "deny-list"):
        module.add_scan_folder_with_status(str(denied))
    with pytest.raises(ValueError, match = "deny-list"):
        module.add_scan_folder_with_status(str(denied / "."))
    assert [row for row in module.list_scan_folders() if "dwarfstar" in row["path"]] == []


@pytest.mark.parametrize("module", [studio_db, hub_scan_folders])
def test_an_allowed_folder_still_registers(tree, module, monkeypatch):
    # The system-folder prefixes include /private/var, where macOS keeps tmp_path.
    monkeypatch.setattr(module, "_denied_path_prefixes", lambda: [])
    _, _, allowed = tree
    row, inserted = module.add_scan_folder_with_status(str(allowed))
    assert inserted is True
    assert Path(row["path"]) == allowed.resolve()


@pytest.mark.parametrize("module", [studio_db, hub_scan_folders])
def test_already_registered_denied_folder_is_hidden_from_the_list(tree, module, monkeypatch):
    monkeypatch.setattr(module, "_denied_path_prefixes", lambda: [])
    models, denied, allowed = tree
    monkeypatch.setattr(settings, "get_config", lambda: _config())
    module.add_scan_folder_with_status(str(denied))
    module.add_scan_folder_with_status(str(allowed))
    monkeypatch.setattr(settings, "get_config", lambda: _config(str(denied)))
    paths = [Path(row["path"]) for row in module.list_scan_folders()]
    assert paths == [allowed.resolve()]
    monkeypatch.setattr(settings, "get_config", lambda: _config())
    assert len(module.list_scan_folders()) == 2


def test_companion_state_skips_denied_files(tree):
    from core.inference.local_model_resolver import local_gguf_companion_state

    models, denied, allowed = tree
    state = local_gguf_companion_state((str(models),))
    paths = [Path(entry[0]) for entry in state]
    assert paths == [allowed / "model-Q4_K_M.gguf"]
