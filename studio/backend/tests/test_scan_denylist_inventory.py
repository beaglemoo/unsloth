# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The deny-list holds at the combined results of every inventory and index, not only in one scanner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.attached_engines_settings as settings  # noqa: E402
from utils import scan_denylist  # noqa: E402


def _config(*entries: str) -> settings.AttachedEnginesConfig:
    return settings.AttachedEnginesConfig(
        enabled = False,
        omlx_url = settings.DEFAULT_OMLX_URL,
        scan_denylist = tuple(entries),
        arbitrate_local_loads = True,
    )


@pytest.fixture
def lmstudio(tmp_path, monkeypatch):
    """A registered scan folder in LM Studio layout: <folder>/acme/{denied-model,ok-model}/*.gguf."""
    folder = tmp_path / "sf"
    for name in ("denied-model", "ok-model"):
        model = folder / "acme" / name
        model.mkdir(parents = True)
        (model / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    rows = [{"id": 1, "path": str(folder), "created_at": "2026-01-01"}]
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: rows)
    monkeypatch.setattr(settings, "get_config", lambda: _config())
    return folder, rows, folder / "acme" / "denied-model"


def deny(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(settings, "get_config", lambda: _config(str(path)))


def _names(rows) -> set[str]:
    return {Path(row.path).name for row in rows}


def test_filter_denied_drops_rows_by_path_and_keeps_pathless_rows(lmstudio, monkeypatch):
    folder, _, denied = lmstudio

    class Row:
        def __init__(self, path):
            self.path = path

    rows = [Row(str(denied)), Row(str(denied / "x.gguf")), Row(str(folder)), Row(None)]
    assert len(scan_denylist.filter_denied(rows)) == 4
    deny(monkeypatch, denied)
    kept = scan_denylist.filter_denied(rows)
    assert [row.path for row in kept] == [str(folder), None]


def test_legacy_inventory_drops_a_denied_lmstudio_model(lmstudio, monkeypatch, tmp_path):
    from routes.models import collect_local_models

    folder, rows, denied = lmstudio
    assert _names(collect_local_models(tmp_path / "m", custom_folders = rows)) >= {
        "denied-model",
        "ok-model",
    }
    deny(monkeypatch, denied)
    names = _names(collect_local_models(tmp_path / "m", custom_folders = rows))
    assert "denied-model" not in names and "ok-model" in names


def test_hub_inventory_drops_a_denied_model(lmstudio, monkeypatch):
    from hub.services.models import local_inventory

    folder, _, denied = lmstudio
    scanned = local_inventory._scan_lmstudio_dir(folder)
    assert _names(local_inventory._filter_and_dedupe_local_models(scanned)) == {
        "denied-model",
        "ok-model",
    }
    deny(monkeypatch, denied)
    assert _names(local_inventory._filter_and_dedupe_local_models(scanned)) == {"ok-model"}


def test_auto_switch_index_drops_a_denied_model(lmstudio, monkeypatch):
    from core.inference import local_model_resolver

    _, _, denied = lmstudio
    index = local_model_resolver._build_index()
    assert "acme/denied-model" in index and "acme/ok-model" in index
    deny(monkeypatch, denied)
    index = local_model_resolver._build_index()
    assert "acme/ok-model" in index
    assert not [key for key in index if "denied-model" in key]


def test_recommended_folder_probe_ignores_denied_weights(tmp_path, monkeypatch):
    from routes.models import _dir_has_downloaded_model

    root = tmp_path / "recommended"
    (root / "dwarf").mkdir(parents = True)
    (root / "dwarf" / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    assert _dir_has_downloaded_model(root) is True
    deny(monkeypatch, root / "dwarf")
    assert _dir_has_downloaded_model(root) is False


def test_folder_browser_hides_a_denied_directory(lmstudio, monkeypatch):
    from routes.models import browse_folders
    from storage import studio_db

    # The system-folder prefixes include /private/var, where macOS keeps tmp_path.
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    folder, _, denied = lmstudio
    listing = browse_folders(path = str(denied.parent), show_hidden = False, current_subject = "owner")
    assert {entry.name for entry in listing.entries} == {"denied-model", "ok-model"}
    deny(monkeypatch, denied)
    listing = browse_folders(path = str(denied.parent), show_hidden = False, current_subject = "owner")
    assert {entry.name for entry in listing.entries} == {"ok-model"}
