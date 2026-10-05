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

import routes.settings as routes  # noqa: E402
import utils.attached_engines_settings as settings  # noqa: E402
from storage import studio_db  # noqa: E402


@pytest.fixture(autouse = True)
def stored_settings(monkeypatch):
    stored: dict = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: stored.get(key, fallback)
    )
    monkeypatch.setattr(studio_db, "upsert_app_settings", stored.update)
    for var in (
        settings.ENABLED_ENV_VAR,
        settings.OMLX_URL_ENV_VAR,
        settings.SCAN_DENYLIST_ENV_VAR,
    ):
        monkeypatch.delenv(var, raising = False)
    settings._invalidate()
    yield stored
    settings._invalidate()


def test_defaults_are_off_and_loopback():
    config = settings.get_config()
    assert config == settings.DEFAULT_CONFIG
    assert config.enabled is False
    assert config.omlx_url == "http://127.0.0.1:8843"
    assert config.scan_denylist == ()
    assert config.arbitrate_local_loads is True


def test_set_config_round_trips_and_keeps_other_fields(stored_settings):
    config = settings.set_config(enabled = True, omlx_url = "http://localhost:9000/v1/")
    assert config.enabled is True
    assert config.omlx_url == "http://localhost:9000"
    config = settings.set_config(scan_denylist = ["/a", " /b ", "/a", ""])
    assert config.scan_denylist == ("/a", "/b")
    assert config.enabled is True
    assert stored_settings[settings.ATTACHED_ENGINES_SETTING_KEY]["omlx_url"] == (
        "http://localhost:9000"
    )


def test_none_leaves_stored_value_untouched():
    settings.set_config(enabled = True)
    assert settings.set_config(enabled = None).enabled is True


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.2.5:8843",
        "http://example.com",
        "ftp://127.0.0.1:1",
        "http://127.0.0.1:8843/other",
        "http://user@127.0.0.1:8843",
        "http://127.0.0.1:notaport",
        "",
    ],
)
def test_non_loopback_or_malformed_url_rejected(url):
    with pytest.raises(ValueError):
        settings.set_config(omlx_url = url)


def test_ipv6_and_127_range_loopback_accepted():
    assert settings.normalize_loopback_url("http://[::1]:8001/v1") == "http://[::1]:8001"
    assert settings.normalize_loopback_url("http://127.0.0.2:1") == "http://127.0.0.2:1"


def test_set_config_rejects_unknown_key_and_non_bool():
    with pytest.raises(ValueError):
        settings.set_config(bogus = 1)
    with pytest.raises(ValueError):
        settings.set_config(enabled = "maybe")
    with pytest.raises(ValueError):
        settings.set_config(scan_denylist = [1])


def test_env_overrides_win_over_stored(monkeypatch):
    settings.set_config(enabled = False, omlx_url = "http://127.0.0.1:1111")
    monkeypatch.setenv(settings.ENABLED_ENV_VAR, "1")
    monkeypatch.setenv(settings.OMLX_URL_ENV_VAR, "http://127.0.0.1:2222/v1")
    monkeypatch.setenv(settings.SCAN_DENYLIST_ENV_VAR, os.pathsep.join(["/x", "/y"]))
    settings._invalidate()
    config = settings.get_config()
    assert config.enabled is True
    assert config.omlx_url == "http://127.0.0.1:2222"
    assert config.scan_denylist == ("/x", "/y")
    monkeypatch.setenv(settings.ENABLED_ENV_VAR, "0")
    settings._invalidate()
    assert settings.get_config().enabled is False


def test_invalid_env_values_are_ignored(monkeypatch):
    monkeypatch.setenv(settings.ENABLED_ENV_VAR, "banana")
    monkeypatch.setenv(settings.OMLX_URL_ENV_VAR, "http://10.0.0.5:8843")
    settings._invalidate()
    config = settings.get_config()
    assert config.enabled is False
    assert config.omlx_url == settings.DEFAULT_OMLX_URL


def test_corrupt_stored_value_falls_back_per_field(stored_settings):
    stored_settings[settings.ATTACHED_ENGINES_SETTING_KEY] = {
        "enabled": True,
        "omlx_url": "http://evil.example",
        "scan_denylist": "nope-not-a-list-but-a-string",
        "arbitrate_local_loads": "banana",
    }
    settings._invalidate()
    config = settings.get_config()
    assert config.enabled is True
    assert config.omlx_url == settings.DEFAULT_OMLX_URL
    assert config.arbitrate_local_loads is True
    stored_settings[settings.ATTACHED_ENGINES_SETTING_KEY] = "garbage"
    settings._invalidate()
    assert settings.get_config() == settings.DEFAULT_CONFIG


def test_reads_are_cached_until_write(stored_settings):
    assert settings.get_config().enabled is False
    stored_settings[settings.ATTACHED_ENGINES_SETTING_KEY] = {"enabled": True}
    assert settings.get_config().enabled is False  # cached
    settings.set_config(scan_denylist = [])  # write invalidates
    assert settings.get_config().enabled is True


def test_route_handlers_round_trip():
    response = routes.get_attached_engines_settings(current_subject = "owner")
    assert response.enabled is False
    response = routes.update_attached_engines_settings(
        routes.AttachedEnginesPayload(enabled = True, omlx_url = "http://127.0.0.1:9001/v1"),
        current_subject = "owner",
    )
    assert response.enabled is True
    assert response.omlx_url == "http://127.0.0.1:9001"
    assert routes.get_attached_engines_settings(current_subject = "owner").enabled is True


def test_route_rejects_non_loopback_with_400():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as err:
        routes.update_attached_engines_settings(
            routes.AttachedEnginesPayload(omlx_url = "http://192.168.1.2:8843"),
            current_subject = "owner",
        )
    assert err.value.status_code == 400
    assert "loopback" in str(err.value.detail)


def test_routes_registered_under_owner_router():
    paths = {
        (route.path, tuple(sorted(route.methods))) for route in routes._owner_settings_router.routes
    }
    assert ("/attached-engines", ("GET",)) in paths
    assert ("/attached-engines", ("PUT",)) in paths


def test_legacy_stored_settings_are_ignored_without_losing_omlx(stored_settings, monkeypatch):
    stored_settings[settings.ATTACHED_ENGINES_SETTING_KEY] = {
        "enabled": True,
        "omlx_url": "http://localhost:9000",
        "ds4_url": {"invalid": "ignored"},
        "prewarm_ds4_on_select": "obsolete",
    }
    monkeypatch.setenv("UNSLOTH_DS4_URL", "http://invalid.example")
    settings._invalidate()
    config = settings.get_config()
    assert config.enabled
    assert config.omlx_url == "http://localhost:9000"
    assert config.scan_denylist == ()
    settings.set_config(arbitrate_local_loads = False)
    assert settings.get_config().omlx_url == "http://localhost:9000"
