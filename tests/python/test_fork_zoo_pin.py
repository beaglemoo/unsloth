"""The fork install pins unsloth-zoo to studio/fork-pins.toml and never tracks upstream main."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips

PINS = STUDIO_DIR / "fork-pins.toml"
COMMIT = "867a86383371ebeb8ab1948d085540d134c07b4c"
GOOD = f'''# comment
[unsloth-zoo]
version = "2026.9.9"   # trailing comment
url = "https://github.com/unslothai/unsloth-zoo"
commit = "{COMMIT}"
'''


@pytest.fixture
def pins_file(tmp_path, monkeypatch):
    path = tmp_path / "fork-pins.toml"
    path.write_text(GOOD, encoding = "utf-8")
    monkeypatch.setattr(ips, "_fork_pins_path", lambda: path)
    return path


def test_the_committed_pin_file_parses_to_a_full_commit():
    pin = ips._parse_fork_pins(PINS.read_text(encoding = "utf-8"))["unsloth-zoo"]
    assert pin["url"] == "https://github.com/unslothai/unsloth-zoo"
    assert len(pin["commit"]) == 40 and set(pin["commit"]) <= set("0123456789abcdef")
    assert pin["version"][0] == "2"


def test_the_pin_file_is_found_next_to_the_installer():
    assert ips._fork_pins_path() == PINS


def test_the_committed_pin_file_survives_tomllib_when_available():
    tomllib = pytest.importorskip("tomllib")
    parsed = tomllib.loads(PINS.read_text(encoding = "utf-8"))
    assert parsed["unsloth-zoo"] == ips._parse_fork_pins(PINS.read_text(encoding = "utf-8"))["unsloth-zoo"]


def test_fork_mode_installs_the_pinned_commit(monkeypatch, pins_file):
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", "1")
    monkeypatch.delenv("UNSLOTH_ZOO_REF", raising = False)
    assert ips._unsloth_zoo_git_spec() == (
        f"unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo@{COMMIT}"
    )


def test_fork_mode_ignores_unsloth_zoo_ref(monkeypatch, pins_file):
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", "1")
    monkeypatch.setenv("UNSLOTH_ZOO_REF", "main")
    assert ips._unsloth_zoo_git_spec().endswith("@" + COMMIT)


def test_the_overlay_installs_the_pin_and_verifies_the_version(monkeypatch, pins_file):
    installs = []
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", "1")
    monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
    monkeypatch.setattr(ips, "pip_install", lambda label, *a, **k: installs.append((label, a, k)))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2026.9.9")
    assert ips._overlay_local_core_package("unsloth-zoo", "/src/unsloth") is True
    assert installs[0][1] == (
        "--no-cache-dir",
        "--no-deps",
        "--force-reinstall",
        f"unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo@{COMMIT}",
    )
    assert "main" not in installs[0][1][-1]


def test_a_zoo_version_that_differs_from_the_pin_fails_the_install(monkeypatch, pins_file):
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", "1")
    monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
    monkeypatch.setattr(ips, "pip_install", lambda *a, **k: None)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2099.1.1")
    with pytest.raises(SystemExit, match = "pins 2026.9.9"):
        ips._overlay_local_core_package("unsloth-zoo", "/src/unsloth")


def test_the_repair_overlay_reports_a_version_mismatch_instead_of_exiting(monkeypatch, pins_file):
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", "1")
    monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
    monkeypatch.setattr(ips, "_note", lambda *a, **k: None)
    monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: True)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2099.1.1")
    assert ips._overlay_local_core_package("unsloth-zoo", "/src/unsloth", strict = False) is False


def test_the_repair_source_spec_is_the_pin_too(monkeypatch, pins_file):
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", "1")
    assert ips._overlay_source_spec("unsloth_zoo", "/src/unsloth").endswith("@" + COMMIT)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[unsloth-zoo]\nversion = \"2026.9.9\"\n",
        GOOD.replace(COMMIT, "main"),
        GOOD.replace('url = "https://github.com/unslothai/unsloth-zoo"', 'url = "git://x"'),
    ],
)
def test_a_missing_or_malformed_pin_fails_closed(monkeypatch, tmp_path, text):
    path = tmp_path / "fork-pins.toml"
    path.write_text(text, encoding = "utf-8")
    monkeypatch.setattr(ips, "_fork_pins_path", lambda: path)
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", "1")
    with pytest.raises(SystemExit, match = "refusing to install unsloth-zoo from upstream main"):
        ips._unsloth_zoo_git_spec()


def test_a_missing_pin_file_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(ips, "_fork_pins_path", lambda: tmp_path / "nope.toml")
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", "1")
    with pytest.raises(SystemExit, match = "cannot read"):
        ips._unsloth_zoo_git_spec()


@pytest.mark.parametrize("env", ["", "0", "false"])
def test_without_fork_mode_the_overlay_is_unchanged(monkeypatch, env):
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", env)
    monkeypatch.delenv("UNSLOTH_ZOO_REF", raising = False)
    assert ips._unsloth_zoo_git_spec() == "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"
    monkeypatch.setenv("UNSLOTH_ZOO_REF", "v1")
    assert ips._unsloth_zoo_git_spec().endswith("@v1")
