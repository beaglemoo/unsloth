"""The fork install pins unsloth-zoo to studio/fork-pins.toml and never tracks upstream main."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips

PINS = STUDIO_DIR / "fork-pins.toml"
COMMIT = "aeb831ae1da152bfee9bd580df2e70c858fe2a71"
GOOD = f'''# comment
[unsloth-zoo]
version = "2026.10.9"   # trailing comment
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
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2026.10.9")
    monkeypatch.setattr(ips, "_recorded_direct_url", lambda name: {"vcs_info": {"commit_id": COMMIT}})
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
    with pytest.raises(SystemExit, match = "pins 2026.10.9"):
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
        "[unsloth-zoo]\nversion = \"2026.10.9\"\n",
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


def test_the_right_version_from_the_wrong_commit_fails_the_install(monkeypatch, pins_file):
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", "1")
    monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
    monkeypatch.setattr(ips, "pip_install", lambda *a, **k: None)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2026.10.9")
    monkeypatch.setattr(ips, "_recorded_direct_url", lambda name: {"vcs_info": {"commit_id": "0" * 40}})
    with pytest.raises(SystemExit, match = "pins 2026.10.9 @"):
        ips._overlay_local_core_package("unsloth-zoo", "/src/unsloth")


@pytest.mark.parametrize("repo", ["", "/src/unsloth"])
def test_the_core_index_guard_only_refuses_a_fork_install_without_a_checkout(monkeypatch, repo):
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", "1")
    if repo:
        ips._refuse_fork_index_install(repo)
        return
    with pytest.raises(SystemExit, match = "release index"):
        ips._refuse_fork_index_install(repo)


@pytest.mark.parametrize("env", ["", "0", "false"])
def test_the_core_index_guard_is_off_outside_fork_mode(monkeypatch, env):
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", env)
    ips._refuse_fork_index_install("")


@pytest.mark.parametrize("no_torch", [True, False])
@pytest.mark.parametrize("package", ["unsloth", "other"])
def test_a_fork_install_with_no_checkout_never_reaches_a_pypi_install(monkeypatch, no_torch, package):
    """Upstream's `_desktop_unsloth_spec` branches (no-torch, final else, --package) install unsloth
    and a bare unsloth-zoo from the index; in fork mode the installer exits before any of them."""
    import contextlib

    calls = []
    monkeypatch.setenv("STUDIO_LOCAL_NONEDITABLE", "1")
    monkeypatch.delenv("STUDIO_LOCAL_REPO", raising = False)
    monkeypatch.delenv("SKIP_STUDIO_BASE", raising = False)
    monkeypatch.setenv("STUDIO_PACKAGE_NAME", package)
    monkeypatch.setattr(ips, "NO_TORCH", no_torch)
    monkeypatch.setattr(
        ips.install_manifest, "pass_lock", lambda *a, **k: contextlib.nullcontext(True)
    )
    for name in ("pip_install", "pip_install_try", "_desktop_unsloth_spec"):
        monkeypatch.setattr(ips, name, lambda *a, _n = name, **k: calls.append(_n))
    monkeypatch.setattr(ips.subprocess, "run", lambda *a, **k: calls.append("subprocess"))
    with pytest.raises(SystemExit, match = "release index"):
        ips.install_python_stack()
    assert calls == []


# The fork is detected from the shipped marker, not from STUDIO_LOCAL_NONEDITABLE (a direct run of the
# installer and `setup.sh --local` set nothing).
@pytest.fixture
def bare_fork_env(monkeypatch, pins_file):
    for var in ("STUDIO_LOCAL_NONEDITABLE", "STUDIO_LOCAL_REPO", "UNSLOTH_ZOO_REF"):
        # setenv first so the teardown restores a variable the code under test assigns directly.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)


@pytest.mark.fork
def test_the_marker_alone_makes_the_install_non_editable_and_pinned(bare_fork_env):
    assert ips._is_fork_install()
    assert ips._local_overlay_non_editable()
    assert ips._unsloth_zoo_git_spec().endswith(f"@{COMMIT}")


def test_without_the_marker_nothing_is_forced(bare_fork_env):
    assert not ips._is_fork_install()
    assert not ips._local_overlay_non_editable()
    assert ips._resolve_fork_local_repo("") == ""
    assert ips._unsloth_zoo_git_spec() == "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"


@pytest.mark.fork
def test_a_direct_run_defaults_to_this_checkout_and_sets_the_fork_env(bare_fork_env):
    assert ips._resolve_fork_local_repo("") == str(STUDIO_DIR.parent)
    assert os.environ["STUDIO_LOCAL_NONEDITABLE"] == "1"
    assert os.environ["STUDIO_LOCAL_REPO"] == str(STUDIO_DIR.parent)


@pytest.mark.fork
def test_an_explicit_checkout_is_kept(bare_fork_env):
    assert ips._resolve_fork_local_repo("/src/unsloth") == "/src/unsloth"


@pytest.mark.fork
def test_a_fork_install_without_any_checkout_refuses_an_index_install(bare_fork_env, monkeypatch):
    monkeypatch.setattr(ips, "__file__", "/nonexistent/studio/install_python_stack.py")
    monkeypatch.setattr(ips, "_FORK_DEFAULT_REPO", "/nonexistent/fork")
    monkeypatch.setattr(ips, "_is_fork_install", lambda: True)
    with pytest.raises(SystemExit, match = "release index"):
        ips._resolve_fork_local_repo("")


def test_the_installer_entry_resolves_the_fork_checkout_before_any_install_step():
    """install_python_stack() defaults the checkout (and forces non-editable) before it reads the
    environment any further, so a run with no env vars cannot reach the index branches."""
    body = ips.install_python_stack.__wrapped__ if hasattr(ips.install_python_stack, "__wrapped__") else ips.install_python_stack
    import inspect

    src = inspect.getsource(body)
    resolve = src.index('_resolve_fork_local_repo(os.environ.get("STUDIO_LOCAL_REPO", ""))')
    assert resolve < src.index("_refuse_fork_index_install(local_repo)")
    assert resolve < src.index("_core_index_install_allowed(local_repo)")
