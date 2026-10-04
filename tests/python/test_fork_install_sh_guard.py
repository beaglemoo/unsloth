"""install.sh is disabled in the beaglemoo fork: it would install PyPI unsloth over the fork."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

INSTALL_SH = Path(__file__).resolve().parents[2] / "install.sh"


def _run(*args, env_extra = None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_FORK")}
    env.update(env_extra or {})
    return subprocess.run(
        ["sh", str(INSTALL_SH), *args],
        capture_output = True,
        text = True,
        env = env,
        timeout = 30,
        stdin = subprocess.DEVNULL,
    )


@pytest.mark.parametrize("args", [[], ["--local"], ["--tauri"], ["--no-torch", "--local"]])
def test_install_sh_refuses_without_touching_anything(args, tmp_path):
    result = _run(*args, env_extra = {"HOME": str(tmp_path), "UNSLOTH_STUDIO_HOME": str(tmp_path / "s")})
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "disabled in the beaglemoo fork" in combined
    assert "studio/scripts/update-fork.sh" in combined
    assert list(tmp_path.iterdir()) == []


def test_the_guard_runs_before_any_flag_is_parsed_or_anything_is_downloaded():
    text = INSTALL_SH.read_text(encoding = "utf-8")
    guard = text.index("UNSLOTH_FORK_ALLOW_INSTALL_SH")
    assert guard < text.index("# ── Parse flags ──")
    assert guard < text.index("uv pip install")


SETUP_SH = INSTALL_SH.parent / "studio" / "setup.sh"


def test_setup_sh_without_local_is_refused(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UNSLOTH_FORK", "STUDIO_LOCAL"))}
    env["HOME"] = str(tmp_path)
    result = subprocess.run(
        ["bash", str(SETUP_SH)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 30,
        stdin = subprocess.DEVNULL,
    )
    assert result.returncode == 1
    assert "disabled in the beaglemoo fork" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_setup_sh_guard_lets_the_update_local_flow_through():
    text = SETUP_SH.read_text(encoding = "utf-8")
    guard = text.index('[ "${STUDIO_LOCAL_INSTALL:-0}" != "1" ] && [ "${UNSLOTH_FORK_ALLOW_INSTALL_SH:-}" != "1" ]')
    # `unsloth studio update --local` exports STUDIO_LOCAL_INSTALL=1; the guard sits after --local is parsed
    assert text.index("export STUDIO_LOCAL_INSTALL=1") < guard
    assert guard < text.index("_DEFAULT_LLAMA_PR_FORCE=")
