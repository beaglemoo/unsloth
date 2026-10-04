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
