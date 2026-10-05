# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for studio/scripts/update-fork.sh. Every run uses a scratch git repo with a fake origin
(a bare repo in tmp) and fake build / install scripts that only log their arguments. The scratch
checkout also has an `upstream` remote that points at a path that does not exist: any command
that contacts it fails the run.

    studio/scripts/tests: python -m pytest test_update_fork.py -q
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "update-fork.sh"
BRANCH = "feat/omlx-ds4-engines"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash script")

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "protocol.file.allow",
    "GIT_CONFIG_VALUE_0": "always",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}


def git(cwd, *args, check=True):
    env = {**os.environ, **GIT_ENV}
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, env=env, check=check
    )


def commit(cwd, name, text="x"):
    (Path(cwd) / name).write_text(text)
    git(cwd, "add", name)
    git(cwd, "commit", "-q", "-m", f"add {name}")
    return git(cwd, "rev-parse", "HEAD").stdout.strip()


def fake_script(path: Path, log: Path, exit_code=0):
    path.write_text(f'#!/bin/sh\necho "{path.name} $*" >> "{log}"\nexit {exit_code}\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class Env:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.origin = tmp_path / "origin.git"
        self.work = tmp_path / "work"
        self.other = tmp_path / "other"
        self.log = tmp_path / "calls.log"
        self.build = fake_script(tmp_path / "build-fork-mac.sh", self.log)
        self.engines = fake_script(tmp_path / "update-engines-mac.sh", self.log)
        git(tmp_path, "init", "-q", "--bare", "-b", BRANCH, str(self.origin))
        git(tmp_path, "clone", "-q", str(self.origin), str(self.work))
        git(self.work, "checkout", "-q", "-b", BRANCH)
        commit(self.work, "base.txt")
        git(self.work, "push", "-q", "origin", BRANCH)
        git(self.work, "remote", "add", "upstream", str(tmp_path / "no-such-upstream.git"))
        git(tmp_path, "clone", "-q", str(self.origin), str(self.other))
        git(self.other, "checkout", "-q", BRANCH)

    def push_from_other(self, *names):
        for name in names:
            commit(self.other, name)
        git(self.other, "push", "-q", "origin", BRANCH)

    def run(self, *args, env_extra=None, **kw):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("FORK_", "UPDATE_ENGINES", "BUILD_FORK"))}
        env.update(GIT_ENV)
        env.update(
            {
                "FORK_REPO": str(self.work),
                "BUILD_FORK": str(self.build),
                "UPDATE_ENGINES": str(self.engines),
                "HOME": str(self.tmp),
            }
        )
        env.update(env_extra or {})
        kw.setdefault("stdin", subprocess.DEVNULL)
        return subprocess.run(
            ["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=60, **kw
        )

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def head(self, repo=None):
        return git(repo or self.work, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def e(tmp_path):
    return Env(tmp_path)


def out(r):
    return r.stdout + r.stderr


# ---- --check ------------------------------------------------------------------------------


def test_check_up_to_date_exits_zero(e):
    r = e.run("--check")
    assert r.returncode == 0, out(r)
    assert "up to date" in out(r)


def test_check_local_ahead_of_origin_is_up_to_date(e):
    commit(e.work, "local.txt")
    r = e.run("--check")
    assert r.returncode == 0, out(r)
    assert "1 commit(s) ahead" in out(r)


def test_check_reports_the_commit_count_and_exits_ten(e):
    e.push_from_other("a.txt", "b.txt")
    before = e.head()
    r = e.run("--check")
    assert r.returncode == 10, out(r)
    assert "2 update(s) available" in out(r)
    assert "add a.txt" in out(r) and "add b.txt" in out(r)
    assert e.head() == before, "--check must not move HEAD"
    assert e.calls() == []


def test_check_works_on_a_dirty_tree(e):
    e.push_from_other("a.txt")
    (e.work / "base.txt").write_text("edited")
    assert e.run("--check").returncode == 10


def test_check_a_diverged_branch_is_an_error(e):
    e.push_from_other("a.txt")
    commit(e.work, "local.txt")
    r = e.run("--check")
    assert r.returncode == 1
    assert "diverged" in out(r)


def test_check_without_the_remote_branch_is_an_error(e):
    r = e.run("--check", env_extra={"FORK_BRANCH": "no/such-branch"})
    assert r.returncode == 1
    assert "is on" in out(r) or "does not exist" in out(r)


def test_check_with_a_missing_remote_branch_names_it(e):
    git(e.work, "checkout", "-q", "-b", "feat/local-only")
    r = e.run("--check", env_extra={"FORK_BRANCH": "feat/local-only"})
    assert r.returncode == 1
    assert "does not exist" in out(r)


def test_check_in_a_non_repo_is_an_error(e, tmp_path):
    r = e.run("--check", env_extra={"FORK_REPO": str(tmp_path / "not-a-repo")})
    assert r.returncode == 1
    assert "not a git checkout" in out(r)


def test_check_with_an_unreachable_origin_is_an_error(e):
    git(e.work, "remote", "set-url", "origin", str(e.tmp / "gone.git"))
    r = e.run("--check")
    assert r.returncode == 1
    assert "fetch origin failed" in out(r)


# ---- preconditions ------------------------------------------------------------------------


def test_the_wrong_branch_is_refused(e):
    git(e.work, "checkout", "-q", "-b", "chore/other")
    r = e.run("--check")
    assert r.returncode == 1
    assert "not 'feat/omlx-ds4-engines'" in out(r)


def test_a_detached_head_is_refused(e):
    git(e.work, "checkout", "-q", "--detach")
    r = e.run("--check")
    assert r.returncode == 1
    assert "detached" in out(r)


def test_the_branch_can_be_overridden(e):
    git(e.work, "checkout", "-q", "-b", "feat/other")
    git(e.work, "push", "-q", "origin", "feat/other")
    r = e.run("--check", env_extra={"FORK_BRANCH": "feat/other"})
    assert r.returncode == 0, out(r)


def test_an_origin_that_is_upstream_unsloth_is_refused(e):
    git(e.work, "remote", "set-url", "origin", "https://github.com/unslothai/unsloth.git")
    r = e.run("--check")
    assert r.returncode == 1
    assert "upstream Unsloth" in out(r)


def test_a_dirty_tree_is_refused_for_an_update(e):
    e.push_from_other("a.txt")
    (e.work / "base.txt").write_text("edited")
    before = e.head()
    r = e.run("--yes")
    assert r.returncode == 1
    assert "uncommitted changes" in out(r)
    assert e.head() == before
    assert e.calls() == []


def test_untracked_files_do_not_block_an_update(e):
    e.push_from_other("a.txt")
    (e.work / "scratch.log").write_text("build output")
    assert e.run("--yes").returncode == 0


def test_a_diverged_branch_is_refused_for_an_update(e):
    e.push_from_other("a.txt")
    commit(e.work, "local.txt")
    before = e.head()
    r = e.run("--yes")
    assert r.returncode == 1
    assert "diverged" in out(r)
    assert e.head() == before
    assert e.calls() == []


# ---- the update ---------------------------------------------------------------------------


def test_the_update_fast_forwards_then_builds_installs_and_swaps_engines_in_order(e):
    e.push_from_other("a.txt", "b.txt")
    old = e.head()
    r = e.run("--yes")
    assert r.returncode == 0, out(r)
    assert e.head() == e.head(e.other)
    assert (e.work / "b.txt").exists()
    assert e.calls() == [
        "build-fork-mac.sh ",
        "build-fork-mac.sh --install",
        "update-engines-mac.sh --yes",
    ]
    text = out(r)
    assert "2 update(s) available" in text
    assert "add a.txt" in text
    assert f"git -C {e.work} reset --hard {old}" in text
    assert "summary" in text


def test_dry_run_changes_nothing_and_prints_the_steps(e):
    e.push_from_other("a.txt")
    before = e.head()
    r = e.run("--dry-run")
    assert r.returncode == 0, out(r)
    assert e.head() == before
    assert not (e.work / "a.txt").exists()
    assert e.calls() == []
    text = out(r)
    assert "[dry-run]" in text
    assert "merge --ff-only" in text
    assert "submodule update --init --recursive" in text
    assert "--install" in text
    assert "update-engines-mac.sh --yes" in text


def test_up_to_date_does_nothing_without_force(e):
    r = e.run("--yes")
    assert r.returncode == 0, out(r)
    assert "nothing to do" in out(r)
    assert e.calls() == []


def test_force_rebuilds_a_current_checkout(e):
    commit(e.work, "local.txt")
    r = e.run("--yes", "--force")
    assert r.returncode == 0, out(r)
    assert e.calls() == [
        "build-fork-mac.sh ",
        "build-fork-mac.sh --install",
        "update-engines-mac.sh --yes",
    ]


def test_without_a_terminal_the_install_needs_yes(e):
    e.push_from_other("a.txt")
    r = e.run()
    assert r.returncode == 1
    assert "needs --yes" in out(r)
    assert e.calls() == ["build-fork-mac.sh "], "the build ran, the install did not"


def test_a_failing_build_stops_before_the_install_and_names_the_rollback(e):
    e.push_from_other("a.txt")
    old = e.head()
    fake_script(e.build, e.log, exit_code=3)
    r = e.run("--yes")
    assert r.returncode == 3
    assert e.calls() == ["build-fork-mac.sh "]
    assert f"reset --hard {old}" in out(r)


def test_a_failing_install_does_not_swap_the_engines(e):
    e.push_from_other("a.txt")
    e.build.write_text(
        f'#!/bin/sh\necho "build-fork-mac.sh $*" >> "{e.log}"\n[ "${{1:-}}" = --install ] && exit 4\nexit 0\n'
    )
    r = e.run("--yes")
    assert r.returncode == 4
    assert e.calls() == ["build-fork-mac.sh ", "build-fork-mac.sh --install"]


def test_the_submodules_are_updated_to_the_pinned_commits(tmp_path):
    e = Env(tmp_path)
    sub = tmp_path / "sub.git"
    subwork = tmp_path / "subwork"
    git(tmp_path, "init", "-q", "-b", "main", str(subwork))
    pinned = commit(subwork, "engine.txt")
    git(tmp_path, "clone", "-q", "--bare", str(subwork), str(sub))
    # the other clone adds the submodule and pushes
    git(e.other, "submodule", "add", "-q", str(sub), "studio/engines/omlx")
    git(e.other, "commit", "-q", "-m", "add the omlx submodule")
    git(e.other, "push", "-q", "origin", BRANCH)
    # a newer submodule commit that the superproject does NOT pin
    commit(subwork, "later.txt")
    git(subwork, "push", "-q", str(sub), "HEAD:main")
    r = e.run("--yes")
    assert r.returncode == 0, out(r)
    checked_out = git(e.work / "studio/engines/omlx", "rev-parse", "HEAD").stdout.strip()
    assert checked_out == pinned
    assert (e.work / "studio/engines/omlx/engine.txt").exists()
    assert not (e.work / "studio/engines/omlx/later.txt").exists()


# ---- never the upstream remote -------------------------------------------------------------


def test_the_upstream_remote_is_never_contacted(e):
    # `upstream` points at a path that does not exist, so `git fetch --all` would fail the run.
    e.push_from_other("a.txt")
    assert e.run("--check").returncode == 10
    assert e.run("--dry-run").returncode == 0
    assert e.run("--yes").returncode == 0
    assert git(e.work, "for-each-ref", "refs/remotes/upstream").stdout.strip() == ""


def test_the_script_never_names_upstream_or_pypi_in_a_command():
    text = SCRIPT.read_text()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "fetch --all" not in code
    assert "pip install" not in code
    assert "pypi" not in code.lower()
    assert "git+https" not in code
    # the one mention of the upstream URL is the refusal that guards origin
    assert code.count("unslothai") == 1


# ---- usage ---------------------------------------------------------------------------------


def test_an_unknown_argument_is_a_usage_error(e):
    assert e.run("--bogus").returncode == 2


def test_check_and_dry_run_cannot_be_combined(e):
    assert e.run("--check", "--dry-run").returncode == 2


def test_help_prints_the_header(e):
    r = e.run("--help")
    assert r.returncode == 0
    assert "update-fork.sh [--check]" in r.stdout
