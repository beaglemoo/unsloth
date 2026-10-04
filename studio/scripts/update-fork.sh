#!/bin/bash
# Update the beaglemoo fork of Unsloth from ITS OWN repo (origin = github.com/beaglemoo/unsloth),
# rebuild, and install. This is the only supported update path: it never touches the `upstream`
# remote, PyPI or unslothai. Upstream is merged by hand (rebase procedure in CLAUDE.md).
#
#   update-fork.sh [--check] [--dry-run] [--yes] [--force]
#
#   --check    read-only: `git fetch origin`, then report whether updates exist and how many
#              commits. Exit 0 = up to date, 10 = updates available, anything else = an error
#              (1 = precondition or git failure, 2 = usage). Works on a dirty tree.
#   --dry-run  fetch, show the incoming commits and print every step that would change anything
#   --yes      do not ask before build-fork-mac.sh --install (it quits Unsloth and replaces the app)
#   --force    run the build and install even when the checkout is already up to date (for
#              example after committing locally)
#
# Steps:
#   1. preconditions: the checkout is on feat/omlx-ds4-engines (override with FORK_BRANCH)
#   2. git fetch origin; refuse if the working tree is dirty (tracked files) or the branch has
#      diverged from origin. Fast-forward only. A checkout that is ahead of origin is up to date.
#   3. show the incoming commit list
#   4. git merge --ff-only, then git submodule update --init --recursive (the pinned commits)
#   5. build-fork-mac.sh              (build the app; nothing live is touched)
#   6. build-fork-mac.sh --install    (quits Unsloth, installs the backend and the app)
#   7. update-engines-mac.sh --yes    (swaps the staged engine venvs)
#   8. summary, including the commit to `git reset --hard` to if you want to go back
#
# Run it in Terminal: the build needs cargo and npm. The app's Settings > About > Updates button
# opens Terminal on this script.
#
# Environment (all optional; the overrides exist for tests): FORK_REPO (default: this checkout),
#   FORK_BRANCH (default feat/omlx-ds4-engines), BUILD_FORK (default build-fork-mac.sh),
#   UPDATE_ENGINES (default update-engines-mac.sh)
set -euo pipefail

CHECK=0 DRY=0 YES=0 FORCE=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --dry-run) DRY=1 ;;
    --yes|-y) YES=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
[ "$CHECK" = 0 ] || [ "$DRY" = 0 ] || { echo "--check and --dry-run cannot be combined (--check is already read-only)" >&2; exit 2; }

# Terminal opened from the app may not have the login PATH; the build needs cargo and npm.
export PATH="/opt/homebrew/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FORK_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BRANCH="${FORK_BRANCH:-feat/omlx-ds4-engines}"
BUILD_FORK="${BUILD_FORK:-$SCRIPT_DIR/build-fork-mac.sh}"
UPDATE_ENGINES="${UPDATE_ENGINES:-$SCRIPT_DIR/update-engines-mac.sh}"
REMOTE=origin
MAX_LIST=40

log() { printf '[fork-update] %s\n' "$*"; }
warn() { printf '[fork-update] warning: %s\n' "$*" >&2; }
die() { printf '[fork-update] error: %s\n' "$*" >&2; exit 1; }
run() {
  if [ "$DRY" = 1 ]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}
# Every git call goes to the fork checkout and names the remote explicitly; there is no
# `git fetch --all`, so the `upstream` remote is never contacted.
g() { git -C "$REPO" "$@"; }

# ---------------------------------------------------------------------------------------------
# 1. preconditions
g rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$REPO is not a git checkout"
current_branch="$(g symbolic-ref --quiet --short HEAD || true)"
[ -n "$current_branch" ] || die "HEAD is detached; check out $BRANCH first"
[ "$current_branch" = "$BRANCH" ] \
  || die "the checkout is on '$current_branch', not '$BRANCH' (set FORK_BRANCH to update another branch)"
origin_url="$(g remote get-url "$REMOTE" 2>/dev/null)" || die "the checkout has no '$REMOTE' remote"
case "$origin_url" in
  *unslothai/unsloth*) die "'$REMOTE' points at upstream Unsloth ($origin_url); the fork updates only from beaglemoo/unsloth" ;;
esac

# ---------------------------------------------------------------------------------------------
# 2. fetch origin only, then classify
log "fetching $REMOTE ($origin_url)"
g fetch --quiet --no-tags --recurse-submodules=no "$REMOTE" || die "git fetch $REMOTE failed"
remote_ref="refs/remotes/$REMOTE/$BRANCH"
g rev-parse --verify --quiet "$remote_ref^{commit}" >/dev/null \
  || die "$REMOTE/$BRANCH does not exist (is the branch pushed?)"
head_sha="$(g rev-parse HEAD)"
remote_sha="$(g rev-parse "$remote_ref")"
short() { printf '%.9s' "$1"; }

STATE=""
BEHIND=0 AHEAD=0
if [ "$head_sha" = "$remote_sha" ]; then
  STATE=current
elif g merge-base --is-ancestor "$remote_sha" "$head_sha"; then
  STATE=current
  AHEAD="$(g rev-list --count "$remote_sha..$head_sha")"
elif g merge-base --is-ancestor "$head_sha" "$remote_sha"; then
  STATE=behind
  BEHIND="$(g rev-list --count "$head_sha..$remote_sha")"
else
  BEHIND="$(g rev-list --count "$head_sha..$remote_sha")"
  AHEAD="$(g rev-list --count "$remote_sha..$head_sha")"
  die "$BRANCH has diverged from $REMOTE/$BRANCH ($AHEAD local, $BEHIND remote commits). Fast-forward only: reconcile it by hand (git log $BRANCH...$REMOTE/$BRANCH)."
fi

print_incoming() {
  g log --no-decorate --format='  %h %s' -n "$MAX_LIST" "$head_sha..$remote_sha"
  [ "$BEHIND" -le "$MAX_LIST" ] || log "  ... and $((BEHIND - MAX_LIST)) more"
}

if [ "$STATE" = current ]; then
  if [ "$AHEAD" -gt 0 ]; then
    log "up to date: $BRANCH is $AHEAD commit(s) ahead of $REMOTE/$BRANCH (nothing to pull)"
  else
    log "up to date at $(short "$head_sha")"
  fi
else
  log "$BEHIND update(s) available: $(short "$head_sha") -> $(short "$remote_sha")"
  print_incoming
fi

if [ "$CHECK" = 1 ]; then
  [ "$STATE" = current ] && exit 0
  exit 10
fi

if [ "$STATE" = current ] && [ "$FORCE" = 0 ]; then
  log "nothing to do (use --force to rebuild and reinstall the current checkout)"
  exit 0
fi

# ---------------------------------------------------------------------------------------------
# Tracked changes block the fast-forward. Untracked files and build output do not count, and
# submodules are left to `git submodule update` below, so a half-finished earlier run (submodules
# at the wrong commit) can be resumed.
dirty="$(g status --porcelain --untracked-files=no --ignore-submodules=all)"
if [ -n "$dirty" ]; then
  printf '%s\n' "$dirty" >&2
  die "the working tree has uncommitted changes (above); commit or stash them first"
fi

# ---------------------------------------------------------------------------------------------
# 4. fast-forward and the pinned submodules
log "rollback target if anything below fails: git -C $REPO reset --hard $head_sha"
if [ "$STATE" = behind ]; then
  run git -C "$REPO" merge --ff-only --quiet "$remote_ref"
fi
run git -C "$REPO" submodule update --init --recursive

# ---------------------------------------------------------------------------------------------
# 5-7. build, install, engines
[ -x "$BUILD_FORK" ] || die "$BUILD_FORK is not executable"
log "build: $BUILD_FORK"
run "$BUILD_FORK"

if [ "$YES" = 0 ] && [ "$DRY" = 0 ]; then
  if [ ! -t 0 ]; then
    die "the build is done; the install quits Unsloth and replaces the app, so it needs --yes when there is no terminal"
  fi
  printf '[fork-update] Install now? This quits Unsloth and replaces /Applications/Unsloth.app. [y/N] '
  read -r answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) log "stopped before the install. The build is ready; run $BUILD_FORK --install when you want it."; exit 0 ;;
  esac
fi
log "install: $BUILD_FORK --install"
run "$BUILD_FORK" --install

[ -x "$UPDATE_ENGINES" ] || die "$UPDATE_ENGINES is not executable"
log "engines: $UPDATE_ENGINES --yes"
run "$UPDATE_ENGINES" --yes

# ---------------------------------------------------------------------------------------------
# 8. summary
new_sha="$(g rev-parse HEAD)"
echo
log "summary"
log "  checkout      $REPO ($BRANCH)"
if [ "$DRY" = 1 ]; then
  log "  dry run: nothing was changed (would go $(short "$head_sha") -> $(short "$remote_sha"))"
else
  log "  commit        $(short "$head_sha") -> $(short "$new_sha") ($BEHIND commit(s) pulled)"
  log "  built and installed the app and backend, engine venvs updated"
  log "  roll back     git -C $REPO reset --hard $head_sha, then build-fork-mac.sh --install"
  log "                (the previous app is kept at ~/Applications/Unsloth-prev.app.bak)"
  log "Open Unsloth again to use it."
fi
