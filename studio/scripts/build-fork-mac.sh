#!/bin/bash
# Build (and later install) the fork's Unsloth.app with the bundled oMLX and ComfyUI engines.
#
#   build-fork-mac.sh [--build] [--dry-run] [--adhoc]     default phase
#   build-fork-mac.sh --install [--dry-run]               explicit, separate phase
#
# build phase (touches neither /Applications, ~/.unsloth/studio nor any running app):
#   1. build-engines-mac.sh (engine venvs and engines.toml)
#   2. UNSLOTH_DESKTOP_BACKEND_VERSION from unsloth/_version.py in THIS repo; it must be
#      numeric and >= 2026.8.4 (never read from the installed ~/.unsloth/studio)
#   3. frontend build
#   4. tauri build --features attached-engines --config src-tauri/tauri.fork.conf.json
#      signed with $SIGNING_IDENTITY (--adhoc signs ad hoc, "-", for verification only)
#   5. copy the app to ~/Homelab/unsloth/dist/Unsloth.app
#
# install phase (changes the machine; run it deliberately):
#   1. quit Unsloth (never killed; aborts if it does not exit within 120 s, QUIT_WAIT) and stop a
#      surviving :8888 backend; the engines must be idle (nothing generating, loading or starting).
#      With the engine lifetime on with_app (the default) the app's own quit first unloads the
#      oMLX models, interrupts and frees ComfyUI, and unregisters the helpers, so the quit can
#      take up to about a minute
#   2. build the fork wheel into a scratch dir first (uv build, else pip wheel) as a fail-fast
#      gate: nothing is touched when the checkout does not build
#   3. snapshot the backend venv (~/.unsloth/studio/unsloth_studio) with a clonefile copy into
#      ~/.unsloth/backend-backups/<timestamp>/ (the newest two are kept)
#   4. install the backend from this checkout, NON-editable, replacing the old wheel's files:
#      STUDIO_LOCAL_NONEDITABLE=1 STUDIO_LOCAL_REPO=<repo> unsloth studio update --local
#      (a direct `pip install --force-reinstall <wheel>` when the venv CLI cannot start; nothing
#      is ever uninstalled first), then check the version
#   5. verify the way the app's preflight runs it, from / : `unsloth -h`,
#      `unsloth studio desktop-capabilities --json`, and the studio.backend imports
#   6. back up /Applications/Unsloth.app to ~/Applications/Unsloth-upstream-0.1.815.app.bak (once;
#      copied to a .partial name and renamed, so an interrupted copy is never taken for a backup)
#   7. ditto the built app to /Applications/Unsloth.app.new and codesign-verify it, then unload
#      oMLX models and free ComfyUI, take the helpers out of launchd (they exec from inside the
#      bundle: `unsloth-studio --engine-helpers unregister` of the installed app, or launchctl
#      bootout for an app from before that CLI), move the old app to
#      ~/Applications/Unsloth-prev.app.bak, move the new one into place and register the helpers
#      that were running again, by name, with `--engine-helpers register <omlx|comfyui>` of the NEW
#      app (a helper that was not running, such as ComfyUI before the user turns it on, is not
#      started; macOS refuses launchctl bootstrap of a bundled helper; if register does not bring
#      them back, use
#      Settings > Engines > Engines enabled, off and on). In with_app mode the helpers
#      are only registered again when Unsloth is running (`pgrep -x unsloth-studio`); otherwise
#      they stay stopped and the app registers them at its next launch. The mode is
#      engine_lifetime in ~/.unsloth/engines/desktop.json (with_app when missing)
#   8. codesign --verify --deep --strict
#
# Steps 4-8 are ONE transaction. A rollback handler is armed once the snapshot exists and is
# disarmed only after step 8. On any failure in between (the install or its checks, the app backup
# copy, staging or verifying the new bundle, the quiesce, a bootout, the port gate, the swap, the
# final verify) it restores the backend snapshot AND the previous app together (the broken venv is
# kept next to the snapshot as failed-unsloth_studio), booting the helpers out again first if they
# had already come up from the new bundle, and then restarts every helper this run owes. The
# handler is armed before the first bootout, and a helper counts as owed before its bootout runs.
#
# The helper stop/restart goes through helpers_stop and helpers_start in engines-lib.sh. The
# `--engine-helpers` CLI has been run against the live helpers only by hand; the scripts have only
# run it with --dry-run and a fake app.
#
# The build bakes UNSLOTH_FORK_REPO (this checkout) into the app: its managed repair runs the
# same non-editable `update --local` from it and refuses, rather than installing from PyPI.
# It also bakes UNSLOTH_FORK_COMMIT (HEAD) and UNSLOTH_FORK_BRANCH (the checked-out branch, or the
# FORK_BRANCH environment variable): Settings > About > Updates compares them with
# github.com/beaglemoo/unsloth and opens studio/scripts/update-fork.sh in Terminal.
#
# Environment: FORK_BRANCH (branch to bake, default the checked-out one), SIGNING_IDENTITY (default "Developer ID Application: james beesley (D6VHKTRR33)"),
#              DIST_DIR (default ~/Homelab/unsloth/dist); for tests: UNSLOTH_STUDIO_VENV,
#              BACKUP_ROOT, INSTALLED_APP, APP_PREV, QUIT_WAIT, and the engines-lib.sh variables
set -euo pipefail

PHASE=build DRY=0 ADHOC=0
for arg in "$@"; do
  case "$arg" in
    --build) PHASE=build ;;
    --install) PHASE=install ;;
    --dry-run) DRY=1 ;;
    --adhoc) ADHOC=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
STUDIO="$REPO/studio"
DIST_DIR="${DIST_DIR:-$HOME/Homelab/unsloth/dist}"
DIST_APP="$DIST_DIR/Unsloth.app"
BUILT_APP="$STUDIO/src-tauri/target/release/bundle/macos/Unsloth.app"
SIGNING_IDENTITY="${SIGNING_IDENTITY:-Developer ID Application: james beesley (D6VHKTRR33)}"
MIN_BACKEND_VERSION="2026.8.4"
APP_BACKUP="$HOME/Applications/Unsloth-upstream-0.1.815.app.bak"
INSTALLED_APP="${INSTALLED_APP:-/Applications/Unsloth.app}"
APP_PREV="${APP_PREV:-$HOME/Applications/Unsloth-prev.app.bak}"
BACKUP_ROOT="${BACKUP_ROOT:-$HOME/.unsloth/backend-backups}"
TS="$(date +%Y%m%d-%H%M%S)"

log() { printf '[fork-build] %s\n' "$*"; }
warn() { printf '[fork-build] warning: %s\n' "$*" >&2; }
die() { printf '[fork-build] error: %s\n' "$*" >&2; exit 1; }
run() {
  if [ "$DRY" = 1 ]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

[ "$(uname -s)" = "Darwin" ] || die "macOS only"
# shellcheck source=engines-lib.sh
. "$SCRIPT_DIR/engines-lib.sh"

# Prints the version when it is numeric (dot separated integers) and >= the minimum.
check_backend_version() { # <version>
  python3 - "$1" "$MIN_BACKEND_VERSION" <<'PY'
import re, sys
version, minimum = sys.argv[1:3]
if not re.fullmatch(r"[0-9]+(\.[0-9]+)+", version):
    sys.exit(f"backend version {version!r} is not numeric")
key = lambda v: tuple(int(p) for p in v.split("."))
if key(version) < key(minimum):
    sys.exit(f"backend version {version} is older than {minimum}")
print(version)
PY
}

repo_backend_version() {
  local line
  line="$(sed -n 's/^__version__ = ["'"'"']\(.*\)["'"'"']$/\1/p' "$REPO/unsloth/_version.py" | head -n 1)"
  [ -n "$line" ] || die "no __version__ in $REPO/unsloth/_version.py"
  check_backend_version "$line"
}

# ---------------------------------------------------------------------------------------------
phase_build() {
  command -v npm >/dev/null || die "npm not found"
  command -v cargo >/dev/null || { [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"; }
  command -v cargo >/dev/null || die "cargo not found (install rustup)"

  log "1/5 engines"
  run "$SCRIPT_DIR/build-engines-mac.sh"

  log "2/5 backend version"
  local version
  version="$(repo_backend_version)"
  log "UNSLOTH_DESKTOP_BACKEND_VERSION=$version (from $REPO/unsloth/_version.py)"

  log "3/5 frontend"
  [ -d "$STUDIO/frontend/node_modules" ] || run npm ci --prefix "$STUDIO/frontend"
  run npm run build --prefix "$STUDIO/frontend"

  log "4/5 tauri build"
  # Tauri copies resources beside the release executable without pruning old entries.
  # Recreate that generated engine resource directory so retired resources cannot linger.
  run rm -rf "$STUDIO/src-tauri/target/release/engines"
  [ -x "$STUDIO/node_modules/.bin/tauri" ] || run npm ci --prefix "$STUDIO"
  local identity="$SIGNING_IDENTITY"
  [ "$ADHOC" = 1 ] && identity="-"
  log "signing identity: $identity"
  # Baked into the app for Settings > About > Updates: the commit this build was made from and
  # the branch it is compared against on github.com/beaglemoo/unsloth.
  local fork_commit fork_branch
  fork_commit="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)" || die "cannot read the checkout's HEAD commit"
  fork_branch="${FORK_BRANCH:-$(git -C "$REPO" symbolic-ref --quiet --short HEAD 2>/dev/null || true)}"
  [ -n "$fork_branch" ] || fork_branch="feat/omlx-ds4-engines"
  log "fork build: ${fork_commit:0:9} on $fork_branch"
  # The frontend is already built above, so tauri's own beforeBuildCommand is blanked.
  ( cd "$STUDIO" && run env \
      UNSLOTH_DESKTOP_BACKEND_VERSION="$version" \
      UNSLOTH_FORK_REPO="$REPO" \
      UNSLOTH_FORK_COMMIT="$fork_commit" \
      UNSLOTH_FORK_BRANCH="$fork_branch" \
      APPLE_SIGNING_IDENTITY="$identity" \
      "$STUDIO/node_modules/.bin/tauri" build \
        --features attached-engines \
        --config src-tauri/tauri.fork.conf.json \
        --config '{"build":{"beforeBuildCommand":""}}' )

  log "5/5 copy to $DIST_APP"
  run mkdir -p "$DIST_DIR"
  run rm -rf "$DIST_APP"
  run ditto "$BUILT_APP" "$DIST_APP"
  log "done: $DIST_APP"
}

# ---------------------------------------------------------------------------------------------
STUDIO_VENV="${UNSLOTH_STUDIO_VENV:-$HOME/.unsloth/studio/unsloth_studio}"
WHEEL=""
BACKEND_BACKUP=""
APP_BUNDLE_ID="ai.unsloth.studio"

quit_unsloth() {
  if app_running || [ "$DRY" = 1 ]; then
    run osascript -e "tell application id \"$APP_BUNDLE_ID\" to quit"
    if [ "$DRY" = 0 ]; then
      local i
      # A with_app quit first unloads the oMLX models, then reaps the backend.
      for i in $(seq 1 "${QUIT_WAIT:-120}"); do app_running || break; sleep 1; done
      app_running && die "Unsloth did not quit within ${QUIT_WAIT:-120} s; not forcing it"
    fi
  fi
  # The app stops its own backend on quit; a backend it did not own stays up and would hold
  # the files being replaced. Ask it to stop through the CLI, never with a signal.
  if [ "$DRY" = 0 ] && curl -s -m 2 -o /dev/null 127.0.0.1:8888/api/health; then
    log "a backend is still serving :8888, stopping it gracefully"
    "$STUDIO_VENV/bin/unsloth" studio stop || true
    local i
    for i in $(seq 1 60); do curl -s -m 2 -o /dev/null 127.0.0.1:8888/api/health || break; sleep 1; done
    curl -s -m 2 -o /dev/null 127.0.0.1:8888/api/health && die "backend on :8888 did not stop within 60 s; not forcing it"
  fi
}

# Build the fork wheel in a scratch dir before anything is touched: proves the checkout builds
# and carries the fork's modules. Sets WHEEL.
build_fork_wheel() {
  local out
  out="$(mktemp -d "${TMPDIR:-/tmp}/unsloth-fork-wheel.XXXXXX")"
  if [ "$DRY" = 1 ]; then
    run uv build --wheel --out-dir "$out" "$REPO"
    WHEEL="$out/unsloth-<version>-py3-none-any.whl"
    rmdir "$out"
    return 0
  fi
  if command -v uv >/dev/null 2>&1; then
    uv build --wheel --out-dir "$out" "$REPO" >"$out/build.log" 2>&1 \
      || { tail -n 30 "$out/build.log" >&2; die "the fork wheel does not build (log: $out/build.log)"; }
  else
    python3 -m pip wheel --no-deps -w "$out" "$REPO" >"$out/build.log" 2>&1 \
      || { tail -n 30 "$out/build.log" >&2; die "the fork wheel does not build (log: $out/build.log)"; }
  fi
  WHEEL="$(find "$out" -maxdepth 1 -name 'unsloth-*.whl' | head -n 1)"
  [ -n "$WHEEL" ] || die "no wheel was produced in $out"
  # Capture first: `zipfile -l | grep -q` trips pipefail when grep exits early (SIGPIPE).
  local listing
  listing="$(python3 -m zipfile -l "$WHEEL")" || die "cannot list $WHEEL"
  grep -q 'studio/backend/routes/attached_engines.py' <<<"$listing" \
    || die "$WHEEL does not contain the fork's attached-engines backend"
  # Without these two a rebase that dropped them would reopen the PyPI and upstream-zoo paths.
  grep -q 'studio/_fork.py' <<<"$listing" \
    || die "$WHEEL does not contain the fork marker studio/_fork.py"
  grep -q 'studio/fork-pins.toml' <<<"$listing" \
    || die "$WHEEL does not contain studio/fork-pins.toml"
  log "fork wheel built: $WHEEL"
}

# A clonefile copy (APFS: instant, no extra space) of the whole backend venv, taken before it is
# mutated. The newest two snapshots are kept.
snapshot_backend() {
  BACKEND_BACKUP="$BACKUP_ROOT/$TS/unsloth_studio"
  run mkdir -p "$BACKUP_ROOT/$TS"
  if ! run cp -cR "$STUDIO_VENV" "$BACKEND_BACKUP" 2>/dev/null; then
    log "clonefile copy not possible here; using ditto"
    run rm -rf "$BACKEND_BACKUP"
    run ditto "$STUDIO_VENV" "$BACKEND_BACKUP"
  fi
  [ "$DRY" = 1 ] || [ -x "$BACKEND_BACKUP/bin/python" ] || die "the backend snapshot at $BACKEND_BACKUP is incomplete"
  log "backend snapshot: $BACKEND_BACKUP"
}

# Put the snapshot back; the venv it replaces is kept next to it for inspection.
restore_backend() {
  [ -d "$BACKEND_BACKUP" ] || die "no backend snapshot to restore (BACKEND_BACKUP=$BACKEND_BACKUP)"
  warn "restoring the backend from $BACKEND_BACKUP"
  local failed="$BACKUP_ROOT/$TS/failed-unsloth_studio"
  rm -rf "$failed" || return 1
  # Explicit returns: this runs inside `if ( restore_backend )`, where set -e is off.
  if [ -e "$STUDIO_VENV" ]; then mv "$STUDIO_VENV" "$failed" || return 1; fi
  [ ! -e "$STUDIO_VENV" ] || return 1
  mv "$BACKEND_BACKUP" "$STUDIO_VENV" || return 1
  warn "restored. The failed install is kept at $failed"
}

prune_backend_backups() {
  local old
  [ -d "$BACKUP_ROOT" ] || return 0
  find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' | sort -r | tail -n +3 | while IFS= read -r old; do
    rm -rf "$old"
  done
}

# Replaces the installed backend with this checkout as a regular (non-editable) package. An
# editable install leaves the wheel's unrecorded runtime files (data_recipe/oxc-validator/
# node_modules) behind; those bare directories become a namespace package for `studio` and
# shadow the editable finder ("No module named 'studio.backend.utils'"). Nothing is uninstalled
# first: the replacement was built (build_fork_wheel) before this runs, and a failure here is
# undone from the snapshot by the caller.
install_backend() {
  local py="$STUDIO_VENV/bin/python" cli="$STUDIO_VENV/bin/unsloth"
  if [ "$DRY" = 1 ] || (cd / && "$cli" -h >/dev/null 2>&1); then
    # UNSLOTH_TAURI_UPDATE=1 is what the app sets: the desktop owns its shortcuts and frontend.
    run env STUDIO_LOCAL_REPO="$REPO" STUDIO_LOCAL_NONEDITABLE=1 UNSLOTH_TAURI_UPDATE=1 \
      SKIP_STUDIO_FRONTEND=1 "$cli" studio update --local
  else
    log "the venv CLI cannot start; reinstalling the backend directly from the wheel"
    run "$py" -m pip install --no-deps --force-reinstall --no-cache-dir "$WHEEL"
  fi
}

check_installed_version() {
  local installed
  installed="$("$STUDIO_VENV/bin/python" -c 'from importlib.metadata import version; print(version("unsloth"))')" \
    || die "could not read the installed backend version"
  check_backend_version "$installed" >/dev/null || die "installed backend version $installed is not acceptable"
  log "installed backend version: $installed"
}

# What the app's preflight runs (preflight/managed.rs: `unsloth -h`, then
# `unsloth studio desktop-capabilities --json`), from / like a Finder launch, plus imports of
# the fork's own modules. studio.backend.routes cannot be imported from outside the backend
# (routes/__init__.py uses bare `from routes...` imports that run.py makes resolvable), so the
# fork's modules are located (importlib find_spec), not imported.
verify_backend() {
  local py="$STUDIO_VENV/bin/python" cli="$STUDIO_VENV/bin/unsloth"
  [ "$DRY" = 0 ] || { log "[dry-run] verify from /: unsloth -h, desktop-capabilities, imports"; return 0; }
  ( cd / && env -u PYTHONPATH -u PYTHONHOME "$cli" -h >/dev/null ) || die "preflight probe 'unsloth -h' failed"
  ( cd / && env -u PYTHONPATH -u PYTHONHOME "$cli" studio desktop-capabilities --json >/dev/null ) \
    || die "preflight probe 'unsloth studio desktop-capabilities --json' failed"
  ( cd / && env -u PYTHONPATH -u PYTHONHOME "$py" - <<'PY'
import importlib.metadata as md, json, os, sys
import studio.backend.utils
import studio.backend
root = list(studio.backend.__path__)[0]
if "site-packages" not in root:
    sys.exit(f"studio.backend resolves outside site-packages: {root}")
route = os.path.join(root, "routes", "attached_engines.py")
for path in (route, os.path.join(root, "utils", "attached_engines_settings.py")):
    # Located, not imported: these modules use backend-relative imports (`from loggers import
    # ...`, `from routes.x import ...`) that only resolve once run.py has set sys.path.
    if not os.path.isfile(path):
        sys.exit(f"missing {path}")
dist = md.distribution("unsloth")
direct = json.loads(dist.read_text("direct_url.json") or "{}")
if direct.get("dir_info", {}).get("editable"):
    sys.exit("unsloth is an editable install; the fork backend must be a regular install")
print("fork backend installed at", os.path.dirname(route), "version", dist.version)
PY
  ) || die "the installed backend is not the fork build"
}

# The install is one transaction from the moment the backend is replaced until the final
# verification: the backend and the app are replaced together and put back together. The EXIT
# handler is armed before the first mutation (phase_install) and disarmed after the last check.
# Whatever the failure (the app backup copy, staging or verifying the new bundle, the quiesce, a
# bootout, the port gate, the swap, the final verify), it restores the backend snapshot AND the
# previous app, then restarts every helper this run owes a restart. It never dies: each step
# warns and the next one still runs.
INSTALL_ARMED=0
APP_STAGED=0 APP_PREV_MOVED=0 APP_NEW_IN_PLACE=0

rollback_install() {
  [ "$DRY" = 0 ] || return 0
  warn "the install failed; rolling back the backend and the app together"
  local app_note=""
  # Past the swap the helpers may already run from the new bundle: stop them before it goes.
  if [ "$APP_NEW_IN_PLACE" = 1 ]; then
    helpers_stop || warn "could not boot every helper out before restoring the app"
    # Each engine's gate is soft and separate: one stuck engine (its unregister failed and it keeps
    # its port) must not keep the previous app and the other engine from being restored. Nothing is killed.
    local gate_label
    for gate_label in ${STOPPED_HELPERS[@]+"${STOPPED_HELPERS[@]}"}; do
      ports_free_wait "$gate_label" || warn "the ${gate_label##*.} helper still holds port $(label_port "$gate_label"); it is left running and the previous app is restored over it"
    done
    if rm -rf "$INSTALLED_APP"; then APP_NEW_IN_PLACE=0; else warn "could not remove the new app at $INSTALLED_APP"; fi
  fi
  [ "$APP_STAGED" = 0 ] || rm -rf "$INSTALLED_APP.new" || true
  rm -rf "$APP_BACKUP.partial" || true
  if [ "$APP_PREV_MOVED" = 1 ]; then
    if [ -e "$INSTALLED_APP" ]; then
      warn "$INSTALLED_APP exists; the previous app stays at $APP_PREV"
    elif mv "$APP_PREV" "$INSTALLED_APP"; then
      APP_PREV_MOVED=0
      app_note=" and the previous app"
    else
      warn "could not put the previous app back from $APP_PREV"
    fi
  fi
  if [ -n "$BACKEND_BACKUP" ] && [ -d "$BACKEND_BACKUP" ]; then
    # restore_backend dies on a problem; keep that out of the handler
    if ( restore_backend ); then
      warn "rolled back: the previous backend was restored$app_note"
    else
      warn "the backend could not be restored; the snapshot is at $BACKEND_BACKUP"
    fi
  fi
  helpers_start || warn "the helpers did not restart; use Settings > Engines > Engines enabled, off and on"
}

install_on_exit() {
  local rc=$?
  trap - EXIT
  trap '' INT TERM HUP
  if [ "$rc" -ne 0 ] && [ "$INSTALL_ARMED" = 1 ]; then
    INSTALL_ARMED=0
    rollback_install
  fi
  exit "$rc"
}
arm_install_rollback() {
  INSTALL_ARMED=1
  trap install_on_exit EXIT
  trap 'exit 130' INT TERM HUP
}
disarm_install_rollback() { INSTALL_ARMED=0; trap - EXIT INT TERM HUP; }

# Swap the app bundle with the helpers out of launchd: they exec from inside it. The new app is
# copied and verified beside the old one first, so the window with no app is two renames.
install_app() {
  local staged="$INSTALLED_APP.new" own=0
  # Standalone use arms (and disarms) its own handler; inside phase_install it is already armed.
  [ "$INSTALL_ARMED" = 1 ] || { arm_install_rollback; own=1; }
  APP_STAGED=1
  run rm -rf "$staged"
  run ditto "$DIST_APP" "$staged"
  run codesign --verify --deep --strict --verbose=2 "$staged"

  log "unload oMLX models and free ComfyUI, then boot the helpers out of launchd"
  engines_quiesce
  # The old bundle still contains the legacy SMAppService plist. Its CLI must unregister it
  # before this swap: a new bundle cannot address a plist it no longer contains.
  HELPER_APP="$INSTALLED_APP" helpers_stop
  # One-time migration over a pre-removal app: no legacy ds4 job may survive into the new bundle.
  HELPER_APP="$INSTALLED_APP" helpers_clear_legacy_ds4
  # Only the ports of the helpers this run stopped (plus the legacy ds4 ones when that launcher was seen).
  wait_ports_free ${STOPPED_HELPERS[@]+"${STOPPED_HELPERS[@]}"}

  if [ -e "$INSTALLED_APP" ]; then
    run mkdir -p "$(dirname "$APP_PREV")"
    run rm -rf "$APP_PREV"
    run mv "$INSTALLED_APP" "$APP_PREV"
    APP_PREV_MOVED=1
  fi
  run mv "$staged" "$INSTALLED_APP" || die "could not move the new app into place"
  APP_NEW_IN_PLACE=1

  log "restart the helpers"
  helpers_start || warn "the helpers did not restart; use Settings > Engines > Engines enabled, off and on"
  [ "$own" = 0 ] || disarm_install_rollback
}

phase_install() {
  [ -d "$DIST_APP" ] || [ "$DRY" = 1 ] || die "$DIST_APP not found; run the build phase first"
  [ -x "$STUDIO_VENV/bin/python" ] || [ "$DRY" = 1 ] || die "no managed backend venv at $STUDIO_VENV"

  log "1/8 quit Unsloth and check the engines are idle"
  quit_unsloth
  # update.rs: a pending staged-update journal has the next idle launch restore the old trees.
  if grep -qi -E 'stag|rollback' <<<"$(ls "$HOME/.unsloth/studio" 2>/dev/null || true)"; then
    die "a staged-update journal exists in ~/.unsloth/studio; resolve it before installing"
  fi
  engines_require_idle

  log "2/8 build the fork wheel (nothing is changed if this fails)"
  build_fork_wheel

  log "3/8 snapshot the backend venv"
  snapshot_backend

  # From here the backend and the app are one transaction. Any failure up to the final
  # verification restores the backend snapshot and the previous app together, then restarts the
  # helpers (see rollback_install). Disarmed only after step 8.
  arm_install_rollback

  log "4/8 install the backend from this checkout (non-editable)"
  if ! ( install_backend && { [ "$DRY" = 1 ] || check_installed_version; } ); then
    die "the backend install failed; rolling back"
  fi

  log "5/8 verify the backend as the app's preflight sees it"
  if ! ( verify_backend ); then
    die "the installed backend did not verify; rolling back"
  fi

  log "6/8 back up $INSTALLED_APP"
  if [ -e "$APP_BACKUP" ]; then
    log "backup already exists, keeping it: $APP_BACKUP"
  else
    run mkdir -p "$HOME/Applications"
    # copied beside the final name and renamed, so an interrupted copy is never mistaken for a backup
    run rm -rf "$APP_BACKUP.partial"
    run ditto "$INSTALLED_APP" "$APP_BACKUP.partial"
    run mv "$APP_BACKUP.partial" "$APP_BACKUP"
  fi

  log "7/8 install the new app"
  install_app

  log "8/8 verify"
  run codesign --verify --deep --strict --verbose=2 "$INSTALLED_APP"
  disarm_install_rollback
  prune_backend_backups
  log "installed. The previous app is kept at $APP_PREV, the backend snapshot under $BACKUP_ROOT."
  log "Open Unsloth: with the engine lifetime on with_app (the default) it registers and starts the engines at launch; with always, Settings > Engines > Engines enabled must be on."
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "$PHASE" in
    build) phase_build ;;
    install) phase_install ;;
  esac
fi
