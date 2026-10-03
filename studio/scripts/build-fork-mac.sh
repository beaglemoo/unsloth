#!/bin/bash
# Build (and later install) the fork's Unsloth.app with the bundled oMLX and ds4 engines.
#
#   build-fork-mac.sh [--build] [--dry-run] [--adhoc]     default phase
#   build-fork-mac.sh --install [--dry-run]               explicit, separate phase
#
# build phase (touches neither /Applications, ~/.unsloth/studio nor any running app):
#   1. build-engines-mac.sh (ds4 build, engine venvs, engines.toml)
#   2. UNSLOTH_DESKTOP_BACKEND_VERSION from unsloth/_version.py in THIS repo; it must be
#      numeric and >= 2026.8.4 (never read from the installed ~/.unsloth/studio)
#   3. frontend build
#   4. tauri build --features attached-engines --config src-tauri/tauri.fork.conf.json
#      signed with $SIGNING_IDENTITY (--adhoc signs ad hoc, "-", for verification only)
#   5. copy the app to ~/Homelab/unsloth/dist/Unsloth.app
#
# install phase (changes the machine; run it deliberately):
#   1. quit Unsloth (never killed; aborts if it does not exit) and stop a surviving :8888 backend
#   2. install the backend from this checkout, NON-editable, replacing the old wheel's files:
#      STUDIO_LOCAL_NONEDITABLE=1 STUDIO_LOCAL_REPO=<repo> unsloth studio update --local
#      (a direct pip reinstall when the venv CLI cannot start), then check the version
#   3. verify the way the app's preflight runs it, from / : `unsloth -h`,
#      `unsloth studio desktop-capabilities --json`, and the studio.backend imports
#   4. back up /Applications/Unsloth.app to ~/Applications/Unsloth-upstream-0.1.815.app.bak
#   5. ditto the built app into /Applications
#   6. codesign --verify --deep --strict
#
# The build bakes UNSLOTH_FORK_REPO (this checkout) into the app: its managed repair runs the
# same non-editable `update --local` from it and refuses, rather than installing from PyPI.
#
# Environment: SIGNING_IDENTITY (default "Developer ID Application: james beesley (D6VHKTRR33)"),
#              DIST_DIR (default ~/Homelab/unsloth/dist)
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
INSTALLED_APP="/Applications/Unsloth.app"

log() { printf '[fork-build] %s\n' "$*"; }
die() { printf '[fork-build] error: %s\n' "$*" >&2; exit 1; }
run() {
  if [ "$DRY" = 1 ]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

[ "$(uname -s)" = "Darwin" ] || die "macOS only"

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
  [ -x "$STUDIO/node_modules/.bin/tauri" ] || run npm ci --prefix "$STUDIO"
  local identity="$SIGNING_IDENTITY"
  [ "$ADHOC" = 1 ] && identity="-"
  log "signing identity: $identity"
  # The frontend is already built above, so tauri's own beforeBuildCommand is blanked.
  ( cd "$STUDIO" && run env \
      UNSLOTH_DESKTOP_BACKEND_VERSION="$version" \
      UNSLOTH_FORK_REPO="$REPO" \
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
STUDIO_VENV="$HOME/.unsloth/studio/unsloth_studio"
APP_BUNDLE_ID="ai.unsloth.studio"

quit_unsloth() {
  if pgrep -x unsloth-studio >/dev/null 2>&1 || [ "$DRY" = 1 ]; then
    run osascript -e "tell application id \"$APP_BUNDLE_ID\" to quit"
    if [ "$DRY" = 0 ]; then
      local i
      for i in $(seq 1 60); do pgrep -x unsloth-studio >/dev/null 2>&1 || break; sleep 1; done
      pgrep -x unsloth-studio >/dev/null 2>&1 && die "Unsloth did not quit within 60 s; not forcing it"
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

# Replaces the installed backend with this checkout as a regular (non-editable) package. An
# editable install leaves the wheel's unrecorded runtime files (data_recipe/oxc-validator/
# node_modules) behind; those bare directories become a namespace package for `studio` and
# shadow the editable finder ("No module named 'studio.backend.utils'").
install_backend() {
  local py="$STUDIO_VENV/bin/python" cli="$STUDIO_VENV/bin/unsloth"
  if [ "$DRY" = 1 ] || (cd / && "$cli" -h >/dev/null 2>&1); then
    # UNSLOTH_TAURI_UPDATE=1 is what the app sets: the desktop owns its shortcuts and frontend.
    run env STUDIO_LOCAL_REPO="$REPO" STUDIO_LOCAL_NONEDITABLE=1 UNSLOTH_TAURI_UPDATE=1 \
      SKIP_STUDIO_FRONTEND=1 "$cli" studio update --local
  else
    log "the venv CLI cannot start; reinstalling the backend directly"
    run "$py" -m pip uninstall -y unsloth
    run "$py" -m pip install --no-deps --force-reinstall --no-cache-dir "$REPO"
  fi
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

phase_install() {
  [ -d "$DIST_APP" ] || [ "$DRY" = 1 ] || die "$DIST_APP not found; run the build phase first"
  [ -x "$STUDIO_VENV/bin/python" ] || [ "$DRY" = 1 ] || die "no managed backend venv at $STUDIO_VENV"

  log "1/6 quit Unsloth"
  quit_unsloth
  # update.rs: a pending staged-update journal has the next idle launch restore the old trees.
  if ls "$HOME/.unsloth/studio" 2>/dev/null | grep -qi -E 'stag|rollback'; then
    die "a staged-update journal exists in ~/.unsloth/studio; resolve it before installing"
  fi

  log "2/6 install the backend from this checkout (non-editable)"
  install_backend
  if [ "$DRY" = 0 ]; then
    local installed
    installed="$("$STUDIO_VENV/bin/python" -c 'from importlib.metadata import version; print(version("unsloth"))')"
    check_backend_version "$installed" >/dev/null || die "installed backend version $installed is not acceptable"
    log "installed backend version: $installed"
  fi

  log "3/6 verify the backend as the app's preflight sees it"
  verify_backend

  log "4/6 back up $INSTALLED_APP"
  if [ -e "$APP_BACKUP" ]; then
    log "backup already exists, keeping it: $APP_BACKUP"
  else
    run mkdir -p "$HOME/Applications"
    run ditto "$INSTALLED_APP" "$APP_BACKUP"
  fi

  log "5/6 install the new app"
  run rm -rf "$INSTALLED_APP"
  run ditto "$DIST_APP" "$INSTALLED_APP"

  log "6/6 verify"
  run codesign --verify --deep --strict --verbose=2 "$INSTALLED_APP"
  log "installed. Open Unsloth and enable Settings > Background engines."
}

case "$PHASE" in
  build) phase_build ;;
  install) phase_install ;;
esac
