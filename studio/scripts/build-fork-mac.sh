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
#   1. STUDIO_LOCAL_REPO=<repo> unsloth studio update --local, then check the backend version
#   2. back up /Applications/Unsloth.app to ~/Applications/Unsloth-upstream-0.1.815.app.bak
#   3. quit Unsloth (never killed; aborts if it does not exit)
#   4. ditto the built app into /Applications
#   5. codesign --verify --deep --strict
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
phase_install() {
  [ -d "$DIST_APP" ] || [ "$DRY" = 1 ] || die "$DIST_APP not found; run the build phase first"
  command -v unsloth >/dev/null || [ -x "$HOME/.local/bin/unsloth" ] || die "unsloth CLI not found"
  local unsloth_bin; unsloth_bin="$(command -v unsloth || echo "$HOME/.local/bin/unsloth")"

  log "1/5 install the backend from this checkout"
  run env STUDIO_LOCAL_REPO="$REPO" "$unsloth_bin" studio update --local
  if [ "$DRY" = 0 ]; then
    local installed
    installed="$("$HOME/.unsloth/studio/unsloth_studio/bin/python" -c 'from importlib.metadata import version; print(version("unsloth"))')"
    check_backend_version "$installed" >/dev/null || die "installed backend version $installed is not acceptable"
    log "installed backend version: $installed"
  fi

  log "2/5 back up $INSTALLED_APP"
  if [ -e "$APP_BACKUP" ]; then
    log "backup already exists, keeping it: $APP_BACKUP"
  else
    run mkdir -p "$HOME/Applications"
    run ditto "$INSTALLED_APP" "$APP_BACKUP"
  fi

  log "3/5 quit Unsloth"
  if pgrep -x unsloth-studio >/dev/null 2>&1 || [ "$DRY" = 1 ]; then
    run osascript -e 'tell application id "ai.unsloth.studio" to quit'
    if [ "$DRY" = 0 ]; then
      local i
      for i in $(seq 1 60); do pgrep -x unsloth-studio >/dev/null 2>&1 || break; sleep 1; done
      pgrep -x unsloth-studio >/dev/null 2>&1 && die "Unsloth did not quit within 60 s; not forcing it"
    fi
  fi

  log "4/5 install the new app"
  run rm -rf "$INSTALLED_APP"
  run ditto "$DIST_APP" "$INSTALLED_APP"

  log "5/5 verify"
  run codesign --verify --deep --strict --verbose=2 "$INSTALLED_APP"
  log "installed. Open Unsloth and enable Settings > Background engines."
}

case "$PHASE" in
  build) phase_build ;;
  install) phase_install ;;
esac
