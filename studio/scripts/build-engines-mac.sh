#!/bin/bash
# Provision the attached engines (oMLX and ds4) for the Unsloth desktop app on macOS.
#
#   1. Build ds4-server (make) from the studio/engines/ds4 submodule and stage it,
#      with its metal/ shaders and the on-demand launcher, in the Tauri resources
#      staging dir (default studio/src-tauri/engines-staging, bundled as
#      Contents/Resources/engines by tauri.fork.conf.json).
#   2. Create or refresh the user-side venvs in ~/.unsloth/engines:
#        omlx/          oMLX installed non-editable from the submodule with its pinned deps
#        ds4-ondemand/  fastapi/uvicorn/httpx for ds4_ondemand.py
#   3. Write ~/.unsloth/engines/engines.toml only if it is missing.
#
# Idempotent: work is skipped when the submodule commit recorded in
# ~/.unsloth/engines/.provisioned still matches and the outputs exist.
#
# Usage: studio/scripts/build-engines-mac.sh [--force] [--skip-ds4] [--skip-venvs]
#
# Environment:
#   UNSLOTH_ENGINES_HOME     engines home (default ~/.unsloth/engines)
#   ENGINES_STAGING          staging dir for the app bundle resources
#   OMLX_EXTRAS              pip extras for oMLX, e.g. "audio,mcp" (default none)
#   OMLX_WITH_CUSTOM_KERNEL  1 (default) builds oMLX's optional Metal custom kernels;
#                            falls back to a build without them if that fails. 0 skips.
#   OMLX_PYTHON              python spec for the oMLX venv (default 3.13)
#   FORCE_REPLACE_RUNNING=1  refresh a venv even while a helper is running from it
set -euo pipefail

FORCE=0 SKIP_DS4=0 SKIP_VENVS=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --skip-ds4) SKIP_DS4=1 ;;
    --skip-venvs) SKIP_VENVS=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENGINES_SRC="$REPO/studio/engines"
WRAPPERS="$SCRIPT_DIR/engines"
ENGINES_HOME="${UNSLOTH_ENGINES_HOME:-$HOME/.unsloth/engines}"
STAGING="${ENGINES_STAGING:-$REPO/studio/src-tauri/engines-staging}"
MARKER="$ENGINES_HOME/.provisioned"
BUILD_ROOT="$ENGINES_HOME/.build"
OMLX_PYTHON="${OMLX_PYTHON:-3.13}"
OMLX_EXTRAS="${OMLX_EXTRAS:-}"
OMLX_WITH_CUSTOM_KERNEL="${OMLX_WITH_CUSTOM_KERNEL:-1}"

log() { printf '[engines] %s\n' "$*"; }
die() { printf '[engines] error: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "macOS only"

mkdir -p "$ENGINES_HOME" "$BUILD_ROOT"
touch "$MARKER"

marker_get() { sed -n "s/^$1=//p" "$MARKER" | tail -n 1; }
marker_set() {
  local tmp
  tmp="$(mktemp "$ENGINES_HOME/.provisioned.XXXXXX")"
  grep -v "^$1=" "$MARKER" > "$tmp" || true
  printf '%s=%s\n' "$1" "$2" >> "$tmp"
  mv "$tmp" "$MARKER"
}

# --- submodules -----------------------------------------------------------
for sub in omlx ds4; do
  if ! [ -f "$ENGINES_SRC/$sub/.git" ] && ! [ -d "$ENGINES_SRC/$sub/.git" ]; then
    log "initialising submodule studio/engines/$sub"
    git -C "$REPO" submodule update --init "studio/engines/$sub"
  fi
done
OMLX_COMMIT="$(git -C "$ENGINES_SRC/omlx" rev-parse HEAD)"
DS4_COMMIT="$(git -C "$ENGINES_SRC/ds4" rev-parse HEAD)"

# Export the committed tree (no .git, no untracked files) so builds never dirty the submodule.
export_tree() { # <submodule dir> <dest>
  rm -rf "$2"; mkdir -p "$2"
  git -C "$1" archive HEAD | tar -x -C "$2"
}

# True when a running engine has files open under the given venv. pgrep -f on the
# interpreter path is not enough: oMLX renames itself to "omlx-server" (setproctitle)
# and a venv python resolves to its base interpreter in the process list.
venv_in_use() { # <venv dir>
  local pid
  for pid in $(pgrep -x omlx-server; pgrep -f ds4_ondemand.py; pgrep -f "$1/bin/python"); do
    lsof -p "$pid" -Fn 2>/dev/null | grep "^n$1/" >/dev/null && return 0
  done
  return 1
}
guard_running() { # <venv dir>
  if venv_in_use "$1" && [ "${FORCE_REPLACE_RUNNING:-0}" != "1" ]; then
    die "$1 is in use by a running engine; stop the helper first (Settings, Background engines) or set FORCE_REPLACE_RUNNING=1"
  fi
}

# --- python selection -----------------------------------------------------
HAVE_UV=0; command -v uv >/dev/null 2>&1 && HAVE_UV=1

make_venv() { # <dir> <python spec, e.g. 3.13 or 3.11+>
  local dir="$1" spec="$2"
  if [ "$HAVE_UV" = 1 ]; then
    uv venv --quiet --clear --python "$spec" "$dir"
  else
    local py="" cand
    for cand in python3.13 python3.12 python3.11; do
      command -v "$cand" >/dev/null 2>&1 && { py="$cand"; break; }
    done
    [ -n "$py" ] || die "no python 3.11-3.13 found and uv is not installed"
    rm -rf "$dir"
    "$py" -m venv "$dir"
  fi
}

pip_install() { # <venv dir> <pip args...>
  local dir="$1"; shift
  if [ "$HAVE_UV" = 1 ]; then
    uv pip install --quiet --python "$dir/bin/python" "$@"
  else
    "$dir/bin/python" -m pip install --quiet "$@"
  fi
}

# --- ds4 build + staging --------------------------------------------------
stage_wrappers() {
  mkdir -p "$STAGING"
  local f
  for f in omlx-launch ds4-ondemand-launch engines_launch.py; do
    cmp -s "$WRAPPERS/$f" "$STAGING/$f" || install -m 0755 "$WRAPPERS/$f" "$STAGING/$f"
  done
}

ds4_staged_ok() {
  [ -x "$STAGING/ds4/ds4-server" ] && [ -d "$STAGING/ds4/metal" ] && [ -f "$STAGING/ds4/ds4_ondemand.py" ]
}

build_ds4() {
  local want="$DS4_COMMIT"
  if [ "$FORCE" = 0 ] && [ "$(marker_get ds4_build)" = "$want" ] && ds4_staged_ok; then
    log "ds4: up to date ($want)"
    return
  fi
  log "ds4: building $want"
  local src="$BUILD_ROOT/ds4"
  export_tree "$ENGINES_SRC/ds4" "$src"
  local jobs; jobs="$(sysctl -n hw.ncpu)"
  make -C "$src" -j"$jobs" ds4-server >"$BUILD_ROOT/ds4-build.log" 2>&1 \
    || { tail -n 30 "$BUILD_ROOT/ds4-build.log" >&2; die "ds4 make failed (log: $BUILD_ROOT/ds4-build.log)"; }
  [ -x "$src/ds4-server" ] || die "ds4-server was not produced"
  local dest="$STAGING/ds4"
  rm -rf "$dest"; mkdir -p "$dest"
  install -m 0755 "$src/ds4-server" "$dest/ds4-server"
  cp -R "$src/metal" "$dest/metal"
  install -m 0644 "$src/ondemand/ds4_ondemand.py" "$dest/ds4_ondemand.py"
  # ds4-server compiles its shaders from metal/*.metal in its cwd; fail early if the set is empty.
  ls "$dest"/metal/*.metal >/dev/null
  marker_set ds4_build "$want"
  log "ds4: staged in $dest"
}

# --- venvs ----------------------------------------------------------------
provision_ds4_ondemand() {
  local venv="$ENGINES_HOME/ds4-ondemand"
  local want="$DS4_COMMIT"
  if [ "$FORCE" = 0 ] && [ "$(marker_get ds4_ondemand)" = "$want" ] && [ -x "$venv/bin/python" ]; then
    log "ds4-ondemand venv: up to date ($want)"
    return
  fi
  guard_running "$venv"
  log "ds4-ondemand venv: creating ($want)"
  make_venv "$venv" "3.11+"
  pip_install "$venv" -r "$ENGINES_SRC/ds4/ondemand/requirements.txt"
  "$venv/bin/python" -c "import fastapi, uvicorn, httpx"
  marker_set ds4_ondemand "$want"
}

provision_omlx() {
  local venv="$ENGINES_HOME/omlx"
  local want="$OMLX_COMMIT:py$OMLX_PYTHON:kernels$OMLX_WITH_CUSTOM_KERNEL:extras[$OMLX_EXTRAS]"
  if [ "$FORCE" = 0 ] && [ "$(marker_get omlx)" = "$want" ] && [ -x "$venv/bin/python" ]; then
    log "omlx venv: up to date ($OMLX_COMMIT)"
    return
  fi
  guard_running "$venv"
  log "omlx venv: creating from $OMLX_COMMIT (this resolves and builds the pinned MLX stack; it can take several minutes)"
  local src="$BUILD_ROOT/omlx"
  export_tree "$ENGINES_SRC/omlx" "$src"
  local target="$src"; [ -n "$OMLX_EXTRAS" ] && target="$src[$OMLX_EXTRAS]"
  local kernels="$OMLX_WITH_CUSTOM_KERNEL"
  make_venv "$venv" "$OMLX_PYTHON"
  if [ "$kernels" = 1 ]; then
    if ! OMLX_WITH_CUSTOM_KERNEL=1 pip_install "$venv" "$target" >"$BUILD_ROOT/omlx-install.log" 2>&1; then
      log "omlx: custom kernel build failed (see $BUILD_ROOT/omlx-install.log); retrying without them"
      kernels=0
      make_venv "$venv" "$OMLX_PYTHON"
    fi
  fi
  if [ "$kernels" = 0 ]; then
    pip_install "$venv" "$target" >"$BUILD_ROOT/omlx-install.log" 2>&1 \
      || { tail -n 30 "$BUILD_ROOT/omlx-install.log" >&2; die "omlx install failed (log: $BUILD_ROOT/omlx-install.log)"; }
  fi
  "$venv/bin/python" -c "import omlx, mlx.core, mlx_lm, mlx_vlm; print('omlx', omlx.__version__)"
  # Record the requested key (not the fallback result) so a fallback does not loop forever.
  marker_set omlx "$OMLX_COMMIT:py$OMLX_PYTHON:kernels$OMLX_WITH_CUSTOM_KERNEL:extras[$OMLX_EXTRAS]"
  marker_set omlx_kernels "$kernels"
  rm -rf "$src"
}

write_default_config() {
  local cfg="$ENGINES_HOME/engines.toml"
  if [ -e "$cfg" ]; then
    log "engines.toml: present, left untouched"
  else
    install -m 0644 "$WRAPPERS/engines.default.toml" "$cfg"
    log "engines.toml: wrote defaults to $cfg"
  fi
}

stage_wrappers
[ "$SKIP_DS4" = 1 ] || build_ds4
if [ "$SKIP_VENVS" = 0 ]; then
  provision_ds4_ondemand
  provision_omlx
fi
write_default_config
log "done (omlx=${OMLX_COMMIT:0:8} ds4=${DS4_COMMIT:0:8}); staging: $STAGING"
