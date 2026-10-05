#!/bin/bash
# Provision the oMLX attached engine for the Unsloth desktop app on macOS.
#
#   1. Create or refresh the user-side venv:
#        omlx/          oMLX installed non-editable from the submodule with its pinned deps
#   2. Write ~/.unsloth/engines/engines.toml if it is missing. An existing one is kept, but an
#      old generated file is migrated (see --migrate-config).
#
# Idempotent: work is skipped when the submodule commit recorded in
# ~/.unsloth/engines/.provisioned still matches and the outputs exist.
#
# The build fails when `git submodule status` shows a mismatch (+/-/U) or a submodule is dirty.
#
# Venvs are staged, never refreshed in place: each is built in <venv>.new, validated (imports plus
# a startup smoke test), and only then swapped in (live -> <venv>.old, .new -> live), and only
# while no engine runs from it. .provisioned is written after the swap. If an engine is running
# the staged venv waits in <venv>.new: stop the helper and run update-engines-mac.sh (which does
# this safely), or --swap-only. <venv>.old is kept for rollback (--rollback-venvs).
#
# Usage: studio/scripts/build-engines-mac.sh [--force] [--skip-venvs]
#                                            [--stage-only | --swap-only | --rollback-venvs [name...]
#                                             | --migrate-config]
#   --stage-only       build and validate the venvs in <venv>.new, do not swap
#   --swap-only        swap the venvs already staged (no build, no submodule checks)
#   --rollback-venvs   put <venv>.old back (name: omlx)
#   --migrate-config   only migrate an existing engines.toml (the full and --stage-only runs do it
#                      too; update-engines-mac.sh calls this explicitly). It removes the legacy
#                      [omlx.env] OMLX_PEER_EVICT_URLS = "http://127.0.0.1:8001" line.
#                      The original is copied to engines.toml.bak-<timestamp> first, every change is
#                      logged, comments and other keys are preserved, and a second run is a no-op.
#                      Helpers read the file when they start, so it takes effect at their next restart.
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

FORCE=0 SKIP_VENVS=0 MODE=full
ROLLBACK_NAMES=()
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --skip-venvs) SKIP_VENVS=1 ;;
    --stage-only) MODE=stage ;;
    --swap-only) MODE=swap ;;
    --rollback-venvs) MODE=rollback ;;
    --migrate-config) MODE=migrate ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    omlx)
      [ "$MODE" = rollback ] || { echo "unexpected argument: $arg" >&2; exit 2; }
      ROLLBACK_NAMES+=("$arg") ;;
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

OMLX_COMMIT=""

log() { printf '[engines] %s\n' "$*"; }
die() { printf '[engines] error: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "macOS only"

marker_get() { sed -n "s/^$1=//p" "$MARKER" 2>/dev/null | tail -n 1 || true; }
marker_set() {
  local tmp
  tmp="$(mktemp "$ENGINES_HOME/.provisioned.XXXXXX")"
  grep -v "^$1=" "$MARKER" > "$tmp" || true
  printf '%s=%s\n' "$1" "$2" >> "$tmp"
  mv "$tmp" "$MARKER"
}

# --- submodules -----------------------------------------------------------
init_submodules() {
  local sub
  for sub in omlx; do
    if ! [ -f "$ENGINES_SRC/$sub/.git" ] && ! [ -d "$ENGINES_SRC/$sub/.git" ]; then
      log "initialising submodule studio/engines/$sub"
      git -C "$REPO" submodule update --init "studio/engines/$sub"
    fi
  done
  OMLX_COMMIT="$(git -C "$ENGINES_SRC/omlx" rev-parse HEAD)"
}

# The engines are built from the submodule checkouts, so those must be exactly what the repo
# pins and untouched: a '+' (checked out elsewhere), '-' (not initialised) or 'U' prefix in
# `git submodule status`, or any local change, would provision a venv from a commit nothing records.
check_submodule_pins() {
  local line sub dirty
  while IFS= read -r line; do
    case "${line:0:1}" in
      " ") ;;
      *) die "submodule pin mismatch: $line (check out the pinned commit, or commit the submodule bump)" ;;
    esac
  done < <(git -C "$REPO" submodule status -- studio/engines/omlx)
  for sub in omlx; do
    dirty="$(git -C "$ENGINES_SRC/$sub" status --porcelain)"
    [ -z "$dirty" ] || die "submodule studio/engines/$sub has local changes; commit or discard them first"
  done
}

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
  for pid in $(pgrep -x omlx-server; pgrep -f "$1/bin/python"); do
    lsof -p "$pid" -Fn 2>/dev/null | grep "^n$1/" >/dev/null && return 0
  done
  return 1
}
guard_running() { # <venv dir>
  if venv_in_use "$1" && [ "${FORCE_REPLACE_RUNNING:-0}" != "1" ]; then
    die "$1 is in use by a running engine; stop the helper first (Settings, Engines enabled) or set FORCE_REPLACE_RUNNING=1"
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

# --- staging --------------------------------------------------------------
stage_wrappers() {
  # Staging is generated bundle input. Recreate it so a previous build cannot retain
  # resources for an engine that is no longer bundled.
  rm -rf "$STAGING"
  mkdir -p "$STAGING"
  local f
  for f in omlx-launch engines_launch.py engines-common.sh; do
    cmp -s "$WRAPPERS/$f" "$STAGING/$f" || install -m 0755 "$WRAPPERS/$f" "$STAGING/$f"
  done
}

# --- venvs ----------------------------------------------------------------
# A staged venv carries .staged: line 1 the .provisioned key, line 2 its value, then any extra
# "key=value" markers to record with it.
staged_ok() { # <new dir> <want>
  [ -x "$1/bin/python" ] && [ -f "$1/.staged" ] && [ "$(sed -n 2p "$1/.staged")" = "$2" ]
}

record_staged() { # <new dir> <key> <value> [extra key=value ...]
  local dir="$1"; shift
  printf '%s\n' "$@" > "$dir/.staged"
}

# The live venv already matches the pin, so a staged <venv>.new is left over from a pin that was
# since reverted: swapping it in would record a commit the repo does not pin.
discard_stale_staged() { # <new dir>
  [ -d "$1" ] || return 0
  log "discarding stale staged venv $1 (the live venv already matches the pin)"
  rm -rf "$1"
}

validate_omlx() { # <venv dir>
  "$1/bin/python" -c "import omlx, mlx.core, mlx_lm, mlx_vlm; print('omlx', omlx.__version__)"
  # Startup smoke test: the entry point the helper execs must load and parse its CLI.
  "$1/bin/python" -m omlx.cli serve --help >/dev/null
}

stage_omlx() {
  local venv="$ENGINES_HOME/omlx" new="$ENGINES_HOME/omlx.new"
  local want="$OMLX_COMMIT:py$OMLX_PYTHON:kernels$OMLX_WITH_CUSTOM_KERNEL:extras[$OMLX_EXTRAS]"
  if [ "$FORCE" = 0 ] && [ "$(marker_get omlx)" = "$want" ] && [ -x "$venv/bin/python" ]; then
    log "omlx venv: up to date ($OMLX_COMMIT)"
    discard_stale_staged "$new"
    return
  fi
  if [ "$FORCE" = 0 ] && staged_ok "$new" "$want"; then
    log "omlx venv: already staged ($OMLX_COMMIT)"
    return
  fi
  log "omlx venv: staging $OMLX_COMMIT in $new (this resolves and builds the pinned MLX stack; it can take several minutes)"
  local src="$BUILD_ROOT/omlx"
  export_tree "$ENGINES_SRC/omlx" "$src"
  local target="$src"; [ -n "$OMLX_EXTRAS" ] && target="${src}[$OMLX_EXTRAS]"
  local kernels="$OMLX_WITH_CUSTOM_KERNEL"
  make_venv "$new" "$OMLX_PYTHON"
  if [ "$kernels" = 1 ]; then
    if ! OMLX_WITH_CUSTOM_KERNEL=1 pip_install "$new" "$target" >"$BUILD_ROOT/omlx-install.log" 2>&1; then
      log "omlx: custom kernel build failed (see $BUILD_ROOT/omlx-install.log); retrying without them"
      kernels=0
      make_venv "$new" "$OMLX_PYTHON"
    fi
  fi
  if [ "$kernels" = 0 ]; then
    pip_install "$new" "$target" >"$BUILD_ROOT/omlx-install.log" 2>&1 \
      || { tail -n 30 "$BUILD_ROOT/omlx-install.log" >&2; die "omlx install failed (log: $BUILD_ROOT/omlx-install.log)"; }
  fi
  validate_omlx "$new"
  # Record the requested key (not the fallback result) so a fallback does not loop forever.
  record_staged "$new" omlx "$want" "omlx_kernels=$kernels"
  rm -rf "$src"
}

# Venv scripts (entry points, activate) embed the path the venv was built at; after the move
# from <venv>.new to <venv> they must point at the live path. `python -m` needs no fix.
fix_venv_paths() { # <live dir> <built-at dir>
  local file tmp
  while IFS= read -r file; do
    tmp="$(mktemp "$ENGINES_HOME/.fixpath.XXXXXX")"
    sed "s|$2|$1|g" "$file" >"$tmp" && cat "$tmp" >"$file"
    rm -f "$tmp"
  done < <(LC_ALL=C grep -rIl -- "$2" "$1/bin" "$1/pyvenv.cfg" 2>/dev/null || true)
}

# True when the venv can be replaced right now.
swap_allowed() { # <live dir>
  [ ! -x "$1/bin/python" ] && return 0
  ! venv_in_use "$1" || [ "${FORCE_REPLACE_RUNNING:-0}" = "1" ]
}

# Swap a staged venv in: live -> .old, .new -> live, then record the markers. Only while no
# engine runs from the live venv; otherwise the staged venv stays in .new.
swap_venv() { # <live dir>
  local venv="$1" new="$1.new" old="$1.old" key want line extra
  [ -d "$new" ] || return 0
  key="$(sed -n 1p "$new/.staged" 2>/dev/null || true)"
  want="$(sed -n 2p "$new/.staged" 2>/dev/null || true)"
  [ -n "$key" ] && [ -n "$want" ] || die "$new has no valid .staged record (an interrupted build?); delete it and re-run"
  if ! swap_allowed "$venv"; then
    log "$(basename "$venv"): staged in $new but an engine is running from $venv; not swapping."
    log "  stop the helper and run studio/scripts/update-engines-mac.sh (or re-run with --swap-only once it is stopped)"
    PENDING_SWAPS=$((PENDING_SWAPS + 1))
    return 0
  fi
  log "$(basename "$venv"): swapping the staged venv in (previous kept as $(basename "$old"))"
  rm -rf "$old" "$venv.failed"
  if [ -d "$venv" ]; then
    mv "$venv" "$old"
    {
      printf '%s=%s\n' "$key" "$(marker_get "$key")"
      tail -n +3 "$new/.staged" | while IFS= read -r line; do
        extra="${line%%=*}"; printf '%s=%s\n' "$extra" "$(marker_get "$extra")"
      done
    } >"$old/.prev-markers"
  fi
  if ! mv "$new" "$venv"; then
    [ ! -d "$old" ] || mv "$old" "$venv"
    die "could not move $new into place; the previous venv was put back"
  fi
  fix_venv_paths "$venv" "$new"
  marker_set "$key" "$want"
  tail -n +3 "$venv/.staged" | while IFS= read -r line; do marker_set "${line%%=*}" "${line#*=}"; done
  rm -f "$venv/.staged"
}

# Put <venv>.old back as the live venv (the failed one is kept as <venv>.failed) and restore the
# markers it was swapped in with. Only while no engine runs from the live venv.
rollback_venv() { # <live dir>
  local venv="$1" old="$1.old" line
  if [ ! -d "$old" ]; then
    log "$(basename "$venv"): no $(basename "$old") to roll back to"
    return 0
  fi
  swap_allowed "$venv" || die "$venv is in use by a running engine; stop the helper first"
  log "$(basename "$venv"): rolling back to $(basename "$old")"
  rm -rf "$venv.failed"
  [ ! -d "$venv" ] || mv "$venv" "$venv.failed"
  mv "$old" "$venv"
  if [ -f "$venv/.prev-markers" ]; then
    while IFS= read -r line; do marker_set "${line%%=*}" "${line#*=}"; done <"$venv/.prev-markers"
    rm -f "$venv/.prev-markers"
  fi
}

# Brings an engines.toml written by an older release up to date without touching other settings.
# The retired peer URL is removed only when it has the old generated value.
# Nothing is written when there is nothing to change, so a second run is a no-op; otherwise the
# original is copied to engines.toml.bak-<timestamp> first. The result must still parse (tomllib).
migrate_config() {
  local cfg="$ENGINES_HOME/engines.toml"
  if [ ! -e "$cfg" ]; then
    log "engines.toml: not present, nothing to migrate"
    return 0
  fi
  python3 - "$cfg" "$(date +%Y%m%d-%H%M%S)" <<'PY'
import os
import re
import shutil
import sys

cfg, ts = sys.argv[1:3]
try:
    import tomllib
except ImportError:  # python < 3.11: skip the validation, the edit is still conservative
    tomllib = None


def say(msg):
    print(f"[engines] engines.toml: {msg}", flush=True)


def parses(text):
    if tomllib is None:
        return True
    try:
        tomllib.loads(text)
        return True
    except tomllib.TOMLDecodeError:
        return False


with open(cfg, newline="") as fh:
    text = fh.read()
if not parses(text):
    say("is not valid TOML; migration skipped, file left untouched")
    sys.exit(0)

lines = text.splitlines(keepends=True)
eol = "\r\n" if "\r\n" in text else "\n"
HEADER = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(#.*)?$")


def span(name):
    """(header index, end index exclusive) of a [section], or None."""
    start = None
    for i, line in enumerate(lines):
        m = HEADER.match(line.rstrip("\r\n"))
        if not m:
            continue
        if start is not None:
            return start, i
        if re.sub(r"\s+", "", m.group(1)) == name:
            start = i
    return (start, len(lines)) if start is not None else None


changes = []
PEER = 'OMLX_PEER_EVICT_URLS = "http://127.0.0.1:8001"'
env = span("omlx.env")
if env:
    peer_lines = [i for i in range(env[0] + 1, env[1]) if lines[i].strip() == PEER]
    if peer_lines:
        # The comment the old migration wrote right above the line goes with it.
        drop = set(peer_lines)
        drop.update(i - 1 for i in peer_lines if i - 1 > env[0] and lines[i - 1].startswith("# oMLX asks the ds4 launcher"))
        for i in sorted(drop, reverse=True):
            del lines[i]
        changes.append(f"[omlx.env] removed {PEER}")

if not changes:
    say("up to date, nothing to migrate")
    sys.exit(0)

new_text = "".join(lines)
if not parses(new_text):
    say("the migrated file would not parse; left untouched")
    sys.exit(1)

backup = f"{cfg}.bak-{ts}"
n = 0
while os.path.exists(backup):
    n += 1
    backup = f"{cfg}.bak-{ts}-{n}"
shutil.copy2(cfg, backup)
say(f"backup written: {backup}")
tmp = f"{cfg}.migrate.tmp"
with open(tmp, "w", newline="") as fh:
    fh.write(new_text)
shutil.copymode(cfg, tmp)
os.replace(tmp, cfg)
for change in changes:
    say(f"migrated {change}")
say("takes effect when the helpers next restart")
PY
}

write_default_config() {
  local cfg="$ENGINES_HOME/engines.toml"
  if [ -e "$cfg" ]; then
    log "engines.toml: present, keeping your settings"
    migrate_config
  else
    install -m 0644 "$WRAPPERS/engines.default.toml" "$cfg"
    log "engines.toml: wrote defaults to $cfg"
  fi
}

PENDING_SWAPS=0

main() {
  mkdir -p "$ENGINES_HOME" "$BUILD_ROOT"
  touch "$MARKER"
  case "$MODE" in
    swap)
      swap_venv "$ENGINES_HOME/omlx"
      [ "$PENDING_SWAPS" = 0 ] || die "$PENDING_SWAPS staged venv(s) not swapped (an engine is running)"
      log "swap done"
      return 0 ;;
    migrate)
      migrate_config
      return 0 ;;
    rollback)
      local name
      [ "${#ROLLBACK_NAMES[@]}" -gt 0 ] || ROLLBACK_NAMES=(omlx)
      for name in "${ROLLBACK_NAMES[@]}"; do rollback_venv "$ENGINES_HOME/$name"; done
      log "rollback done"
      return 0 ;;
  esac
  init_submodules
  check_submodule_pins
  stage_wrappers
  if [ "$SKIP_VENVS" = 0 ]; then
    stage_omlx
    if [ "$MODE" = full ]; then
      swap_venv "$ENGINES_HOME/omlx"
    fi
  fi
  write_default_config
  log "done (omlx=${OMLX_COMMIT:0:8}); staging: $STAGING"
  [ "$PENDING_SWAPS" = 0 ] || log "$PENDING_SWAPS staged venv(s) are waiting in $ENGINES_HOME/*.new"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main
fi
