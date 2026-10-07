#!/bin/bash
# Safely update the engine venvs (oMLX, ComfyUI) from their pinned submodules while Unsloth's
# background helpers keep their place in launchd. Run it after bumping studio/engines/omlx or
# studio/engines/comfyui (or its constraints file).
#
#   update-engines-mac.sh [--dry-run] [--yes] [--force]
#
#   1. preflight: the engines must be idle (nothing generating, loading or starting), then
#      build-engines-mac.sh --migrate-config removes the old generated OMLX_PEER_EVICT_URLS line; a
#      backup engines.toml.bak-<timestamp> is written first and every change is logged)
#   2. stage: build-engines-mac.sh --stage-only builds and validates <venv>.new while the helpers
#      keep serving, so they are down only for the swap, not for the multi-minute build
#      (the submodule pin check runs here; the build fails on a mismatch or a dirty submodule)
#   3. idle again, then free the swapped engines' memory gracefully (idle only; nothing is
#      killed): unload the oMLX models, POST /free to ComfyUI
#   4. stop the helpers whose venv is being swapped, and only those (`unsloth-studio
#      --engine-helpers unregister <name>` with the named-target CLI, else launchctl bootout),
#      wait until their ports (:8843 oMLX, :8844 ComfyUI) are free
#   5. swap: build-engines-mac.sh --swap-only (live -> <venv>.old, .new -> live, then .provisioned)
#   6. start the helpers again (`unsloth-studio --engine-helpers register`) and wait for health.
#      In with_app mode (engine_lifetime in ~/.unsloth/engines/desktop.json, the default) they are
#      only registered again while Unsloth is running (`pgrep -x unsloth-studio`); with the app
#      closed they stay stopped, because the app registers them at its next launch
#   7. on any failure after the swap: stop the helpers, build-engines-mac.sh --rollback-venvs,
#      start them again, wait for health, and exit 1. <venv>.failed keeps the broken venv for inspection.
#
# Only the staged venvs change, and only their helpers bounce: a ComfyUI-only update leaves oMLX and
# its helper alone, and an oMLX-only update leaves ComfyUI running. A ComfyUI that is not a helper
# (started by hand, or the helper not loaded) blocks its swap while it runs from the venv (nothing
# is killed). The swapped ComfyUI venv is smoke-tested (ComfyUI's --quick-test-for-ci from the live
# path; COMFYUI_CHECK overrides the command). A failure rolls every venv this run swapped back.
# UNSLOTH_BUILD_COMFYUI=0 leaves ComfyUI out of all of it.
#
# The helper stop and start live in engines-lib.sh and use the app's `--engine-helpers` CLI
# (macOS refuses `launchctl bootstrap` of a bundled helper). They have only been exercised with
# --dry-run and a fake app. If a helper does not come back, use Unsloth > Settings >
# Engines > Engines enabled, off and on.
#
# Environment: UNSLOTH_ENGINES_HOME (default ~/.unsloth/engines), OMLX_URL, COMFYUI_URL, ENGINE_PORTS,
#   HELPER_APP, LAUNCHCTL, PGREP, BUILD_ENGINES (the build script), HEALTH_TIMEOUT (default 240 s),
#   UNSLOTH_BUILD_COMFYUI (0: ignore ComfyUI), COMFYUI_CHECK (command run after a ComfyUI swap)
set -euo pipefail

DRY=0 YES=0 FORCE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --yes|-y) YES=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINES_HOME="${UNSLOTH_ENGINES_HOME:-$HOME/.unsloth/engines}"
BUILD_ENGINES="${BUILD_ENGINES:-$SCRIPT_DIR/build-engines-mac.sh}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-240}"
VENVS=(omlx)
[ "${UNSLOTH_BUILD_COMFYUI:-1}" = 0 ] || VENVS+=(comfyui)

log() { printf '[engines-update] %s\n' "$*"; }
warn() { printf '[engines-update] warning: %s\n' "$*" >&2; }
die() { printf '[engines-update] error: %s\n' "$*" >&2; exit 1; }
run() {
  if [ "$DRY" = 1 ]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

[ "$(uname -s)" = "Darwin" ] || die "macOS only"
command -v python3 >/dev/null || die "python3 not found"
# shellcheck source=engines-lib.sh
. "$SCRIPT_DIR/engines-lib.sh"

# pre | stopped | swapping: how far the run got, for the EXIT handler.
STAGE=pre
WANT_OMLX=0
WANT_COMFYUI=0
ROSTER=""
SWAPPED=()
SWAP_OMLX=0
SWAP_COMFYUI=0

swapped_has() { # <name>
  local n
  for n in ${SWAPPED[@]+"${SWAPPED[@]}"}; do [ "$n" != "$1" ] || return 0; done
  return 1
}

# A ComfyUI process running from the venv (venv python or src/main.py in its argv). The caller
# asks this only when the ComfyUI helper is not loaded: a helper is stopped before the swap, and
# nothing is ever killed here, so the update waits for a process that is not one.
comfyui_running() {
  local venv="$ENGINES_HOME/comfyui"
  pgrep -f "$venv/src/main.py" >/dev/null 2>&1 || pgrep -f "$venv/bin/python" >/dev/null 2>&1
}

# Smoke test of the swapped ComfyUI venv at its live path (relocation can break a venv):
# ComfyUI's own --quick-test-for-ci, which initialises every node and exits before binding a port.
comfyui_check() {
  local venv="$ENGINES_HOME/comfyui"
  if [ "$DRY" = 1 ]; then log "dry-run: would smoke-test the swapped ComfyUI venv ($venv/src/main.py --quick-test-for-ci)"; return 0; fi
  if [ -n "${COMFYUI_CHECK:-}" ]; then "$COMFYUI_CHECK"; return; fi
  mkdir -p "$ENGINES_HOME/.build/comfyui-smoke"
  python3 - "$venv" "$ENGINES_HOME/.build/comfyui-smoke" <<'PY'
import subprocess, sys
venv, smoke = sys.argv[1:3]
try:
    run = subprocess.run(
        [f"{venv}/bin/python", "main.py", "--quick-test-for-ci", "--base-directory", smoke,
         "--disable-all-custom-nodes", "--cpu"],
        cwd=f"{venv}/src", timeout=300, capture_output=True, text=True,
    )
except subprocess.TimeoutExpired:
    print("ComfyUI smoke test timed out after 300 s", file=sys.stderr)
    sys.exit(1)
if run.returncode != 0:
    print((run.stdout + run.stderr)[-2000:], file=sys.stderr)
    sys.exit(1)
PY
}

staged_names() {
  local name
  for name in "${VENVS[@]}"; do
    if [ -d "$ENGINES_HOME/$name.new" ]; then printf '%s\n' "$name"; fi
  done
}

confirm() {
  [ "$YES" = 1 ] && return 0
  if [ "$DRY" = 1 ]; then log "dry-run: would ask: $1"; return 0; fi
  [ -t 0 ] || die "no terminal for the confirmation; re-run with --yes"
  local answer
  read -r -p "[engines-update] $1 [y/N] " answer
  case "$answer" in y|Y|yes|YES) return 0 ;; *) die "aborted by the user" ;; esac
}

# The labels of the helpers whose venv this run swaps (only these are stopped and restarted).
swap_labels() {
  [ "$SWAP_OMLX" = 0 ] || printf '%s\n' ai.unsloth.studio.omlx
  [ "$SWAP_COMFYUI" = 0 ] || printf '%s\n' ai.unsloth.studio.comfyui
}

# Nothing to wait for when helpers_start left the helpers stopped (with_app, Unsloth closed), or for
# an engine that was not swapped or whose helper was not running before.
healthy() {
  local want_omlx=0 want_comfyui=0
  if [ "$HELPERS_LEFT_STOPPED" = 1 ]; then return 0; fi
  if [ "$SWAP_OMLX" = 1 ] && [ "$WANT_OMLX" = 1 ]; then want_omlx=1; fi
  if [ "$SWAP_COMFYUI" = 1 ] && [ "$WANT_COMFYUI" = 1 ]; then want_comfyui=1; fi
  if [ "$want_omlx" = 0 ] && [ "$want_comfyui" = 0 ]; then return 0; fi
  wait_engines_healthy "$HEALTH_TIMEOUT" "$want_omlx" "$want_comfyui"
}

rollback_swap() {
  log "rolling back: restoring the previous venvs"
  local labels=()
  while IFS= read -r _l; do labels+=("$_l"); done < <(swap_labels)
  if [ "${#labels[@]}" -gt 0 ]; then
    helpers_stop "${labels[@]}"
    wait_ports_free "${labels[@]}"
  fi
  local name args=()
  for name in ${SWAPPED[@]+"${SWAPPED[@]}"}; do
    # only venvs this run actually replaced (their .new is gone and a .old exists)
    if [ ! -d "$ENGINES_HOME/$name.new" ] && [ -d "$ENGINES_HOME/$name.old" ]; then args+=("$name"); fi
  done
  if [ "${#args[@]}" -gt 0 ]; then
    run "$BUILD_ENGINES" --rollback-venvs "${args[@]}" || warn "venv rollback failed; restore $ENGINES_HOME/*.old by hand"
  fi
  helpers_start || warn "helpers did not restart after the rollback"
  if [ "$HELPERS_LEFT_STOPPED" = 1 ]; then
    log "rolled back; the helpers stay stopped until Unsloth registers them at its next launch"
  elif healthy; then
    log "rolled back; the previous engines are serving again"
  else
    warn "engines are not healthy after the rollback"
    [ "$SWAP_OMLX" = 0 ] || show_launch_failure omlx
    [ "$SWAP_COMFYUI" = 0 ] || show_launch_failure comfyui
  fi
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [ "$rc" -ne 0 ]; then
    case "$STAGE" in
      stopped)
        warn "failed before the swap; restarting the helpers"
        helpers_start || true ;;
      swapping)
        rollback_swap || true ;;
    esac
  fi
  exit "$rc"
}
trap on_exit EXIT

# 1 -------------------------------------------------------------------------------------------
log "1/6 preflight$([ "$DRY" = 1 ] && echo ' (dry-run)')"
[ -x "$BUILD_ENGINES" ] || die "$BUILD_ENGINES not found"
engines_require_idle
run "$BUILD_ENGINES" --migrate-config
for _label in "${HELPER_LABELS[@]}"; do
  if helper_loaded "$_label"; then
    case "$_label" in
      *.omlx) WANT_OMLX=1 ;;
      *.comfyui) WANT_COMFYUI=1 ;;
    esac
    log "$_label is loaded"
  else
    log "$_label is not loaded"
  fi
done
if [ "$WANT_OMLX" = 1 ] && omlx_up; then ROSTER="$(fetch_roster 2>/dev/null || true)"; fi
if [ "$DRY" = 1 ]; then WANT_OMLX=1 WANT_COMFYUI=1; fi

# 2 -------------------------------------------------------------------------------------------
log "2/6 stage the new venvs (the helpers keep serving)"
build_args=(--stage-only)
[ "$FORCE" = 0 ] || build_args+=(--force)
run "$BUILD_ENGINES" "${build_args[@]}"
if [ "$DRY" = 0 ]; then
  while IFS= read -r _name; do SWAPPED+=("$_name"); done < <(staged_names)
  if [ "${#SWAPPED[@]}" -eq 0 ]; then
    log "the engine venvs are already up to date; nothing to do"
    exit 0
  fi
  log "staged: ${SWAPPED[*]}"
else
  SWAPPED=("${VENVS[@]}")
fi
if swapped_has omlx; then SWAP_OMLX=1; fi
if swapped_has comfyui; then SWAP_COMFYUI=1; fi

# 3 -------------------------------------------------------------------------------------------
log "3/6 idle check, then free the memory of what is swapped"
if [ "$SWAP_COMFYUI" = 1 ] && ! helper_loaded ai.unsloth.studio.comfyui && comfyui_running; then
  die "ComfyUI is running from $ENGINES_HOME/comfyui outside the helper; stop it first (nothing was changed; the staged venv waits in comfyui.new)"
fi
STOP_LABELS=()
while IFS= read -r _l; do STOP_LABELS+=("$_l"); done < <(swap_labels)
quiesce_args=()
[ "$SWAP_OMLX" = 0 ] || quiesce_args+=(omlx)
[ "$SWAP_COMFYUI" = 0 ] || quiesce_args+=(comfyui)
untouched=""
[ "$SWAP_OMLX" = 1 ] || untouched="oMLX and its helper"
[ "$SWAP_COMFYUI" = 1 ] || untouched="${untouched:+$untouched, }ComfyUI and its helper"
confirm "This stops the engine helpers for a moment to swap in ${SWAPPED[*]}.${untouched:+ $untouched are not touched.} Continue?"
engines_quiesce "${quiesce_args[@]}"
[ -z "$untouched" ] || log "$untouched are not being updated; leaving them running"

# 4 -------------------------------------------------------------------------------------------
log "4/6 stop the helpers"
STAGE=stopped
helpers_stop "${STOP_LABELS[@]}"
wait_ports_free "${STOP_LABELS[@]}"

# 5 -------------------------------------------------------------------------------------------
log "5/6 swap the venvs"
STAGE=swapping
run "$BUILD_ENGINES" --swap-only

# 6 -------------------------------------------------------------------------------------------
log "6/6 restart the helpers and wait for health"
helpers_start || die "the helpers did not restart"
if [ "$DRY" = 1 ]; then
  if [ "$SWAP_OMLX" = 1 ]; then log "dry-run: would poll $OMLX_URL/v1/models for up to ${HEALTH_TIMEOUT} s, then compare the roster"; fi
  if [ "$SWAP_COMFYUI" = 1 ]; then log "dry-run: would poll $COMFYUI_URL/system_stats for up to ${HEALTH_TIMEOUT} s"; comfyui_check; fi
  log "dry-run complete"
  STAGE=pre
  exit 0
fi
if [ "$SWAP_COMFYUI" = 1 ]; then
  comfyui_check || die "the swapped ComfyUI venv failed its smoke test"
  log "comfyui: the swapped venv passed its smoke test"
fi
if [ "$HELPERS_LEFT_STOPPED" = 1 ]; then
  STAGE=pre
  log "done. The helpers are stopped (with_app and Unsloth is closed); the app registers them at its next launch, so the new venvs are not health-checked until then."
  log "The previous venvs are kept as *.old for build-engines-mac.sh --rollback-venvs."
  exit 0
fi
if ! healthy; then
  [ "$SWAP_OMLX" = 0 ] || show_launch_failure omlx
  [ "$SWAP_COMFYUI" = 0 ] || show_launch_failure comfyui
  die "the engines did not become healthy within ${HEALTH_TIMEOUT} s with the new venvs"
fi
if [ "$SWAP_OMLX" = 1 ] && [ -n "$ROSTER" ]; then
  if [ "$ROSTER" = "$(fetch_roster 2>/dev/null || true)" ]; then
    log "the oMLX model roster is unchanged"
  else
    warn "the oMLX roster differs from before the update (models.json or the models directory changed?)"
  fi
fi
STAGE=pre
log "done.${untouched:+ $untouched were not touched.} The previous venvs are kept as *.old for build-engines-mac.sh --rollback-venvs."
