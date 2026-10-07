#!/bin/bash
# Safely update the engine venvs (oMLX, ComfyUI) from their pinned submodules while Unsloth's
# background helper keeps its place in launchd. Run it after bumping studio/engines/omlx or
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
#   3. idle again, then unload oMLX models gracefully (idle only; nothing is killed)
#   4. stop the helpers (`unsloth-studio --engine-helpers unregister`, else launchctl bootout),
#      wait until :8843 is free
#   5. swap: build-engines-mac.sh --swap-only (live -> <venv>.old, .new -> live, then .provisioned)
#   6. start the helpers again (`unsloth-studio --engine-helpers register`) and wait for health.
#      In with_app mode (engine_lifetime in ~/.unsloth/engines/desktop.json, the default) they are
#      only registered again while Unsloth is running (`pgrep -x unsloth-studio`); with the app
#      closed they stay stopped, because the app registers them at its next launch
#   7. on any failure after the swap: stop the helpers, build-engines-mac.sh --rollback-venvs,
#      start them again, wait for health, and exit 1. <venv>.failed keeps the broken venv for inspection.
#
# Only the staged venvs change. A ComfyUI-only update leaves oMLX and its helper alone: there is
# no ComfyUI helper yet (Phase 2), so the swap is guarded instead. It is refused while a ComfyUI
# process runs from the venv (nothing is killed), and the swapped venv is smoke-tested
# (ComfyUI's --quick-test-for-ci from the live path; COMFYUI_CHECK overrides the command). A failure
# rolls every venv this run swapped back. UNSLOTH_BUILD_COMFYUI=0 leaves ComfyUI out of all of it.
#
# The helper stop and start live in engines-lib.sh and use the app's `--engine-helpers` CLI
# (macOS refuses `launchctl bootstrap` of a bundled helper). They have only been exercised with
# --dry-run and a fake app. If a helper does not come back, use Unsloth > Settings >
# Attached engines > Engines enabled, off and on.
#
# Environment: UNSLOTH_ENGINES_HOME (default ~/.unsloth/engines), OMLX_URL, ENGINE_PORTS,
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
ROSTER=""
SWAPPED=()
SWAP_OMLX=0
SWAP_COMFYUI=0

swapped_has() { # <name>
  local n
  for n in ${SWAPPED[@]+"${SWAPPED[@]}"}; do [ "$n" != "$1" ] || return 0; done
  return 1
}

# A ComfyUI process running from the venv (venv python or src/main.py in its argv). There is no
# ComfyUI helper to stop yet, and nothing is ever killed here, so the update waits for it.
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

# Nothing to wait for when helpers_start left the helpers stopped (with_app, Unsloth closed).
healthy() {
  if [ "$SWAP_OMLX" = 0 ] || [ "$HELPERS_LEFT_STOPPED" = 1 ]; then return 0; fi
  wait_engines_healthy "$HEALTH_TIMEOUT" "$WANT_OMLX"
}

rollback_swap() {
  log "rolling back: restoring the previous venvs"
  if [ "$SWAP_OMLX" = 1 ]; then
    helpers_stop
    wait_ports_free
  fi
  local name args=()
  for name in ${SWAPPED[@]+"${SWAPPED[@]}"}; do
    # only venvs this run actually replaced (their .new is gone and a .old exists)
    if [ ! -d "$ENGINES_HOME/$name.new" ] && [ -d "$ENGINES_HOME/$name.old" ]; then args+=("$name"); fi
  done
  if [ "${#args[@]}" -gt 0 ]; then
    run "$BUILD_ENGINES" --rollback-venvs "${args[@]}" || warn "venv rollback failed; restore $ENGINES_HOME/*.old by hand"
  fi
  if [ "$SWAP_OMLX" = 1 ]; then helpers_start || warn "helpers did not restart after the rollback"; fi
  if [ "$SWAP_OMLX" = 1 ] && [ "$HELPERS_LEFT_STOPPED" = 1 ]; then
    log "rolled back; the helpers stay stopped until Unsloth registers them at its next launch"
  elif healthy; then
    log "rolled back; the previous engines are serving again"
  else
    warn "engines are not healthy after the rollback"
    [ "$SWAP_OMLX" = 0 ] || show_launch_failure omlx
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
    WANT_OMLX=1
    log "$_label is loaded"
  else
    log "$_label is not loaded"
  fi
done
if [ "$WANT_OMLX" = 1 ] && omlx_up; then ROSTER="$(fetch_roster 2>/dev/null || true)"; fi
if [ "$DRY" = 1 ]; then WANT_OMLX=1; fi

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
log "3/6 idle check, then unload oMLX models"
if [ "$SWAP_COMFYUI" = 1 ] && comfyui_running; then
  die "ComfyUI is running from $ENGINES_HOME/comfyui; stop it first (nothing was changed; the staged venv waits in comfyui.new)"
fi
if [ "$SWAP_OMLX" = 1 ]; then
  confirm "This stops the oMLX helper for a moment to swap in ${SWAPPED[*]}. Continue?"
  engines_quiesce
else
  confirm "This swaps in ${SWAPPED[*]}. oMLX and its helper are not touched. Continue?"
  log "oMLX is not being updated; leaving it running"
fi

# 4 -------------------------------------------------------------------------------------------
log "4/6 stop the helpers"
if [ "$SWAP_OMLX" = 1 ]; then
  STAGE=stopped
  helpers_stop
  wait_ports_free
else
  log "no helper serves ${SWAPPED[*]}; nothing to stop"
fi

# 5 -------------------------------------------------------------------------------------------
log "5/6 swap the venvs"
STAGE=swapping
run "$BUILD_ENGINES" --swap-only

# 6 -------------------------------------------------------------------------------------------
log "6/6 restart the helpers and wait for health"
if [ "$SWAP_OMLX" = 1 ]; then
  helpers_start || die "the helpers did not restart"
fi
if [ "$DRY" = 1 ]; then
  if [ "$SWAP_OMLX" = 1 ]; then log "dry-run: would poll $OMLX_URL/v1/models for up to ${HEALTH_TIMEOUT} s, then compare the roster"; fi
  if [ "$SWAP_COMFYUI" = 1 ]; then comfyui_check; fi
  log "dry-run complete"
  STAGE=pre
  exit 0
fi
if [ "$SWAP_COMFYUI" = 1 ]; then
  comfyui_check || die "the swapped ComfyUI venv failed its smoke test"
  log "comfyui: the swapped venv passed its smoke test"
fi
if [ "$SWAP_OMLX" = 0 ]; then
  STAGE=pre
  log "done. oMLX was not touched. The previous venv is kept as comfyui.old for build-engines-mac.sh --rollback-venvs comfyui."
  exit 0
fi
if [ "$HELPERS_LEFT_STOPPED" = 1 ]; then
  STAGE=pre
  log "done. The helpers are stopped (with_app and Unsloth is closed); the app registers them at its next launch, so the new venvs are not health-checked until then."
  log "The previous venvs are kept as *.old for build-engines-mac.sh --rollback-venvs."
  exit 0
fi
if ! healthy; then
  show_launch_failure omlx
  die "the engines did not become healthy within ${HEALTH_TIMEOUT} s with the new venvs"
fi
if [ -n "$ROSTER" ]; then
  if [ "$ROSTER" = "$(fetch_roster 2>/dev/null || true)" ]; then
    log "the oMLX model roster is unchanged"
  else
    warn "the oMLX roster differs from before the update (models.json or the models directory changed?)"
  fi
fi
STAGE=pre
log "done. The previous venvs are kept as *.old for build-engines-mac.sh --rollback-venvs."
