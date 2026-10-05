#!/bin/bash
# Safely update the oMLX venv from its pinned submodule while Unsloth's
# background helper keeps its place in launchd. Run it after bumping studio/engines/omlx.
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
# Only the oMLX venv changes.
#
# The helper stop and start live in engines-lib.sh and use the app's `--engine-helpers` CLI
# (macOS refuses `launchctl bootstrap` of a bundled helper). They have only been exercised with
# --dry-run and a fake app. If a helper does not come back, use Unsloth > Settings >
# Attached engines > Engines enabled, off and on.
#
# Environment: UNSLOTH_ENGINES_HOME (default ~/.unsloth/engines), OMLX_URL, ENGINE_PORTS,
#   HELPER_APP, LAUNCHCTL, PGREP, BUILD_ENGINES (the build script), HEALTH_TIMEOUT (default 240 s)
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
  if [ "$HELPERS_LEFT_STOPPED" = 1 ]; then return 0; fi
  wait_engines_healthy "$HEALTH_TIMEOUT" "$WANT_OMLX"
}

rollback_swap() {
  log "rolling back: restoring the previous venvs"
  helpers_stop
  wait_ports_free
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
    show_launch_failure omlx
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

# 3 -------------------------------------------------------------------------------------------
log "3/6 idle check, then unload oMLX models"
confirm "This stops the oMLX helper for a moment to swap in ${SWAPPED[*]}. Continue?"
engines_quiesce

# 4 -------------------------------------------------------------------------------------------
log "4/6 stop the helpers"
STAGE=stopped
helpers_stop
wait_ports_free

# 5 -------------------------------------------------------------------------------------------
log "5/6 swap the venvs"
STAGE=swapping
run "$BUILD_ENGINES" --swap-only

# 6 -------------------------------------------------------------------------------------------
log "6/6 restart the helpers and wait for health"
helpers_start || die "the helpers did not restart"
if [ "$DRY" = 1 ]; then
  log "dry-run: would poll $OMLX_URL/v1/models for up to ${HEALTH_TIMEOUT} s, then compare the roster"
  log "dry-run complete"
  STAGE=pre
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
