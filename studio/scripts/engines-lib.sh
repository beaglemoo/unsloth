# shellcheck shell=bash
# Shared helpers for update-engines-mac.sh and build-fork-mac.sh --install (sourced, never run).
#
# The caller defines log, warn, die, run (run honours DRY) and DRY before sourcing. Everything
# that changes state goes through `run`/`post`, so --dry-run prints it instead.
#
# Environment (all optional; the overrides exist for tests):
#   OMLX_URL            engine base URL (default 127.0.0.1:8843)
#   ENGINE_PORTS        ports that must be free while the helper is stopped (default 8843)
#   LEGACY_DS4_URL      one-time migration: the DwarfStar launcher of a pre-removal app, if one still
#                       answers (default 127.0.0.1:8001); LEGACY_DS4_PORTS its ports (default 8001 8000);
#                       LEGACY_DS4_WAIT seconds to wait for its launchd job to go (default 20)
#   LAUNCHCTL           launchctl binary (only ever used to look at, or boot out, the helpers)
#   HELPER_APP          the Unsloth.app whose Contents/MacOS/unsloth-studio starts and stops the
#                       helpers (default: INSTALLED_APP when the caller sets it, else
#                       /Applications/Unsloth.app). After build-fork-mac.sh --install that is the NEW app.
#   HELPER_START_WAIT   seconds helpers_start waits for the labels to appear in launchd (default 20)
#   UNSLOTH_ENGINES_HOME  where desktop.json lives (default ~/.unsloth/engines): the app's
#                       engine_lifetime ("with_app" by default, or "always") and engines_enabled
#   PGREP               pgrep binary, to ask whether Unsloth (`unsloth-studio`) is running

OMLX_URL="${OMLX_URL:-http://127.0.0.1:8843}"
ENGINE_PORTS="${ENGINE_PORTS:-8843}"
# One-time migration for an install over a pre-removal app, which still ran the DwarfStar (ds4)
# launcher as a second helper. The new bundle has no ds4 launcher, so a left-over job would
# crash-loop once the bundle is swapped. Remove this block with the migration once no install
# can predate the removal.
LEGACY_DS4_URL="${LEGACY_DS4_URL:-http://127.0.0.1:8001}"
LEGACY_DS4_PORTS="${LEGACY_DS4_PORTS:-8001 8000}"
LEGACY_DS4_LABEL=ai.unsloth.studio.ds4
LEGACY_DS4_WAIT="${LEGACY_DS4_WAIT:-20}"
# 1 once the legacy launcher answered /admin/status in this run
LEGACY_DS4_SEEN=0
LAUNCHCTL="${LAUNCHCTL:-launchctl}"
HELPER_START_WAIT="${HELPER_START_WAIT:-20}"
PGREP="${PGREP:-pgrep}"
HELPER_DOMAIN="gui/$(id -u)"
HELPER_LABELS=(ai.unsloth.studio.omlx)
PORT_GATE_TIMEOUT="${PORT_GATE_TIMEOUT:-150}"
# Helpers this run booted out and still owes a restart.
STOPPED_HELPERS=()
# 1 after helpers_start deliberately left the helpers stopped (with_app mode, Unsloth closed).
# Read by the sourcing scripts.
# shellcheck disable=SC2034
HELPERS_LEFT_STOPPED=0
POST_CODE=""

urlencode() { python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }

# POST <url> <max seconds>; the HTTP status ends up in POST_CODE (000 when unreachable).
post() {
  if [ "$DRY" = 1 ]; then
    printf '[dry-run] curl -s -m %s -X POST %q\n' "$2" "$1"
    POST_CODE=200
    return 0
  fi
  POST_CODE="$(curl -s -o /dev/null -w '%{http_code}' -m "$2" -X POST "$1" || true)"
  [ -n "$POST_CODE" ] || POST_CODE=000
}

# stdin: oMLX /api/status JSON; stdout: line 1 = busy count, then one loaded model id per line
omlx_status_summary() {
  python3 -c '
import json, sys
s = json.load(sys.stdin)
print(int(s.get("active_requests", 0)) + int(s.get("waiting_requests", 0)) + int(s.get("models_loading", 0)))
print("\n".join(s.get("loaded_models", [])))'
}

# stdin: /v1/models JSON; stdout: sorted ids
roster_from_json() {
  python3 -c 'import json, sys; print("\n".join(sorted(m["id"] for m in json.load(sys.stdin)["data"])))'
}

fetch_roster() { curl -fsS -m 10 "$OMLX_URL/v1/models" | roster_from_json; }

# True when oMLX answers /api/status.
omlx_up() { curl -fsS -m 5 -o /dev/null "$OMLX_URL/api/status" 2>/dev/null; }

# Migration only: die unless the legacy ds4 launcher, when one answers, is idle (nothing in
# flight and not starting). An answer that cannot be read counts as busy.
legacy_ds4_require_idle() {
  local status verdict
  status="$(curl -fsS -m 3 "$LEGACY_DS4_URL/admin/status" 2>/dev/null)" || return 0
  LEGACY_DS4_SEEN=1
  verdict="$(printf '%s' "$status" | python3 -c '
import json, sys
try:
    s = json.load(sys.stdin)
    busy = int(s.get("in_flight") or 0) > 0 or bool(s.get("starting"))
except Exception:
    busy = True
print("busy" if busy else "idle")' 2>/dev/null || echo busy)"
  [ "$verdict" = idle ] || die "the legacy DwarfStar launcher at $LEGACY_DS4_URL is busy or starting; wait for it to go idle"
}

# die unless oMLX (and, for the migration, the legacy ds4 launcher) is idle: nothing generating,
# loading or starting.
engines_require_idle() {
  local status summary busy
  if status="$(curl -fsS -m 5 "$OMLX_URL/api/status" 2>/dev/null)"; then
    summary="$(printf '%s' "$status" | omlx_status_summary)"
    busy="$(printf '%s\n' "$summary" | head -n 1)"
    [ "$busy" = 0 ] || die "oMLX is busy (active + waiting + loading = $busy); wait for it to go idle"
  fi
  legacy_ds4_require_idle
}

# Unload every oMLX model gracefully and only while idle, so nothing holding
# weights is ever killed. A model that stays loaded (for example a pinned one) is a warning:
# the helper's SIGTERM releases it.
engines_quiesce() {
  engines_require_idle
  local status summary loaded id
  if status="$(curl -fsS -m 5 "$OMLX_URL/api/status" 2>/dev/null)"; then
    summary="$(printf '%s' "$status" | omlx_status_summary)"
    loaded="$(printf '%s\n' "$summary" | tail -n +2)"
    if [ -z "$loaded" ]; then
      log "oMLX: no models loaded"
    else
      while IFS= read -r id; do
        [ -n "$id" ] || continue
        log "oMLX: unloading $id"
        post "$OMLX_URL/v1/models/$(urlencode "$id")/unload" 90
        case "$POST_CODE" in
          200|202|400|404) ;;
          409) die "oMLX refused to unload $id (busy); try again when it is idle" ;;
          *) die "oMLX unload of $id failed with HTTP $POST_CODE" ;;
        esac
      done <<<"$loaded"
      if [ "$DRY" = 1 ]; then
        log "dry-run: would poll $OMLX_URL/api/status for up to 60 s until no model is loaded"
      else
        for _ in $(seq 1 60); do
          status="$(curl -fsS -m 5 "$OMLX_URL/api/status" 2>/dev/null || true)"
          [ -n "$status" ] || break
          loaded="$(printf '%s' "$status" | omlx_status_summary | tail -n +2)"
          [ -z "$loaded" ] && break
          sleep 1
        done
        if [ -n "$loaded" ]; then
          warn "still loaded after 60 s: $(printf '%s' "$loaded" | tr '\n' ' ')(released when oMLX exits)"
        else
          log "oMLX: all models unloaded"
        fi
      fi
    fi
  else
    log "oMLX: $OMLX_URL not reachable, nothing to unload"
  fi
  # Migration only: the legacy ds4 launcher was idle above; ask it to unload (best effort, the
  # helper unregister that follows stops it either way).
  if [ "$LEGACY_DS4_SEEN" = 1 ]; then
    log "legacy DwarfStar: stopping it while idle"
    post "$LEGACY_DS4_URL/admin/stop?if_idle=1" 30
    case "$POST_CODE" in
      200|202) ;;
      *) warn "legacy DwarfStar stop answered HTTP $POST_CODE; continuing (the helper unregister stops it)" ;;
    esac
  fi
}

# desktop.json is written by the app (Settings > Attached engines): engine_lifetime and
# engines_enabled. The scripts only read it, plus seed engines_enabled once (see below).
engines_desktop_file() { printf '%s/desktop.json' "${UNSLOTH_ENGINES_HOME:-$HOME/.unsloth/engines}"; }

# with_app (the default, also for a missing or unreadable file) or always.
engine_lifetime() {
  python3 - "$(engines_desktop_file)" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1])).get("engine_lifetime")
except Exception:
    value = None
print("always" if value == "always" else "with_app")
PY
}

# True while Unsloth itself runs. Asked of pgrep, never of the app binary.
app_running() { "$PGREP" -x unsloth-studio >/dev/null 2>&1; }

# The app remembers "engines enabled" in desktop.json because, in with_app mode, its quit
# unregisters the helpers and the registration alone no longer says what the user wants. An app
# from before that file leaves no record, so the first time these scripts take registered
# helpers down they write engines_enabled=true (only when the key is absent), or the app would
# not start the engines at its next launch.
seed_engines_enabled() {
  [ "$DRY" = 1 ] && return 0
  local seeded
  seeded="$(python3 - "$(engines_desktop_file)" <<'PY'
import json, os, sys
path = sys.argv[1]
try:
    with open(path) as handle:
        data = json.load(handle)
except FileNotFoundError:
    data = {}
except Exception:
    sys.exit(0)
if not isinstance(data, dict) or "engines_enabled" in data:
    sys.exit(0)
data["engines_enabled"] = True
os.makedirs(os.path.dirname(path), exist_ok=True)
tmp = "%s.tmp.%d" % (path, os.getpid())
with open(tmp, "w") as handle:
    json.dump(data, handle)
    handle.write("\n")
os.replace(tmp, path)
print("seeded")
PY
)" || return 0
  [ -z "$seeded" ] || log "recorded engines_enabled=true in $(engines_desktop_file) (the helpers are registered)"
  return 0
}

helper_loaded() { "$LAUNCHCTL" print "$HELPER_DOMAIN/$1" >/dev/null 2>&1; }

# The Unsloth.app that owns the helpers, resolved when used so a caller may set INSTALLED_APP
# after sourcing this file.
helper_app() { printf '%s' "${HELPER_APP:-${INSTALLED_APP:-/Applications/Unsloth.app}}"; }
helper_cli() { printf '%s/Contents/MacOS/unsloth-studio' "$(helper_app)"; }

# True when <binary> supports `--engine-helpers`: it must answer `--engine-helpers status` with
# exit 0 and a JSON object. An app without the flag would start its whole GUI for an unknown
# argument, so the binary is searched for the flag first and the probe is time-limited.
helper_cli_supported() { # <binary>
  [ -x "$1" ] || return 1
  grep -aqF -- "--engine-helpers" "$1" || return 1
  python3 - "$1" <<'PY'
import json, subprocess, sys
try:
    run = subprocess.run([sys.argv[1], "--engine-helpers", "status"], capture_output=True, text=True, timeout=20)
    data = json.loads(run.stdout)
except Exception:
    sys.exit(1)
sys.exit(0 if run.returncode == 0 and isinstance(data, dict) and "helpers" in data else 1)
PY
}

# Record <label> as owed a restart unless it already is. The debt accumulates: a helper stopped
# earlier and not yet restarted stays owed.
helper_owe() {
  local label="$1" owed
  for owed in ${STOPPED_HELPERS[@]+"${STOPPED_HELPERS[@]}"}; do
    [ "$owed" != "$label" ] || return 0
  done
  STOPPED_HELPERS+=("$label")
}

# Stop the helpers, and remember which ones (STOPPED_HELPERS, cleared by a successful
# helpers_start). When none is loaded, the old app still gets a best-effort unregister.
#   - When the app supports it: `unsloth-studio --engine-helpers unregister`, the SMAppService
#     call the Settings toggle makes. It also clears a legacy bundled helper registration.
#   - Otherwise (an app from before the CLI): `launchctl bootout` of each loaded helper.
# Everything is recorded as owed BEFORE the call that stops it: a call that fails part way may
# still have taken a helper down, and helpers_start leaves a helper that is still loaded alone.
# Under `set -e` a failing call ends the caller; what was recorded is still owed a restart.
# Never kickstart -k and never a signal of our own.
helpers_stop() {
  local label bin loaded=() any=0
  for label in "${HELPER_LABELS[@]}"; do
    if [ "$DRY" = 1 ] || helper_loaded "$label"; then
      loaded+=("$label")
      any=1
    else
      log "$label is not loaded, leaving it alone"
    fi
  done
  bin="$(helper_cli)"
  if [ "$any" = 0 ]; then
    # An old bundle may still have a registered legacy helper which the new bundle cannot
    # address. Its own CLI must get one best-effort chance to unregister it before the swap.
    if helper_cli_supported "$bin"; then
      log "unregister legacy helper registration: $bin --engine-helpers unregister"
      run "$bin" --engine-helpers unregister || true
    fi
    return 0
  fi
  seed_engines_enabled
  if helper_cli_supported "$bin"; then
    for label in "${HELPER_LABELS[@]}"; do helper_owe "$label"; done
    log "unregister the helpers: $bin --engine-helpers unregister"
    run "$bin" --engine-helpers unregister
  else
    log "$bin has no --engine-helpers; using launchctl bootout"
    for label in "${loaded[@]}"; do
      helper_owe "$label"
      log "bootout $label"
      run "$LAUNCHCTL" bootout "$HELPER_DOMAIN/$label"
    done
  fi
}

# Migration only (install over a pre-removal app): the legacy ds4 helper can outlive the old
# app's unregister. The new bundle has no ds4 launcher, so a job still in launchd would crash-loop
# after the swap. When it is still loaded, unregister once more through the old app's CLI (the
# app that registered it; bootout when that app has no CLI), wait up to LEGACY_DS4_WAIT s, and die
# when it is still there (the install's rollback then runs).
helpers_clear_legacy_ds4() {
  local bin start=$SECONDS
  helper_loaded "$LEGACY_DS4_LABEL" || return 0
  warn "the legacy $LEGACY_DS4_LABEL job is still loaded; unregistering it once more"
  bin="$(helper_cli)"
  if helper_cli_supported "$bin"; then
    run "$bin" --engine-helpers unregister || true
  else
    run "$LAUNCHCTL" bootout "$HELPER_DOMAIN/$LEGACY_DS4_LABEL" || true
  fi
  [ "$DRY" = 0 ] || return 0
  while helper_loaded "$LEGACY_DS4_LABEL"; do
    [ $((SECONDS - start)) -lt "$LEGACY_DS4_WAIT" ] || die "the legacy $LEGACY_DS4_LABEL job is still loaded after ${LEGACY_DS4_WAIT} s; the new bundle has no ds4 launcher, so it would crash-loop. Not swapping the app"
    sleep "${HEALTH_POLL:-1}"
  done
  log "the legacy $LEGACY_DS4_LABEL job is gone"
}

# True once every label in STOPPED_HELPERS is loaded, waiting up to HELPER_START_WAIT seconds.
owed_helpers_loaded() {
  local start=$SECONDS label up
  while :; do
    up=1
    for label in ${STOPPED_HELPERS[@]+"${STOPPED_HELPERS[@]}"}; do
      helper_loaded "$label" || up=0
    done
    [ "$up" = 1 ] && return 0
    [ $((SECONDS - start)) -lt "$HELPER_START_WAIT" ] || return 1
    sleep "${HEALTH_POLL:-1}"
  done
}

# Bring back the helpers helpers_stop took down with `unsloth-studio --engine-helpers register`
# from the app at helper_app (after --install, the NEW app). macOS refuses `launchctl bootstrap`
# and `kickstart` for a helper bundled for SMAppService, so those are never tried. A helper that
# was only booted out (an app without the CLI) can leave its registration in place, which makes
# `register` a no-op; one `restart` (unregister, wait, register) covers that. Returns 1 when a
# helper could not be brought back, with the manual fallback in the warning (Settings > Attached
# engines > Engines enabled, off and on).
helpers_start() {
  local label bin all_up=1
  HELPERS_LEFT_STOPPED=0
  [ "${#STOPPED_HELPERS[@]}" -gt 0 ] || return 0
  if [ "$DRY" = 0 ]; then
    for label in "${STOPPED_HELPERS[@]}"; do helper_loaded "$label" || all_up=0; done
    if [ "$all_up" = 1 ]; then
      log "the helpers are already loaded"
      STOPPED_HELPERS=()
      return 0
    fi
  fi
  # In with_app mode the helpers run only while Unsloth does: the app registers them at its
  # next launch, so with the app closed they stay stopped. Asked before the CLI probe, which
  # spawns the same binary name.
  if [ "$(engine_lifetime)" = with_app ] && ! app_running; then
    log "engine lifetime is with_app and Unsloth is not running: leaving the helpers stopped (the app registers them at its next launch)"
    STOPPED_HELPERS=()
    # shellcheck disable=SC2034
    HELPERS_LEFT_STOPPED=1
    return 0
  fi
  bin="$(helper_cli)"
  if ! helper_cli_supported "$bin"; then
    warn "$bin has no --engine-helpers, so the helpers cannot be restarted from here. Turn Unsloth > Settings > Attached engines > Engines enabled off and on."
    return 1
  fi
  log "register the helpers: $bin --engine-helpers register"
  if run "$bin" --engine-helpers register; then
    if [ "$DRY" = 1 ] || owed_helpers_loaded; then
      STOPPED_HELPERS=()
      return 0
    fi
    warn "register did not bring the helpers back; trying restart"
  else
    warn "register failed; trying restart"
  fi
  if run "$bin" --engine-helpers restart && owed_helpers_loaded; then
    STOPPED_HELPERS=()
    return 0
  fi
  warn "the helpers did not restart. Turn Unsloth > Settings > Attached engines > Engines enabled off and on."
  return 1
}

# The ports the gate watches: the engine ports, plus the legacy ds4 ones only when that launcher
# was seen in this run (migration).
gate_ports() {
  if [ "$LEGACY_DS4_SEEN" = 1 ]; then printf '%s %s' "$ENGINE_PORTS" "$LEGACY_DS4_PORTS"; else printf '%s' "$ENGINE_PORTS"; fi
}

port_owners() {
  local port
  for port in $(gate_ports); do
    lsof -nP "-iTCP:$port" -sTCP:LISTEN 2>/dev/null || true
  done
}

# Wait until none of the engine ports listens. Never kills the owner.
wait_ports_free() {
  log "port-free gate: ports $(gate_ports) must stop listening (${PORT_GATE_TIMEOUT} s)"
  if [ "$DRY" = 1 ]; then
    printf '[dry-run] poll lsof for ports %s every 2 s until empty (timeout %s s)\n' "$(gate_ports)" "$PORT_GATE_TIMEOUT"
    return 0
  fi
  local start=$SECONDS owners
  while :; do
    owners="$(port_owners)"
    [ -z "$owners" ] && break
    if [ $((SECONDS - start)) -ge "$PORT_GATE_TIMEOUT" ]; then
      printf '%s\n' "$owners" >&2
      die "ports still held after ${PORT_GATE_TIMEOUT} s (owners above). Not killing anything"
    fi
    sleep 2
  done
  log "ports are free"
}

# wait_engines_healthy <timeout s> <expect oMLX 0|1>
wait_engines_healthy() {
  local timeout="$1" want_omlx="$2" start=$SECONDS last=-15 o
  while :; do
    o=1
    if [ "$want_omlx" = 1 ]; then curl -fsS -m 3 -o /dev/null "$OMLX_URL/v1/models" 2>/dev/null || o=0; fi
    [ "$o" = 1 ] && return 0
    [ $((SECONDS - start)) -lt "$timeout" ] || return 1
    if [ $((SECONDS - last - start)) -ge 15 ]; then
      last=$((SECONDS - start))
      log "waiting (${last} s): oMLX $([ "$o" = 1 ] && echo up || echo down)"
    fi
    sleep "${HEALTH_POLL:-3}"
  done
}

# Print the launch failure the wrappers recorded for an engine, if any.
show_launch_failure() { # <name: omlx>
  local file="${UNSLOTH_ENGINES_HOME:-$HOME/.unsloth/engines}/$1.fail"
  [ -f "$file" ] && warn "$1 launch failure: $(cat "$file")"
  return 0
}
