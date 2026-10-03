# shellcheck shell=bash
# Shared helpers for update-engines-mac.sh and build-fork-mac.sh --install (sourced, never run).
#
# The caller defines log, warn, die, run (run honours DRY) and DRY before sourcing. Everything
# that changes state goes through `run`/`post`, so --dry-run prints it instead.
#
# Environment (all optional; the overrides exist for tests):
#   OMLX_URL, DS4_URL   engine base URLs (default 127.0.0.1:8843 and :8001)
#   ENGINE_PORTS        ports that must be free while the helpers are stopped (default 8843 8001 8000)
#   LAUNCHCTL           launchctl binary
#   HELPER_PLIST_DIR    where the bundled helper plists live
#                       (default /Applications/Unsloth.app/Contents/Library/LaunchAgents)

OMLX_URL="${OMLX_URL:-http://127.0.0.1:8843}"
DS4_URL="${DS4_URL:-http://127.0.0.1:8001}"
ENGINE_PORTS="${ENGINE_PORTS:-8843 8001 8000}"
LAUNCHCTL="${LAUNCHCTL:-launchctl}"
HELPER_PLIST_DIR="${HELPER_PLIST_DIR:-/Applications/Unsloth.app/Contents/Library/LaunchAgents}"
HELPER_DOMAIN="gui/$(id -u)"
HELPER_LABELS=(ai.unsloth.studio.omlx ai.unsloth.studio.ds4)
PORT_GATE_TIMEOUT="${PORT_GATE_TIMEOUT:-150}"
# Helpers this run booted out and still owes a restart.
STOPPED_HELPERS=()
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

# stdin: ds4 launcher /admin/status JSON; stdout: <loaded 0|1> <in_flight> <starting 0|1>
ds4_status_summary() {
  python3 -c '
import json, sys
s = json.load(sys.stdin)
print(int(bool(s.get("loaded"))), int(s.get("in_flight") or 0), int(bool(s.get("starting"))))'
}

# stdin: /v1/models JSON; stdout: sorted ids
roster_from_json() {
  python3 -c 'import json, sys; print("\n".join(sorted(m["id"] for m in json.load(sys.stdin)["data"])))'
}

fetch_roster() { curl -fsS -m 10 "$OMLX_URL/v1/models" | roster_from_json; }

# True when oMLX answers /api/status.
omlx_up() { curl -fsS -m 5 -o /dev/null "$OMLX_URL/api/status" 2>/dev/null; }
ds4_up() { curl -fsS -m 5 -o /dev/null "$DS4_URL/admin/status" 2>/dev/null; }

# die unless both engines are idle: nothing generating, loading or starting.
engines_require_idle() {
  local status summary busy ds4 loaded in_flight starting
  if status="$(curl -fsS -m 5 "$OMLX_URL/api/status" 2>/dev/null)"; then
    summary="$(printf '%s' "$status" | omlx_status_summary)"
    busy="$(printf '%s\n' "$summary" | head -n 1)"
    [ "$busy" = 0 ] || die "oMLX is busy (active + waiting + loading = $busy); wait for it to go idle"
  fi
  if ds4="$(curl -fsS -m 5 "$DS4_URL/admin/status" 2>/dev/null)"; then
    read -r loaded in_flight starting <<<"$(printf '%s' "$ds4" | ds4_status_summary)"
    [ "$in_flight" = 0 ] || die "ds4 has $in_flight request(s) in flight; wait for it to go idle"
    [ "$starting" = 0 ] || die "ds4 is starting a model; wait for it to finish"
  fi
}

# Unload every oMLX model and stop ds4, gracefully and only while idle, so nothing holding
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
  local ds4 loaded_flag in_flight starting
  if ds4="$(curl -fsS -m 5 "$DS4_URL/admin/status" 2>/dev/null)"; then
    read -r loaded_flag in_flight starting <<<"$(printf '%s' "$ds4" | ds4_status_summary)"
    if [ "$loaded_flag" = 1 ]; then
      log "ds4: stopping (POST /admin/stop?if_idle=1)"
      post "$DS4_URL/admin/stop?if_idle=1" 150
      case "$POST_CODE" in
        200|202|204) log "ds4: stopped" ;;
        409) die "ds4 became busy; try again when it is idle" ;;
        *) die "ds4 /admin/stop failed with HTTP $POST_CODE" ;;
      esac
    else
      log "ds4: not loaded, nothing to stop"
    fi
  else
    log "ds4: $DS4_URL not reachable, nothing to stop"
  fi
}

helper_loaded() { "$LAUNCHCTL" print "$HELPER_DOMAIN/$1" >/dev/null 2>&1; }

# Boot out every loaded helper (SIGTERM, the engines unload gracefully) and remember which ones
# (STOPPED_HELPERS, cleared by a successful helpers_start).
# Never kickstart -k and never a signal of our own.
helpers_stop() {
  local label seen owed
  for label in "${HELPER_LABELS[@]}"; do
    if [ "$DRY" = 1 ] || helper_loaded "$label"; then
      log "bootout $label"
      run "$LAUNCHCTL" bootout "$HELPER_DOMAIN/$label"
      # the restart debt accumulates: a helper booted out earlier and not yet restarted stays owed
      seen=0
      for owed in ${STOPPED_HELPERS[@]+"${STOPPED_HELPERS[@]}"}; do
        [ "$owed" != "$label" ] || seen=1
      done
      [ "$seen" = 1 ] || STOPPED_HELPERS+=("$label")
    else
      log "$label is not loaded, leaving it alone"
    fi
  done
}

# Bring back the helpers helpers_stop booted out: bootstrap from the bundled plist, else kickstart
# the label if launchd still has it. Returns 1 when one could not be restarted (the manual
# fallback is Settings > API Keys > Background engines, off and on).
helpers_start() {
  local label plist failed=0
  for label in ${STOPPED_HELPERS[@]+"${STOPPED_HELPERS[@]}"}; do
    plist="$HELPER_PLIST_DIR/$label.plist"
    if [ "$DRY" = 0 ] && helper_loaded "$label"; then
      log "$label is already loaded"
      continue
    fi
    log "bootstrap $label"
    if run "$LAUNCHCTL" bootstrap "$HELPER_DOMAIN" "$plist"; then
      continue
    fi
    warn "bootstrap of $plist failed; trying kickstart"
    if ! run "$LAUNCHCTL" kickstart "$HELPER_DOMAIN/$label"; then
      warn "$label did not restart. Turn Unsloth > Settings > API Keys > Background engines off and on."
      failed=1
    fi
  done
  [ "$failed" = 0 ] && STOPPED_HELPERS=()
  return "$failed"
}

port_owners() {
  local port
  for port in $ENGINE_PORTS; do
    lsof -nP "-iTCP:$port" -sTCP:LISTEN 2>/dev/null || true
  done
}

# Wait until none of the engine ports listens. Never kills the owner.
wait_ports_free() {
  log "port-free gate: ports $ENGINE_PORTS must stop listening (${PORT_GATE_TIMEOUT} s)"
  if [ "$DRY" = 1 ]; then
    printf '[dry-run] poll lsof for ports %s every 2 s until empty (timeout %s s)\n' "$ENGINE_PORTS" "$PORT_GATE_TIMEOUT"
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

# wait_engines_healthy <timeout s> <expect oMLX 0|1> <expect ds4 0|1>
wait_engines_healthy() {
  local timeout="$1" want_omlx="$2" want_ds4="$3" start=$SECONDS last=-15 o d
  while :; do
    o=1; d=1
    if [ "$want_omlx" = 1 ]; then curl -fsS -m 3 -o /dev/null "$OMLX_URL/v1/models" 2>/dev/null || o=0; fi
    if [ "$want_ds4" = 1 ]; then curl -fsS -m 3 -o /dev/null "$DS4_URL/admin/status" 2>/dev/null || d=0; fi
    [ "$o" = 1 ] && [ "$d" = 1 ] && return 0
    [ $((SECONDS - start)) -lt "$timeout" ] || return 1
    if [ $((SECONDS - last - start)) -ge 15 ]; then
      last=$((SECONDS - start))
      log "waiting (${last} s): oMLX $([ "$o" = 1 ] && echo up || echo down), ds4 $([ "$d" = 1 ] && echo up || echo down)"
    fi
    sleep "${HEALTH_POLL:-3}"
  done
}

# Print the launch failure the wrappers recorded for an engine, if any.
show_launch_failure() { # <name: omlx|ds4>
  local file="${UNSLOTH_ENGINES_HOME:-$HOME/.unsloth/engines}/$1.fail"
  [ -f "$file" ] && warn "$1 launch failure: $(cat "$file")"
  return 0
}
