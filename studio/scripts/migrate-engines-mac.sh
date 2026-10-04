#!/bin/bash
# Live cutover from the old oMLX / EngineBar / ds4 launchd setup to the fork's Unsloth.app,
# which bundles both engines as SMAppService helpers. Also the matching rollback.
#
#   migrate-engines-mac.sh [--dry-run] [--yes]                      cutover
#   migrate-engines-mac.sh --rollback [<state dir>] [--dry-run] [--yes]
#
# cutover (state is kept in ~/.unsloth/engines/migration/<timestamp>/):
#   1. preflight: dist/Unsloth.app passes codesign, ~/.unsloth/engines/.provisioned exists,
#      the oMLX roster (/v1/models ids) is recorded
#   2. back up ~/.unsloth/studio (ditto, size reported first) and the old plists
#   3. unload every loaded oMLX model, stop ds4 (POST :8001/admin/stop, up to 135 s)
#      -- asks for confirmation first unless --yes
#   4. bootout dev.sillymoo.{omlx-tuned,enginebar,ds4-ondemand} (and ds4-menubar if present),
#      move their plists to *.retired
#   5. quit oMLX.app and EngineBar.app via osascript (never killed)
#   6. port-free gate: :8843, :8001, :8000 must stop listening (120 s)
#   7. build-fork-mac.sh --install
#   8. open /Applications/Unsloth.app; the user enables Settings > Attached engines > Background
#      engines and approves the Login Items prompt (the app is not notarized); poll up to
#      10 minutes for :8843 and :8001
#   9. verify: roster matches the pre-migration ids, ds4 runs the bundled binary, both
#      ai.unsloth.studio.* agents are running
#
# --rollback restores the old setup from the state dir (default: the latest). Engines are
# unloaded and stopped first so no process holding weights is ever killed, then the helpers
# are disabled, Unsloth quits, the port gate runs, /Applications/Unsloth.app is restored from
# ~/Applications/Unsloth-upstream-0.1.815.app.bak, ~/.unsloth/studio is restored from
# <state>/studio.bak (the current copy is moved aside, never deleted), oMLX.app and EngineBar.app
# are moved back from ~/.unsloth/engines/migration/retired-apps/ (to /Applications and
# ~/Applications; the old oMLX agent opens oMLX.app, so this comes first), and the old plists are
# restored and bootstrapped.
#
# --dry-run prints every mutating command and still runs the read-only checks.
# Environment: DIST_DIR (default ~/Homelab/unsloth/dist), APPS_DIR (default /Applications)
set -euo pipefail

MODE=cutover DRY=0 YES=0 STATE_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rollback)
      MODE=rollback
      if [ $# -gt 1 ] && [ "${2#-}" = "$2" ]; then STATE_ARG="$2"; shift; fi ;;
    --dry-run) DRY=1 ;;
    --yes|-y) YES=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="${DIST_DIR:-$HOME/Homelab/unsloth/dist}"
DIST_APP="$DIST_DIR/Unsloth.app"
APPS_DIR="${APPS_DIR:-/Applications}"
INSTALLED_APP="$APPS_DIR/Unsloth.app"
OMLX_APP="$APPS_DIR/oMLX.app"
ENGINEBAR_APP="$HOME/Applications/EngineBar.app"
APP_BACKUP="$HOME/Applications/Unsloth-upstream-0.1.815.app.bak"
ENGINES_HOME="$HOME/.unsloth/engines"
MIG_ROOT="$ENGINES_HOME/migration"
RETIRED_APPS="$MIG_ROOT/retired-apps"
STUDIO_DIR="$HOME/.unsloth/studio"
LA_DIR="$HOME/Library/LaunchAgents"
RUNTIME_PY="$HOME/Homelab/omlx-stack/scripts/omlx_tuned_runtime.py"
DOMAIN="gui/$(id -u)"
OMLX_URL="http://127.0.0.1:8843"
DS4_URL="http://127.0.0.1:8001"
OLD_LABELS=(dev.sillymoo.omlx-tuned dev.sillymoo.enginebar dev.sillymoo.ds4-ondemand)
OPTIONAL_LABEL="dev.sillymoo.ds4-menubar"
NEW_OMLX="ai.unsloth.studio.omlx"
NEW_DS4="ai.unsloth.studio.ds4"
STUDIO_BUNDLE_ID="ai.unsloth.studio"
GATE_TIMEOUT=120
BOOT_TIMEOUT=600

STATE=""        # state dir, set per mode
HINT=""         # extra line appended to die messages
TS="$(date +%Y%m%d-%H%M%S)"
POST_CODE=""

log() { printf '[engine-migrate] %s\n' "$*"; }
warn() { printf '[engine-migrate] warning: %s\n' "$*" >&2; }
die() {
  printf '[engine-migrate] error: %s\n' "$*" >&2
  [ -z "$HINT" ] || printf '[engine-migrate] %s\n' "$HINT" >&2
  exit 1
}
run() {
  if [ "$DRY" = 1 ]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

[ "$(uname -s)" = "Darwin" ] || die "macOS only"
command -v python3 >/dev/null || die "python3 not found"

# ---------------------------------------------------------------------------------------------
# helpers

confirm() { # <prompt>
  [ "$YES" = 1 ] && return 0
  if [ "$DRY" = 1 ]; then log "dry-run: would ask: $1"; return 0; fi
  [ -t 0 ] || die "no terminal for the confirmation; re-run with --yes"
  local answer
  read -r -p "[engine-migrate] $1 [y/N] " answer
  case "$answer" in y|Y|yes|YES) return 0 ;; *) die "aborted by the user" ;; esac
}

label_loaded() { launchctl print "$DOMAIN/$1" >/dev/null 2>&1; }
label_running() { grep -q 'state = running' <<<"$(launchctl print "$DOMAIN/$1" 2>/dev/null || true)"; }
app_running() { pgrep -x "$1" >/dev/null 2>&1; }

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

urlencode() { python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }

# stdin: /v1/models JSON; stdout: sorted ids
roster_from_json() {
  python3 -c 'import json, sys; print("\n".join(sorted(m["id"] for m in json.load(sys.stdin)["data"])))'
}

fetch_roster() { curl -fsS -m 10 "$OMLX_URL/v1/models" | roster_from_json; }

# stdin: oMLX /api/status JSON; stdout: line 1 = busy count, then one loaded model id per line
omlx_status_summary() {
  python3 -c '
import json, sys
s = json.load(sys.stdin)
print(int(s.get("active_requests", 0)) + int(s.get("waiting_requests", 0)) + int(s.get("models_loading", 0)))
print("\n".join(s.get("loaded_models", [])))'
}

# stdin: ds4 launcher /admin/status JSON; stdout: <loaded 0|1> <in_flight>
ds4_status_summary() {
  python3 -c '
import json, sys
s = json.load(sys.stdin)
print(int(bool(s.get("loaded"))), int(s.get("in_flight") or 0))'
}

# Unload every loaded oMLX model and stop ds4. Quiescence is the hard gate; a model that stays
# loaded (for example a pinned one) is only a warning because the engines are then stopped
# gracefully, never killed.
stop_engines() {
  local status summary busy loaded id encoded i
  if status="$(curl -fsS -m 5 "$OMLX_URL/api/status" 2>/dev/null)"; then
    summary="$(printf '%s' "$status" | omlx_status_summary)"
    busy="$(printf '%s\n' "$summary" | head -n 1)"
    loaded="$(printf '%s\n' "$summary" | tail -n +2)"
    [ "$busy" = 0 ] || die "oMLX is busy (active + waiting + loading = $busy); wait for it to go idle"
    if [ -z "$loaded" ]; then
      log "oMLX: no models loaded"
    else
      while IFS= read -r id; do
        [ -n "$id" ] || continue
        encoded="$(urlencode "$id")"
        log "oMLX: unloading $id"
        post "$OMLX_URL/v1/models/$encoded/unload" 60
        case "$POST_CODE" in
          200|202|400|404) ;;
          *) die "oMLX unload of $id failed with HTTP $POST_CODE" ;;
        esac
      done <<<"$loaded"
      if [ "$DRY" = 1 ]; then
        log "dry-run: would poll $OMLX_URL/api/status for up to 60 s until no model is loaded"
      else
        for i in $(seq 1 60); do
          status="$(curl -fsS -m 5 "$OMLX_URL/api/status" 2>/dev/null || true)"
          [ -n "$status" ] || break
          loaded="$(printf '%s' "$status" | omlx_status_summary | tail -n +2)"
          [ -n "$loaded" ] || break
          sleep 1
        done
        if [ -n "$loaded" ]; then
          warn "still loaded after 60 s: $(printf '%s' "$loaded" | tr '\n' ' ')(they are released when oMLX quits)"
        else
          log "oMLX: all models unloaded"
        fi
      fi
    fi
  else
    log "oMLX: $OMLX_URL not reachable, nothing to unload"
  fi

  local ds4
  if ds4="$(curl -fsS -m 5 "$DS4_URL/admin/status" 2>/dev/null)"; then
    local loaded_flag in_flight
    read -r loaded_flag in_flight <<<"$(printf '%s' "$ds4" | ds4_status_summary)"
    [ "$in_flight" = 0 ] || die "ds4 has $in_flight request(s) in flight; wait for it to go idle"
    if [ "$loaded_flag" = 1 ]; then
      log "ds4: stopping (POST /admin/stop, may block up to 135 s)"
      post "$DS4_URL/admin/stop" 150
      case "$POST_CODE" in
        200|202|204) log "ds4: stopped" ;;
        *) die "ds4 /admin/stop failed with HTTP $POST_CODE" ;;
      esac
    else
      log "ds4: not loaded, nothing to stop"
    fi
  else
    log "ds4: $DS4_URL not reachable, nothing to stop"
  fi
}

port_owners() { lsof -nP -iTCP:8843 -iTCP:8001 -iTCP:8000 -sTCP:LISTEN 2>/dev/null || true; }

port_gate() {
  log "port-free gate: :8843, :8001 and :8000 must stop listening (${GATE_TIMEOUT} s)"
  if [ "$DRY" = 1 ]; then
    printf '[dry-run] poll lsof -nP -iTCP:8843 -iTCP:8001 -iTCP:8000 -sTCP:LISTEN every 2 s until empty (timeout %s s)\n' "$GATE_TIMEOUT"
    local owners; owners="$(port_owners)"
    if [ -n "$owners" ]; then log "dry-run: current listeners:"; printf '%s\n' "$owners" | sed 's/^/    /'; else log "dry-run: all three ports are free"; fi
    return 0
  fi
  local start=$SECONDS owners
  while :; do
    owners="$(port_owners)"
    [ -z "$owners" ] && break
    if [ $((SECONDS - start)) -ge "$GATE_TIMEOUT" ]; then
      printf '%s\n' "$owners" >&2
      die "ports still held after ${GATE_TIMEOUT} s (owners above). Not killing anything: stop the owner gracefully, then re-run"
    fi
    sleep 2
  done
  log "ports are free"
}

# quit_app <process name> <bundle id> <label>; a graceful quit only, and only when running
# (telling an app to quit would launch it).
quit_app() {
  if app_running "$1" || [ "$DRY" = 1 ]; then
    log "quitting $3"
    run osascript -e "tell application id \"$2\" to quit"
    if [ "$DRY" = 0 ]; then
      local i
      for i in $(seq 1 60); do app_running "$1" || break; sleep 1; done
      app_running "$1" && die "$3 did not quit within 60 s; not forcing it"
    fi
  else
    log "$3 is not running"
  fi
  return 0
}

bundle_id() { # <app path> <fallback>
  defaults read "$1/Contents/Info.plist" CFBundleIdentifier 2>/dev/null || printf '%s' "$2"
}

wait_for_endpoints() { # <timeout s>
  local start=$SECONDS last=-15 omlx_ok ds4_ok
  while :; do
    omlx_ok=0; ds4_ok=0
    curl -fsS -m 3 -o /dev/null "$OMLX_URL/v1/models" 2>/dev/null && omlx_ok=1
    curl -fsS -m 3 -o /dev/null "$DS4_URL/admin/status" 2>/dev/null && ds4_ok=1
    [ "$omlx_ok" = 1 ] && [ "$ds4_ok" = 1 ] && return 0
    if [ $((SECONDS - start)) -ge "$1" ]; then
      return 1
    fi
    if [ $((SECONDS - last - start)) -ge 15 ]; then
      last=$((SECONDS - start))
      log "waiting ($last s): :8843 /v1/models $([ "$omlx_ok" = 1 ] && echo up || echo down), :8001 /admin/status $([ "$ds4_ok" = 1 ] && echo up || echo down)"
    fi
    sleep 3
  done
}

# ---------------------------------------------------------------------------------------------
cutover() {
  local rollback_cmd
  rollback_cmd="$0 --rollback"
  log "mode: cutover$([ "$DRY" = 1 ] && echo ' (dry-run)')"

  # 1/9 preflight --------------------------------------------------------------------------
  log "1/9 preflight"
  [ -d "$DIST_APP" ] || die "$DIST_APP not found; run build-fork-mac.sh first"
  codesign --verify --deep --strict "$DIST_APP" 2>/dev/null || die "codesign verification failed for $DIST_APP"
  log "$DIST_APP passes codesign --verify --deep --strict"
  [ -e "$ENGINES_HOME/.provisioned" ] || die "$ENGINES_HOME/.provisioned missing; run build-engines-mac.sh first"
  log "engines are provisioned ($ENGINES_HOME/.provisioned)"

  STATE="$MIG_ROOT/$TS"
  local roster
  if roster="$(fetch_roster 2>/dev/null)" && [ -n "$roster" ]; then
    :
  elif [ "$DRY" = 1 ]; then
    roster=""
    warn "dry-run: could not read $OMLX_URL/v1/models"
  else
    die "could not read the oMLX roster from $OMLX_URL/v1/models; is oMLX running? Nothing was changed"
  fi
  log "state dir: $STATE"
  run mkdir -p "$STATE"
  if [ "$DRY" = 1 ]; then
    printf '[dry-run] write %s/pre-roster.txt\n' "$STATE"
  else
    printf '%s\n' "$roster" >"$STATE/pre-roster.txt"
  fi
  log "pre-migration oMLX roster ($(printf '%s\n' "$roster" | grep -c . || true) ids):"
  printf '%s\n' "$roster" | sed 's/^/    /'

  # 2/9 back up ----------------------------------------------------------------------------
  log "2/9 back up"
  if [ -d "$STUDIO_DIR" ]; then
    local size_kb avail_kb
    size_kb="$(du -sk "$STUDIO_DIR" 2>/dev/null | cut -f1)"
    avail_kb="$(df -k "$HOME" | awk 'NR==2 {print $4}')"
    log "$STUDIO_DIR is $(du -sh "$STUDIO_DIR" 2>/dev/null | cut -f1) ($size_kb KiB); $((avail_kb / 1024)) MiB free"
    [ "$avail_kb" -gt $((size_kb * 2)) ] || die "not enough free disk space for the backup"
    run ditto "$STUDIO_DIR" "$STATE/studio.bak"
  else
    warn "$STUDIO_DIR does not exist; nothing to back up"
  fi
  local label plist
  for label in "${OLD_LABELS[@]}" "$OPTIONAL_LABEL"; do
    plist="$LA_DIR/$label.plist"
    if [ -f "$plist" ]; then
      run cp -p "$plist" "$STATE/$label.plist"
    elif [ "$label" != "$OPTIONAL_LABEL" ]; then
      warn "$plist not found"
    fi
  done

  # 3/9 stop engines -----------------------------------------------------------------------
  log "3/9 unload oMLX models and stop ds4"
  confirm "This unloads all oMLX models and stops DwarfStar (ds4), then replaces the old engine agents. Continue?"
  HINT="to roll back: $rollback_cmd $STATE"
  stop_engines

  # 4/9 bootout old agents -----------------------------------------------------------------
  log "4/9 bootout the old agents and retire their plists"
  for label in "${OLD_LABELS[@]}" "$OPTIONAL_LABEL"; do
    plist="$LA_DIR/$label.plist"
    if label_loaded "$label"; then
      log "bootout $label"
      run launchctl bootout "$DOMAIN/$label"
    else
      log "$label is not loaded in launchd"
    fi
    if [ -f "$plist" ]; then
      run mv "$plist" "$plist.retired"
    fi
  done

  # 5/9 quit apps --------------------------------------------------------------------------
  log "5/9 quit oMLX.app and EngineBar.app"
  quit_app oMLX "$(bundle_id "$OMLX_APP" app.omlx)" "oMLX.app"
  quit_app EngineBar "$(bundle_id "$HOME/Applications/EngineBar.app" dev.sillymoo.enginebar)" "EngineBar.app"

  # 6/9 port gate --------------------------------------------------------------------------
  log "6/9 port-free gate"
  port_gate

  # 7/9 install ----------------------------------------------------------------------------
  log "7/9 install the fork app"
  if [ "$DRY" = 1 ]; then
    "$SCRIPT_DIR/build-fork-mac.sh" --install --dry-run
  else
    "$SCRIPT_DIR/build-fork-mac.sh" --install
  fi

  # 8/9 open the app -----------------------------------------------------------------------
  log "8/9 open the new Unsloth.app"
  run open -a "$INSTALLED_APP"
  log "ACTION NEEDED in Unsloth: Settings > Attached engines > Engines enabled: turn it on."
  log "ACTION NEEDED in System Settings > General > Login Items: approve the Unsloth items"
  log "(the app is not notarized, so macOS asks for approval)."
  if [ "$DRY" = 1 ]; then
    log "dry-run: would poll $OMLX_URL/v1/models and $DS4_URL/admin/status for up to ${BOOT_TIMEOUT} s"
  else
    wait_for_endpoints "$BOOT_TIMEOUT" || die "engines did not answer within ${BOOT_TIMEOUT} s (:8843 /v1/models, :8001 /admin/status)"
    log "both endpoints answer"
  fi

  # 9/9 verify -----------------------------------------------------------------------------
  log "9/9 verify"
  if [ "$DRY" = 1 ]; then
    log "dry-run: would compare the /v1/models ids with $STATE/pre-roster.txt and print the diff"
    log "dry-run: would check /admin/status config.ds4_binary is under $INSTALLED_APP/Contents/Resources/engines/ds4/"
    log "dry-run: would check launchctl print $DOMAIN/$NEW_OMLX and $DOMAIN/$NEW_DS4 show state = running"
    log "dry-run complete"
    return 0
  fi
  local post
  post="$(fetch_roster)" || die "could not read the roster after migration"
  printf '%s\n' "$post" >"$STATE/post-roster.txt"
  if diff -u "$STATE/pre-roster.txt" "$STATE/post-roster.txt"; then
    log "roster matches the pre-migration ids"
  else
    die "roster differs from the pre-migration ids (diff above)"
  fi
  local binary expected
  expected="$INSTALLED_APP/Contents/Resources/engines/ds4/"
  binary="$(curl -fsS -m 5 "$DS4_URL/admin/status" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("config", {}).get("ds4_binary", ""))')"
  log "ds4 binary: $binary"
  case "$binary" in "$expected"*) ;; *) die "ds4 binary is not the bundled one (expected under $expected)" ;; esac
  for label in "$NEW_OMLX" "$NEW_DS4"; do
    launchctl print "$DOMAIN/$label" | grep -E 'state =|pid =' | sed 's/^/    /' || true
    label_running "$label" || die "$label is not running"
  done
  HINT=""
  log "cutover complete. State and backups: $STATE"
  log "to roll back: $rollback_cmd $STATE"
}

# Cutover state dirs are named by timestamp; retired-apps/ and anything else beside them is not one.
latest_state_dir() {
  find "$MIG_ROOT" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sort | tail -n 1 || true
}

# oMLX.app and EngineBar.app were moved to retired-apps/ after the cutover; the old launchd agents
# open them, so they go back before anything is bootstrapped. An app already in place is kept.
restore_retired_apps() {
  local name dest src
  for name in oMLX EngineBar; do
    if [ "$name" = oMLX ]; then dest="$OMLX_APP"; else dest="$ENGINEBAR_APP"; fi
    src="$RETIRED_APPS/$name.app"
    if [ -e "$dest" ]; then
      log "$dest is already in place"
    elif [ -d "$src" ]; then
      run mkdir -p "$(dirname "$dest")"
      run mv "$src" "$dest"
      log "restored $dest from $src"
    else
      warn "$dest is missing and there is no $src to restore it from"
    fi
  done
}

# ---------------------------------------------------------------------------------------------
rollback() {
  log "mode: rollback$([ "$DRY" = 1 ] && echo ' (dry-run)')"
  if [ -n "$STATE_ARG" ]; then
    STATE="${STATE_ARG%/}"
    [ -d "$STATE" ] || die "state dir not found: $STATE"
  else
    STATE="$(latest_state_dir)"
    STATE="${STATE%/}"
    if [ -z "$STATE" ]; then
      [ "$DRY" = 1 ] || die "no state dir under $MIG_ROOT; pass one explicitly"
      STATE="$MIG_ROOT/<latest>"
      warn "dry-run: no state dir under $MIG_ROOT yet, using the placeholder $STATE"
    fi
  fi
  log "state dir: $STATE"
  [ -d "$STATE" ] || [ "$DRY" = 1 ] || die "state dir not found: $STATE"
  [ -d "$STATE/studio.bak" ] || warn "$STATE/studio.bak not found; ~/.unsloth/studio will not be restored"
  [ -d "$APP_BACKUP" ] || warn "$APP_BACKUP not found; /Applications/Unsloth.app will not be restored"
  confirm "Rollback: stop the bundled engines and restore the old Unsloth, Studio and launchd agents from $STATE. Continue?"
  HINT="re-run: $0 --rollback $STATE (every step skips what is already done)"

  # 0 quiesce: bootout and the helper toggle SIGTERM then SIGKILL after ExitTimeOut -------
  log "0/7 unload oMLX models and stop ds4 first, so nothing holding weights is killed"
  stop_engines

  # 1 disable helpers ---------------------------------------------------------------------
  log "1/7 disable the bundled helpers"
  if app_running unsloth-studio; then
    log "ACTION NEEDED in Unsloth: Settings > Attached engines > Engines enabled: turn it off."
    if [ "$DRY" = 1 ]; then
      log "dry-run: would wait up to 180 s for $NEW_OMLX and $NEW_DS4 to leave launchd, then fall back to bootout"
    else
      local i
      for i in $(seq 1 60); do
        label_loaded "$NEW_OMLX" || label_loaded "$NEW_DS4" || break
        [ $((i % 5)) -ne 1 ] || log "waiting for the helpers to be disabled ($((i * 3)) s)"
        sleep 3
      done
    fi
  else
    log "Unsloth is not running, so the toggle is not reachable; using launchctl bootout"
  fi
  local label
  for label in "$NEW_OMLX" "$NEW_DS4"; do
    if label_loaded "$label"; then
      warn "$label is still loaded; booting it out. SMAppService may re-register it while the app is installed;"
      warn "remove it for good in System Settings > General > Login Items if it comes back."
      run launchctl bootout "$DOMAIN/$label"
    else
      log "$label is not loaded"
    fi
  done

  # 2 quit Unsloth ------------------------------------------------------------------------
  log "2/7 quit Unsloth"
  quit_app unsloth-studio "$STUDIO_BUNDLE_ID" "Unsloth"

  # 3 port gate ---------------------------------------------------------------------------
  log "3/7 port-free gate"
  port_gate

  # 4 restore the app ---------------------------------------------------------------------
  log "4/7 restore $INSTALLED_APP"
  if [ -d "$APP_BACKUP" ]; then
    if [ -e "$INSTALLED_APP" ]; then
      run mv "$INSTALLED_APP" "$STATE/Unsloth-fork.app.$TS"
    fi
    run ditto "$APP_BACKUP" "$INSTALLED_APP"
    run codesign --verify --deep --strict --verbose=2 "$INSTALLED_APP"
  else
    warn "skipped: no $APP_BACKUP"
  fi

  # 5 restore studio ----------------------------------------------------------------------
  log "5/7 restore $STUDIO_DIR"
  if [ -d "$STATE/studio.bak" ]; then
    if [ -e "$STUDIO_DIR" ]; then
      run mv "$STUDIO_DIR" "$HOME/.unsloth/studio.fork-$TS"
      log "current Studio kept at $HOME/.unsloth/studio.fork-$TS"
    fi
    run ditto "$STATE/studio.bak" "$STUDIO_DIR"
  else
    warn "skipped: no $STATE/studio.bak"
  fi

  # 6 restore apps and plists --------------------------------------------------------------
  log "6/7 restore oMLX.app and EngineBar.app, then the old launchd agents"
  restore_retired_apps
  local plist saved
  local -a restored=()
  for label in dev.sillymoo.ds4-ondemand dev.sillymoo.omlx-tuned dev.sillymoo.enginebar "$OPTIONAL_LABEL"; do
    plist="$LA_DIR/$label.plist"
    saved="$STATE/$label.plist"
    if [ -f "$plist" ]; then
      log "$plist already in place"
    elif [ -f "$plist.retired" ]; then
      run mv "$plist.retired" "$plist"
    elif [ -f "$saved" ]; then
      run cp -p "$saved" "$plist"
    else
      [ "$label" = "$OPTIONAL_LABEL" ] || warn "no plist to restore for $label"
      continue
    fi
    if [ -f "$saved" ] && [ -f "$plist" ] && ! cmp -s "$saved" "$plist"; then
      warn "$plist differs from the saved copy; using the saved copy"
      run cp -p "$saved" "$plist"
    fi
    restored+=("$label")
  done
  for label in ${restored[@]+"${restored[@]}"}; do
    if label_loaded "$label"; then
      log "$label is already loaded"
    else
      log "bootstrap $label"
      run launchctl bootstrap "$DOMAIN" "$LA_DIR/$label.plist"
    fi
  done

  if [ -f "$RUNTIME_PY" ]; then
    log "oMLX runtime status"
    local rstatus need=0
    if [ "$DRY" = 1 ]; then
      printf '[dry-run] python3 %q status\n' "$RUNTIME_PY"
      printf '[dry-run] python3 %q activate   (only if the API is unreachable or metadata_state is not active)\n' "$RUNTIME_PY"
    else
      local i
      for i in $(seq 1 20); do curl -fsS -m 3 -o /dev/null "$OMLX_URL/v1/models" 2>/dev/null && break; sleep 3; done
      rstatus="$(python3 "$RUNTIME_PY" status)" || die "$RUNTIME_PY status failed"
      printf '%s\n' "$rstatus"
      need="$(printf '%s' "$rstatus" | python3 -c '
import json, sys
s = json.load(sys.stdin)
print(0 if s.get("api", {}).get("reachable") and s.get("metadata_state") == "active" else 1)')"
      if [ "$need" = 1 ]; then
        log "oMLX is not in the active tuned state; running activate"
        python3 "$RUNTIME_PY" activate || die "$RUNTIME_PY activate failed"
      fi
    fi
  else
    warn "$RUNTIME_PY not found; skipping the oMLX runtime status/activate"
  fi

  # 7 verify ------------------------------------------------------------------------------
  log "7/7 verify :8843 and :8001"
  if [ "$DRY" = 1 ]; then
    log "dry-run: would poll $OMLX_URL/v1/models and $DS4_URL/admin/status for up to 180 s"
    log "dry-run complete"
    return 0
  fi
  wait_for_endpoints 180 || die "the old engines did not answer on :8843 / :8001; check launchctl print $DOMAIN/dev.sillymoo.ds4-ondemand and the oMLX runtime status"
  fetch_roster | sed 's/^/    /'
  HINT=""
  log "rollback complete. The fork's Studio copy is kept next to ~/.unsloth for inspection."
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "$MODE" in
    cutover) cutover ;;
    rollback) rollback ;;
  esac
fi
