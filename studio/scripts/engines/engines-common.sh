# shellcheck shell=bash
# Shared by omlx-launch (sourced, never run). Needs ENGINES_HOME set.

# engines_rotate_log <file> [max bytes] [keep]: when the file is over the limit (default 20 MB),
# shift <file>.N -> <file>.N+1 (keeping 3) and move the file to <file>.1. Run on start, before
# the wrapper redirects its output to the file; a running engine's log is never cut under it.
engines_rotate_log() {
  local file="$1" max="${2:-20971520}" keep="${3:-3}" size prev i
  [ -f "$file" ] || return 0
  size="$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)"
  [ "$size" -gt "$max" ] || return 0
  rm -f "$file.$keep"
  i=$keep
  while [ "$i" -gt 1 ]; do
    prev=$((i - 1))
    if [ -f "$file.$prev" ]; then mv -f "$file.$prev" "$file.$i"; fi
    i=$prev
  done
  mv -f "$file" "$file.1"
}

# engines_fail <name> <reason>: crash-loop backoff for a fatal precondition. Writes
# $ENGINES_HOME/<name>.fail as {"reason","ts","count"} (the tray, the engines panel and
# /api/engines/attached/status show it), sleeps min(10*2^(count-1), 300) s so launchd's KeepAlive
# respawn backs off, then exits 78. engines_launch.py does the same for its own failures and
# clears the file right before it execs the engine. ENGINES_FAIL_NO_SLEEP=1 skips the sleep.
engines_fail() {
  local name="$1" reason="$2" file count wait=10 i
  file="$ENGINES_HOME/$name.fail"
  count="$(sed -n 's/.*"count": *\([0-9][0-9]*\).*/\1/p' "$file" 2>/dev/null | head -n 1 || true)"
  count=$(( ${count:-0} + 1 ))
  printf '%s: %s\n' "$name" "$reason" >&2
  reason="${reason//\\/\\\\}"
  reason="${reason//\"/\\\"}"
  reason="${reason//$'\n'/ }"
  mkdir -p "$ENGINES_HOME"
  printf '{"reason": "%s", "ts": %s, "count": %s}\n' "$reason" "$(date +%s)" "$count" >"$file.tmp.$$" \
    && mv -f "$file.tmp.$$" "$file"
  i=1
  while [ "$i" -lt "$count" ] && [ "$wait" -lt 300 ]; do wait=$((wait * 2)); i=$((i + 1)); done
  [ "$wait" -le 300 ] || wait=300
  if [ "${ENGINES_FAIL_NO_SLEEP:-0}" != 1 ]; then
    printf '%s: failure %s, backing off %s s before exiting\n' "$name" "$count" "$wait" >&2
    sleep "$wait"
  fi
  exit 78
}
