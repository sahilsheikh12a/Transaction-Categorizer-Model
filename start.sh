#!/usr/bin/env bash
# Boots the PhonePe Categorizer UI on http://127.0.0.1:8000
#
#   ./start.sh                 # UI on :8000, corrections saved to categorizer.db
#   ./start.sh --port 9000     # different port
#   ./start.sh --no-db         # scratch session, corrections not saved
#   ./start.sh --check         # run tests + lint instead of serving
#
# Creates the venv and installs requirements on first run.
set -euo pipefail
cd "$(dirname "$0")"

PORT=8000
HOST=127.0.0.1
EXTRA=()
CHECK=0

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --no-db) EXTRA+=(--no-db); shift ;;
    --check) CHECK=1; shift ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY=.venv/bin/python

# ── setup ────────────────────────────────────────────────────────────────
if [ ! -x "$PY" ]; then
  echo "[setup] creating .venv and installing requirements…"
  python3 -m venv .venv
  ./.venv/bin/pip install -q -U pip
  ./.venv/bin/pip install -q -r requirements.txt
fi

# Reinstall if requirements changed since the venv was built.
if [ requirements.txt -nt .venv ]; then
  echo "[setup] requirements.txt changed — reinstalling…"
  "$PY" -m pip install -q -r requirements.txt
  touch .venv
fi

# ── --check: tests + lint, then exit ─────────────────────────────────────
if [ "$CHECK" -eq 1 ]; then
  echo "[check] pytest"
  "$PY" -m pytest
  echo "[check] ruff"
  "$PY" -m ruff check .
  exit 0
fi

# ── free the port ────────────────────────────────────────────────────────
# A previous run left in the background keeps the socket bound, so the new
# process dies on "address already in use" while the stale one carries on
# serving old code — confusing to debug, trivial to prevent.
if command -v lsof >/dev/null 2>&1; then
  STALE=$(lsof -ti "tcp:${PORT}" 2>/dev/null || true)
  if [ -n "$STALE" ]; then
    echo "[server] killing stale process(es) on :${PORT} — ${STALE}"
    # shellcheck disable=SC2086
    kill -9 $STALE 2>/dev/null || true
    sleep 0.4
  fi
fi

# ── go ───────────────────────────────────────────────────────────────────
URL="http://${HOST}:${PORT}"
echo "[server] ${URL}"

# Open a browser once the port answers, without blocking the server.
(
  for _ in $(seq 1 50); do
    if "$PY" - "$HOST" "$PORT" <<'PROBE' 2>/dev/null; then
import socket, sys
s = socket.socket()
s.settimeout(0.2)
sys.exit(0 if s.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0 else 1)
PROBE
      for opener in xdg-open open; do
        command -v "$opener" >/dev/null 2>&1 && "$opener" "$URL" >/dev/null 2>&1 && break
      done
      break
    fi
    sleep 0.2
  done
) &

# -u: unbuffered. Without it Python holds the server's own startup lines
# in a pipe buffer whenever output is redirected to a file or a log,
# so "which port / which DB" never shows up until the process exits.
exec "$PY" -u -m phonepe_categorizer serve --host "$HOST" --port "$PORT" ${EXTRA+"${EXTRA[@]}"}
