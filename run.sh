#!/usr/bin/env bash
# Timbreforge launcher (macOS / Linux).
#
# Creates a local virtual environment on first run, installs the dependencies
# into it and then starts the app. Nothing is installed system-wide and no
# network access happens after the first run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3.10+ is required but '$PY' was not found." >&2
  exit 1
fi

VENV=".venv"
if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment in $VENV ..."
  "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
  echo "Installing dependencies ..."
  "$VENV/bin/python" -m pip install -r requirements.txt
fi

exec "$VENV/bin/python" -m timbreforge "$@"
