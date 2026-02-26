#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-$PWD}"
VENV_DIR="${VENV_DIR:-$APP_DIR/.venv}"
PORT="${PORT:-8000}"

cd "$APP_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
