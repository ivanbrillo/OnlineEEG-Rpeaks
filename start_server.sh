#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found at: $VENV_DIR"
  echo "Create it first with: python3 -m venv venv"
  exit 1
fi

source "$VENV_DIR/bin/activate"

if ! command -v gunicorn >/dev/null 2>&1; then
  echo "gunicorn is not installed in this venv. Installing requirements..."
  pip install -r "$PROJECT_DIR/requirements.txt"
fi

export FLASK_DEBUG="${FLASK_DEBUG:-0}"
export PORT="${PORT:-5000}"
export SECRET_KEY="${SECRET_KEY:-change-me-in-production}"

cd "$PROJECT_DIR"
exec gunicorn \
  --bind "0.0.0.0:${PORT}" \
  --workers "${GUNICORN_WORKERS:-2}" \
  --threads "${GUNICORN_THREADS:-4}" \
  --timeout "${GUNICORN_TIMEOUT:-300}" \
  app:app