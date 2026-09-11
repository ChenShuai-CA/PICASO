#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
SCENARIO_PYTHON="${SCENARIO_PYTHON:-/home/shuai/.venvs/scenario-gpu/bin/python}"
if [[ ! -x "$SCENARIO_PYTHON" ]]; then
  echo "Missing Ubuntu scenario-gpu interpreter: $SCENARIO_PYTHON" >&2
  exit 1
fi
exec "$SCENARIO_PYTHON" -m scenario_lab "$@"
