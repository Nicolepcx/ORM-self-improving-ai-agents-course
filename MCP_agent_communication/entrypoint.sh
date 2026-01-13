#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-streamable-http}"
PORT="${PORT:-3333}"
DEPS="${DEPS:-}"   # e.g. "numpy,pandas"

# Build args for mcp-run-python
ARGS=()
if [ -n "${DEPS}" ]; then
  ARGS+=(--deps "${DEPS}")
fi

if [ "$MODE" = "stdio" ]; then
  exec python -m mcp_run_python stdio "${ARGS[@]}"
elif [ "$MODE" = "streamable-http" ]; then
  exec python -m mcp_run_python streamable-http --port "${PORT}" "${ARGS[@]}"
elif [ "$MODE" = "example" ]; then
  exec python -m mcp_run_python ${ARGS:+--deps "${DEPS}"} example
else
  echo "Unknown mode: $MODE (use stdio | streamable-http | example)"
  exit 1
fi
