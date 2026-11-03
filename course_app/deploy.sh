#!/usr/bin/env bash
# Spin the lab app up before class and tear it down afterwards.
#
#   COURSE_PASSWORD='...' ACCESS_UNTIL='2026-11-05T18:00:00+01:00' ./deploy.sh up
#   ./deploy.sh status
#   ./deploy.sh down
#
# Needs doctl (brew install doctl && doctl auth init) and DigitalOcean's
# GitHub app connected to the course repository.
set -euo pipefail

APP_NAME="rl-course-lab"
SPEC="$(cd "$(dirname "$0")/.." && pwd)/.do/app.yaml"

app_id() { doctl apps list --format ID,Spec.Name --no-header | awk -v n="$APP_NAME" '$2 == n {print $1}'; }

render_spec() {
  : "${COURSE_PASSWORD:?Set COURSE_PASSWORD}"
  : "${ACCESS_UNTIL:?Set ACCESS_UNTIL, e.g. 2026-11-05T18:00:00+01:00}"
  # API keys come from the environment or course_app/.env (never written into the spec file)
  for var in OPENROUTER_API_KEY COMPOSIO_API_KEY; do
    if [[ -z "${!var:-}" && -f "$(dirname "$0")/.env" ]]; then
      export "$var=$(grep -E "^${var}=" "$(dirname "$0")/.env" | cut -d= -f2-)"
    fi
    [[ -n "${!var:-}" ]] || { echo "Set $var or add it to course_app/.env" >&2; exit 1; }
  done
  python3 -c 'import os, sys; print(os.path.expandvars(open(sys.argv[1]).read()))' "$SPEC"
}

case "${1:-}" in
  up)
    if [[ -n "$(app_id)" ]]; then
      echo "Updating existing app…"
      render_spec | doctl apps update "$(app_id)" --spec - --wait
    else
      echo "Creating app…"
      render_spec | doctl apps create --spec - --wait
    fi
    doctl apps get "$(app_id)" --format DefaultIngress --no-header
    ;;
  status)
    doctl apps get "$(app_id)" --format ID,DefaultIngress,ActiveDeployment.Phase
    ;;
  down)
    id="$(app_id)"
    [[ -z "$id" ]] && { echo "No app named $APP_NAME"; exit 0; }
    doctl apps delete "$id" --force
    echo "Deleted $APP_NAME. Nothing is billed any more."
    ;;
  *)
    echo "Usage: $0 up|status|down"; exit 1 ;;
esac
