# Course lab app

A Streamlit companion to the hands-on notebooks: visuals and controls up front, the relevant code in collapsible panels.
The notebooks stay the "take it home" version.

| Page | Status | Needs |
|---|---|---|
| Demo 1 · Relative policies | ✅ | nothing (pure Python; reproduces the notebook's run exactly) |
| 1 · Frozen Lake | ✅ | nothing (pure Python) |
| 2 · Reward function | ✅ | OpenRouter (`moonshotai/kimi-k2` for inputs, `deepseek/deepseek-r1-0528` as judge, as in the notebook) |
| 3 · Rock Paper Lizard Spock | ✅ | OpenRouter (`qwen/qwen-2.5-7b-instruct`, the notebook's base model) |
| 4 · Additional histories | ✅ | nothing |
| 5 · MCP tools | ✅ | Composio (Tool Router search + notebook fallback), OpenRouter (`openai/gpt-6-luna`, replacing the notebook's retired `openai/o4-mini`), ART `generate_scenarios` |

## Run locally

```bash
cd course_app
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
ALLOW_NO_PASSWORD=1 .venv/bin/streamlit run app.py
```

## Access control

| Env var | What it does |
|---|---|
| `COURSE_PASSWORD` | Shared password you announce in class. Without it the app refuses everyone. Avoid `"` and `\` in it. |
| `ACCESS_UNTIL` | ISO timestamp, e.g. `2026-11-05T18:00:00+01:00`. After it the app shows "lab closed" to everyone. |
| `ALLOW_NO_PASSWORD=1` | Local development only. |
| `OPENROUTER_API_KEY` | Needed by the LLM pages. Locally put it in `course_app/.env` (gitignored); `deploy.sh` reads it from there. Give the key a credit limit on OpenRouter. |
| `COMPOSIO_API_KEY` | Needed by session 5. Same handling as the OpenRouter key. The course user (`COMPOSIO_USER_ID`, default `course-user@example.com`) must have **no connected accounts**; the lab refuses to execute tools otherwise. |
| `LLM_MAX_CALLS_SESSION` / `LLM_MAX_CALLS_TOTAL` | Call caps per browser session (default 400) and per server process (default 20000). |

## Deploy on DigitalOcean App Platform

One-time setup: `brew install doctl`, `doctl auth init`, and connect DigitalOcean's GitHub app to this repository.

```bash
COURSE_PASSWORD='…' ACCESS_UNTIL='2026-11-05T18:00:00+01:00' course_app/deploy.sh up
course_app/deploy.sh status
course_app/deploy.sh down   # delete the app after the course, so nothing is billed
```

The spec lives in [`.do/app.yaml`](../.do/app.yaml) (Frankfurt, 1 vCPU / 1 GB). Pushing to `main` redeploys while the app exists.
Running `up` again on an existing app updates the password or expiry in place.
