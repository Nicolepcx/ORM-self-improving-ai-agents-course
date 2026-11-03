"""
Composio tool discovery, the OpenRouter + Composio workflow and ART scenario
generation from hands_on/session_05_HANDS_ON_MCP_server_composio.ipynb.

Uses the notebook's own stack: the Composio SDK, `openai/o4-mini` on OpenRouter,
and ART's real `art.mcp.generate_scenarios`. Differences, all forced by running
outside the notebook:
  - ART's `info/ok/warn` console logging goes to a callback so the page can show it
  - `await` becomes `asyncio.run(...)`
  - tool discovery goes through Composio's Tool Router search first (the "meta
    tools for discovering" the notebook mentions). `tools.get(search=...)` only
    matches short queries, so sentence-length use cases like the notebook's
    default always fell through to the hard-coded email tools. The notebook's
    lookup chain is kept as the fallback.
  - one safety guard: tool calls are only executed when the Composio user has
    no connected accounts. Then nothing can act on a real account and execution
    fails exactly as in the notebook ("connect the required app account"). If the
    instructor ever connects an account to this user, the lab skips execution
    instead of letting attendees trigger real actions through it.
"""

import asyncio
import contextlib
import io
import json
import os
import re
from typing import Any, Callable, Dict, List

from art.mcp import generate_scenarios

COMPOSIO_USER_ID = os.getenv("COMPOSIO_USER_ID", "course-user@example.com")
COMPOSIO_TOOL_LIMIT = 10
COMPOSIO_FALLBACK_TOOLKITS = ["GMAIL", "OUTLOOK", "SENDGRID"]
COMPOSIO_FALLBACK_TOOLS = [
    "GMAIL_SEND_EMAIL",
    "GMAIL_CREATE_EMAIL_DRAFT",
    "GMAIL_SEND_DRAFT",
    "OUTLOOK_SEND_EMAIL",
    "SENDGRID_SEND_EMAIL_WITH_TWILIO_SEND_GRID",
]
WORKFLOW_MODEL = "openai/gpt-6-luna"  # notebook: "openai/o4-mini", no longer on OpenRouter
GENERATOR_MODEL = "openai/gpt-6-luna"  # notebook: "openai/o4-mini", no longer on OpenRouter
DEFAULT_USE_CASE = ("I want to search for tools that can help me send an email. "
                    "Show me what tools are available and what the workflow would look like.")

Log = Callable[[str, str], None]  # (level, message)


# ------------------------------------------------------------------ helpers (verbatim from the notebook)
def _tool_function(tool: Any) -> Dict[str, Any]:
    """Return the function payload for dict or SDK model tools."""
    if hasattr(tool, "model_dump"):
        tool = tool.model_dump()
    elif hasattr(tool, "dict"):
        tool = tool.dict()

    if not isinstance(tool, dict):
        return {}
    if "function" in tool and isinstance(tool["function"], dict):
        return tool["function"]
    if tool.get("type") == "function" and "name" in tool:
        # Some providers use the Responses API shape: {type, name, description, parameters}.
        return tool
    return tool


def _clean_json_schema(schema: Any) -> Dict[str, Any]:
    """Keep tool parameters in the JSON Schema subset accepted by Chat Completions."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    allowed = {
        "type", "properties", "required", "description", "enum", "items", "anyOf",
        "oneOf", "allOf", "default", "additionalProperties", "format", "title",
        "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems",
    }
    cleaned = {}
    for key, value in schema.items():
        if key not in allowed:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {name: _clean_json_schema(prop) for name, prop in value.items()}
        elif key in {"items", "additionalProperties"} and isinstance(value, dict):
            cleaned[key] = _clean_json_schema(value)
        elif key in {"anyOf", "oneOf", "allOf"} and isinstance(value, list):
            cleaned[key] = [_clean_json_schema(item) for item in value]
        else:
            cleaned[key] = value

    cleaned.setdefault("type", "object")
    if cleaned.get("type") == "object":
        cleaned.setdefault("properties", {})
    return cleaned


def composio_tool_to_art_info(tool: Any) -> Dict[str, Any]:
    """Convert a Composio/OpenAI tool into the schema shape expected by ART."""
    fn = _tool_function(tool)
    return {
        "name": fn.get("name", "UNKNOWN_TOOL"),
        "description": fn.get("description", ""),
        "parameters": _clean_json_schema(fn.get("parameters", {"type": "object", "properties": {}})),
    }


def composio_tool_to_openai_chat_tool(tool: Any) -> Dict[str, Any]:
    """Normalize Composio tools to Chat Completions/OpenRouter's strict tool shape."""
    fn = composio_tool_to_art_info(tool)
    return {
        "type": "function",
        "function": {
            "name": fn["name"],
            "description": fn["description"],
            "parameters": fn["parameters"],
        },
    }


def _short_json(value: Any, max_chars: int = 500) -> str:
    text = json.dumps(value, indent=2, default=str) if not isinstance(value, str) else value
    return text if len(text) <= max_chars else text[:max_chars] + "... [truncated]"


def _get_composio_tools(composio, **kwargs):
    """Call tools.get across SDK versions that differ on user_id placement."""
    try:
        return composio.tools.get(user_id=COMPOSIO_USER_ID, **kwargs)
    except TypeError:
        return composio.tools.get(COMPOSIO_USER_ID, **kwargs)


def get_composio_tools_for_use_case(composio, use_case: str, limit: int = COMPOSIO_TOOL_LIMIT,
                                    log: Log = lambda level, msg: None):
    """Fetch Composio tools, trying fallbacks when semantic search returns no tools."""
    lookup_attempts = [
        ("semantic search", {"search": use_case, "limit": limit}),
        ("semantic search, latest versions", {"search": use_case, "limit": limit, "toolkit_versions": "latest"}),
        ("exact email tool slugs", {"tools": COMPOSIO_FALLBACK_TOOLS}),
        ("exact email tool slugs, latest versions", {"tools": COMPOSIO_FALLBACK_TOOLS, "toolkit_versions": "latest"}),
        ("email toolkits", {"toolkits": COMPOSIO_FALLBACK_TOOLKITS, "limit": limit}),
        ("email toolkits lowercase", {"toolkits": [t.lower() for t in COMPOSIO_FALLBACK_TOOLKITS], "limit": limit}),
        ("email toolkits, latest versions", {"toolkits": COMPOSIO_FALLBACK_TOOLKITS, "limit": limit, "toolkit_versions": "latest"}),
    ]

    last_error = None
    for label, kwargs in lookup_attempts:
        try:
            tools = _get_composio_tools(composio, **kwargs)
        except TypeError as e:
            last_error = e
            log("skip", f"{label}: not supported by this SDK version ({e})")
            continue
        except Exception as e:
            last_error = e
            log("warn", f"Composio lookup failed for {label}: {e}")
            continue

        if tools:
            log("ok", f"Loaded {len(tools)} Composio tool(s) via {label}")
            return tools, label

        log("info", f"No tools returned via {label}; trying next lookup strategy.")

    if last_error:
        log("warn", f"No Composio tools found. Last lookup error: {last_error}")
    else:
        log("warn", "No Composio tools found after all lookup strategies.")
    return [], None


def discover_tools(composio, use_case: str, limit: int = COMPOSIO_TOOL_LIMIT,
                   log: Log = lambda level, msg: None) -> Dict[str, Any]:
    """
    Find tools for a natural-language use case with Composio's Tool Router search,
    then load their schemas. Falls back to the notebook's lookup chain.
    """
    router_results: List[Dict] = []
    try:
        session = composio.create(user_id=COMPOSIO_USER_ID, manage_connections=False)
        response = session.search(query=use_case).model_dump()
        router_results = [r for r in response.get("results") or [] if not r.get("error")]
        log("ok", f"Tool Router split the use case into {len(router_results)} sub-task(s)")
    except Exception as e:
        log("warn", f"Tool Router search failed: {type(e).__name__}: {e}")

    slugs: List[str] = []
    for result in router_results:
        slugs += result.get("primary_tool_slugs") or []
    for result in router_results:
        slugs += result.get("related_tool_slugs") or []
    slugs = list(dict.fromkeys(slugs))[:limit]

    tools, via = [], None
    if slugs:
        try:
            tools = _get_composio_tools(composio, tools=slugs)
            via = "Tool Router"
            log("ok", f"Loaded {len(tools)} tool schema(s) for {len(slugs)} recommended slug(s)")
        except Exception as e:
            log("warn", f"Loading tool schemas failed: {type(e).__name__}: {e}")
    if not tools:
        log("info", "Falling back to the notebook's lookup chain")
        tools, via = get_composio_tools_for_use_case(composio, use_case, limit=limit, log=log)
    return {"tools": tools, "via": via, "router": router_results}


def tools_by_toolkit(tool_infos: List[Dict]) -> Dict[str, List[Dict]]:
    """Group tools the way `list_composio_tools_example` prints them."""
    groups: Dict[str, List[Dict]] = {}
    for tool in tool_infos:
        toolkit = tool["name"].split("_")[0] if "_" in tool["name"] else "OTHER"
        groups.setdefault(toolkit, []).append(tool)
    return groups


def connected_toolkits(composio) -> List[str]:
    """Toolkits with a connected account for the course user (should be none)."""
    accounts = composio.connected_accounts.list(user_ids=[COMPOSIO_USER_ID])
    items = getattr(accounts, "items", accounts) or []
    return [str(getattr(getattr(a, "toolkit", None), "slug", getattr(a, "toolkit", "?"))) for a in items]


# ------------------------------------------------------------------ Example 2: the workflow
def composio_workflow(composio, llm, use_case: str, tools: List[Any], max_turns: int,
                      log: Log = lambda level, msg: None) -> tuple[List[Dict], int]:
    """Search Composio tools for a use case, then let the model use them. Returns messages and calls used."""
    chat_tools = [composio_tool_to_openai_chat_tool(tool) for tool in tools]
    chat_tools = [tool for tool in chat_tools if tool["function"]["name"] != "UNKNOWN_TOOL"]
    if not chat_tools:
        log("warn", "No Chat Completions-compatible Composio tools available")
        return [], 0

    tool_names = [tool["function"]["name"] for tool in chat_tools]
    system_prompt = f"""You are an AI assistant with access to Composio tools for workflow automation.

The notebook has already searched Composio for tools relevant to the user's use case and exposed only those tools to you.

Available tool names:
{json.dumps(tool_names, indent=2)}

Workflow pattern:
1. Explain which available tools match the user's goal.
2. If a tool requires account authentication and execution fails or is not possible, explain which app connection is needed.
3. Only execute a tool when the user has provided enough concrete inputs.
4. If the user is only asking to discover tools or see a workflow, summarize the tools and recommended workflow instead of performing a real-world action.
"""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": use_case},
    ]
    calls = 0
    for turn in range(max_turns):
        response = llm.chat.completions.create(
            model=WORKFLOW_MODEL,
            messages=messages,
            tools=chat_tools,
            tool_choice="auto" if chat_tools else None,
        )
        calls += 1

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        log("info", f"Turn {turn + 1}: {len(msg.tool_calls or [])} tool call(s)")

        if not msg.tool_calls:
            break

        connected = connected_toolkits(composio)
        if connected:
            results = [{
                "successful": False,
                "error": f"Skipped by the lab: the course Composio user has connected accounts ({', '.join(connected)}).",
                "note": "The lab never executes tools on a connected account. Run the notebook with your own Composio user.",
            } for _ in msg.tool_calls]
        else:
            try:
                results = composio.provider.handle_tool_calls(response=response, user_id=COMPOSIO_USER_ID)
            except Exception as e:
                # For discovery/planning demos, keep the notebook moving even if the
                # account is not connected or the SDK cannot execute normalized calls.
                results = [
                    {
                        "successful": False,
                        "error": str(e),
                        "note": "Tool call was generated, but execution was skipped or failed. Connect the required app account before real execution.",
                    }
                    for _ in msg.tool_calls
                ]

        for tc, result in zip(msg.tool_calls, results):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    return messages, calls


# ------------------------------------------------------------------ scenario generation (ART's real generate_scenarios)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run_generate_scenarios(tool_infos: List[Dict], num_scenarios: int, api_key: str) -> tuple[List[Dict], str]:
    """Call ART's generate_scenarios exactly like the notebook, capturing its console log."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        collection = asyncio.run(generate_scenarios(
            tools=tool_infos,
            resources=[],
            num_scenarios=num_scenarios,
            show_preview=False,
            generator_model=GENERATOR_MODEL,
            generator_api_key=api_key,
        ))
    scenarios = [{"task": s.task, "difficulty": s.difficulty} for s in collection.scenarios]
    return scenarios, _ANSI.sub("", buffer.getvalue())
