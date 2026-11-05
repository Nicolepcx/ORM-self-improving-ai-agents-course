import json
import os
import time

import plotly.graph_objects as go
import streamlit as st

from core import llm
from core.trajectory_view import render_trajectory
from core.ui import code_panel, page_header, takeaway, task_card
from labs import mcp_tools as mt
from labs import s5_notebook_source as src
from labs.trajectory import Trajectory, from_openai_message

page_header(
    session=5,
    title="MCP tools: from hundreds of tools to a plan",
    subtitle=(
        "Instead of hard-coding every integration, the agent searches **Composio** for tools that match a "
        "natural-language use case. The focus is not execution, but **tool discovery, planning, "
        "authentication and orchestration**, and the scenarios you would later train on."
    ),
    minutes=15,
    notebook_path="hands_on/session_05_HANDS_ON_MCP_server_composio.ipynb",
)
task_card(
    task="Add more comparisons and better tools to improve the agent's MCP server usage.",
    todos=[
        "**Set `USE_CASE`** to something you'd want your agent to do, ideally a workflow that needs two apps "
        "(e.g. GitHub and Slack).",
        "**Better tools:** search Composio and look at which tools, plan and pitfalls the agent gets for it.",
        "**Run the workflow** and see how the agent plans with those tools (and what it says when a tool is missing).",
        "**More comparisons:** generate scenarios. `NUM_SCENARIOS` sets how many different tasks you'd roll out and "
        "compare, `MAX_TURNS` how deep each workflow may go.",
    ],
    why=(
        "This exercise shows how easily you can give an agent **more tools**: instead of hard-coding every "
        "integration, the agent searches **Composio** for tools that match a natural-language use case, "
        "across hundreds of app toolkits.\n"
        "- Agents move from *“I have one tool”* to *“I have hundreds of tools, how do I decide?”*: tool discovery, "
        "selection, workflow planning, connection management, multi-turn tool use.\n"
        "- **Better tools:** complementary toolkits make realistic, multi-tool workflows possible.\n"
        "- **More comparisons:** more scenarios and more trajectories per scenario give RULER a richer, relative "
        "learning signal when you train."
    ),
    remember="Modern agents don't just call tools. They discover, plan, authenticate and compose them dynamically, "
             "and Composio is one layer that makes that possible.",
)

if not llm.is_configured() or not os.getenv("COMPOSIO_API_KEY"):
    st.error("This page needs the instructor's OpenRouter and Composio keys, which aren't configured on the server yet.")
    st.stop()


@st.cache_resource
def composio_client():
    from composio import Composio
    from composio_openai import OpenAIProvider
    return Composio(api_key=os.environ["COMPOSIO_API_KEY"], provider=OpenAIProvider())


@st.cache_data(ttl=3600, show_spinner=False, max_entries=500)
def discover(use_case: str) -> dict:
    logs = []
    found = mt.discover_tools(composio_client(), use_case, log=lambda level, msg: logs.append((level, msg)))
    found["tool_infos"] = [mt.composio_tool_to_art_info(t) for t in found["tools"]]
    found["logs"] = logs
    return found


LOG_ICON = {"ok": "✅", "info": "ℹ️", "warn": "⚠️", "skip": "⏭️"}

# ---------------------------------------------------------------- TODO
st.markdown("#### TODO · Your use case")
use_case = st.text_area("USE_CASE", mt.DEFAULT_USE_CASE, height=90, key="s5_use_case").strip()
c1, c2 = st.columns(2)
num_scenarios = c1.slider("NUM_SCENARIOS", 1, 15, 5, help="Breadth: how many different tasks to generate")
max_turns = c2.slider("MAX_TURNS", 1, 10, 5, help="Depth: how many model turns the workflow may take")
st.code(f"NUM_SCENARIOS = {num_scenarios}\nMAX_TURNS = {max_turns}\nUSE_CASE = {use_case!r}",
        language="python", wrap_lines=True)
st.caption(f"The notebook used `openai/o4-mini`, which OpenRouter no longer serves. This lab uses `{mt.WORKFLOW_MODEL}` instead.")
code_panel("Composio setup (notebook)", src.SETUP)

st.divider()

# ---------------------------------------------------------------- 1 · discover
st.markdown("#### 1 · Discover tools")
st.markdown("Composio's **Tool Router** reads the use case, splits it into sub-tasks and recommends tools, "
            "a plan and known pitfalls for each. Only those tools are exposed to the agent.")
if st.button("Search Composio", icon=":material/travel_explore:", disabled=not use_case):
    with st.spinner("Searching the Composio catalog…"):
        st.session_state.s5_found = {"use_case": use_case, **discover(use_case)}
    st.session_state.pop("s5_workflow", None)
    st.session_state.pop("s5_scenarios", None)

found = st.session_state.get("s5_found")
if found:
    if found["use_case"] != use_case:
        st.caption("These tools were found for a different use case. Search again to match your current one.")
    infos = found["tool_infos"]
    groups = mt.tools_by_toolkit(infos)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sub-tasks", len(found["router"]) or "–")
    m2.metric("Toolkits", len(groups))
    m3.metric("Tools exposed to the agent", len(infos))
    m4.metric("Found via", found["via"] or "nothing")

    with st.expander("Lookup log"):
        for level, msg in found["logs"]:
            st.markdown(f"{LOG_ICON.get(level, '•')} {msg}")

    for i, result in enumerate(found["router"], 1):
        with st.container(border=True):
            st.markdown(f"**Sub-task {i}** · {result.get('use_case', '')}")
            st.markdown(" ".join(f"`{t}`" for t in result.get("toolkits") or []) +
                        (f" · difficulty: *{result['difficulty']}*" if result.get("difficulty") else ""))
            left, right = st.columns(2)
            with left:
                st.markdown("**Recommended plan**")
                for step in result.get("recommended_plan_steps") or []:
                    st.markdown(f"- {step}")
            with right:
                st.markdown("**Primary tools** " + " ".join(f"`{s}`" for s in result.get("primary_tool_slugs") or []))
                st.markdown("**Related tools** " + " ".join(f"`{s}`" for s in result.get("related_tool_slugs") or []))
                if result.get("known_pitfalls"):
                    with st.expander("Known pitfalls"):
                        for pitfall in result["known_pitfalls"]:
                            st.markdown(f"- {pitfall}")

    st.markdown("**The tools the agent gets**, grouped by toolkit")
    cols = st.columns(min(3, max(1, len(groups))))
    for col, (toolkit, toolkit_tools) in zip(cols * 10, groups.items()):
        with col.container(border=True):
            st.markdown(f"**{toolkit.title()}**")
            for tool in toolkit_tools:
                params = list(tool["parameters"].get("properties", {}))
                required = set(tool["parameters"].get("required", []))
                st.markdown(f"`{tool['name']}`")
                st.caption(tool["description"].split("\n")[0][:140])
                if params:
                    st.caption("params: " + ", ".join(f"**{p}**" if p in required else p for p in params[:8])
                               + (" …" if len(params) > 8 else ""))
    code_panel("Helper functions (notebook)", src.HELPERS)
    code_panel("Tool Router discovery (this lab)", mt.discover_tools,
               "`tools.get(search=…)` only matches short queries, so sentence-length use cases fell back to the "
               "email tools. The Tool Router search handles the full sentence.")

    st.divider()

    # ------------------------------------------------------------ 2 · workflow
    st.markdown("#### 2 · Plan the workflow")
    st.markdown(f"`{mt.WORKFLOW_MODEL}` gets your use case and only the tools above, for up to {max_turns} turn(s). "
                "Tool calls are executed through Composio; the course user has no connected accounts, so "
                "execution fails safely and the agent has to explain which connection is needed.")
    if st.button("Run the workflow", icon=":material/route:", disabled=not infos):
        try:
            llm.reserve_calls(max_turns)
        except llm.BudgetExceeded as e:
            st.warning(str(e))
        else:
            start, calls = time.time(), max_turns
            with st.spinner("The agent is planning…"):
                try:
                    messages, calls = mt.composio_workflow(composio_client(), llm.client(), use_case,
                                                           found["tools"], max_turns)
                    st.session_state.s5_workflow = {"messages": messages, "seconds": time.time() - start}
                except Exception as e:
                    st.session_state.s5_workflow = {"error": f"{type(e).__name__}: {e}"}
            llm.release_calls(max_turns - calls)

    workflow = st.session_state.get("s5_workflow")
    if workflow:
        if "error" in workflow:
            st.error(f"`{workflow['error']}`")
        else:
            messages = workflow["messages"]
            final = next((m.get("content") for m in reversed(messages)
                          if m.get("role") == "assistant" and m.get("content")), "")
            tool_calls = sum(len(m.get("tool_calls") or []) for m in messages)
            st.markdown(f"**The agent's answer** · {workflow['seconds']:.0f}s · {tool_calls} tool call(s)")
            with st.container(border=True):
                st.markdown(final or "*(no text answer)*")
            with st.expander("The full trajectory"):
                render_trajectory(Trajectory(messages=[from_openai_message(m) for m in messages],
                                             additional_histories=[], reward="–",
                                             metrics={"turns": sum(m["role"] == "assistant" for m in messages),
                                                      "tool_calls": tool_calls}),
                                  title="Workflow trajectory")
    code_panel("composio_workflow_example (notebook)", src.WORKFLOW)

    st.divider()

    # ------------------------------------------------------------ 3 · scenarios
    st.markdown("#### 3 · Generate training scenarios")
    st.markdown(f"ART's `generate_scenarios` turns tool schemas into realistic tasks of varying difficulty: "
                f"the inputs you would later roll out, judge with RULER and train on.")
    b1, b2 = st.columns(2)
    focused = b1.button(f"Focused: {num_scenarios} scenarios for the first tool", icon=":material/target:",
                        disabled=not infos)
    enhanced = b2.button("Enhanced: 15 scenarios using all tools", icon=":material/hub:", disabled=not infos)
    if focused or enhanced:
        try:
            llm.reserve_calls(1)
        except llm.BudgetExceeded as e:
            st.warning(str(e))
        else:
            tools_for_generation = infos[:1] if focused else [
                {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}
                for t in infos
            ]
            n = num_scenarios if focused else 15
            with st.spinner(f"{mt.GENERATOR_MODEL} is writing {n} scenarios…"):
                try:
                    scenarios, log_text = mt.run_generate_scenarios(tools_for_generation, n, os.environ["OPENROUTER_API_KEY"])
                    st.session_state.s5_scenarios = {
                        "kind": f"Focused on {infos[0]['name']}" if focused else f"Enhanced: all {len(infos)} tools",
                        "scenarios": scenarios, "log": log_text}
                except Exception as e:
                    st.session_state.s5_scenarios = {"error": f"{type(e).__name__}: {e}"}

    generated = st.session_state.get("s5_scenarios")
    if generated:
        if "error" in generated:
            st.error(f"Scenario generation failed: `{generated['error']}`")
        else:
            scenarios = sorted(generated["scenarios"], key=lambda s: s["difficulty"])
            st.markdown(f"**{len(scenarios)} scenarios** · {generated['kind']}")
            counts = [sum(s["difficulty"] == d for s in scenarios) for d in range(1, 6)]
            fig = go.Figure(go.Bar(x=[f"{d}/5" for d in range(1, 6)], y=counts, marker_color="#7c5cd6"))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="difficulty",
                              yaxis=dict(dtick=1))
            st.plotly_chart(fig, config={"displayModeBar": False})
            cols = st.columns(3)
            for i, s in enumerate(scenarios):
                with cols[i % 3].container(border=True):
                    st.caption("difficulty " + "●" * s["difficulty"] + "○" * (5 - s["difficulty"]))
                    st.markdown(s["task"])
            with st.expander("generate_scenarios log"):
                st.code(generated["log"], language="text")
    code_panel("Focused scenario generation (notebook)", src.FOCUSED_SCENARIOS)
    code_panel("Enhanced scenario generation (notebook)", src.ENHANCED_SCENARIOS)

st.divider()
st.markdown("#### Check yourself")
for question, answer in [
    ("Try a use case that needs two apps (e.g. GitHub and Slack). How does the Tool Router's plan change?",
     "It splits the request into sub-tasks and recommends tools per app, plus prerequisites such as finding a "
     "channel ID first. Planning across tools is what the agent has to learn."),
    ("The agent can only use the tools it was given. What happens if discovery misses one?",
     "The agent can't do that step: it either says so or works around it. Tool discovery quality directly "
     "limits what the agent can achieve, so it is part of what you evaluate."),
    ("How do NUM_SCENARIOS and MAX_TURNS relate to training?",
     "NUM_SCENARIOS is breadth: how many different tasks you roll out. MAX_TURNS is depth: how long each "
     "trajectory may get. More scenarios and more comparisons per scenario give RULER a better learning signal."),
]:
    st.markdown(f"**{question}**")
    with st.expander("Reveal"):
        st.markdown(answer)

takeaway("<b>RULER learns what \"good\" means from your specific tools and scenarios, no labeled data required.</b> "
         "Discovery decides which tools exist for the agent, the plan decides how they combine, and generated "
         "scenarios become the training set.")
