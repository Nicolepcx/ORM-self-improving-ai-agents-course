import json

import pandas as pd
import streamlit as st

from core import llm
from core.code_input import editor, parse_kwarg_value, parse_message_content
from core.trajectory_view import render_trajectory
from core.ui import code_panel, page_header, takeaway, task_card
from labs import rpls
from labs import rpls_viz as viz
from labs import s3_notebook_source as src

PLAYERS = ["Player 1 · trainable model", "Player 2 · base model"]

page_header(
    session=3,
    title="Rock Paper Scissors Lizard Spock: trajectories and metrics",
    subtitle=(
        "Two copies of **Qwen2.5-7B-Instruct** play each other by calling a `play_move` tool. "
        "Your job is to fill in the two TODOs of the notebook's `rollout()`: the **system instruction** "
        "that tells the agent how to behave, and the **metrics** that make its behaviour measurable."
    ),
    minutes=20,
    notebook_path="hands_on/session_03_HANDS_ON_rock_paper_lizard_spock.ipynb",
)
task_card(
    task="Design a training loop to teach your agent to play Rock Paper Scissors Lizard Spock.",
    todos=[
        "**TODO 1, the system instruction:** tell the agent its role, that it **must use the `play_move` tool**, "
        "and that only valid moves are allowed.",
        "**TODO 2, the metrics dictionary:** decide what to measure: rounds, how often each move is played, "
        "invalid or missing moves (`nothing`).",
        "**Run `rollout()`** to create trajectories, and watch the metrics update round by round.",
        "**Change your instruction** and run again: compare the trajectories and metrics in *Your runs*.",
    ],
    why=(
        "This exercise teaches the core skill of every RL-for-agents system: **turning agent behaviour into "
        "measurable signals**.\n"
        "- **Trajectories** describe what happened, step by step: tool calls, arguments, environment feedback, "
        "multiple turns.\n"
        "- **Metrics are not the reward.** They are your debugging interface. That's why `normalize_move` turns "
        "every outcome, even failure, into something you can count.\n"
        "- **The system instruction is the behavioural contract.** Small wording changes can drastically change "
        "learning outcomes.\n\n"
        "A reward function is only as good as the trajectory and metrics behind it. If you can't reliably capture "
        "what the agent did, no RL algorithm will save you. (Training the model itself needs a GPU and is optional, "
        "after the session.)"
    ),
    remember="Trajectories describe behaviour. Metrics make behaviour debuggable.",
)

if not llm.is_configured():
    st.error("This page needs the instructor's OpenRouter key, which isn't configured on the server yet.")
    st.stop()

# ---------------------------------------------------------------- the game
left, right = st.columns([1, 1.4], gap="large")
with left:
    st.image(viz.rules_svg(), caption="Arrows point from winner to loser")
with right:
    st.markdown(
        """
**The game**: each move beats two others and loses to two.
The arrows point from winner to loser.

- 🪨 rock beats ✂️ scissors and 🦎 lizard
- 📄 paper beats 🪨 rock and 🖖 spock
- ✂️ scissors beats 📄 paper and 🦎 lizard
- 🦎 lizard beats 🖖 spock and 📄 paper
- 🖖 spock beats ✂️ scissors and 🪨 rock

**One rollout**: up to 10 rounds, the first player with 3 wins ends it.
Each player's **reward** is its number of wins. A player that doesn't call
the tool, or calls it with something that isn't a valid move, plays `nothing`.
"""
    )
    st.link_button("Sheldon explains the game (YouTube, from 0:44)",
                   "https://www.youtube.com/watch?v=jnfz_9d9BUA&t=44s", icon=":material/play_circle:")
code_panel("The tool the model must call", json.dumps(rpls.TOOLS, indent=2),
           "The model can only play by calling `play_move`. The `enum` lists the valid moves.")

st.divider()

# ---------------------------------------------------------------- TODO 1
st.markdown("#### TODO 1 · The system instruction")
st.markdown("This is the agent's **behavioral contract**. Both players get the same instruction.")
st.caption("Write it as plain text, or paste the notebook's Python: a string, `( \"...\" \"...\" )` or `\"content\": (...)`.")
instruction_code = editor("", key="rpls_instruction_editor", lines=(5, 14),
                          ghost_text="You are playing Rock Paper Scissors Lizard Spock…")
instruction_parsed = parse_message_content(instruction_code)
instruction = instruction_parsed.value or ""
if instruction_parsed.error:
    st.error(instruction_parsed.error)
elif instruction:
    st.caption(f"Read as {instruction_parsed.read_as}. The model receives: {instruction!r}")
with st.expander("Help: what should the instruction do?"):
    st.markdown(
        "It tells the agent:\n"
        "- what its role is\n"
        "- that it **must use the tool**\n"
        "- that only **valid moves** are allowed\n\n"
        "Small wording changes here can drastically change learning outcomes."
    )

# ---------------------------------------------------------------- TODO 2
st.markdown("#### TODO 2 · The metrics dictionary")
st.markdown("Metrics are **not** the reward. They are your **debugging interface**. "
            "Fill in the dict, exactly as you would in the notebook.")
metrics_code = editor("metrics={\n    \n},", key="rpls_metrics_editor", lines=(5, 16))
metrics_parsed = parse_kwarg_value(metrics_code, "metrics", dict)
metrics_template, metrics_error = metrics_parsed.value, metrics_parsed.error
if metrics_error:
    st.error(metrics_error)
else:
    st.caption(f"Each trajectory starts with: metrics = {metrics_template!r}")
with st.expander("Help: what should the metrics track?"):
    st.markdown(
        "In this notebook they track:\n"
        "- number of rounds\n"
        "- frequency of each move\n"
        "- invalid or missing moves (`nothing`)\n\n"
        "Look at the rollout code below: which keys does the game loop update, and which of them "
        "must already exist before the first round?"
    )

st.markdown("Your TODOs, filled into the notebook's code:")
st.code(
    "trajectories = [\n"
    "    art.Trajectory(\n"
    "        messages_and_choices=[\n"
    "            {\n"
    '                "role": "system",\n'
    f'                "content": {instruction!r},\n'
    "            },\n"
    '            {"role": "user", "content": "What will your first move be?"},\n'
    "        ],\n"
    "        tools=tools,\n"
    "        reward=0,\n"
    f"        metrics={metrics_template if metrics_template is not None else '...'!r},\n"
    "    )\n"
    "    for _ in range(2)\n"
    "]",
    language="python", wrap_lines=True,
)
code_panel("The full rollout() from the notebook", src.ROLLOUT,
           "This is the code that runs when you press the button, against the same base model. "
           "Only the GPU training around it is skipped.")
code_panel("normalize_move and get_tool_call_id_and_move", src.HELPERS)

run = st.button("Run rollout()", type="primary", icon=":material/play_arrow:", disabled=metrics_error is not None)

# ---------------------------------------------------------------- run
if run:
    try:
        llm.reserve_calls(rpls.MAX_CALLS)
    except llm.BudgetExceeded as e:
        st.warning(str(e))
    else:
        board = st.empty()

        def show_progress(result: rpls.RolloutResult) -> None:
            last = result.rounds[-1]
            board.info(
                f"Round {last.number}: {viz.ICON[last.moves[0]]} {last.moves[0]} vs "
                f"{viz.ICON[last.moves[1]]} {last.moves[1]} · score {result.rewards[0]} : {result.rewards[1]}",
                icon=":material/sports_esports:",
            )

        with st.spinner("Playing… each round is two real calls to Qwen2.5-7B-Instruct"):
            result = rpls.rollout(llm.client(), instruction, metrics_template, on_round=show_progress)
        board.empty()
        llm.release_calls(rpls.MAX_CALLS - result.calls_used)
        st.session_state.rpls_result = result
        st.session_state.setdefault("rpls_runs", []).append({
            "instruction": instruction[:60] + ("…" if len(instruction) > 60 else ""),
            "metrics": repr(metrics_template),
            "rounds": len(result.rounds),
            "score": f"{result.rewards[0]} : {result.rewards[1]}",
            "draws": sum(r.winner is None for r in result.rounds),
            "'nothing' moves": sum(r.moves.count("nothing") for r in result.rounds),
            "error": result.error or "",
        })

result: rpls.RolloutResult | None = st.session_state.get("rpls_result")
if result is not None:
    st.divider()
    st.markdown("#### The match")

    if result.error:
        st.error(f"**The rollout crashed**, exactly as it would in the notebook:\n\n`{result.error}`",
                 icon=":material/error:")
        if result.error.startswith("KeyError"):
            st.markdown(
                "The game loop runs `trajectory.metrics[\"num_rounds\"] += 1` every round. "
                "If that key isn't in your metrics dict, the first round crashes. Fix TODO 2 and run again."
            )
        else:
            st.markdown(
                "In training, ART collects many rollouts and tolerates some failures "
                "(`max_exceptions`), but a crashed rollout produces no trajectory to learn from."
            )

    if result.rounds:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(f"Reward · {PLAYERS[0]}", result.rewards[0])
        s2.metric(f"Reward · {PLAYERS[1]}", result.rewards[1])
        s3.metric("Rounds played", len(result.rounds))
        s4.metric("Draws", sum(r.winner is None for r in result.rounds))

        mirrored = sum(r.moves[0] == r.moves[1] for r in result.rounds)
        if mirrored == len(result.rounds) and len(result.rounds) > 1:
            st.info(
                "Both players chose **the same move in every round**. Same model, same instruction, "
                "same conversation: they behave identically, so nobody can win. This is the starting "
                "point of training; RL is what makes the trainable player diverge from the base model.",
                icon=":material/content_copy:",
            )

        left, right = st.columns([1.1, 1], gap="large")
        with left:
            st.dataframe(pd.DataFrame([
                {"round": r.number,
                 "player 1": f"{viz.ICON[r.moves[0]]} {r.moves[0]}",
                 "player 2": f"{viz.ICON[r.moves[1]]} {r.moves[1]}",
                 "result": "draw" if r.winner is None else f"player {r.winner + 1} wins"}
                for r in result.rounds
            ]), hide_index=True)
        with right:
            st.plotly_chart(viz.move_distribution(result.metrics, ["player 1", "player 2"]),
                            config={"displayModeBar": False})
            st.caption("Move frequencies, read straight from the metrics dicts.")

    st.markdown("**The metrics dicts after the rollout**")
    m1, m2 = st.columns(2)
    m1.code(f"# {PLAYERS[0]}\nmetrics = {result.metrics[0]!r}", language="python", wrap_lines=True)
    m2.code(f"# {PLAYERS[1]}\nmetrics = {result.metrics[1]!r}", language="python", wrap_lines=True)

    st.markdown("#### The trajectories")
    st.markdown("`rollout()` returns **player 1's** trajectory for training. Every tool call is a generated "
                "(trained) message; tool results and user turns are context.")
    tab1, tab2 = st.tabs([PLAYERS[0] + " · returned for training", PLAYERS[1]])
    with tab1:
        render_trajectory(result.trajectory(0), title="Trajectory · player 1")
    with tab2:
        render_trajectory(result.trajectory(1), title="Trajectory · player 2")

runs = st.session_state.get("rpls_runs", [])
if len(runs) > 1:
    st.markdown("#### Your runs")
    st.dataframe(pd.DataFrame(runs), hide_index=True)

st.divider()
st.markdown("#### Check yourself")
for question, answer in [
    ("Your rollout played 10 rounds and both rewards are 0. Was it a bad rollout?",
     "Not necessarily: the reward says *who won*, not *how they played*. The metrics tell you whether "
     "the agents called the tool, played valid moves and how varied their moves were."),
    ("Why does `normalize_move` turn anything unexpected into `nothing` instead of raising an error?",
     "So every outcome becomes measurable, even failure. A missing or invalid move shows up as a "
     "`nothing` count in the metrics instead of silently disappearing."),
    ("What would you need to change to train an agent that actually wins?",
     "Nothing in the trajectory or metrics structure: you'd run many rollouts, group them and train "
     "on the reward. That's the part of the notebook that runs on a GPU after the session."),
]:
    st.markdown(f"**{question}**")
    with st.expander("Reveal"):
        st.markdown(answer)

takeaway("<b>Trajectories describe behavior. Metrics make behavior debuggable.</b> Once those two are "
         "correct, you can scale to better rewards, judge models, group-based methods and multi-step tool "
         "agents without changing the core structure.")
