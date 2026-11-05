import streamlit as st

from core.trajectory_view import render_trajectory
from core.ui import code_panel, page_header, takeaway, task_card
from labs import additional_histories as ah
from labs import s4_notebook_source as src

page_header(
    session=4,
    title="Additional histories: one reward, many conversations",
    subtitle=(
        "In real agent systems the important work often happens **outside** the main conversation: "
        "in sub-agents, in reasoning tokens the chat template drops, or in turns that were later summarized. "
        "**Additional histories** attach those conversations to the trajectory, so they share its reward "
        "and get trained too."
    ),
    minutes=15,
    notebook_path="hands_on/session_04_HANDS_ON_additional_histories.ipynb",
)
task_card(
    task="Create a trajectory with additional history from multi-turn conversations within your multi-agent "
         "collaboration.",
    todos=[
        "**Open the three trajectories:** sub-agent delegation, history compaction, reasoning tokens.",
        "**Inspect their structure:** the main `messages_and_choices`, the `additional_histories` and the `metrics`. "
        "The ART code panel under each shows how the trajectory is created.",
        "**Flip “What does training see?”** and note what the model would never learn from with the main "
        "conversation alone.",
        "**④ Reward:** see how one reward, computed from the main conversation, is shared by every history. "
        "To create your own, copy one example in the notebook and add your agents' conversations.",
    ],
    why=(
        "In real agent systems the main user-facing conversation is often **not** where the work happens. "
        "It happens in sub-agent conversations, tool-call traces, reasoning tokens the chat template removes, "
        "and long conversations that later get summarized.\n"
        "- If you train only on the final coordinator transcript, you lose the information that **caused** the "
        "reward. That breaks **credit assignment** and slows learning.\n"
        "- **Additional histories** attach those conversations to the trajectory, so the model can learn how to "
        "delegate, how to evaluate sub-agent output and how to combine results.\n"
        "- RULER can't score additional histories yet, so the reward comes from the main conversation; the "
        "histories carry the **training signal**."
    ),
    remember="The final chat transcript is not the full behaviour of an agent. Additional histories are how you "
             "train on what actually happened.",
)

st.markdown("##### How to read a trajectory")
c1, c2, c3 = st.columns(3)
c1.markdown("**Context** · plain dicts\n\nSystem, user and tool messages. The model reads them but is **not** trained on them.")
c2.markdown("**● Trained** · `Choice` objects\n\nWhat the model *generated*. Only these tokens are pushed up or down by the reward.")
c3.markdown("**Additional history** · `History`\n\nAnother conversation inside the same trajectory. Same reward, its own trained tokens.")

VIEW_OPTIONS = ["Main conversation only", "Main + additional histories"]


def example_tab(key: str, trajectory: ah.Trajectory, intro: str, missed: str, code: str) -> None:
    st.markdown(intro)
    view = st.segmented_control("What does training see?", VIEW_OPTIONS, default=VIEW_OPTIONS[1], key=f"view_{key}")
    with_histories = view != VIEW_OPTIONS[0]

    main_tokens = ah.trained_tokens(trajectory.messages)
    history_tokens = sum(ah.trained_tokens(h.messages) for h in trajectory.additional_histories)
    seen = main_tokens + (history_tokens if with_histories else 0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Messages in main conversation", len(trajectory.messages))
    m2.metric("Additional histories", len(trajectory.additional_histories))
    m3.metric("Trained tokens (≈)", seen, help="Rough estimate: ~4 characters per token, generated messages only")
    m4.metric("Training signal kept", f"{seen / (main_tokens + history_tokens):.0%}")

    if with_histories:
        st.success("Every generated message, in every conversation, is trained with the trajectory's reward.",
                   icon=":material/check_circle:")
    else:
        st.warning(missed, icon=":material/visibility_off:")

    render_trajectory(trajectory, histories_trained=with_histories)
    code_panel("The ART code for this trajectory", code,
               "Dicts are context, `Choice` objects are trained, `additional_histories` hold the extra conversations.")


tab_sub, tab_compact, tab_think, tab_reward, tab_check = st.tabs(
    ["① Sub-agent delegation", "② History compaction", "③ Reasoning tokens", "④ Reward", "⑤ Check yourself"]
)

with tab_sub:
    example_tab(
        "sub",
        ah.SUB_AGENTS,
        "The coordinator only *calls* `analyze_code` and `fix_issues`. The actual analysis and fixing "
        "happen in two sub-agent conversations, each hanging off the tool call that started it.",
        "The model only learns *that* it called two tools and wrote a summary. How the sub-agents found "
        "and fixed the bug, which is what earned the reward, is never trained. That breaks credit assignment.",
        src.SUB_AGENTS,
    )

with tab_compact:
    example_tab(
        "compact",
        ah.COMPACTION,
        "A long-running agent replaced its first turns with a one-line summary to save context. "
        "The original turns are kept as an additional history, attached to the summary that replaced them.",
        "The first two answers only survive as a one-line summary, which is context, not something the model "
        "generated. Those answers would never be trained, even though they were part of the conversation that got the reward.",
        src.COMPACTION,
    )

with tab_think:
    example_tab(
        "think",
        ah.THINK_TOKENS,
        "Reasoning models write `<think>` blocks (highlighted). On the next turn, the Qwen 3 chat template "
        "**removes** the earlier `<think>` from the prompt. So the conversation the model saw on turn 2 differs "
        "from turn 1's transcript, and is stored as its own history.",
        "Only turn 1 is trained. Turn 2, which the model generated from a prompt with the earlier reasoning "
        "stripped, is lost, and so is its reasoning.",
        src.THINK_TOKENS,
    )

# ---------------------------------------------------------------- ④ reward
with tab_reward:
    st.markdown(
        "One trajectory gets **one reward**, and every history in it is trained with that number. "
        "RULER can't score trajectories with additional histories yet, so the notebook uses a simple "
        "reward function that looks only at the **final answer of the main conversation**."
    )
    examples = {"Sub-agent delegation": ah.SUB_AGENTS, "History compaction": ah.COMPACTION,
                "Reasoning tokens": ah.THINK_TOKENS}

    rows = []
    for name, traj in examples.items():
        score, _ = ah.simple_reward_function(ah.final_answer(traj), traj.metrics)
        rows.append({"Example": name, "Reward stored in the trajectory": traj.reward,
                     "simple_reward_function": round(score, 2),
                     "Additional histories": len(traj.additional_histories)})
    st.dataframe(rows, hide_index=True)
    st.caption("Why does *Reasoning tokens* score 0.0? Its final answer is “4”: none of the keyword rules fit a math task. "
               "A generic heuristic is not a good reward for every task.")

    st.divider()
    st.markdown("#### Try it: edit the final answer and watch the reward")
    name = st.selectbox("Trajectory", list(examples), key="reward_example")
    traj = examples[name]
    answer = st.text_area("Final assistant message (main conversation)", ah.final_answer(traj),
                          height=130, key=f"answer_{name}")
    score, rules = ah.simple_reward_function(answer, traj.metrics)
    left, right = st.columns([1, 2], gap="large")
    left.metric("Reward", f"{score:.2f}")
    with right:
        for rule, points, fired in rules:
            st.markdown(f"{'✅' if fired else '⬜'} {rule} &nbsp; **+{points}**")
    st.caption("Notice what the reward can't see: nothing in the sub-agent or original histories changes this number.")
    code_panel("simple_reward_function (from the notebook)", src.REWARD_FUNCTION)

# ---------------------------------------------------------------- ⑤ check
with tab_check:
    questions = [
        ("A coordinator gets reward 1.0 after delegating to a research sub-agent. "
         "Without additional histories, which behaviour gets reinforced?",
         "Only the coordinator's own tokens: the decision to call the tool and the final write-up. "
         "The research that actually made the answer good is never reinforced, so the sub-agent can't improve."),
        ("Why are the summary message in example ② and the tool results in example ① not trained?",
         "They are plain dicts, i.e. context the model read, not text it generated. Training only "
         "adjusts the probability of tokens the model itself produced (the `Choice` messages)."),
        ("All histories share one reward. When could that be a problem?",
         "When a history did poorly but the overall trajectory still scored well (or the reverse). The "
         "bad sub-agent gets reinforced too. That's the price of a single trajectory-level reward; "
         "per-history rewards would need a finer-grained evaluation."),
        ("RULER doesn't support additional histories yet. What can you do?",
         "Score the main conversation (with RULER or a custom function) and train the full trajectory with "
         "that reward, as the notebook does, or build a simplified trajectory just for scoring."),
    ]
    for question, answer in questions:
        st.markdown(f"**{question}**")
        with st.expander("Reveal"):
            st.markdown(answer)

    takeaway(
        "<b>If it influenced the reward, it should be in the trajectory.</b> Additional histories let one "
        "trajectory carry every conversation that produced the outcome (sub-agents, compacted turns, "
        "per-turn views of reasoning), so credit flows to the work that earned it."
    )
