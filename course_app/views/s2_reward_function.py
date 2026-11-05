import time

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import llm
from core.ui import code_panel, page_header, takeaway, task_card
from labs import reward_function as rf
from labs import s2_notebook_source as src

page_header(
    session=2,
    title="Reward functions: let a judge rank the answers",
    subtitle=(
        "You will train a small model on a **custom task**. Nobody writes labels: a model generates the "
        "training inputs, and a **judge model** scores the policy's answers in groups. Those scores are the reward. "
        "The task description is the one thing you control, so it is your **reward specification**."
    ),
    minutes=15,
    notebook_path="hands_on/session_02_HANDS_ON_reward_function.ipynb",
)
task_card(
    task="Write a task description for your reward function to teach your AI agent a new skill.",
    todos=[
        "**Write your `TASK_DESCRIPTION`**: pick a sample task or write your own. Be specific about the input "
        "and the exact output format.",
        "**Generate the training inputs** and check them: are they realistic *inputs* for your task?",
        "**Run the judge demo** and read how the judge scores each answer against your description.",
        "**Change the description** (or switch task) and judge again. Does the ranking change?",
    ],
    why=(
        "In this setup nobody writes labels. A model generates the training inputs, the policy produces a **group** "
        "of answers per input, and a **judge** (RULER-style) scores them. Training then makes the higher-scored "
        "answers more likely.\n"
        "- That makes the **task description your reward specification**, the most important control knob: "
        "the judge only knows what “good” means from what you write.\n"
        "- You don't need one perfect label, you need **comparisons and ranking signals**.\n"
        "- Making the description more precise about priorities, constraints and format is a first-class way to "
        "improve the trained model."
    ),
    remember="RULER learns what “good” means entirely from your task description and relative comparisons. "
             "No labeled outputs required.",
)

if not llm.is_configured():
    st.error("This page needs the instructor's OpenRouter key, which isn't configured on the server yet.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.markdown(f"**A · Generate rollouts**\n\nThe policy writes `rollouts_per_group = 2` answers per input. "
            "You need *comparisons*, not one perfect label.")
c2.markdown(f"**B · Judge rollouts**\n\n`{rf.RULER_MODEL}` scores every answer in a group from 0 to 1 "
            "and returns strict JSON.")
c3.markdown("**C · Update the policy**\n\nART trains a LoRA adapter so high-reward answers become more likely. "
            "*This step needs a GPU: run it in Colab.*")
st.caption(f"The notebook uses `moonshotai/kimi-k2` to generate inputs and `deepseek/deepseek-r1-0528` as judge. "
           f"This lab uses `{rf.INPUT_GENERATION_MODEL}` for both, so it responds within a live session.")
code_panel("Notebook settings", src.SETTINGS)

st.divider()

# ---------------------------------------------------------------- TODO
st.markdown("#### TODO · Choose your TASK_DESCRIPTION")
names = list(rf.TASK_DESCRIPTIONS) + ["CUSTOM_TASK_DESCRIPTION"]
choice = st.radio("Sample tasks", names, format_func=lambda n: rf.task_label(n) if n in rf.TASK_DESCRIPTIONS else "Custom",
                  horizontal=True, key="s2_task_choice")
if choice == "CUSTOM_TASK_DESCRIPTION":
    task_description = st.text_area("Describe your custom task (be specific!)", key="s2_custom_task", height=200,
                                    placeholder="Describe the input the model gets and the exact output format you expect…")
else:
    task_description = rf.TASK_DESCRIPTIONS[choice]
    st.code(task_description.strip(), language="text", wrap_lines=True)
st.code(f"TASK_DESCRIPTION = {choice}", language="python")
task_ready = bool(task_description.strip())

st.divider()

# ---------------------------------------------------------------- generate dataset
st.markdown("#### 1 · Generate the training dataset")
st.markdown(
    f"`{rf.INPUT_GENERATION_MODEL}` invents {rf.NUM_TRAINING_INPUTS} realistic, different inputs for your task. "
    "Nothing is labelled: the judge will provide the signal later."
)
code_panel("generate_training_inputs and extract_json_object", [rf.generate_training_inputs, rf.extract_json_object])

if st.button(f"Generate {rf.NUM_TRAINING_INPUTS} training inputs", icon=":material/dataset:", disabled=not task_ready):
    try:
        llm.reserve_calls(rf.MAX_GENERATION_ATTEMPTS)
    except llm.BudgetExceeded as e:
        st.warning(str(e))
    else:
        logs, calls = [], rf.MAX_GENERATION_ATTEMPTS
        start = time.time()
        with st.status(f"Calling {rf.INPUT_GENERATION_MODEL}…", expanded=True) as status:
            try:
                inputs, calls = rf.generate_training_inputs(
                    llm.client(), task_description, num_examples=rf.NUM_TRAINING_INPUTS,
                    log=lambda line: (logs.append(line), st.text(line)),
                )
                st.session_state.s2_inputs = {"task": task_description, "inputs": inputs}
                status.update(label=f"Generated {len(inputs)} training inputs in {time.time() - start:.0f}s",
                              state="complete", expanded=False)
            except Exception as e:
                st.session_state.s2_inputs = {"task": task_description, "error": f"{type(e).__name__}: {e}"}
                status.update(label="Generation failed", state="error")
        llm.release_calls(rf.MAX_GENERATION_ATTEMPTS - calls)

generated = st.session_state.get("s2_inputs")
if generated:
    if generated.get("task") != task_description:
        st.caption("These inputs were generated for a different task description. Generate again to match your current task.")
    if "error" in generated:
        st.error(f"`{generated['error']}`")
    else:
        st.markdown(f"**Generated {len(generated['inputs'])} training inputs**")
        cols = st.columns(3)
        for i, text in enumerate(generated["inputs"]):
            with cols[i % 3].container(border=True):
                st.caption(f"Example {i + 1}")
                st.markdown(text)

st.divider()

# ---------------------------------------------------------------- judge demo
st.markdown("#### 2 · Judge demo")
st.markdown(
    "The notebook gives the judge one input and three hand-written answers (good, mediocre and bad bug reports) "
    "as a **group**. `robust_score_group` asks the judge to score each answer against **your TASK_DESCRIPTION**."
)
st.markdown(f"**test_input**: {rf.JUDGE_DEMO_INPUT}")
cols = st.columns(3)
for col, (name, text) in zip(cols, rf.JUDGE_DEMO_CANDIDATES.items()):
    with col.container(border=True):
        st.caption(name)
        st.text(text)
st.caption("These three answers are bug reports. Predict how the judge will score them for the task you selected.")
code_panel("robust_score_group", rf.robust_score_group,
           "The rubric is the reward function. `extract_json_object` makes it robust to code fences and extra text.")
code_panel("The judge demo cell", src.JUDGE_DEMO)

if st.button("Score the group with the judge", type="primary", icon=":material/gavel:", disabled=not task_ready):
    try:
        llm.reserve_calls(1)
    except llm.BudgetExceeded as e:
        st.warning(str(e))
    else:
        start = time.time()
        with st.spinner(f"{rf.RULER_MODEL} is judging the answers…"):
            try:
                judgement = rf.robust_score_group(llm.client(), list(rf.JUDGE_DEMO_CANDIDATES.values()),
                                                  judge_model=rf.RULER_MODEL, task_description=task_description)
                st.session_state.s2_judgement = {"task": task_description, "result": judgement,
                                                 "seconds": time.time() - start}
            except Exception as e:
                st.session_state.s2_judgement = {"task": task_description, "error": f"{type(e).__name__}: {e}"}
        runs = st.session_state.setdefault("s2_runs", [])
        j = st.session_state.s2_judgement
        runs.append({"task": rf.task_label(choice) if choice in rf.TASK_DESCRIPTIONS else "Custom",
                     **({k: round(s, 2) for k, s in zip(rf.JUDGE_DEMO_CANDIDATES, j["result"].scores)}
                        if "result" in j else {"error": j["error"]})})

judged = st.session_state.get("s2_judgement")
if judged:
    if judged["task"] != task_description:
        st.caption("This judgement was made for a different task description. Score again to match your current task.")
    if "error" in judged:
        st.error(f"Scoring failed, as it would in the notebook: `{judged['error']}`. "
                 "In the training loop the notebook retries up to 10 times.")
    else:
        result: rf.Judgement = judged["result"]
        st.markdown(f"**The judge's verdict** · {judged['seconds']:.0f}s")
        ranked = sorted(zip(rf.JUDGE_DEMO_CANDIDATES, result.scores, result.rationales), key=lambda r: -r[1])
        for name, score, rationale in ranked:
            rank = 1 + sum(s > score for s in result.scores)
            tied = result.scores.count(score) > 1
            with st.container(border=True):
                left, right = st.columns([1, 3])
                left.metric(f"Rank {rank}{' (tied)' if tied else ''} · {name}", f"{score:.2f}")
                left.progress(score)
                right.markdown(f"**Rationale**: {rationale}")

        advantages = rf.group_relative_advantages(result.scores)
        labels = list(rf.JUDGE_DEMO_CANDIDATES)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Reward = judge score", "Group-relative advantage"))
        fig.add_trace(go.Bar(x=labels, y=result.scores, marker_color="#2f7de1"), row=1, col=1)
        fig.add_trace(go.Bar(x=labels, y=advantages,
                             marker_color=["#1fa37a" if a > 0 else "#d64545" for a in advantages]), row=1, col=2)
        fig.update_yaxes(range=[0, 1.05], row=1, col=1)
        fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, config={"displayModeBar": False})
        if max(result.scores) - min(result.scores) < 1e-9:
            st.warning(
                "All answers got the **same score**, so every advantage is 0: this group carries **no learning "
                "signal**. Group-relative methods only learn from differences *within* a group.",
                icon=":material/drag_handle:",
            )
        else:
            st.caption("Like Frozen Lake's advantage bars: answers above the group average are pushed up, the others down.")

        with st.expander("Raw judge output (before extract_json_object)"):
            st.code(result.raw, language="json", wrap_lines=True)

runs = st.session_state.get("s2_runs", [])
if len(runs) > 1:
    st.markdown("#### Your judge runs")
    st.dataframe(runs, hide_index=True)

st.divider()
st.markdown("#### Check yourself")
for question, answer in [
    ("Run the judge demo with the Grammarly task and with another task. Why do the same three answers get different scores?",
     "The judge scores answers against the TASK_DESCRIPTION. Change the description and you change the reward: "
     "the task description is your reward specification."),
    ("Why does the notebook generate 2 rollouts per input instead of 1?",
     "Relative methods need comparisons. With one answer per input there is nothing to rank within the group."),
    ("What would happen in training if the judge gave every answer in a group the same score?",
     "All group-relative advantages would be 0, so that group would not change the model at all."),
]:
    st.markdown(f"**{question}**")
    with st.expander("Reveal"):
        st.markdown(answer)

takeaway("<b>Task descriptions are your reward specification.</b> A judge turns them into scores, the group turns "
         "scores into relative advantages, and training makes the better answers more likely.")
