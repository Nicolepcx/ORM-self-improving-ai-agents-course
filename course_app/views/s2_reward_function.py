import time

import plotly.graph_objects as go
import streamlit as st

from core import llm
from core.ui import code_panel, page_header, takeaway, task_card
from labs import reward_function as rf
from labs import s2_notebook_source as src

ss = st.session_state
CFG = rf.TRAINING_CONFIG

page_header(
    session=2,
    title="Reward functions: teach a new skill with just a task description",
    subtitle=(
        "With ART + RULER you don't write labels or a scoring function. You **describe the task in words**. "
        "From that description a model generates the training inputs, the policy answers each one several times, "
        "and a **judge** scores the answers against your description. Those scores are the reward."
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
        "**Run one training step** on your task: the policy answers a few inputs, the judge scores each group. "
        "Read which answers would be reinforced and why.",
        "**Sharpen the description** (format, priorities, constraints) and run the step again. Do the scores and "
        "the ranking change?",
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

c1, c2, c3, c4 = st.columns(4)
c1.markdown("**1 · Training inputs**\n\nGenerated from your description: the prompts the model practises on. "
            "No answers.")
c2.markdown(f"**2 · Rollouts**\n\nThe policy answers each input **{CFG['rollouts_per_group']}×** "
            "(`rollouts_per_group`): a group to compare.")
c3.markdown("**3 · Judge**\n\nScores every answer in the group from 0 to 1 against **your description**.")
c4.markdown("**4 · Update**\n\nART trains a LoRA adapter so above-average answers become more likely. "
            "*Needs a GPU: run it in Colab.*")
st.caption(f"Models: the policy is `{rf.POLICY_MODEL}`, one of the notebook's `BASE_MODEL` options (its default "
           f"Qwen3-4B-Instruct isn't available on OpenRouter). Inputs, system prompt and judge use "
           f"`{rf.INPUT_GENERATION_MODEL}`; the notebook uses kimi-k2 and deepseek-r1, which were too slow for a live session.")
code_panel("Notebook settings", src.SETTINGS)

st.divider()

# ---------------------------------------------------------------- TODO
st.markdown("#### TODO · Write your TASK_DESCRIPTION")
names = list(rf.TASK_DESCRIPTIONS) + ["CUSTOM_TASK_DESCRIPTION"]
choice = st.radio("Sample tasks", names, format_func=lambda n: rf.task_label(n) if n in rf.TASK_DESCRIPTIONS else "Custom",
                  horizontal=True, key="s2_task_choice")
if choice == "CUSTOM_TASK_DESCRIPTION":
    task_description = st.text_area("Describe your custom task (be specific!)", key="s2_custom_task", height=200,
                                    placeholder="Describe the input the model gets and the exact output format you expect…")
else:
    task_description = st.text_area("Task description (edit it to sharpen your reward specification)",
                                    rf.TASK_DESCRIPTIONS[choice].strip(), height=220, key=f"s2_desc_{choice}")
st.code(f"TASK_DESCRIPTION = {choice if choice != 'CUSTOM_TASK_DESCRIPTION' else 'CUSTOM_TASK_DESCRIPTION'}",
        language="python")
task_ready = bool(task_description.strip())

st.divider()

# ---------------------------------------------------------------- 1 · training inputs
st.markdown("#### 1 · Generate the training inputs")
st.markdown(
    f"`{rf.INPUT_GENERATION_MODEL}` reads your description and invents {CFG['num_training_inputs']} realistic, "
    "different **inputs**: what a user would send. They have **no answers attached**: the model writes the answers "
    "during training, and the judge decides how good they are."
)
code_panel("generate_training_inputs and extract_json_object", [rf.generate_training_inputs, rf.extract_json_object],
           "The prompt is a little clearer than the notebook's: it insists that inputs never contain the expected output.")

if st.button(f"Generate {CFG['num_training_inputs']} training inputs", icon=":material/dataset:", disabled=not task_ready):
    try:
        llm.reserve_calls(rf.MAX_GENERATION_ATTEMPTS)
    except llm.BudgetExceeded as e:
        st.warning(str(e))
    else:
        calls, start = rf.MAX_GENERATION_ATTEMPTS, time.time()
        with st.status(f"Calling {rf.INPUT_GENERATION_MODEL}…", expanded=True) as status:
            try:
                inputs, calls = rf.generate_training_inputs(llm.client(), task_description,
                                                            num_examples=CFG["num_training_inputs"], log=st.text)
                ss.s2_inputs = {"task": task_description, "inputs": inputs}
                status.update(label=f"Generated {len(inputs)} training inputs in {time.time() - start:.0f}s",
                              state="complete", expanded=False)
            except Exception as e:
                ss.s2_inputs = {"task": task_description, "error": f"{type(e).__name__}: {e}"}
                status.update(label="Generation failed", state="error")
        llm.release_calls(rf.MAX_GENERATION_ATTEMPTS - calls)
        ss.pop("s2_step", None)

generated = ss.get("s2_inputs")
inputs_ready = bool(generated and "inputs" in generated)
if generated:
    if generated.get("task") != task_description:
        st.caption("These inputs were generated for a different task description. Generate again to match it.")
    if "error" in generated:
        st.error(f"`{generated['error']}`")
    else:
        with st.expander(f"The {len(generated['inputs'])} training inputs", expanded=True):
            cols = st.columns(3)
            for i, text in enumerate(generated["inputs"]):
                with cols[i % 3].container(border=True):
                    st.caption(f"Input {i + 1}")
                    st.markdown(text)

st.markdown("##### How these inputs become training")
steps = CFG["num_training_inputs"] * CFG["num_epochs"] // CFG["groups_per_step"]
st.markdown(
    f"With the notebook's `TRAINING_CONFIG`, each **training step** takes `groups_per_step = {CFG['groups_per_step']}` "
    f"input, lets the model answer it `rollouts_per_group = {CFG['rollouts_per_group']}` times, has the judge score "
    f"that group, and updates the model. `num_epochs = {CFG['num_epochs']}` passes over the "
    f"{CFG['num_training_inputs']} inputs make **{steps} training steps**, "
    f"**{steps * CFG['rollouts_per_group']} answers** and **{steps} judge calls**."
)
fig = go.Figure(go.Heatmap(
    z=[[e + 1] * CFG["num_training_inputs"] for e in range(CFG["num_epochs"])],
    x=[f"{i + 1}" for i in range(CFG["num_training_inputs"])], y=[f"epoch {e + 1}" for e in range(CFG["num_epochs"])],
    text=[[f"step {e * CFG['num_training_inputs'] + i + 1}" for i in range(CFG["num_training_inputs"])]
          for e in range(CFG["num_epochs"])],
    hovertemplate="input %{x} · %{y} · %{text}<extra></extra>", colorscale=[[0, "#cfe0f7"], [1, "#2f7de1"]],
    showscale=False, xgap=3, ygap=3))
fig.update_layout(height=190, margin=dict(l=10, r=10, t=10, b=30), xaxis_title="training input")
st.plotly_chart(fig, config={"displayModeBar": False})
st.caption("Each cell is one training step. The same input comes back in every epoch, answered by the model as it "
           "is *at that point*: if training works, the later answers score higher.")
code_panel("The notebook's training loop", src.TRAINING_LOOP)

st.divider()

# ---------------------------------------------------------------- 2 · one training step
st.markdown("#### 2 · Run one training step on your task")
st.markdown(
    "This runs the notebook's loop for real on your first inputs, up to `model.train()`: the description becomes the "
    "policy's **system prompt**, the policy writes a **group of answers** per input, and the judge scores each group."
)
c1, c2 = st.columns(2)
n_inputs = c1.slider("Inputs (one group each)", 1, 3, 2, key="s2_n_inputs")
group_size = c2.slider("Answers per input (rollouts_per_group)", 2, 6, 4, key="s2_group",
                       help=f"The notebook uses {CFG['rollouts_per_group']}. More comparisons give the judge a richer "
                            "ranking, one of the notebook's suggested improvements.")
code_panel("generate_system_prompt and rollout (notebook)", src.SYSTEM_PROMPT_GENERATION + "\n\n" + src.ROLLOUT)
code_panel("robust_score_group (the judge)", rf.robust_score_group,
           "The rubric is the reward function. `extract_json_object` makes it robust to code fences and extra text. "
           "Unlike the notebook, the judge also sees the input the answers respond to, so it can check they're faithful to it.")

if st.button("Run one training step", type="primary", icon=":material/play_arrow:",
             disabled=not (task_ready and inputs_ready)):
    calls = 1 + n_inputs * group_size + n_inputs
    try:
        llm.reserve_calls(calls)
    except llm.BudgetExceeded as e:
        st.warning(str(e))
    else:
        start = time.time()
        with st.status("Running one training step…", expanded=True) as status:
            try:
                prompts = ss.setdefault("s2_system_prompts", {})
                if task_description not in prompts:
                    st.text(f"Turning your description into a system prompt ({rf.SYSTEM_PROMPT_GENERATION_MODEL})…")
                    prompts[task_description] = rf.generate_system_prompt(llm.client(), task_description)
                st.text(f"{rf.POLICY_MODEL} writes {group_size} answers to each of {n_inputs} input(s); "
                        f"{rf.RULER_MODEL} judges each group…")
                groups = rf.training_step_preview(llm.client(), task_description, prompts[task_description],
                                                  generated["inputs"][:n_inputs], group_size)
                ss.s2_step = {"task": task_description, "system_prompt": prompts[task_description], "groups": groups,
                              "seconds": time.time() - start}
                status.update(label=f"Training step done in {time.time() - start:.0f}s", state="complete",
                              expanded=False)
            except Exception as e:
                ss.s2_step = {"task": task_description, "error": f"{type(e).__name__}: {e}"}
                status.update(label="The training step failed", state="error")
        step = ss.s2_step
        if "groups" in step:
            scored = [g for g in step["groups"] if g.judgement]
            all_scores = [s for g in scored for s in g.judgement.scores]
            ss.setdefault("s2_runs", []).append({
                "task": rf.task_label(choice) if choice in rf.TASK_DESCRIPTIONS else "Custom",
                "description edited": choice in rf.TASK_DESCRIPTIONS
                                      and task_description.strip() != rf.TASK_DESCRIPTIONS[choice].strip(),
                "mean score": round(sum(all_scores) / len(all_scores), 2) if all_scores else None,
                "groups with no signal": sum(1 for g in scored if max(g.judgement.scores) - min(g.judgement.scores) < 1e-9),
                "groups": len(step["groups"]),
            })

step = ss.get("s2_step")
if step:
    if step["task"] != task_description:
        st.caption("This step was run for a different task description. Run it again to match it.")
    if "error" in step:
        st.error(f"`{step['error']}`")
    else:
        with st.expander("The system prompt generated from your description", expanded=False):
            st.markdown(step["system_prompt"])
        for gi, group in enumerate(step["groups"], 1):
            st.markdown(f"##### Group {gi} · input: *{group.input_text}*")
            if group.error:
                st.error(f"`{group.error}`")
                continue
            scores, advs = group.judgement.scores, group.advantages
            per_row = 2
            cols = []
            for _ in range(0, len(group.answers), per_row):
                cols += st.columns(per_row, gap="medium")
            for k, (col, answer, score, adv, why) in enumerate(
                    zip(cols, group.answers, scores, advs, group.judgement.rationales), 1):
                with col.container(border=True):
                    st.caption(f"Answer {k}")
                    st.text(answer[:600] + ("…" if len(answer) > 600 else ""))
                    st.metric("Judge score (reward)", f"{score:.2f}")
                    if abs(adv) < 1e-9:
                        st.markdown("**± 0** · no push")
                    elif adv > 0:
                        st.markdown(f":green[**▲ {adv:+.2f}** · more likely]")
                    else:
                        st.markdown(f":red[**▼ {adv:+.2f}** · less likely]")
                    st.caption(why)
            if max(scores) - min(scores) < 1e-9:
                st.info("Every answer in this group got the **same score**, so all advantages are 0: this group would "
                        "**not change the model at all**. Learning only comes from differences *within* a group. A "
                        "stricter description, or more answers per input, can create those differences.",
                        icon=":material/drag_handle:")
            else:
                best = max(range(len(scores)), key=scores.__getitem__)
                st.caption(f"Advantage = (score − group mean {sum(scores) / len(scores):.2f}) / group std. Training "
                           f"pushes answer {best + 1} up the most, even if every answer is decent: only the ranking "
                           "inside the group matters.")
            with st.expander("Raw judge output"):
                st.code(group.judgement.raw, language="json", wrap_lines=True)
        st.markdown(
            "**What `model.train()` would do next:** nudge the LoRA weights so that answers like the green ones "
            "become more likely for inputs like these, and the red ones less likely. Then the next training step "
            "takes the next input. Over all steps the model learns your task, guided only by your description."
        )

runs = ss.get("s2_runs", [])
if len(runs) > 1:
    st.markdown("##### Your training steps")
    st.dataframe(runs, hide_index=True)
    st.caption("Compare runs after editing the description: a sharper reward specification changes the scores, and "
               "which answers get reinforced.")

with st.expander("The notebook's judge demo (three hand-written bug reports)"):
    st.markdown(
        "The notebook also tests the judge on one input and three fixed answers: good, mediocre and bad **bug "
        "reports**. They only make sense for a bug-report task description; for the sample tasks above the judge "
        "correctly gives all three 0. That's why this page scores answers the policy wrote **for your task** instead."
    )
    st.markdown(f"**test_input**: {rf.JUDGE_DEMO_INPUT}")
    cols = st.columns(3)
    for col, (name, text) in zip(cols, rf.JUDGE_DEMO_CANDIDATES.items()):
        with col.container(border=True):
            st.caption(name)
            st.text(text)
    code_panel("The judge demo cell", src.JUDGE_DEMO)

st.divider()
st.markdown("#### Check yourself")
for question, answer in [
    ("Where does the reward come from in this setup?",
     "From the judge, which scores the policy's answers against your task description. Change the description and "
     "you change the reward: the description is your reward specification."),
    ("Why does the model answer each input several times?",
     "Relative methods need comparisons. With one answer per input there is nothing to rank inside the group, so "
     "there is no signal for which way to push."),
    ("A group where every answer scores 1.0: good news for training?",
     "No: all advantages are 0, so that step teaches nothing. A more demanding description, or more answers per "
     "input, creates the differences learning needs."),
    ("Why do the training inputs not need answers?",
     "The model writes the answers itself during training, and the judge decides how good they are. You only need "
     "realistic prompts for the skill."),
]:
    st.markdown(f"**{question}**")
    with st.expander("Reveal"):
        st.markdown(answer)

takeaway("<b>Task descriptions are your reward specification.</b> A judge turns them into scores, the group turns "
         "scores into relative advantages, and training makes the better answers more likely.")
