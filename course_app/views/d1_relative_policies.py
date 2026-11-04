import pandas as pd
import streamlit as st

from core.ui import code_panel, page_header, takeaway
from labs import relative_policies as rp
from labs import relative_policies_viz as viz

NB = rp.NOTEBOOK_SETTINGS

page_header(
    session=1,
    kind="DEMO",
    title="Relative policies: learning from rankings, not scores",
    subtitle=(
        "The same agent is trained twice on the same arithmetic prompts with the same **miscalibrated reward model**. "
        "One run uses **absolute feedback** (`reward − baseline`), the other **relative feedback within each prompt "
        "group** (z-score, as in GRPO). Same rewards, different advantages, and very different policies."
    ),
    minutes=10,
    notebook_path="demo/session_01_DEMO_relative_policies_RL.ipynb",
)


@st.cache_resource(max_entries=200, show_spinner="Training both methods…")
def run(settings: tuple) -> dict:
    return rp.run_demo(**dict(settings))


def settings_key(s: dict) -> tuple:
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in s.items()))


tab_setup, tab_train, tab_step, tab_rank, tab_why = st.tabs(
    ["① The setup", "② Train both", "③ Step by step", "④ Inside one prompt group", "⑤ Why GRPO"]
)

# ---------------------------------------------------------------- ① setup
with tab_setup:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(
            "**The agent** answers arithmetic prompts by choosing one of six behaviours. "
            "Some are reliable, some aren't (error rates from `maybe_corrupt_answer`):"
        )
        st.dataframe(pd.DataFrame([
            {"action": "use_tool", "what it does": "calls a calculator", "error rate": "2%"},
            {"action": "verify", "what it does": "answers, then checks", "error rate": "3%"},
            {"action": "plan", "what it does": "step-by-step", "error rate": "8%"},
            {"action": "direct_answer", "what it does": "just answers", "error rate": "10%"},
            {"action": "ramble", "what it does": "long-winded", "error rate": "35%"},
            {"action": "refuse", "what it does": "won't answer", "error rate": "100%"},
        ]), hide_index=True)
        st.markdown(f"**{len(rp.PROMPTS)} prompts** (8 additions, 6 multiplications, 5 subtractions), "
                    f"e.g. *What is 17 + 25?* Every step, each prompt gets {NB['group_size']} rollouts: "
                    "**one prompt group**, the unit the relative method compares within.")
    with right:
        st.markdown("**The reward model** scores each transcript:")
        st.markdown(
            "- **+1** if the final answer is correct\n"
            "- small bonuses for using a tool (+0.06), verifying (+0.04), planning (+0.02)\n"
            "- a **length penalty** of 0.0009 per character\n"
            "- **−0.7** for refusing"
        )
        code_panel("score_rollout (the reward model)", rp.score_rollout)

    st.markdown("##### Question to all")
    st.markdown("**If we removed the length penalty, would the model learn to ramble just to sound smarter?**")
    with st.expander("Reveal the answer"):
        st.markdown(f"### {rp.reveal_answer()}")

    st.divider()
    st.markdown("##### The noise map: a miscalibrated reward model")
    st.markdown(
        "A real reward model is just another LLM, and it's biased: some prompts get inflated scores, others deflated. "
        "The demo simulates that with a per-prompt **scale** and **bias**: "
        "`observed = clip(scale × raw + bias + noise)`. At step 40 the reward model **drifts**: every prompt "
        "gets a new, wider scale and bias, like a reward model update mid-training."
    )
    demo = run(settings_key(NB))
    drift_map = demo["abs_maps"][NB["noise_shift_step"]]
    st.plotly_chart(viz.noise_map_figure(rp.PROMPTS, demo["noise_map"], drift_map), config={"displayModeBar": False})
    st.caption("An absolute baseline can't tell “this prompt is scored generously” from “this answer is good”. "
               "Normalizing within each prompt group cancels the scale and bias.")
    code_panel("create_reward_noise_map and apply_reward_noise", [rp.create_reward_noise_map, rp.apply_reward_noise])

# ---------------------------------------------------------------- ② train
with tab_train:
    st.markdown("Train both methods with the notebook's settings, or change one thing and watch what happens.")
    with st.form("rp_settings"):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            reward_noise = st.toggle("Miscalibrated reward model (noise map)", value=NB["reward_noise"])
            st.caption("Off = every prompt is scored on the same scale.")
            drift = st.toggle("Reward model drift mid-training", value=NB["drift"])
            shift_step = st.slider("Drift at step", 5, 75, NB["noise_shift_step"], 5)
        with c2:
            group_size = st.slider("Group size (rollouts per prompt)", 2, 24, NB["group_size"])
            st.caption("How many answers are compared within each prompt group.")
            lr = st.slider("Learning rate", 0.02, 0.5, NB["lr"], 0.02)
        with c3:
            temperature = st.slider("Sampling temperature", 0.5, 2.0, NB["temperature"], 0.05)
            st.caption("Higher = more exploration, and more mistakes.")
            seed = st.number_input("Seed", 0, 9999, NB["seed"])
        submitted = st.form_submit_button("Train both methods", type="primary", icon=":material/play_arrow:")
    if submitted:
        st.session_state.rp_active_settings = {**NB, "reward_noise": reward_noise, "drift": drift,
                                        "noise_shift_step": shift_step, "group_size": group_size, "lr": lr,
                                        "temperature": temperature, "seed": int(seed)}
    settings = st.session_state.get("rp_active_settings", NB)
    is_notebook = settings == NB
    result = run(settings_key(settings))
    st.session_state.rp_result_settings = settings

    abs_s, rel_s = result["abs_stats"], result["rel_stats"]
    top = lambda pol: max(pol[-1], key=pol[-1].get)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Absolute: final correctness", f"{abs_s[-1]['correctness_rate']:.1%}")
    m2.metric("Relative: final correctness", f"{rel_s[-1]['correctness_rate']:.1%}",
              f"{rel_s[-1]['correctness_rate'] - abs_s[-1]['correctness_rate']:+.1%}")
    m3.metric("Absolute: learned behaviour", top(result["abs_pol"]),
              f"{result['abs_pol'][-1][top(result['abs_pol'])]:.0%}", delta_color="off")
    m4.metric("Relative: learned behaviour", top(result["rel_pol"]),
              f"{result['rel_pol'][-1][top(result['rel_pol'])]:.0%}", delta_color="off")
    st.caption("These are the notebook's exact settings and results." if is_notebook else
               "Custom settings. Press the button with the defaults to get back to the notebook's run.")

    drift_step = settings["noise_shift_step"] if settings["drift"] and settings["reward_noise"] else None
    st.plotly_chart(viz.training_summary(abs_s, rel_s, result["abs_pol"], result["rel_pol"], drift_step),
                    config={"displayModeBar": False})
    st.markdown(
        "**How to read it.** *Top left* is what the reward model says: context, not the learning signal. "
        "*Top right* is ground truth, never seen by training. *Bottom left* is what the optimizer actually uses: "
        "the relative mean advantage is 0 by construction and its spread stays usable, while the absolute advantages "
        "shrink as the baseline catches up with the rewards. *Bottom right* is the result: similar reward curves, "
        "different policies."
    )
    code_panel("train_absolute_baseline", rp.train_absolute_baseline,
               "Advantage = reward − running baseline (exponential moving average).")
    code_panel("train_group_relative and compute_group_advantages",
               [rp.train_group_relative, rp.group_relative_advantages, rp.compute_group_advantages],
               "Advantage = z-score within each prompt group. Policy update for both: `logits[action] += lr * advantage`.")

# shared for tabs ③ and ④
result = run(settings_key(st.session_state.get("rp_result_settings", NB)))
steps = len(result["abs_stats"])

# ---------------------------------------------------------------- ③ step by step
with tab_step:
    st.markdown("Scrub through training. This replaces the notebook's animation and its step-level deep dives.")
    k = st.slider("Training step", 1, steps, 1, key="rp_step")
    i = k - 1
    rel_adv = rp.group_relative_advantages(result["rel_rollouts"][i])
    st.plotly_chart(viz.policy_at_step(result["abs_pol"][i], result["rel_pol"][i], k), config={"displayModeBar": False})
    st.plotly_chart(viz.step_details(i, result["abs_rollouts"], result["abs_stats"], result["rel_rollouts"], rel_adv),
                    config={"displayModeBar": False})
    a, r = result["abs_stats"][i], result["rel_stats"][i]
    st.markdown(
        f"**Step {k}:** absolute advantages have mean **{a['mean_adv']:+.3f}** and std **{a['std_adv']:.3f}** "
        f"(baseline {a['baseline']:.3f}); relative advantages have mean **{r['mean_adv']:+.3f}** and std "
        f"**{r['std_adv']:.3f}**. Try steps 1, 21, 41 and {steps}: the absolute signal changes with the reward "
        "scale, the relative one stays centred and well-scaled."
    )
    with st.expander("Sample rollouts at this step"):
        c1, c2 = st.columns(2)
        for col, name, ros, advs in [
            (c1, "Absolute", result["abs_rollouts"][i], [ro.reward - a["baseline"] for ro in result["abs_rollouts"][i]]),
            (c2, "Relative", result["rel_rollouts"][i], rel_adv),
        ]:
            col.markdown(f"**{name}**")
            col.dataframe(pd.DataFrame([{"action": ro.action, "reward": round(ro.reward, 3), "advantage": round(ad, 3),
                                         "prompt": ro.prompt} for ro, ad in list(zip(ros, advs))[:12]]),
                          hide_index=True)

# ---------------------------------------------------------------- ④ ranking table
with tab_rank:
    st.markdown("Zoom into **one prompt group at one step**: who is reinforced, and why?")
    c1, c2 = st.columns([1, 2])
    k = c1.select_slider("Step", options=list(range(1, steps + 1)), value=1, key="rp_rank_step")
    prompt_id = c2.selectbox("Prompt", [p[0] for p in rp.PROMPTS], format_func=lambda pid: rp.PROMPTS[pid][1])
    i = k - 1
    noise = result["abs_maps"][i]
    if noise:
        scale, bias = noise[prompt_id]
        st.caption(f"Reward noise for this prompt at this step: scale **{scale:.2f}**, bias **{bias:+.2f}** "
                   "(raw reward is reconstructed as (observed − bias) / scale, for teaching only).")

    def ranking(rollouts, method):
        group = [ro for ro in rollouts[i] if ro.prompt_id == prompt_id]
        observed = [ro.reward for ro in group]
        if method == "abs":
            baseline = result["abs_stats"][i]["baseline"]
            advantages = [o - baseline for o in observed]
        else:
            advantages = rp.compute_group_advantages(observed)
        order = sorted(range(len(group)), key=lambda j: observed[j], reverse=True)
        rows = []
        for rank, j in enumerate(order, 1):
            row = {"rank": rank, "action": group[j].action}
            if noise:
                row["raw"] = round((observed[j] - bias) / scale if scale > 0 else observed[j], 3)
            row.update({"observed": round(observed[j], 3), "advantage": round(advantages[j], 3),
                        "correct": "✓" if rp.is_correct_answer(group[j].transcript, group[j].correct_answer) else "✗"})
            rows.append(row)
        return pd.DataFrame(rows), sum(observed) / len(observed), advantages

    def colour(v):
        return f"color: {'#1fa37a' if v > 0 else '#d64545'}; font-weight: 600"

    left, right = st.columns(2, gap="large")
    for col, method, title in [(left, "abs", "Absolute baseline"), (right, "rel", "Group relative")]:
        table, mean_reward, advs = ranking(result[f"{method}_rollouts"], method)
        with col:
            st.markdown(f"**{title}**")
            st.dataframe(table.style.map(colour, subset=["advantage"]), hide_index=True)
            extra = f" · global baseline {result['abs_stats'][i]['baseline']:.3f}" if method == "abs" else ""
            st.caption(f"group mean reward {mean_reward:.3f}{extra} · advantage mean {sum(advs) / len(advs):+.3f}, "
                       f"std {pd.Series(advs).std(ddof=0):.3f}")
    st.markdown(
        "In the **relative** table the best rollout always gets a positive advantage and the worst a negative one, "
        "whatever the absolute reward. *Learning depends on who wins, not by how much.* In the **absolute** table every "
        "rollout is compared with one global baseline, so a generously scored prompt can reinforce mediocre answers."
    )

# ---------------------------------------------------------------- ⑤ why
with tab_why:
    st.markdown("#### Rewards vs advantages")
    st.markdown("Rewards are **environment outputs**. Advantages are **training inputs**. The optimizer never updates "
                "the policy from rewards directly, only from advantages. Two runs can see identical rewards and still "
                "learn very differently.")
    st.markdown("#### PPO vs GRPO")
    st.markdown(
        "PPO estimates advantages with a **value model (critic)**, usually as large as the policy, doubling memory and "
        "compute, and often wrong, so updates can point the wrong way. **GRPO** drops the critic: it samples a group of "
        "answers for the same prompt and uses the group itself as the baseline:"
    )
    st.latex(r"A_i=\frac{r(x, y_i)-\operatorname{mean}\left(\{r(x, y_j)\}_{j=1}^{G}\right)}"
             r"{\operatorname{std}\left(\{r(x, y_j)\}_{j=1}^{G}\right)}")
    st.markdown(
        "- The advantage is **always centred around 0**: the group average is the baseline.\n"
        "- Dividing by the std keeps a **consistent scale** (≈ 1), so the signal doesn't fade late in training.\n"
        "- If the reward model **drifts**, the group mean moves with it: the best answer in the group still wins."
    )
    st.link_button("Group Sequence Policy Optimization (paper)", "https://arxiv.org/pdf/2507.18071",
                   icon=":material/description:")
    takeaway("<b>Relative feedback turns reward into within-prompt advantage, so learning is driven by ranking rather "
             "than absolute scale.</b> “Which is better?” is easier to answer reliably than “how good is this?”, and "
             "that is exactly what makes it robust to miscalibrated and drifting reward models.")
