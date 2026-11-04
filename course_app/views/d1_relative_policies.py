import math
import random

import pandas as pd
import streamlit as st

from core.ui import code_panel, page_header, takeaway
from labs import relative_policies as rp
from labs import relative_policies_viz as viz

NB = rp.NOTEBOOK_SETTINGS
ss = st.session_state

page_header(
    session=1,
    kind="DEMO",
    title="Relative policies: learning from rankings, not scores",
    subtitle=(
        "A **minimal reinforcement learning loop**: a simple policy learns which strategy to use for arithmetic "
        "questions, once from **absolute numeric feedback** and once from **relative feedback**. "
        "The core idea: *relative feedback turns reward into within-prompt advantage, so learning is driven by "
        "ranking rather than absolute scale.*"
    ),
    minutes=10,
    notebook_path="demo/session_01_DEMO_relative_policies_RL.ipynb",
)


@st.cache_resource(max_entries=200, show_spinner="Training both methods…")
def run(settings: tuple) -> dict:
    return rp.run_demo(**dict(settings))


def settings_key(s: dict) -> tuple:
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in s.items()))


def how_to_read(markdown: str, title: str = "How to read this") -> None:
    with st.expander(f":material/menu_book: {title}"):
        st.markdown(markdown)


notebook_run = run(settings_key(NB))

tabs = st.tabs(["① How RL works", "② Why advantages", "③ The reward model", "④ Train both",
                "⑤ Step by step", "⑥ Inside one prompt group", "⑦ Why GRPO"])

# ======================================================================== ① how RL works
with tabs[0]:
    st.markdown("#### The loop: act, get scored, learn")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1 · Policy**\n\nA score (*logit*) per action; softmax turns them into probabilities. "
                "For an LLM, this is the model itself.")
    c2.markdown("**2 · Rollouts**\n\nFor one prompt, sample a **group** of answers. Each picks an action and "
                "produces a transcript.")
    c3.markdown("**3 · Reward → advantage**\n\nA reward model scores each transcript. The **advantage** says how much "
                "*better or worse than expected* each one was.")
    c4.markdown("**4 · Update**\n\n`logits[action] += lr × advantage`: better-than-expected actions become more "
                "likely, worse ones less likely.")
    st.markdown(
        "The agent answers arithmetic prompts with one of six strategies. Some are reliable, some aren't, "
        "but **the agent doesn't know that**: it can only learn it from rewards."
    )
    st.dataframe(pd.DataFrame([
        {"action": "use_tool", "what it does": "calls a calculator", "chance of a wrong answer": "2%"},
        {"action": "verify", "what it does": "answers, then checks", "chance of a wrong answer": "3%"},
        {"action": "plan", "what it does": "solves step by step", "chance of a wrong answer": "8%"},
        {"action": "direct_answer", "what it does": "just answers", "chance of a wrong answer": "10%"},
        {"action": "ramble", "what it does": "long-winded", "chance of a wrong answer": "35%"},
        {"action": "refuse", "what it does": "won't answer", "chance of a wrong answer": "100%"},
    ]), hide_index=True)

    st.divider()
    st.markdown("#### Try it: run the loop one step at a time")
    st.markdown("Everything below runs the notebook's own functions on **one prompt**. Run a step, look at each stage, "
                "then run more and watch the policy change.")

    def reset_loop():
        ss.rl_loop = {"logits": {a: 0.0 for a in rp.ACTIONS}, "baseline": 0.0, "step": 0, "last": None,
                      "history": [rp.softmax({a: 0.0 for a in rp.ACTIONS})]}

    if "rl_loop" not in ss:
        reset_loop()
    c1, c2, c3 = st.columns([1.2, 1.4, 1])
    method = c1.radio("Advantage", ["Relative (GRPO group advantage)", "Absolute (reward − baseline)"],
                      key="rl_method", on_change=reset_loop)
    prompt_id = c2.selectbox("Prompt", [p[0] for p in rp.PROMPTS], format_func=lambda pid: rp.PROMPTS[pid][1],
                             key="rl_prompt", on_change=reset_loop)
    miscalibrated = c3.toggle("Miscalibrated reward model", value=True, key="rl_noise", on_change=reset_loop,
                              help="Uses the notebook's noise map: this prompt's rewards are scaled and shifted.")
    relative = method.startswith("Relative")

    def one_step():
        state = ss.rl_loop
        rng = random.Random(1000 + 7919 * prompt_id + state["step"])
        noise = notebook_run["noise_map"] if miscalibrated else None
        rollouts = rp.generate_rollouts_for_prompts([rp.PROMPTS[prompt_id]], state["logits"], NB["group_size"],
                                                    NB["temperature"], state["step"], rng, noise, NB["noise_std"])
        rewards = [ro.reward for ro in rollouts]
        if relative:
            advantages = rp.compute_group_advantages(rewards)
        else:
            state["baseline"] = 0.95 * state["baseline"] + 0.05 * (sum(rewards) / len(rewards))
            advantages = [r - state["baseline"] for r in rewards]
        before = rp.softmax(state["logits"])
        delta = {a: 0.0 for a in rp.ACTIONS}
        for ro, adv in zip(rollouts, advantages):
            state["logits"][ro.action] += NB["lr"] * adv
            delta[ro.action] += NB["lr"] * adv
        after = rp.softmax(state["logits"])
        state["step"] += 1
        state["history"].append(after)
        state["last"] = dict(rollouts=rollouts, rewards=rewards, advantages=advantages, before=before,
                             after=after, delta=delta, baseline=state["baseline"])

    b1, b2, b3, _ = st.columns([1.2, 1, 1, 2])
    if b1.button("Run one training step", type="primary", icon=":material/play_arrow:"):
        one_step()
    if b2.button("Run 10 steps", icon=":material/fast_forward:"):
        for _ in range(10):
            one_step()
    b3.button("Start over", icon=":material/restart_alt:", on_click=reset_loop)

    last = ss.rl_loop["last"]
    if last is None:
        st.info("Press **Run one training step**. The policy starts uniform: every action has probability 1/6.",
                icon=":material/touch_app:")
    else:
        rollouts, rewards, advs = last["rollouts"], last["rewards"], last["advantages"]
        st.markdown(f"**Training step {ss.rl_loop['step']}** · prompt: *{rp.PROMPTS[prompt_id][1]}* "
                    f"(correct answer {rp.PROMPTS[prompt_id][2]})")
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            st.markdown("**① Policy before the update**")
            st.plotly_chart(viz.signal_bars(rp.ACTIONS, [last["before"][a] for a in rp.ACTIONS], "",
                                            "probability", y_range=[0, 1.1], colors=["#7c5cd6"] * 6,
                                            text_format="{:.2f}"),
                            config={"displayModeBar": False}, key="rl_before")
        with right:
            st.markdown(f"**② {len(rollouts)} rollouts → ③ reward → advantage**")
            st.dataframe(pd.DataFrame([{
                "#": i + 1, "action": ro.action,
                "correct": "✓" if rp.is_correct_answer(ro.transcript, ro.correct_answer) else "✗",
                "reward": round(r, 3), "advantage": round(a, 3)} for i, (ro, r, a) in enumerate(zip(rollouts, rewards, advs))
            ]).style.map(lambda v: f"color: {'#1fa37a' if v > 0 else '#d64545'}; font-weight: 600", subset=["advantage"]).format(precision=3),
                hide_index=True, height=300)

        best, worst = max(range(len(rewards)), key=rewards.__getitem__), min(range(len(rewards)), key=rewards.__getitem__)
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
        if relative:
            st.markdown(
                f"**Relative advantage** compares each rollout with the *other rollouts for the same prompt*: "
                f"group mean **{mean:.3f}**, std **{std:.3f}**.  \n"
                f"Best · `{rollouts[best].action}`: A = ({rewards[best]:.3f} − {mean:.3f}) / {std:.3f} = **{advs[best]:+.2f}**  \n"
                f"Worst · `{rollouts[worst].action}`: A = ({rewards[worst]:.3f} − {mean:.3f}) / {std:.3f} = **{advs[worst]:+.2f}**"
            )
        else:
            st.markdown(
                f"**Absolute advantage** compares each rollout with a *running baseline* of past rewards "
                f"(exponential moving average, now **{last['baseline']:.3f}**).  \n"
                f"Best · `{rollouts[best].action}`: A = {rewards[best]:.3f} − {last['baseline']:.3f} = **{advs[best]:+.2f}**  \n"
                f"Worst · `{rollouts[worst].action}`: A = {rewards[worst]:.3f} − {last['baseline']:.3f} = **{advs[worst]:+.2f}**"
            )
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**④ The update:** `logits[action] += lr × advantage`, summed per action")
            st.plotly_chart(viz.signal_bars(rp.ACTIONS, [last["delta"][a] for a in rp.ACTIONS], "",
                                            "change of the action's logit"), config={"displayModeBar": False},
                            key="rl_delta")
        with right:
            st.markdown("**Policy after the update**")
            st.plotly_chart(viz.policy_before_after(last["before"], last["after"], ""),
                            config={"displayModeBar": False}, key="rl_after")
        with st.expander("Transcripts of the best and the worst rollout"):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Best** · reward {rewards[best]:+.3f} · advantage {advs[best]:+.2f}")
            c1.code(rollouts[best].transcript, language="text")
            c2.markdown(f"**Worst** · reward {rewards[worst]:+.3f} · advantage {advs[worst]:+.2f}")
            c2.code(rollouts[worst].transcript, language="text")
        if len(ss.rl_loop["history"]) > 2:
            st.markdown("**The policy over your training steps**")
            st.line_chart(pd.DataFrame(ss.rl_loop["history"])[rp.ACTIONS], height=240,
                          x_label="training step", y_label="probability")
        if not relative and ss.rl_loop["step"] <= 3:
            st.caption("Notice: early on the baseline is still near 0, so almost every rollout with a positive reward "
                       "gets a positive advantage, even mediocre ones. Run more steps and watch the baseline catch up.")

    code_panel("generate_rollouts_for_prompts", rp.generate_rollouts_for_prompts)
    code_panel("compute_group_advantages", rp.compute_group_advantages)

# ======================================================================== ② why advantages
with tabs[1]:
    st.markdown("#### Rewards are environment outputs. Advantages are training inputs.")
    st.markdown(
        "The optimizer never updates the policy from rewards directly. It updates from **advantages**: "
        "*how much better (or worse) than expected* a rollout was. That's where the name comes from: "
        "the advantage of taking this action over what the policy usually achieves."
    )
    group = [ro for ro in notebook_run["abs_rollouts"][0] if ro.prompt_id == 0]
    raw = [rp.score_rollout(ro.transcript, ro.correct_answer, ro.action) for ro in group]
    labels = [f"#{i + 1} {ro.action}{' ✓' if rp.is_correct_answer(ro.transcript, ro.correct_answer) else ' ✗'}"
              for i, ro in enumerate(group)]
    st.markdown(f"Below is one real group from the notebook's first training step: {len(group)} rollouts for "
                f"*{group[0].prompt}*, scored by the **well-calibrated** reward model (`score_rollout`).")

    st.markdown("##### Part 1 · Absolute vs relative advantage, side by side")
    st.markdown(
        "Both methods turn the **same rewards** into advantages, they just compare against a different reference:\n"
        "- **Absolute:** against a running **baseline**, an average of past rewards over *all* prompts.\n"
        "- **Relative (GRPO/GSPO):** against the **other rollouts for the same prompt**: subtract the group mean, "
        "divide by the group's standard deviation."
    )
    nb_baselines = {k: notebook_run["abs_stats"][k - 1]["baseline"] for k in (1, 21, NB["steps"])}
    baseline = st.slider("Absolute baseline b", -0.5, 1.0, round(nb_baselines[1], 2), 0.01, key="adv_baseline")
    st.caption(f"Starts at the baseline the absolute method actually had at this step in the notebook "
               f"(**{nb_baselines[1]:.3f}**; it was {nb_baselines[21]:.3f} at step 21 and {nb_baselines[NB['steps']]:.3f} "
               f"at step {NB['steps']}). Slide it to see how much the absolute signal depends on it.")
    mean = sum(raw) / len(raw)
    std = math.sqrt(sum((r - mean) ** 2 for r in raw) / len(raw))
    abs_adv = [r - baseline for r in raw]
    rel_adv = rp.compute_group_advantages(raw)
    limit = max(abs(v) for v in abs_adv + rel_adv) * 1.25

    def push(advantages):
        totals = {a: 0.0 for a in rp.ACTIONS}
        for ro, v in zip(group, advantages):
            totals[ro.action] += NB["lr"] * v
        return totals

    abs_push, rel_push = push(abs_adv), push(rel_adv)
    push_limit = max(abs(v) for v in list(abs_push.values()) + list(rel_push.values())) * 1.3
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Absolute**")
        st.caption(f"A = reward − {baseline:.3f}")
        st.plotly_chart(viz.signal_bars(labels, abs_adv, "Advantage per rollout", "advantage", zero_line=0,
                                        y_range=[-limit, limit]), config={"displayModeBar": False}, key="adv_abs")
        st.plotly_chart(viz.signal_bars(rp.ACTIONS, [abs_push[a] for a in rp.ACTIONS], "Push per action",
                                        "logit change", y_range=[-push_limit, push_limit]),
                        config={"displayModeBar": False}, key="adv_abs_push")
    with right:
        st.markdown("**Relative (GRPO)**")
        st.caption(f"A = (reward − {mean:.3f}) / {std:.3f}")
        st.plotly_chart(viz.signal_bars(labels, rel_adv, "Advantage per rollout", "advantage", zero_line=0,
                                        y_range=[-limit, limit]), config={"displayModeBar": False}, key="adv_rel")
        st.plotly_chart(viz.signal_bars(rp.ACTIONS, [rel_push[a] for a in rp.ACTIONS], "Push per action",
                                        "logit change", y_range=[-push_limit, push_limit]),
                        config={"displayModeBar": False}, key="adv_rel_push")

    correct = [rp.is_correct_answer(ro.transcript, ro.correct_answer) for ro in group]
    st.dataframe(pd.DataFrame([{
        "#": i + 1, "action": ro.action, "correct": "✓" if ok else "✗", "reward": round(r, 3),
        "absolute A": round(a, 3), "relative A": round(g, 3),
        "same direction": "yes" if (a > 0) == (g > 0) else "no"}
        for i, (ro, ok, r, a, g) in enumerate(zip(group, correct, raw, abs_adv, rel_adv))]), hide_index=True)

    abs_mean = sum(abs_adv) / len(abs_adv)
    wrong_up_abs = sum(1 for ok, a in zip(correct, abs_adv) if a > 0 and not ok)
    wrong_up_rel = sum(1 for ok, g in zip(correct, rel_adv) if g > 0 and not ok)
    disagree = sum(1 for a, g in zip(abs_adv, rel_adv) if (a > 0) != (g > 0))
    st.markdown(
        f"- **Centre:** the absolute advantages average **{abs_mean:+.3f}**, the relative ones **0.000**. "
        "An absolute signal with a non-zero mean pushes *everything* in the group up (or down) a bit, on top of the "
        "useful part.\n"
        f"- **Direction:** {disagree} of {len(group)} rollouts get a different sign. Wrong answers pushed up: "
        f"absolute **{wrong_up_abs}**, relative **{wrong_up_rel}**.\n"
        f"- **Scale:** relative advantages always have std 1 within the group, whatever the rewards; the absolute "
        f"ones have the rewards' own spread ({std:.3f}). Move the baseline: only the absolute side changes."
    )
    with st.expander("Why not update with the raw reward directly?"):
        positive = sum(r > 0 for r in raw)
        generous_push = push([max(-1.0, min(1.0, r + 0.4)) for r in raw])
        generous_up = [a for a in rp.ACTIONS if generous_push[a] > 1e-9]
        st.markdown(
            f"A raw reward has no sense of *better than what*. Here {positive} of {len(raw)} rewards are positive, "
            "so it happens to point roughly the right way, only because this reward model is centred near zero. "
            f"Give the same answers a generous reward model (+0.4 on every score): then {len(generous_up)} of "
            f"{len(rp.ACTIONS)} actions get pushed up ({', '.join(generous_up)}), including ones that only produced "
            "wrong answers. That is what a baseline, and better still the group, fixes."
        )
    with st.expander("Is this exactly how GRPO and GSPO compute it?"):
        st.markdown(
            "Yes, up to one detail. GRPO (DeepSeekMath) and GSPO (Qwen) both use the group-normalized advantage "
            "*(reward − group mean) / group std*, which statistically is a z-score within the group. "
            "GSPO changes the importance ratio and clipping to the sequence level, not the advantage. The detail: "
            "the notebook divides by the **population** std (and uses 1 when all rewards are equal); typical "
            "GRPO implementations use the **sample** std plus a small ε. That rescales the advantages slightly but "
            "never changes their sign or ranking. Some variants drop the std division altogether "
            "(e.g. Dr. GRPO), keeping only *reward − group mean*."
        )

    st.divider()
    st.markdown("##### Part 2 · Now miscalibrate the reward model")
    st.markdown(
        "Real reward models are biased: one prompt type gets generous scores, another strict ones. "
        "Change how *this prompt* is scored and compare both advantages with the calibrated ones (outlined):"
    )
    c1, c2 = st.columns(2)
    scale = c1.slider("Scale (multiplies the reward)", 0.3, 1.7, 1.0, 0.05, key="adv_scale")
    bias = c2.slider("Bias (added to the reward)", -0.4, 0.4, 0.0, 0.05, key="adv_bias")
    observed = [max(-1.0, min(1.0, scale * r + bias)) for r in raw]
    expected = sum(raw) / len(raw)  # the baseline a calibrated run would have learned
    abs_now, abs_ref = [o - expected for o in observed], [r - expected for r in raw]
    rel_now, rel_ref = rp.compute_group_advantages(observed), rp.compute_group_advantages(raw)
    st.plotly_chart(viz.compare_advantages(labels, abs_now, rel_now, abs_ref, rel_ref),
                    config={"displayModeBar": False}, key="adv_compare")
    clipped = sum(1 for r in raw if abs(scale * r + bias) > 1.0)
    abs_change = max(abs(a - b) for a, b in zip(abs_now, abs_ref))
    rel_change = max(abs(a - b) for a, b in zip(rel_now, rel_ref))
    m1, m2, m3 = st.columns(3)
    m1.metric("Absolute: largest change of an advantage", f"{abs_change:.2f}")
    m2.metric("Relative: largest change of an advantage", f"{rel_change:.2f}")
    m3.metric("Rewards clipped at ±1", f"{clipped} of {len(raw)}")
    st.markdown("Why the relative advantages don't move: for any scale *s* > 0 and bias *b*, the group normalization cancels both:")
    st.latex(r"\frac{(s\,r_i + b) - \operatorname{mean}(s\,r + b)}{\operatorname{std}(s\,r + b)}"
             r"= \frac{s\,(r_i - \bar r)}{s\,\sigma_r} = \frac{r_i - \bar r}{\sigma_r}")
    if scale == 1.0 and bias == 0.0:
        st.info("Move the sliders. Try scale 0.6 with bias −0.3 (a strict reward model), then scale 1.6 with bias +0.3.",
                icon=":material/tune:")
    elif clipped == 0:
        st.success(f"The absolute advantages moved by up to **{abs_change:.2f}**, the relative ones by "
                   f"**{rel_change:.2f}**. The absolute signal shifts with every bias and stretches with every scale, "
                   "so the policy would learn the reward model's quirks instead of which answers are better.",
                   icon=":material/check_circle:")
    else:
        st.warning(f"**{clipped} of {len(raw)} rewards hit the ±1 limit** and become identical, so the ranking among "
                   "them is lost, and now the relative advantages change too. That's **reward saturation**: when a "
                   "reward model can't tell good answers apart any more, no advantage construction can. It's also why "
                   "rewards saturating near the cap late in training weakens the signal.",
                   icon=":material/compress:")
    takeaway("<b>“Which is better?” is easier to answer reliably than “how good is this?”.</b> Relative advantages "
             "only need the ranking inside a group, so they survive reward models that are miscalibrated per prompt.")

# ======================================================================== ③ reward model
with tabs[2]:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("#### The reward model")
        st.markdown(
            "- **+1** if the final answer is correct (the most important signal)\n"
            "- a **length penalty** of 0.0009 per character (discourages verbosity)\n"
            "- small **shaping bonuses**: tool use +0.06, verification +0.04, planning +0.02\n"
            "- **−0.7** for refusing\n\n"
            "Clipped to [−1, 1]."
        )
    with right:
        code_panel("score_rollout (the reward model)", rp.score_rollout)
        code_panel("render_transcript (what each action writes)", rp.render_transcript)

    st.markdown("##### Question to all")
    st.markdown("**If we removed the length penalty, would the model learn to ramble just to sound smarter?**")
    st.caption("`length_penalty = 0.0009 * len(transcript)`")
    with st.expander("Reveal the answer"):
        st.markdown(f"### {rp.reveal_answer()}")

    st.divider()
    st.markdown("#### The noise map: a miscalibrated reward model")
    st.markdown(
        "In the real world a reward model is just another LLM, and it's biased: it might give higher scores to "
        "specific topics or formats simply because of its own training data. The demo simulates that with a "
        "per-prompt **scale** and **bias**, plus a little per-rollout noise:\n\n"
        "`observed = clip(scale × raw + bias + noise)`\n\n"
        "- **Absolute baseline's failure:** if a prompt sits in a *high-bias* zone, the agent over-rewards whatever "
        "it does on that prompt, regardless of quality.\n"
        "- **Relative feedback's success:** normalizing within the prompt group subtracts the bias and divides out "
        "the scale (see ② Why advantages).\n\n"
        f"At step **{NB['noise_shift_step']}** the reward model **drifts**: every prompt gets a new, wider scale and bias, "
        "like a reward model update or a data-mix change in the middle of training."
    )
    st.plotly_chart(viz.noise_map_figure(rp.PROMPTS, notebook_run["noise_map"],
                                         notebook_run["abs_maps"][NB["noise_shift_step"]]),
                    config={"displayModeBar": False})
    code_panel("create_reward_noise_map and apply_reward_noise", [rp.create_reward_noise_map, rp.apply_reward_noise])

# ======================================================================== ④ train both
WIDGETS = {"rp_noise": NB["reward_noise"], "rp_drift": NB["drift"], "rp_shift": NB["noise_shift_step"],
           "rp_group": NB["group_size"], "rp_lr": NB["lr"], "rp_temp": NB["temperature"], "rp_seed": NB["seed"]}


def reset_to_notebook():
    for key, value in WIDGETS.items():
        ss[key] = value


with tabs[3]:
    st.markdown(
        f"Both methods train for **{NB['steps']} steps** on all {len(rp.PROMPTS)} prompts, "
        f"{NB['group_size']} rollouts per prompt per step, with the same miscalibrated reward model and the drift at "
        f"step {NB['noise_shift_step']}. Change a setting and both methods retrain immediately."
    )
    for key, value in WIDGETS.items():
        ss.setdefault(key, value)
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        noise_on = c1.toggle("Miscalibrated reward model", key="rp_noise")
        drift_on = c1.toggle("Drift mid-training", key="rp_drift", disabled=not noise_on)
        shift = c2.slider("Drift at step", 5, 75, step=5, key="rp_shift", disabled=not (noise_on and drift_on))
        group_size = c2.slider("Group size", 2, 24, key="rp_group", help="Rollouts per prompt per step")
        lr = c3.slider("Learning rate", 0.02, 0.5, step=0.02, key="rp_lr")
        temperature = c3.slider("Sampling temperature", 0.5, 2.0, step=0.05, key="rp_temp")
        seed = c4.number_input("Seed", 0, 9999, key="rp_seed")
        c4.button("Notebook settings", icon=":material/restart_alt:", on_click=reset_to_notebook)
    settings = {**NB, "reward_noise": noise_on, "drift": drift_on, "noise_shift_step": shift, "group_size": group_size,
                "lr": lr, "temperature": temperature, "seed": int(seed)}
    result = run(settings_key(settings))
    ss.rp_current = settings

    history = ss.setdefault("rp_history", [])
    if not history or history[-1]["settings"] != settings:
        changes = [k for k in ("reward_noise", "drift", "noise_shift_step", "group_size", "lr", "temperature", "seed")
                   if settings[k] != NB[k]]
        history.append({"settings": settings,
                        "label": "notebook settings" if not changes else ", ".join(f"{k}={settings[k]}" for k in changes),
                        "abs_stats": result["abs_stats"], "rel_stats": result["rel_stats"],
                        "abs_top": max(result["abs_pol"][-1], key=result["abs_pol"][-1].get),
                        "rel_top": max(result["rel_pol"][-1], key=result["rel_pol"][-1].get)})
        del history[:-8]
    previous = history[-2] if len(history) > 1 else None

    abs_s, rel_s = result["abs_stats"], result["rel_stats"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Absolute · final correctness", f"{abs_s[-1]['correctness_rate']:.1%}",
              f"{abs_s[-1]['correctness_rate'] - previous['abs_stats'][-1]['correctness_rate']:+.1%} vs previous"
              if previous else None)
    m2.metric("Relative · final correctness", f"{rel_s[-1]['correctness_rate']:.1%}",
              f"{rel_s[-1]['correctness_rate'] - previous['rel_stats'][-1]['correctness_rate']:+.1%} vs previous"
              if previous else None)
    m3.metric("Absolute · learned", history[-1]["abs_top"])
    m4.metric("Relative · learned", history[-1]["rel_top"])
    st.caption(f"Run: **{history[-1]['label']}**" + (f" · previous: {previous['label']}" if previous else ""))

    drift_step = shift if (noise_on and drift_on) else None
    st.markdown("##### Watch the training")
    st.plotly_chart(viz.training_animation(result, drift_step, every=2), config={"displayModeBar": False},
                    key=f"anim_{settings_key(settings)}")
    how_to_read(
        "This animation makes the **actual optimization signal** visible, not just whether rewards go up.\n\n"
        "- **Top row, policies and their difference:** how probability mass shifts across actions. Green bars in the "
        "difference panel are actions the relative method favours more, red the absolute one. Large bars are "
        "systematic preference differences, not noise.\n"
        "- **Bottom left, reward vs baseline (absolute):** the baseline adapts to the reward scale. As reward and "
        "baseline converge, the absolute advantages shrink toward zero.\n"
        "- **Bottom middle, relative average reward:** context only. Relative optimization doesn't rely on the "
        "absolute reward scale; it keeps learning even when reward magnitudes change after the drift.\n"
        "- **Bottom right, advantages at this step:** the signal the update actually uses. Wide or skewed "
        "distributions mean unstable updates; the relative one is centred by construction and keeps a usable spread.\n\n"
        "The causal chain: **reward and baseline shape advantages → advantages drive the update → the update shapes "
        "the policy.** Even when rewards look similar, the learned behaviour can diverge."
    )

    st.markdown("##### The whole run at a glance")
    st.plotly_chart(viz.training_summary(abs_s, rel_s, result["abs_pol"], result["rel_pol"], drift_step,
                                         previous=previous), config={"displayModeBar": False})
    how_to_read(
        "This figure deliberately mixes **reward-model outputs**, **ground truth** and **optimization signals**, to "
        "show where they align and where they don't.\n\n"
        "- **Average reward (top left):** what the reward model says, with the absolute method's baseline (red dashed) "
        "and the drift point. It is *not* task performance and *not* the learning signal. Similar curves don't imply "
        "similar learning.\n"
        "- **True correctness (top right):** measured against the correct answer, **never used in training**. "
        "The only question that ultimately matters: is the agent actually solving the task?\n"
        "- **Advantages (bottom left), the most important panel:** mean (solid) and spread (dashed) of the signal "
        "used for updates. Absolute: the mean drifts toward zero as the baseline adapts, and the spread changes with "
        "the reward model. Relative: the mean is 0 by construction and the spread stays stable.\n"
        "- **Final policy (bottom right):** the accumulated effect of every update. Similar reward curves but different "
        "policies mean the methods learned different internal preferences.\n\n"
        "**Reward-model outputs are context. Advantages are the learning signal. Policies change because of "
        "advantages, not rewards.**"
    )
    if len(history) > 1:
        st.markdown("##### Your runs")
        st.dataframe(pd.DataFrame([{
            "run": h["label"],
            "absolute correctness": f"{h['abs_stats'][-1]['correctness_rate']:.1%}",
            "relative correctness": f"{h['rel_stats'][-1]['correctness_rate']:.1%}",
            "absolute learned": h["abs_top"], "relative learned": h["rel_top"]} for h in history]), hide_index=True)

    st.markdown("##### Experiments")
    for question, answer in [
        ("Switch the **miscalibrated reward model off**. Which method does better now?",
         "Often the absolute method. Without per-prompt bias there is nothing for the group normalization to cancel, and dividing "
         "by a small within-group spread can amplify tiny reward differences between equally good answers. Relative "
         "feedback is about **robustness to a miscalibrated reward model**, not a free accuracy boost."),
        ("Keep the notebook settings and change **only the seed** a few times. How stable is the gap?",
         "The final correctness numbers move around from seed to seed; the gap in the notebook's run is on the large "
         "side. What stays the same on every seed is the **advantage signal**: centred and well-scaled for the relative "
         "method, shrinking for the absolute one (bottom left panel). Always compare RL methods across several seeds."),
        ("Set the **group size to 2**. What happens to the relative method?",
         "With two rollouts the group advantage is always ±1: it only says which of the two was better. Larger groups give a "
         "more informative ranking, which is why GRPO samples many answers per prompt."),
        ("Move the **drift** early (step 10) or late (step 70). Where do you see it?",
         "In the reward curves and the absolute baseline, which has to re-adapt. The relative advantages keep their "
         "centre and scale through the drift: the best answer in a group still wins."),
    ]:
        st.markdown(f"**{question}**")
        with st.expander("Reveal"):
            st.markdown(answer)

    code_panel("train_absolute_baseline", rp.train_absolute_baseline,
               "Advantage = reward − running baseline (exponential moving average across all prompts).")
    code_panel("train_group_relative", [rp.train_group_relative, rp.group_relative_advantages],
               "Advantage = (reward − group mean) / group std within each prompt group, as in GRPO. Same update for both: `logits[action] += lr * advantage`.")
    st.caption("Policy update note: like the notebook, this uses a bandit-style shortcut (`logits += lr * advantage`) "
               "instead of gradients of log-probabilities, to show the effect of the advantage construction clearly.")

# shared by ⑤ and ⑥: whatever was last trained in ④
current = run(settings_key(ss.get("rp_current", NB)))
steps = len(current["abs_stats"])

# ======================================================================== ⑤ step by step
with tabs[4]:
    st.markdown("A microscope on **one training step**: the exact objects the optimizer sees. "
                "Uses the run from ④ Train both.")
    k = st.slider("Training step", 1, steps, 1, key="rp_step")
    i = k - 1
    rel_adv = rp.group_relative_advantages(current["rel_rollouts"][i])
    st.plotly_chart(viz.policy_at_step(current["abs_pol"][i], current["rel_pol"][i], k), config={"displayModeBar": False})
    st.plotly_chart(viz.step_details(i, current["abs_rollouts"], current["abs_stats"], current["rel_rollouts"], rel_adv),
                    config={"displayModeBar": False})
    a, r = current["abs_stats"][i], current["rel_stats"][i]
    st.markdown(
        f"**Step {k}:** absolute advantages have mean **{a['mean_adv']:+.3f}** and spread **{a['std_adv']:.3f}** "
        f"(baseline {a['baseline']:.3f}). Relative advantages have mean **{r['mean_adv']:+.3f}** and spread "
        f"**{r['std_adv']:.3f}**. Compare steps 1, 21, 41 and {steps}."
    )
    how_to_read(
        "- **Sorted rewards (top left / middle):** each dot is one rollout. For the absolute method the dashed line is "
        "the **global baseline**: dots above it get a positive advantage, below it a negative one, and the distance "
        "sets the size. If the baseline is misplaced or drifts, the whole signal shifts. For the relative method the "
        "dashed line is only the mean: what counts is the order *within each prompt group*.\n"
        "- **Reward → advantage (top right):** absolute is a straight line (advantage = reward − baseline). "
        "Relative is scattered: the same reward can get very different advantages, depending on what else was "
        "sampled for that prompt.\n"
        "- **Advantage distributions (bottom left / middle):** absolute has a mean that isn't zero and a scale tied to "
        "the rewards. Relative is centred at 0 with a stable spread: a well-conditioned signal by construction.\n"
        "- **Mean advantage by action (bottom right):** which behaviours are being reinforced or suppressed right now. "
        "Actions with consistently positive advantage will dominate the policy.\n\n"
        "| | early step | late step |\n|---|---|---|\n"
        "| reward spread | large | saturated |\n| absolute advantage | wide | collapses toward 0 |\n"
        "| relative advantage | centred, std ≈ 1 | still centred, still spread |\n"
        "| learning pressure | both | mainly relative |\n\n"
        "Late in training, absolute advantages vanish when rewards saturate and the baseline catches up: gradients die "
        "and the policy freezes. Relative advantages keep ranking the rollouts, so learning pressure remains."
    )
    with st.expander("Sample rollouts at this step"):
        c1, c2 = st.columns(2)
        for col, name, ros, advs in [
            (c1, "Absolute", current["abs_rollouts"][i], [ro.reward - a["baseline"] for ro in current["abs_rollouts"][i]]),
            (c2, "Relative", current["rel_rollouts"][i], rel_adv),
        ]:
            col.markdown(f"**{name}**")
            col.dataframe(pd.DataFrame([{"action": ro.action, "reward": round(ro.reward, 3), "advantage": round(ad, 3),
                                         "prompt": ro.prompt} for ro, ad in list(zip(ros, advs))[:12]]), hide_index=True)

# ======================================================================== ⑥ ranking table
with tabs[5]:
    st.markdown("Zoom into **one prompt group at one step**: *given several candidate rollouts for the same prompt, "
                "which ones are reinforced, and why?*")
    c1, c2 = st.columns([1, 2])
    k = c1.select_slider("Step", options=list(range(1, steps + 1)), value=1, key="rp_rank_step")
    prompt_id = c2.selectbox("Prompt", [p[0] for p in rp.PROMPTS], format_func=lambda pid: rp.PROMPTS[pid][1],
                             key="rp_rank_prompt")
    i = k - 1
    noise = current["abs_maps"][i]
    if noise:
        scale, bias = noise[prompt_id]
        st.caption(f"Reward noise for this prompt at this step: scale **{scale:.2f}**, bias **{bias:+.2f}**. "
                   "*Raw* is reconstructed as (observed − bias) / scale, for teaching only.")

    def ranking(rollouts, method):
        group = [ro for ro in rollouts[i] if ro.prompt_id == prompt_id]
        observed = [ro.reward for ro in group]
        if method == "abs":
            advantages = [o - current["abs_stats"][i]["baseline"] for o in observed]
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

    left, right = st.columns(2, gap="large")
    for col, method, title in [(left, "abs", "Absolute baseline"), (right, "rel", "Group relative")]:
        table, mean_reward, advs = ranking(current[f"{method}_rollouts"], method)
        with col:
            st.markdown(f"**{title}**")
            st.dataframe(table.style.map(lambda v: f"color: {'#1fa37a' if v > 0 else '#d64545'}; font-weight: 600",
                                         subset=["advantage"]).format(precision=3), hide_index=True)
            extra = f" · global baseline {current['abs_stats'][i]['baseline']:.3f}" if method == "abs" else ""
            st.caption(f"group mean reward {mean_reward:.3f}{extra} · advantage mean {sum(advs) / len(advs):+.3f}, "
                       f"std {pd.Series(advs).std(ddof=0):.3f}")
    how_to_read(
        "- **Rank** by observed reward · **raw** is the reward before noise (teaching only) · **observed** is what the "
        "optimizer sees · **advantage** is the training signal · **correct** is ground truth, never used in training.\n"
        "- **Absolute:** every rollout is compared with one **global** baseline. Early on the baseline is poorly "
        "calibrated, so even mediocre rollouts get a large positive advantage, and a generously scored prompt can "
        "reinforce wrong answers. Late in training, once rewards saturate near the cap, correct rollouts all get almost "
        "the same advantage: no signal *among* good behaviours.\n"
        "- **Relative:** rollouts are compared with the **other rollouts for the same prompt**. The best one always gets "
        "a positive advantage and the worst a negative one, whatever the absolute reward; mean 0, std 1 by construction. "
        "Even when every rollout picks the same action, the ranking still creates a learning signal.\n\n"
        "*Learning depends on who wins, not by how much.* When debugging training, look at rankings, not reward curves: "
        "**who is being reinforced, and why?**"
    )

# ======================================================================== ⑦ why GRPO
with tabs[6]:
    st.markdown("#### PPO vs GRPO")
    st.markdown(
        "PPO often struggles in complex agent training, which is why the field is moving toward **GRPO/GSPO** and "
        "their variants. PPO's effectiveness hinges entirely on its **value estimate**:\n\n"
        "- **Computational burden:** the value model (critic) is usually as large as the policy, doubling memory and "
        "compute.\n"
        "- **Scaling:** making a value model work for long responses and complex reasoning is very hard.\n"
        "- **Noisy advantage estimates:** when the critic is wrong, updates move in the wrong direction, slowing or "
        "even destroying learning, just like the absolute baseline whose advantages have arbitrary mean and scale."
    )
    st.markdown("#### How GRPO solves the absolute problem")
    st.markdown("GRPO skips the value model. It samples a **group of G answers to the same query x** and uses the group "
                "itself as a real-data baseline:")
    st.latex(r"A_i=\frac{r(x, y_i)-\operatorname{mean}\left(\{r(x, y_j)\}_{j=1}^{G}\right)}"
             r"{\operatorname{std}\left(\{r(x, y_j)\}_{j=1}^{G}\right)}")
    st.markdown(
        "- The advantage is **always centred around 0**: the group average is the baseline, so the signal is "
        "comparative, not absolute.\n"
        "- Dividing by the std keeps a **consistent scale (≈ 1)**, preventing the signal collapse seen in the absolute "
        "baseline late in training.\n"
        "- If the reward model **drifts**, the group mean moves with it. The best answer in the group still has the "
        "highest relative score, even if its absolute value shifted."
    )
    st.link_button("Group Sequence Policy Optimization (paper)", "https://arxiv.org/pdf/2507.18071",
                   icon=":material/description:")
    st.divider()
    st.markdown("#### Key takeaways")
    c1, c2 = st.columns(2, gap="large")
    c1.markdown(
        "**Absolute feedback (baseline method)**\n"
        "- numeric rewards with a running baseline: `advantage = reward − baseline`\n"
        "- the moving-average baseline reduces variance but can't handle per-prompt scale differences\n"
        "- advantages can have any scale, depending on the reward distribution"
    )
    c2.markdown(
        "**Relative feedback (group method)**\n"
        "- absolute rewards first, then **normalized within each prompt group** (not DPO preference pairs)\n"
        "- `advantage = (reward − group mean) / group std`: mean 0, std ≈ 1\n"
        "- the best rollout in each group gets a positive advantage regardless of the reward scale"
    )
    st.markdown(
        "**Why this matters for LLMs:** reward values aren't globally comparable (prompts differ in difficulty), "
        "reward models can be miscalibrated per prompt type, and *“which is better?”* is easier to judge than "
        "*“how good is this?”*."
    )
    takeaway("<b>Relative feedback turns reward into within-prompt advantage, so learning is driven by ranking rather "
             "than absolute scale.</b> What survives to the end of training is not reward, it is ranking information.")
