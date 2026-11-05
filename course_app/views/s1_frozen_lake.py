import random

import pandas as pd
import streamlit as st

from core.ui import code_panel, page_header, takeaway, task_card
from labs import frozen_lake as fl
from labs import frozen_lake_viz as viz

DEFAULTS = {"lr": 0.1, "eps": 0.2, "slippery": 0.0, "iterations": 60, "rollouts": 10, "seed": 42}

page_header(
    session=1,
    title="Frozen Lake: the Hello World of RL",
    subtitle=(
        "An agent has to cross a frozen lake from **START** to **GOAL** without falling into a **HOLE**. "
        "It starts out knowing nothing. Through trial, reward and update it learns a *policy*, "
        "exactly the loop we will later run on LLM agents."
    ),
    minutes=15,
    notebook_path="hands_on/session_01_HANDS_ON_frozen_lake.ipynb",
)
task_card(
    task="Run a notebook with a simplified Frozen Lake Game world. Adjust the policy to see how this influences "
         "the trajectories and success rate.",
    todos=[
        "**① The game:** read the rules and the reward: a small cost per step, +1 for the goal, −1 for a hole.",
        "**② Untrained agent:** play a few episodes with the uniform policy and read its trajectory step by step.",
        "**③ Train & play:** change **one** policy setting at a time (learning rate, exploration ε, slippery ice, "
        "iterations, rollouts), predict what happens, press **Train & play**, and compare the trajectories and the "
        "success rate before vs after.",
        "**④ Experiments:** answer the questions, then reveal.",
    ],
    why=(
        "Frozen Lake is the *Hello World* of RL and your foundational blueprint: the mechanics are the same as "
        "training an LLM agent to use tools or reason through a prompt.\n"
        "- **Policy:** the agent's “brain”. Here a table of weights + softmax; in an LLM, the model weights and logits.\n"
        "- **Rollouts → trajectories:** playing the policy produces the transcript of an episode.\n"
        "- **Rewards:** the bridge between performance and optimization. The step cost works like penalizing "
        "looping or redundant reasoning in an LLM.\n"
        "- **Advantage = reward − baseline:** did better than expected → those actions become more likely. "
        "That is the path to GRPO later in the course.\n\n"
        "Adjusting the policy settings makes the trade-offs tangible: exploration vs exploitation, and fast vs stable learning."
    ),
    remember="Rollout → reward → update → repeat. The policy improves by making actions from better-than-expected "
             "trajectories more likely.",
)


# ---------------------------------------------------------------- helpers
@st.cache_data(max_entries=500, show_spinner="Training…")
def run_training(lr, eps, slippery, iterations, rollouts, seed):
    rng = random.Random(seed)
    policy = fl.TabularPolicy(learning_rate=lr, exploration_rate=eps)
    history = fl.train_policy(policy, iterations, rollouts, slippery, rng)
    return history, dict(policy.action_weights)


@st.cache_data(max_entries=500, show_spinner="Evaluating…")
def run_evaluation(lr, eps, slippery, iterations, rollouts, seed, episodes=200):
    _, weights = run_training(lr, eps, slippery, iterations, rollouts, seed)
    return (
        fl.evaluate(fl.TabularPolicy(exploration_rate=eps), episodes, slippery, random.Random(seed)),
        fl.evaluate(trained_policy(weights, eps), episodes, slippery, random.Random(seed)),
    )


def trained_policy(weights, eps) -> fl.TabularPolicy:
    policy = fl.TabularPolicy(exploration_rate=eps)
    policy.action_weights.update({s: dict(w) for s, w in weights.items()})
    return policy


def transcript(game: fl.FrozenLakeGame) -> pd.DataFrame:
    rows, reward = [], 0.0
    for k, action in enumerate(game["actions_taken"]):
        tile = fl.get_tile(game, game["trajectory"][k + 1])
        reward += -0.01 + (1.0 if tile == "G" else -1.0 if tile == "H" else 0.0)
        rows.append({
            "step": k + 1,
            "state": str(game["trajectory"][k]),
            "action": action + (" (slipped)" if game["slipped"][k] else ""),
            "lands on": {"S": "start", "F": "ice", "H": "HOLE", "G": "GOAL"}[tile],
            "reward so far": round(reward, 2),
        })
    return pd.DataFrame(rows)


def outcome(game: fl.FrozenLakeGame) -> str:
    if game["success"]:
        return "🎉 reached the goal"
    if fl.get_tile(game, game["position"]) == "H":
        return "🕳️ fell into a hole"
    return "⏱️ ran out of steps"


def episode_view(game: fl.FrozenLakeGame, key: str) -> None:
    st.plotly_chart(viz.rollout_figure(game), config={"displayModeBar": False}, key=key)
    st.markdown(f"**{outcome(game)}** · {game['steps']} steps · reward **{game['reward']:+.2f}**")


def setting(col, label, tip, **slider_kwargs):
    with col:
        value = st.slider(label, **slider_kwargs)
        st.caption(tip)
    return value


tab_game, tab_untrained, tab_train, tab_exp = st.tabs(
    ["① The game", "② Untrained agent", "③ Train & play", "④ Experiments"]
)

# ---------------------------------------------------------------- ① game
with tab_game:
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.plotly_chart(viz.grid_figure(fl.create_game()["grid"]), config={"displayModeBar": False})
    with right:
        st.markdown(
            """
**Rules**
- 4 actions: ↑ ↓ ← →. Walking into a wall keeps you in place.
- Every step costs **−0.01** (be efficient).
- **GOAL** gives **+1**, a **HOLE** gives **−1** and ends the episode.
- With *slippery ice*, the environment sometimes ignores your action.

**Same loop, different scale**

| | Frozen Lake | LLM agent |
|---|---|---|
| **State** | grid position | conversation so far |
| **Action** | ↑ ↓ ← → | next token / tool call |
| **Policy** | table of weights + softmax | transformer + logits |
| **Trajectory** | path across the lake | the full transcript |
| **Reward** | goal / hole / step cost | judge score, tests, metrics |
| **Noise** | slippery ice | tool failures, flaky APIs |
"""
        )
    code_panel(
        "Environment step & reward",
        fl.apply_action,
        "This is the whole reward function. Everything the agent learns comes from these few numbers.",
    )

# ---------------------------------------------------------------- ② untrained
with tab_untrained:
    st.markdown(
        "Before training, every action is equally likely in every state: the agent is just wandering. "
        "Play a few episodes and notice how rarely it reaches the goal."
    )
    if st.button("Play another episode", key="untrained_btn", icon=":material/casino:"):
        st.session_state.untrained_ep = st.session_state.get("untrained_ep", 0) + 1
    ep = st.session_state.get("untrained_ep", 0)
    game = fl.rollout(fl.TabularPolicy(exploration_rate=DEFAULTS["eps"]), 0.0, random.Random(1000 + ep))

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        episode_view(game, key=f"untrained_{ep}")
    with right:
        st.markdown("**The trajectory**: for an LLM, each row would be a message or tool call.")
        st.dataframe(transcript(game), hide_index=True, height=320)
    code_panel("One rollout = one episode", [fl.rollout, fl.TabularPolicy.get_action])

# ---------------------------------------------------------------- ③ train & play
with tab_train:
    st.markdown("#### 1 · Choose your settings")
    st.markdown("Read what each setting does, make a prediction, then press **Train & play**.")

    with st.form("fl_settings", border=True):
        c1, c2, c3 = st.columns(3, gap="large")
        lr = setting(
            c1, "Learning rate",
            "How big a step the policy takes after each update. "
            "**Try:** 0.01 (very slow) · 0.1 (medium) · 0.5 (fast)",
            min_value=0.01, max_value=0.5, value=DEFAULTS["lr"], step=0.01,
        )
        eps = setting(
            c2, "Exploration rate ε",
            "Chance the agent ignores its policy and moves at random, to discover new paths. "
            "**Try:** 0 (never explore) · 0.2 · 0.5 (explore a lot)",
            min_value=0.0, max_value=0.8, value=DEFAULTS["eps"], step=0.05,
        )
        slippery = setting(
            c3, "Slippery ice",
            "Chance the *environment* swaps the chosen move for a random one. The agent can't control this. "
            "**Try:** 0 (solid ice) · 0.3 (slippery)",
            min_value=0.0, max_value=0.5, value=DEFAULTS["slippery"], step=0.05,
        )
        c1, c2, c3 = st.columns(3, gap="large")
        iterations = setting(
            c1, "Training iterations",
            "How many play → score → update cycles. "
            "**Try:** 10 (quick) · 60 · 150 (long)",
            min_value=5, max_value=200, value=DEFAULTS["iterations"], step=5,
        )
        rollouts = setting(
            c2, "Rollouts per iteration",
            "Episodes played before each update. More episodes = a more reliable signal. "
            "**Try:** 2 (noisy) · 10 · 30 (stable)",
            min_value=2, max_value=50, value=DEFAULTS["rollouts"], step=1,
        )
        with c3:
            seed = st.number_input("Random seed", 0, 9999, DEFAULTS["seed"], 1)
            st.caption(
                "Same seed = same random luck, so you can compare settings fairly. "
                "**Try:** change only the seed to see how much luck matters."
            )
        submitted = st.form_submit_button("Train & play", type="primary", icon=":material/play_arrow:")

    if submitted:
        st.session_state.fl_params = dict(
            lr=lr, eps=eps, slippery=slippery, iterations=iterations, rollouts=rollouts, seed=seed
        )
        st.session_state.fl_episode = 0

    params = st.session_state.get("fl_params")
    if params is None:
        st.info("Pick your settings and press **Train & play** to train your agent.", icon=":material/arrow_upward:")
    else:
        history, weights = run_training(**params)
        untrained_rate, trained_rate = run_evaluation(**params)
        grid = fl.create_game()["grid"]

        # Keep the last few runs so attendees can compare settings.
        runs = st.session_state.setdefault("fl_runs", [])
        label = (f"lr {params['lr']} · ε {params['eps']} · slip {params['slippery']} · "
                 f"{params['iterations']}×{params['rollouts']} · seed {params['seed']}")
        if not runs or runs[-1]["label"] != label:
            runs.append({"label": label, "iterations": [h["iteration"] for h in history],
                         "success": [h["success_rate"] for h in history]})
            del runs[:-6]

        st.divider()
        st.markdown("#### 2 · Watch your agent")
        st.caption(f"Your settings: {label}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Untrained success", f"{untrained_rate:.0%}", help="Over 200 fresh episodes")
        m2.metric("Trained success", f"{trained_rate:.0%}", f"{trained_rate - untrained_rate:+.0%}",
                  help="Over 200 fresh episodes with your trained policy")
        m3.metric("Avg steps (last iteration)", f"{history[-1]['avg_steps']:.1f}")
        m4.metric("Episodes played in training", f"{params['iterations'] * params['rollouts']:,}")

        if st.button("Play another episode", key="trained_btn", icon=":material/casino:"):
            st.session_state.fl_episode += 1
        ep = st.session_state.fl_episode
        before, after = st.columns(2, gap="large")
        with before:
            st.markdown("**Before training**")
            untrained_game = fl.rollout(fl.TabularPolicy(exploration_rate=params["eps"]),
                                        params["slippery"], random.Random(7000 + ep))
            episode_view(untrained_game, key=f"before_{ep}_{label}")
        with after:
            st.markdown("**After training**")
            trained_game = fl.rollout(trained_policy(weights, params["eps"]),
                                      params["slippery"], random.Random(5000 + ep))
            episode_view(trained_game, key=f"after_{ep}_{label}")
        with st.expander("Trajectory of the trained agent, step by step"):
            st.dataframe(transcript(trained_game), hide_index=True)

        st.divider()
        st.markdown("#### 3 · How it learned")
        st.plotly_chart(viz.learning_curves(history, runs[:-1]), config={"displayModeBar": False})
        st.caption("Dotted lines are your previous runs, so you can compare settings.")

        k = st.slider("Look inside training iteration", 1, len(history), len(history), key=f"fl_iter_{label}")
        step = history[k - 1]
        left, right = st.columns([1, 1.1], gap="large")
        with left:
            st.plotly_chart(viz.policy_figure(grid, step["policy"]), config={"displayModeBar": False})
            st.caption("The policy: arrow size = probability of that move, red = most likely. "
                       "Hover a tile for exact numbers. Slide to iteration 1 to see where it started.")
        with right:
            st.plotly_chart(viz.reward_advantage_figure(step["rewards"], step["advantages"]),
                            config={"displayModeBar": False})
            st.markdown(
                "Each bar is one rollout in this iteration. Rollouts that beat the baseline "
                "(green advantage) make their moves **more** likely; the others make theirs **less** likely."
            )
        code_panel(
            "The learning update (REINFORCE)",
            fl.TabularPolicy.update_from_trajectory,
            "The advantage decides the direction, the learning rate decides the step size.",
        )
        code_panel("The RL loop", fl.train_policy)

# ---------------------------------------------------------------- ④ experiments
with tab_exp:
    st.markdown("Run each experiment in **③ Train & play**. Predict first, then reveal.")
    experiments = [
        ("Set **exploration ε = 0.5**, then **ε = 0**. What happens to trained success?",
         "With ε = 0.5 the agent moves at random half the time, *even after it has learned the way*, "
         "so success stays low (around 20%). With ε = 0 it usually does best (often 90%+), because the "
         "softmax policy is itself random at the start and already explores. But try a few seeds: "
         "now and then an ε = 0 run locks into a bad habit and never finds the goal (0%). "
         "That's the exploration vs exploitation trade-off. LLMs face the same one: "
         "sample with higher temperature during training, answer more deterministically in use."),
        ("Set **learning rate = 0.01**, then **0.5**. What changes?",
         "At 0.01 the updates are so small that after 60 iterations the agent has barely learned anything. "
         "At 0.5 it learns faster and here ends up about as good as 0.1, because the rewards are clean. "
         "Bonus: combine 0.5 with slippery ice and compare the curve with 0.1. "
         "Big steps on noisy rewards are where a high learning rate gets risky."),
        ("Set **slippery ice = 0.3**. Why does success drop even for a good policy?",
         "Success falls to roughly 20%. The environment overrides the action, so the same policy gets "
         "different rewards, and some slips land in a hole no matter how good the policy is. Noisy "
         "rewards make learning slower. LLM agents face the same thing with flaky tools and noisy judges."),
        ("Set **rollouts per iteration = 2** vs **30** (same seed). Which is more stable?",
         "More rollouts per update give a better estimate of what is actually good, so the curve is "
         "smoother. This is why GRPO samples a *group* of answers per prompt."),
        ("Set **training iterations = 10**. Is that enough?",
         "Usually not: the agent needs to stumble onto the goal a few times before there is anything "
         "to reinforce. Early training often looks flat, then success takes off."),
        ("Change only the **seed**. Why do runs differ?",
         "RL is stochastic: different random episodes lead to different early rewards and a different "
         "learning path. Always compare settings across several seeds."),
    ]
    for question, answer in experiments:
        st.markdown(f"**{question}**")
        with st.expander("Reveal"):
            st.markdown(answer)

    takeaway(
        "<b>Rollout → reward → update → repeat.</b> The policy improves by making actions from "
        "better-than-expected trajectories more likely. Swap the table for an LLM and the grid for a "
        "conversation, and this is the loop behind self-improving agents."
    )
