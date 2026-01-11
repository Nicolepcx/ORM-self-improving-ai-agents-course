from __future__ import annotations

import random

import matplotlib.pyplot as plt
from matplotlib import animation


def animate_training_comparison(
    abs_policy_history, abs_stats_history, abs_rollout_history,
    rel_policy_history, rel_stats_history, rel_rollout_history,
    fps: int = 8,
    save_path: str = None
):
    """
    Create an animated comparison showing:
    1. Policy distributions evolving
    2. Rewards over time
    3. Advantages distribution
    4. Sample rollouts with their rewards

    Notes:
    - Expects ACTIONS (list[str]) to exist in the caller's global scope
      OR be importable from the same module (see below).
    - Expects compute_group_advantages(list[float]) -> list[float] to exist as well.
    """

    # --------- resolve external dependencies without changing signature ---------
    # Option A: user defines ACTIONS / compute_group_advantages in their notebook
    # Option B: you keep them in the same repo and adjust imports below

    # Try to find ACTIONS
    try:
        ACTIONS = globals()["ACTIONS"]
    except KeyError:
        # If you prefer, replace this with: from .your_module import ACTIONS
        raise NameError(
            "ACTIONS is not defined. Define ACTIONS in your notebook "
            "or provide it in the module where animate_training_comparison is imported from."
        )

    # Try to find compute_group_advantages
    try:
        compute_group_advantages = globals()["compute_group_advantages"]
    except KeyError:
        # If you prefer, replace this with: from .your_module import compute_group_advantages
        raise NameError(
            "compute_group_advantages is not defined. Define it in your notebook "
            "or provide it in the module where animate_training_comparison is imported from."
        )

    # -------------------------------------------------------------------------

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Top row: Policy distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Middle row: Rewards and advantages
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # Bottom row: Sample rollouts
    ax7 = fig.add_subplot(gs[2, :])

    x_actions = list(range(len(ACTIONS)))

    # Policy bars
    bars_abs = ax1.bar(x_actions, [abs_policy_history[0][a] for a in ACTIONS], alpha=0.7)
    bars_rel = ax2.bar(x_actions, [rel_policy_history[0][a] for a in ACTIONS], alpha=0.7, color="orange")

    ax1.set_xticks(x_actions)
    ax1.set_xticklabels(ACTIONS, rotation=45, ha="right", fontsize=8)
    ax2.set_xticks(x_actions)
    ax2.set_xticklabels(ACTIONS, rotation=45, ha="right", fontsize=8)

    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax1.set_title("Absolute Baseline Policy", fontsize=10, fontweight="bold")
    ax2.set_title("Group Relative Policy", fontsize=10, fontweight="bold")

    # Policy comparison (difference)
    bars_diff = ax3.bar(x_actions, [0] * len(ACTIONS), alpha=0.7, color="green")
    ax3.set_xticks(x_actions)
    ax3.set_xticklabels(ACTIONS, rotation=45, ha="right", fontsize=8)
    ax3.set_title("Policy Difference (Rel - Abs)", fontsize=10, fontweight="bold")
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.5)

    # Reward curves
    steps = [s["step"] for s in abs_stats_history]
    ax4.plot([], [], "b-", label="Avg Reward", linewidth=2)
    ax4.plot([], [], "r--", label="Baseline", linewidth=1.5)
    ax4.set_xlim(1, len(steps))
    ax4.set_ylim(-1.1, 1.1)
    ax4.set_title("Absolute: Reward vs Baseline", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Reward")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5.plot([], [], "orange", label="Avg Reward", linewidth=2)
    ax5.set_xlim(1, len(steps))
    ax5.set_ylim(-1.1, 1.1)
    ax5.set_title("Relative: Average Reward", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Reward")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Advantage comparison
    ax6.hist([], bins=20, alpha=0.6, label="Absolute", color="blue", range=(-3, 3))
    ax6.hist([], bins=20, alpha=0.6, label="Relative", color="orange", range=(-3, 3))
    ax6.set_title("Advantage Distribution (Current Step)", fontsize=10, fontweight="bold")
    ax6.set_xlabel("Advantage")
    ax6.set_ylabel("Frequency")
    ax6.legend(fontsize=8)
    ax6.set_xlim(-3, 3)

    # Sample rollouts display
    ax7.axis("off")
    rollout_text = ax7.text(
        0.05, 0.95, "",
        transform=ax7.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    reward_line_abs, = ax4.plot([], [], "b-", linewidth=2)
    baseline_line, = ax4.plot([], [], "r--", linewidth=1.5)
    reward_line_rel, = ax5.plot([], [], "orange", linewidth=2)

    def init():
        return (
            list(bars_abs)
            + list(bars_rel)
            + list(bars_diff)
            + [reward_line_abs, baseline_line, reward_line_rel, rollout_text]
        )

    def update(frame):
        abs_probs = abs_policy_history[frame]
        rel_probs = rel_policy_history[frame]

        for bar, a in zip(bars_abs, ACTIONS):
            bar.set_height(abs_probs[a])
        for bar, a in zip(bars_rel, ACTIONS):
            bar.set_height(rel_probs[a])
        for bar, a in zip(bars_diff, ACTIONS):
            diff = rel_probs[a] - abs_probs[a]
            bar.set_height(diff)
            bar.set_color("green" if diff > 0 else "red")

        xs = steps[: frame + 1]
        abs_rewards = [s["avg_reward"] for s in abs_stats_history[: frame + 1]]
        abs_baselines = [s["baseline"] for s in abs_stats_history[: frame + 1]]
        rel_rewards = [s["avg_reward"] for s in rel_stats_history[: frame + 1]]

        reward_line_abs.set_data(xs, abs_rewards)
        baseline_line.set_data(xs, abs_baselines)
        reward_line_rel.set_data(xs, rel_rewards)

        ax6.clear()
        if frame < len(abs_rollout_history):
            abs_rollouts = abs_rollout_history[frame]
            abs_rewards_step = [r.reward for r in abs_rollouts]
            abs_baseline = abs_stats_history[frame]["baseline"]
            abs_advs = [r - abs_baseline for r in abs_rewards_step]

            rel_rollouts = rel_rollout_history[frame]
            rel_rewards_step = [r.reward for r in rel_rollouts]

            by_prompt = {}
            for idx, ro in enumerate(rel_rollouts):
                by_prompt.setdefault(ro.prompt_id, []).append(idx)

            rel_advs = []
            for _, idxs in by_prompt.items():
                group_rewards = [rel_rewards_step[i] for i in idxs]
                rel_advs.extend(compute_group_advantages(group_rewards))

            ax6.hist(abs_advs, bins=20, alpha=0.6, label="Absolute", color="blue", range=(-3, 3))
            ax6.hist(rel_advs, bins=20, alpha=0.6, label="Relative", color="orange", range=(-3, 3))
            ax6.set_title(f"Advantage Distribution (Step {frame + 1})", fontsize=10, fontweight="bold")
            ax6.set_xlabel("Advantage")
            ax6.set_ylabel("Frequency")
            ax6.legend(fontsize=8)
            ax6.set_xlim(-3, 3)

            abs_samples = random.sample(abs_rollouts, min(3, len(abs_rollouts)))
            rel_samples = random.sample(rel_rollouts, min(3, len(rel_rollouts)))

            text = f"Step {frame + 1} - Sample Rollouts:\n\n"
            text += "ABSOLUTE BASELINE:\n"
            for i, ro in enumerate(abs_samples[:2]):
                text += f"  {i+1}. Action: {ro.action:15s} | Reward: {ro.reward:6.3f} | Prompt: {ro.prompt[:30]}...\n"
            text += "\nGROUP RELATIVE:\n"
            for i, ro in enumerate(rel_samples[:2]):
                text += f"  {i+1}. Action: {ro.action:15s} | Reward: {ro.reward:6.3f} | Prompt: {ro.prompt[:30]}...\n"
            rollout_text.set_text(text)

        ax1.set_title(f"Absolute Baseline Policy (Step {frame + 1})", fontsize=10, fontweight="bold")
        ax2.set_title(f"Group Relative Policy (Step {frame + 1})", fontsize=10, fontweight="bold")

        return (
            list(bars_abs)
            + list(bars_rel)
            + list(bars_diff)
            + [reward_line_abs, baseline_line, reward_line_rel, rollout_text]
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(abs_policy_history),
        init_func=init,
        interval=int(1000 / fps),
        blit=False,
        repeat=True,
    )

    plt.close(fig)

    if save_path:
        Writer = animation.writers["ffmpeg"] if "ffmpeg" in animation.writers.list() else None
        if Writer:
            writer = Writer(fps=fps, metadata=dict(artist="RL Comparison"), bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"Video saved to {save_path}")
        else:
            print("FFmpeg not available. Install with: conda install -c conda-forge ffmpeg")

    return anim
