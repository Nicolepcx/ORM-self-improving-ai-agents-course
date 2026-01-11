from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_training_summary(abs_stats, rel_stats, abs_pol, rel_pol):
    """
    Create comprehensive comparison plots + print summary stats.

    Expects:
      - abs_stats, rel_stats: list[dict] with keys:
          step, avg_reward, baseline (abs only used), correctness_rate, mean_adv, std_adv
      - abs_pol, rel_pol: list[dict] policies per step, keyed by action name

    Optional globals you can inject into this module:
      - ACTIONS: list[str]
      - NOISE_SHIFT_STEP: int
    """

    # Resolve ACTIONS from module globals (inject like: mod.ACTIONS = ACTIONS)
    try:
        ACTIONS = globals()["ACTIONS"]
    except KeyError:
        raise NameError(
            "ACTIONS is not defined in the module namespace. "
            "Inject it after loading, e.g. mod.ACTIONS = ACTIONS"
        )

    # NOISE_SHIFT_STEP is optional
    NOISE_SHIFT_STEP = globals().get("NOISE_SHIFT_STEP", None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = [s["step"] for s in abs_stats]

    # Average reward over time
    ax = axes[0, 0]
    ax.plot(steps, [s["avg_reward"] for s in abs_stats], "b-", label="Absolute Baseline", linewidth=2)
    ax.plot(steps, [s["avg_reward"] for s in rel_stats], "orange", label="Group Relative", linewidth=2)
    ax.plot(steps, [s["baseline"] for s in abs_stats], "r--", label="Baseline (Absolute)", linewidth=1.5, alpha=0.7)

    if NOISE_SHIFT_STEP is not None:
        ax.axvline(NOISE_SHIFT_STEP, color="gray", linestyle=":", linewidth=2, alpha=0.7, label="Reward Model Drift")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")
    ax.set_title("Average Reward Over Training", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    # Correctness rate
    ax = axes[0, 1]
    ax.plot(steps, [s["correctness_rate"] for s in abs_stats], "b-", label="Absolute Baseline", linewidth=2)
    ax.plot(steps, [s["correctness_rate"] for s in rel_stats], "orange", label="Group Relative", linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Correctness Rate")
    ax.set_title("True Correctness Rate Over Training", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Advantage statistics
    ax = axes[1, 0]
    ax.plot(steps, [s["mean_adv"] for s in abs_stats], "b-", label="Absolute Mean Adv", linewidth=2)
    ax.plot(steps, [s["mean_adv"] for s in rel_stats], "orange", label="Relative Mean Adv", linewidth=2)
    ax.plot(steps, [s["std_adv"] for s in abs_stats], "b--", label="Absolute Std Adv", linewidth=1.5, alpha=0.7)
    ax.plot(steps, [s["std_adv"] for s in rel_stats], "orange", linestyle="--", label="Relative Std Adv", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Advantage")
    ax.set_title("Advantage Statistics Over Training", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Final policy comparison
    ax = axes[1, 1]
    x_pos = np.arange(len(ACTIONS))
    width = 0.35

    abs_final = [abs_pol[-1][a] for a in ACTIONS]
    rel_final = [rel_pol[-1][a] for a in ACTIONS]

    ax.bar(x_pos - width / 2, abs_final, width, label="Absolute Baseline", alpha=0.8)
    ax.bar(x_pos + width / 2, rel_final, width, label="Group Relative", alpha=0.8, color="orange")

    ax.set_xlabel("Action")
    ax.set_ylabel("Probability")
    ax.set_title("Final Policy Distribution", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ACTIONS, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    print("\nAbsolute Baseline Method:")
    print(f"  Initial avg reward: {abs_stats[0]['avg_reward']:.3f}")
    print(f"  Final avg reward: {abs_stats[-1]['avg_reward']:.3f}")
    print(f"  Improvement: {abs_stats[-1]['avg_reward'] - abs_stats[0]['avg_reward']:+.3f}")
    print(f"  Final correctness rate: {abs_stats[-1]['correctness_rate']:.1%}")
    print(f"  Final baseline: {abs_stats[-1]['baseline']:.3f}")

    print("\nGroup Relative Method:")
    print(f"  Initial avg reward: {rel_stats[0]['avg_reward']:.3f}")
    print(f"  Final avg reward: {rel_stats[-1]['avg_reward']:.3f}")
    print(f"  Improvement: {rel_stats[-1]['avg_reward'] - rel_stats[0]['avg_reward']:+.3f}")
    print(f"  Final correctness rate: {rel_stats[-1]['correctness_rate']:.1%}")

    print("\nComparison:")
    abs_improvement = abs_stats[-1]["avg_reward"] - abs_stats[0]["avg_reward"]
    rel_improvement = rel_stats[-1]["avg_reward"] - rel_stats[0]["avg_reward"]
    print(f"  Absolute improvement: {abs_improvement:+.3f}")
    print(f"  Relative improvement: {rel_improvement:+.3f}")
    print(f"  Difference: {rel_improvement - abs_improvement:+.3f}")
