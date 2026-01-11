from __future__ import annotations

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def visualize_step_details(step_idx: int, abs_rollouts, abs_stats, rel_rollouts, rel_stats):
    """Show detailed breakdown of rewards and advantages for a specific step.

    External deps you inject into this module namespace:
      - ACTIONS: list[str]
      - compute_group_advantages(rewards: list[float]) -> list[float]
    """

    if step_idx >= len(abs_rollouts):
        print(f"Step {step_idx} not available (max: {len(abs_rollouts)-1})")
        return

    # resolve injected deps
    try:
        ACTIONS = globals()["ACTIONS"]
    except KeyError:
        raise NameError(
            "ACTIONS is not defined in the module namespace. Inject it after loading, e.g. mod.ACTIONS = ACTIONS"
        )

    try:
        compute_group_advantages = globals()["compute_group_advantages"]
    except KeyError:
        raise NameError(
            "compute_group_advantages is not defined in the module namespace. "
            "Inject it after loading, e.g. mod.compute_group_advantages = compute_group_advantages"
        )

    abs_step_rollouts = abs_rollouts[step_idx]
    rel_step_rollouts = rel_rollouts[step_idx]
    abs_step_stats = abs_stats[step_idx]

    # Compute advantages: absolute
    abs_rewards = [r.reward for r in abs_step_rollouts]
    abs_baseline = abs_step_stats["baseline"]
    abs_advantages = [r - abs_baseline for r in abs_rewards]

    # Compute advantages: relative (per prompt group)
    rel_rewards = [r.reward for r in rel_step_rollouts]
    by_prompt = {}
    for idx, ro in enumerate(rel_step_rollouts):
        by_prompt.setdefault(ro.prompt_id, []).append(idx)

    rel_advantages = [0.0] * len(rel_step_rollouts)
    for _, idxs in by_prompt.items():
        group_rewards = [rel_rewards[i] for i in idxs]
        group_adv = compute_group_advantages(group_rewards)
        for i, a in zip(idxs, group_adv):
            rel_advantages[i] = a

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Top row: Rewards
    ax = axes[0, 0]
    ax.scatter(range(len(abs_rewards)), sorted(abs_rewards), alpha=0.6, s=50, label="Rewards")
    ax.axhline(abs_baseline, color="r", linestyle="--", linewidth=2, label=f"Baseline: {abs_baseline:.3f}")
    ax.set_xlabel("Rollout (sorted)")
    ax.set_ylabel("Reward")
    ax.set_title(f"Absolute: Rewards at Step {step_idx+1}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(range(len(rel_rewards)), sorted(rel_rewards), alpha=0.6, s=50, color="orange", label="Rewards")
    ax.axhline(np.mean(rel_rewards), color="r", linestyle="--", linewidth=2, label=f"Mean: {np.mean(rel_rewards):.3f}")
    ax.set_xlabel("Rollout (sorted)")
    ax.set_ylabel("Reward")
    ax.set_title(f"Relative: Rewards at Step {step_idx+1}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reward vs Advantage scatter
    ax = axes[0, 2]
    ax.scatter(abs_rewards, abs_advantages, alpha=0.6, s=50, label="Absolute")
    ax.scatter(rel_rewards, rel_advantages, alpha=0.6, s=50, color="orange", label="Relative")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Advantage")
    ax.set_title("Reward vs Advantage", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)

    # Bottom row: Advantages
    ax = axes[1, 0]
    ax.hist(abs_advantages, bins=20, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Advantage")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Absolute: Advantage Distribution\n(Mean: {np.mean(abs_advantages):.3f}, Std: {np.std(abs_advantages):.3f})",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.hist(rel_advantages, bins=20, alpha=0.7, color="orange", edgecolor="black")
    ax.axvline(0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Advantage")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Relative: Advantage Distribution\n(Mean: {np.mean(rel_advantages):.3f}, Std: {np.std(rel_advantages):.3f})",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Action wise analysis
    ax = axes[1, 2]
    action_abs_adv = defaultdict(list)
    action_rel_adv = defaultdict(list)

    for ro, adv in zip(abs_step_rollouts, abs_advantages):
        action_abs_adv[ro.action].append(adv)

    for ro, adv in zip(rel_step_rollouts, rel_advantages):
        action_rel_adv[ro.action].append(adv)

    actions_sorted = sorted(ACTIONS, key=lambda a: np.mean(action_abs_adv.get(a, [0])))
    x_pos = np.arange(len(actions_sorted))

    abs_means = [np.mean(action_abs_adv.get(a, [0])) for a in actions_sorted]
    rel_means = [np.mean(action_rel_adv.get(a, [0])) for a in actions_sorted]

    width = 0.35
    ax.barh(x_pos - width / 2, abs_means, width, label="Absolute", alpha=0.8)
    ax.barh(x_pos + width / 2, rel_means, width, label="Relative", alpha=0.8, color="orange")

    ax.set_yticks(x_pos)
    ax.set_yticklabels(actions_sorted)
    ax.set_xlabel("Mean Advantage")
    ax.set_title("Mean Advantage by Action", fontweight="bold")
    ax.legend()
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print("=" * 70)
    print(f"DETAILED ANALYSIS: Step {step_idx+1}")
    print("=" * 70)

    print("\nAbsolute Baseline Method:")
    print(f"  Number of rollouts: {len(abs_step_rollouts)}")
    print(f"  Reward range: [{min(abs_rewards):.3f}, {max(abs_rewards):.3f}]")
    print(f"  Average reward: {np.mean(abs_rewards):.3f}")
    print(f"  Baseline: {abs_baseline:.3f}")
    print(f"  Advantage range: [{min(abs_advantages):.3f}, {max(abs_advantages):.3f}]")
    print(f"  Mean advantage: {np.mean(abs_advantages):.3f}")
    print(f"  Std advantage: {np.std(abs_advantages):.3f}")

    print("\nGroup Relative Method:")
    print(f"  Number of rollouts: {len(rel_step_rollouts)}")
    print(f"  Reward range: [{min(rel_rewards):.3f}, {max(rel_rewards):.3f}]")
    print(f"  Average reward: {np.mean(rel_rewards):.3f}")
    print(f"  Advantage range: [{min(rel_advantages):.3f}, {max(rel_advantages):.3f}]")
    print(f"  Mean advantage: {np.mean(rel_advantages):.3f}")
    print(f"  Std advantage: {np.std(rel_advantages):.3f}")

    print("\nKey Insight:")
    print("  Absolute: Advantages are reward - baseline")
    print("  Relative: Advantages are z scores within prompt groups")
    print("  Relative method normalizes across prompt difficulty automatically!")
