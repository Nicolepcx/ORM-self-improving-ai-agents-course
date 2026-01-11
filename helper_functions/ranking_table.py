from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np


def show_ranking_table(
    step_idx: int,
    prompt_id: int,
    rollouts,
    stats,
    method_name: str,
    reward_noise_map: Optional[Dict[int, Tuple[float, float]]] = None,
):
    """
    Show a per-prompt ranking table for one prompt group.
    Displays: rollout index, action, raw reward, observed reward, rank, relative advantage.

    External deps you inject into this module namespace (same pattern as ACTIONS):
      - compute_group_advantages(rewards: list[float]) -> list[float]
      - is_correct_answer(transcript: str, correct_answer: str) -> bool
    """
    if step_idx >= len(rollouts):
        print(f"Step {step_idx} not available")
        return

    # resolve injected functions
    try:
        compute_group_advantages = globals()["compute_group_advantages"]
    except KeyError:
        raise NameError(
            "compute_group_advantages is not defined in the module namespace. "
            "Inject it after loading, e.g. mod.compute_group_advantages = compute_group_advantages"
        )

    try:
        is_correct_answer = globals()["is_correct_answer"]
    except KeyError:
        raise NameError(
            "is_correct_answer is not defined in the module namespace. "
            "Inject it after loading, e.g. mod.is_correct_answer = is_correct_answer"
        )

    step_rollouts = rollouts[step_idx]
    step_stats = stats[step_idx]

    prompt_rollouts = [r for r in step_rollouts if r.prompt_id == prompt_id]

    if not prompt_rollouts:
        print(f"No rollouts found for prompt_id {prompt_id} at step {step_idx}")
        return

    observed_rewards = [r.reward for r in prompt_rollouts]

    raw_rewards = []
    if reward_noise_map is not None and prompt_id in reward_noise_map:
        scale, bias = reward_noise_map[prompt_id]
        for obs_r in observed_rewards:
            raw_rewards.append((obs_r - bias) / scale if scale > 0 else obs_r)
    else:
        raw_rewards = observed_rewards

    rewards = observed_rewards

    if method_name == "Absolute Baseline":
        baseline = step_stats["baseline"]
        advantages = [r - baseline for r in rewards]
        ranked_indices = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)
    else:
        advantages = compute_group_advantages(rewards)
        ranked_indices = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)

    print("=" * 110)
    print(f"{method_name} - Step {step_idx+1}, Prompt ID {prompt_id}: {prompt_rollouts[0].prompt}")
    if reward_noise_map is not None and prompt_id in reward_noise_map:
        scale, bias = reward_noise_map[prompt_id]
        print(f"Reward noise: scale={scale:.2f}, bias={bias:.2f} (showing raw → observed)")
    print("=" * 110)

    if reward_noise_map is not None and prompt_id in reward_noise_map:
        print(f"{'Rank':<6} {'Idx':<6} {'Action':<18} {'Raw Reward':<12} {'Observed':<12} {'Advantage':<12} {'Correct':<8}")
        print("-" * 110)
        for rank, orig_idx in enumerate(ranked_indices, 1):
            ro = prompt_rollouts[orig_idx]
            correct_str = "✓" if is_correct_answer(ro.transcript, ro.correct_answer) else "✗"
            print(
                f"{rank:<6} {orig_idx:<6} {ro.action:<18} "
                f"{raw_rewards[orig_idx]:>10.3f} {rewards[orig_idx]:>10.3f} {advantages[orig_idx]:>10.3f} {correct_str:<8}"
            )
    else:
        print(f"{'Rank':<6} {'Idx':<6} {'Action':<18} {'Observed Reward':<18} {'Relative Adv':<15} {'Correct':<8}")
        print("-" * 110)
        for rank, orig_idx in enumerate(ranked_indices, 1):
            ro = prompt_rollouts[orig_idx]
            correct_str = "✓" if is_correct_answer(ro.transcript, ro.correct_answer) else "✗"
            print(
                f"{rank:<6} {orig_idx:<6} {ro.action:<18} "
                f"{rewards[orig_idx]:>15.3f} {advantages[orig_idx]:>15.3f} {correct_str:<8}"
            )

    print("-" * 110)
    print(f"Group mean reward: {np.mean(rewards):.3f}")
    if method_name == "Absolute Baseline":
        print(f"Global baseline: {baseline:.3f}")
    print(f"Group mean advantage: {np.mean(advantages):.3f}")
    print(f"Group std advantage: {np.std(advantages):.3f}")
    print()
