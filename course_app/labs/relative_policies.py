"""
Relative vs absolute feedback: the simulation from
demo/session_01_DEMO_relative_policies_RL.ipynb.

Line-for-line the notebook's functions. The one change: every function draws
from an explicit `rng: random.Random` instead of the global `random` module.
`random.Random(42)` produces exactly the stream `random.seed(42)` does, so with
the notebook's settings `run_demo()` reproduces the notebook's results, while
many attendees can run it at once on one server.
"""

import codecs
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Available actions the agent can take
ACTIONS = [
    "plan",           # Step-by-step planning
    "direct_answer",  # Direct response
    "verify",         # Answer then verify
    "use_tool",       # Use calculator tool
    "ramble",         # Long-winded response (often wrong)
    "refuse",         # Refuse to answer
]


@dataclass
class Rollout:
    """A single rollout: prompt, action taken, transcript, and reward."""
    prompt_id: int
    prompt: str
    correct_answer: str
    action: str
    transcript: str
    reward: float
    step: int  # Training step when this was generated


def softmax(logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """Convert logits to probabilities using softmax."""
    m = max(logits.values())
    exps = {k: math.exp((v - m) / max(temperature, 1e-6)) for k, v in logits.items()}
    s = sum(exps.values())
    return {k: exps[k] / s for k in exps}


def sample_action(probs: Dict[str, float], rng: random.Random) -> str:
    """Sample an action from the probability distribution."""
    r = rng.random()
    c = 0.0
    for a in ACTIONS:
        c += probs[a]
        if r <= c:
            return a
    return ACTIONS[-1]


def maybe_corrupt_answer(correct: str, action: str, temperature_noise: float, rng: random.Random) -> str:
    """Introduce occasional mistakes - some actions are more reliable than others."""
    base_error = {
        "direct_answer": 0.10,
        "plan": 0.08,
        "verify": 0.03,
        "use_tool": 0.02,
        "ramble": 0.35,
        "refuse": 1.00,
    }[action]

    p_error = min(0.95, base_error + temperature_noise)
    if rng.random() < p_error:
        try:
            v = int(correct)
            delta = rng.choice([-3, -2, -1, 1, 2, 3, 5, 7])
            return str(v + delta)
        except Exception:
            return "wrong"
    return correct


def render_transcript(prompt: str, action: str, final_answer: str, rng: random.Random) -> str:
    """Generate a mock transcript for the given action."""
    filler = [
        "Let me break this down carefully.",
        "I will sanity check the result.",
        "I will keep it concise.",
        "I will avoid unnecessary steps.",
        "I will double check arithmetic.",
    ]
    extra = "\n".join(rng.sample(filler, k=2))

    if action == "plan":
        return (
            f"User: {prompt}\n"
            f"Assistant: I will solve this step by step.\n"
            f"Assistant: {extra}\n"
            f"Assistant: Final answer: {final_answer}\n"
        )
    if action == "direct_answer":
        return f"User: {prompt}\nAssistant: {final_answer}\n"
    if action == "verify":
        return (
            f"User: {prompt}\n"
            f"Assistant: First pass gives {final_answer}. Now verification.\n"
            f"Assistant: Verification complete.\n"
            f"Assistant: Final answer: {final_answer}\n"
        )
    if action == "use_tool":
        return (
            f"User: {prompt}\n"
            f"Assistant: I will use a calculator tool.\n"
            f"Assistant: Tool call: calc({prompt!r})\n"
            f"Tool: {final_answer}\n"
            f"Assistant: Final answer: {final_answer}\n"
        )
    if action == "ramble":
        return (
            f"User: {prompt}\n"
            f"Assistant: Interesting. There are multiple angles here.\n"
            f"Assistant: I will explore possibilities and decide.\n"
            f"Assistant: I think the answer is {final_answer}.\n"
        )
    if action == "refuse":
        return f"User: {prompt}\nAssistant: I cannot help with that.\n"
    return f"User: {prompt}\nAssistant: {final_answer}\n"


def score_rollout(transcript: str, correct_answer: str, action: str) -> float:
    """Compute raw reward for a rollout. This is our 'reward model'."""
    # Correctness signal (most important)
    correct = 1.0 if (f"Final answer: {correct_answer}" in transcript or
                      transcript.strip().endswith(correct_answer)) else 0.0

    # Efficiency penalty (discourage verbosity)
    length_penalty = 0.0009 * len(transcript)

    # Shaping bonuses (encourage helpful behaviors)
    tool_bonus = 0.06 if "Tool:" in transcript else 0.0
    verify_bonus = 0.04 if "Verification" in transcript else 0.0
    plan_bonus = 0.02 if "step by step" in transcript else 0.0

    # Refusal penalty
    refuse_penalty = 0.7 if action == "refuse" else 0.0

    reward = correct + tool_bonus + verify_bonus + plan_bonus - length_penalty - refuse_penalty
    return max(-1.0, min(1.0, reward))


def create_reward_noise_map(prompts: List[Tuple[int, str, str]], rng: random.Random,
                            scale_range: Tuple[float, float] = (0.5, 1.5),
                            bias_range: Tuple[float, float] = (-0.3, 0.3)) -> Dict[int, Tuple[float, float]]:
    """
    Create per-prompt reward scale and offset noise to simulate miscalibrated reward model.
    observed_reward = clip(scale * raw_reward + bias + noise)
    """
    noise_map = {}
    for prompt_id, _, _ in prompts:
        scale = rng.uniform(scale_range[0], scale_range[1])
        bias = rng.uniform(bias_range[0], bias_range[1])
        noise_map[prompt_id] = (scale, bias)
    return noise_map


def apply_reward_noise(raw_reward: float, prompt_id: int, noise_map: Dict[int, Tuple[float, float]],
                       rng: random.Random, noise_std: float = 0.05) -> float:
    """Apply per-prompt scale, bias, and noise to raw reward."""
    scale, bias = noise_map[prompt_id]
    noise = rng.gauss(0.0, noise_std)
    observed_reward = scale * raw_reward + bias + noise
    return max(-1.0, min(1.0, observed_reward))


def is_correct_answer(transcript: str, correct_answer: str) -> bool:
    """Check if the transcript contains the correct answer."""
    return (f"Final answer: {correct_answer}" in transcript or
            transcript.strip().endswith(correct_answer))


def reveal_answer() -> str:
    """The notebook's rot13-encoded answer to the length-penalty question."""
    return codecs.decode("Gur nafjre vf hfhnyyl lrf. EY jvyy rkcybvg nal ybbcbyr va lbhe erjneq shapgvba!", "rot_13")


# ------------------------------------------------------------------ training
def compute_group_advantages(rewards: List[float]) -> List[float]:
    """Compute z-score normalized advantages for a group (relative feedback)."""
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / max(len(rewards), 1)
    sd = math.sqrt(var) if var > 1e-12 else 1.0
    return [(r - mu) / sd for r in rewards]


def generate_rollouts_for_prompts(prompts, logits, group_size, temperature, step, rng,
                                  reward_noise_map=None, noise_std=0.05) -> List[Rollout]:
    """Generate rollouts for all prompts."""
    probs = softmax(logits, temperature=temperature)
    all_rollouts: List[Rollout] = []

    for prompt_id, prompt, correct_answer in prompts:
        for _ in range(group_size):
            action = sample_action(probs, rng)

            # Add noise so "tool" is not always perfect and "verify" is usually better
            temp_noise = max(0.0, (temperature - 1.0) * 0.08)
            final_answer = maybe_corrupt_answer(correct_answer, action, temp_noise, rng)

            transcript = render_transcript(prompt, action, final_answer, rng)
            raw_reward = score_rollout(transcript, correct_answer, action)

            # Apply per-prompt reward noise if provided (simulates miscalibrated reward model)
            if reward_noise_map is not None:
                reward = apply_reward_noise(raw_reward, prompt_id, reward_noise_map, rng, noise_std)
            else:
                reward = raw_reward

            all_rollouts.append(Rollout(prompt_id, prompt, correct_answer, action, transcript, reward, step))

    return all_rollouts


def _step_stats(step_idx, rewards, advantages, rollouts, baseline=None) -> Dict:
    mean_adv = sum(advantages) / len(advantages)
    stats = {
        "step": step_idx + 1,
        "avg_reward": sum(rewards) / len(rewards),
        "best_reward": max(rewards),
        "worst_reward": min(rewards),
        "mean_adv": mean_adv,
        "std_adv": math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)),
        "correct_rate": sum(1 for r in rollouts if r.reward > 0.5) / len(rollouts),  # Reward-based
        "correctness_rate": sum(1 for r in rollouts if is_correct_answer(r.transcript, r.correct_answer)) / len(rollouts),
    }
    if baseline is not None:
        stats["baseline"] = baseline
    return stats


def train_absolute_baseline(prompts, steps, group_size, temperature, lr, rng, baseline_decay=0.95,
                            reward_noise_map=None, noise_std=0.05, noise_map_updater=None):
    """Train using absolute feedback with running baseline (REINFORCE-style)."""
    logits = {a: 0.0 for a in ACTIONS}
    baseline = 0.0
    current_noise_map = reward_noise_map
    policy_history, stats_history, rollout_history, noise_maps = [], [], [], []

    for step_idx in range(steps):
        # Update noise map mid-training if updater provided (simulates reward model drift)
        if noise_map_updater is not None:
            new_map = noise_map_updater(step_idx, current_noise_map)
            if new_map is not None:
                current_noise_map = new_map
        noise_maps.append(current_noise_map)

        rollouts = generate_rollouts_for_prompts(prompts, logits, group_size, temperature, step_idx, rng,
                                                 current_noise_map, noise_std)
        rollout_history.append(rollouts)

        rewards = [r.reward for r in rollouts]
        avg_r = sum(rewards) / len(rewards)

        # Update baseline (exponential moving average)
        baseline = baseline_decay * baseline + (1.0 - baseline_decay) * avg_r

        # Compute advantages: reward - baseline
        advantages = [r - baseline for r in rewards]

        # Update policy (pedagogical shortcut: bandit-style update)
        for ro, adv in zip(rollouts, advantages):
            logits[ro.action] += lr * adv

        policy_history.append(softmax(logits, temperature=1.0))
        stats_history.append(_step_stats(step_idx, rewards, advantages, rollouts, baseline))

    return policy_history, stats_history, rollout_history, noise_maps


def train_group_relative(prompts, steps, group_size, temperature, lr, rng,
                         reward_noise_map=None, noise_std=0.05, noise_map_updater=None):
    """Train using relative feedback within prompt groups (GRPO-style)."""
    logits = {a: 0.0 for a in ACTIONS}
    current_noise_map = reward_noise_map
    policy_history, stats_history, rollout_history, noise_maps = [], [], [], []

    for step_idx in range(steps):
        if noise_map_updater is not None:
            new_map = noise_map_updater(step_idx, current_noise_map)
            if new_map is not None:
                current_noise_map = new_map
        noise_maps.append(current_noise_map)

        rollouts = generate_rollouts_for_prompts(prompts, logits, group_size, temperature, step_idx, rng,
                                                 current_noise_map, noise_std)
        rollout_history.append(rollouts)

        rewards = [r.reward for r in rollouts]

        # Compute advantages per prompt group (relative feedback, z-score normalization)
        advantages = group_relative_advantages(rollouts)

        # Update policy (pedagogical shortcut: bandit-style update)
        for ro, adv in zip(rollouts, advantages):
            logits[ro.action] += lr * adv

        policy_history.append(softmax(logits, temperature=1.0))
        stats_history.append(_step_stats(step_idx, rewards, advantages, rollouts))

    return policy_history, stats_history, rollout_history, noise_maps


def group_relative_advantages(rollouts: List[Rollout]) -> List[float]:
    """z-score advantages within each prompt group, in rollout order."""
    rewards = [r.reward for r in rollouts]
    advantages = [0.0 for _ in rollouts]
    by_prompt: Dict[int, List[int]] = {}
    for idx, ro in enumerate(rollouts):
        by_prompt.setdefault(ro.prompt_id, []).append(idx)
    for idxs in by_prompt.values():
        for i, a in zip(idxs, compute_group_advantages([rewards[i] for i in idxs])):
            advantages[i] = a
    return advantages


# ------------------------------------------------------------------ the notebook's experiment
def make_prompts() -> List[Tuple[int, str, str]]:
    prompts, pid = [], 0
    for a, b in [(17, 25), (9, 7), (100, 58), (32, 11), (84, 16), (6, 13), (55, 9), (41, 1)]:
        prompts.append((pid, f"What is {a} + {b}?", str(a + b)))
        pid += 1
    for a, b in [(8, 9), (12, 6), (7, 11), (5, 14), (9, 13), (4, 17)]:
        prompts.append((pid, f"What is {a} * {b}?", str(a * b)))
        pid += 1
    for a, b in [(90, 37), (120, 49), (75, 33), (60, 19), (200, 111)]:
        prompts.append((pid, f"Compute {a} - {b}", str(a - b)))
        pid += 1
    return prompts


PROMPTS = make_prompts()

NOTEBOOK_SETTINGS = dict(
    seed=42, steps=80, group_size=12, temperature=1.35, lr=0.18,
    reward_noise=True, scale_range=(0.5, 1.5), bias_range=(-0.3, 0.3), noise_std=0.05,
    drift=True, noise_shift_step=40, drift_scale_range=(0.3, 1.7), drift_bias_range=(-0.4, 0.4),
)


def run_demo(seed: int, steps: int, group_size: int, temperature: float, lr: float,
             reward_noise: bool, scale_range, bias_range, noise_std: float,
             drift: bool, noise_shift_step: int, drift_scale_range, drift_bias_range) -> Dict:
    """Cell 16 of the notebook: build the noise map, then train both methods on one random stream."""
    rng = random.Random(seed)
    reward_noise_map: Optional[Dict] = (
        create_reward_noise_map(PROMPTS, rng, tuple(scale_range), tuple(bias_range)) if reward_noise else None
    )

    def updater(step_idx, current_map):
        if drift and reward_noise and step_idx == noise_shift_step:
            return create_reward_noise_map(PROMPTS, rng, tuple(drift_scale_range), tuple(drift_bias_range))
        return None

    abs_pol, abs_stats, abs_rollouts, abs_maps = train_absolute_baseline(
        PROMPTS, steps, group_size, temperature, lr, rng,
        reward_noise_map=reward_noise_map, noise_std=noise_std, noise_map_updater=updater)
    rel_pol, rel_stats, rel_rollouts, rel_maps = train_group_relative(
        PROMPTS, steps, group_size, temperature, lr, rng,
        reward_noise_map=reward_noise_map, noise_std=noise_std, noise_map_updater=updater)

    return dict(abs_pol=abs_pol, abs_stats=abs_stats, abs_rollouts=abs_rollouts, abs_maps=abs_maps,
                rel_pol=rel_pol, rel_stats=rel_stats, rel_rollouts=rel_rollouts, rel_maps=rel_maps,
                noise_map=reward_noise_map)
