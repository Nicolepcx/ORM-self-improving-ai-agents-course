"""
Frozen Lake: the RL logic from hands_on/session_01_HANDS_ON_frozen_lake.ipynb.

Kept as close to the notebook as possible so the code shown in the app is the
code attendees know. The one change: every function takes an explicit
`rng: random.Random` instead of using the global `random` module, so many
attendees can train at the same time on one server without sharing a seed.
"""

import math
import random
from collections import defaultdict
from typing import Dict, List, Literal, Tuple, TypedDict

# Actions
Action = Literal["up", "down", "left", "right"]
ACTIONS: List[Action] = ["up", "down", "left", "right"]

# Default 4x4 Frozen Lake map
# S = Start, F = Frozen (safe), H = Hole, G = Goal
DEFAULT_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]

MAX_STEPS = 100


class FrozenLakeGame(TypedDict):
    grid: List[List[str]]
    position: Tuple[int, int]
    steps: int
    trajectory: List[Tuple[int, int]]
    actions_taken: List[Action]
    slipped: List[bool]
    reward: float
    success: bool


def create_game(map_layout: List[str] = None) -> FrozenLakeGame:
    """Create a new Frozen Lake game, with the agent on the start tile."""
    grid = [list(row) for row in (map_layout or DEFAULT_MAP)]
    start_pos = next(
        ((i, j) for i, row in enumerate(grid) for j, tile in enumerate(row) if tile == "S"),
        (0, 0),
    )
    return {
        "grid": grid,
        "position": start_pos,
        "steps": 0,
        "trajectory": [start_pos],
        "actions_taken": [],
        "slipped": [],
        "reward": 0.0,
        "success": False,
    }


def get_tile(game: FrozenLakeGame, pos: Tuple[int, int]) -> str:
    """Get tile type at position (out of bounds counts as a hole)."""
    i, j = pos
    grid = game["grid"]
    if 0 <= i < len(grid) and 0 <= j < len(grid[0]):
        return grid[i][j]
    return "H"


def move_position(pos: Tuple[int, int], action: Action, grid: List[List[str]]) -> Tuple[int, int]:
    """Calculate new position after action (walls keep you in place)."""
    i, j = pos
    rows, cols = len(grid), len(grid[0])
    if action == "up":
        return (max(0, i - 1), j)
    if action == "down":
        return (min(rows - 1, i + 1), j)
    if action == "left":
        return (i, max(0, j - 1))
    if action == "right":
        return (i, min(cols - 1, j + 1))
    return pos


def apply_action(game: FrozenLakeGame, action: Action, slippery: float, rng: random.Random) -> bool:
    """
    Apply action to game. Returns True if game continues, False if finished.
    Reward = small step cost + terminal reward (+1 goal, -1 hole).
    """
    # Slippery ice: with probability `slippery` the action is replaced by a random one
    slipped = slippery > 0 and rng.random() < slippery
    if slipped:
        action = rng.choice(ACTIONS)

    new_pos = move_position(game["position"], action, game["grid"])
    tile = get_tile(game, new_pos)

    game["position"] = new_pos
    game["steps"] += 1
    game["trajectory"].append(new_pos)
    game["actions_taken"].append(action)
    game["slipped"].append(slipped)

    # Step cost: encourages short paths
    game["reward"] -= 0.01

    if tile == "H":
        game["reward"] += -1.0  # fell into a hole
        return False
    if tile == "G":
        game["success"] = True
        game["reward"] += 1.0  # reached the goal
        return False
    return True


def is_finished(game: FrozenLakeGame) -> bool:
    tile = get_tile(game, game["position"])
    return tile in ("H", "G") or game["steps"] >= MAX_STEPS


class TabularPolicy:
    """
    A tabular policy π(a|s) that learns with REINFORCE-style policy gradient.

    Stores one logit (weight) per state-action pair; softmax turns them
    into probabilities. An LLM does the same thing, only its "table" is
    a transformer and its actions are tokens / tool calls.
    """

    def __init__(self, learning_rate: float = 0.1, exploration_rate: float = 0.2,
                 baseline_decay: float = 0.9):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.baseline_decay = baseline_decay
        self.reward_baseline = 0.0  # running average of rewards
        # state -> action -> logit; all zeros = uniform after softmax
        self.action_weights: Dict[Tuple[int, int], Dict[Action, float]] = defaultdict(
            lambda: {action: 0.0 for action in ACTIONS}
        )

    @staticmethod
    def softmax(weights: Dict[Action, float]) -> Dict[Action, float]:
        max_w = max(weights.values())
        exp_w = {a: math.exp(w - max_w) for a, w in weights.items()}
        total = sum(exp_w.values())
        return {a: exp_w[a] / total for a in ACTIONS}

    def probs(self, state: Tuple[int, int]) -> Dict[Action, float]:
        return self.softmax(self.action_weights[state])

    def get_action(self, state: Tuple[int, int], rng: random.Random) -> Action:
        """Sample an action: explore at random with prob ε, else sample from π(a|s)."""
        if rng.random() < self.exploration_rate:
            return rng.choice(ACTIONS)
        probs = self.probs(state)
        return rng.choices(ACTIONS, weights=[probs[a] for a in ACTIONS])[0]

    def update_from_trajectory(self, trajectory: List[Tuple[int, int]],
                               actions: List[Action], reward: float) -> float:
        """
        REINFORCE update. For every (state, action) in the trajectory:
            advantage      = reward - baseline
            grad_log_prob  = 1 - p(a)  for the taken action
                           =   - p(a)  for all other actions
            weight[a]     += learning_rate * advantage * grad_log_prob
        Returns the advantage that was used.
        """
        # Baseline: exponential moving average of past rewards
        self.reward_baseline = (self.baseline_decay * self.reward_baseline
                                + (1 - self.baseline_decay) * reward)
        advantage = reward - self.reward_baseline

        for state, action in zip(trajectory[:-1], actions):
            weights = self.action_weights[state]
            probs = self.softmax(weights)
            for a in ACTIONS:
                grad_log_prob = (1.0 - probs[a]) if a == action else -probs[a]
                weights[a] += self.learning_rate * advantage * grad_log_prob
                weights[a] = max(-10.0, min(10.0, weights[a]))  # keep it stable
        return advantage

    def snapshot(self, grid: List[List[str]]) -> Dict[Tuple[int, int], Dict[Action, float]]:
        """Action probabilities for every walkable tile (used for the policy map)."""
        return {
            (i, j): self.probs((i, j))
            for i, row in enumerate(grid)
            for j, tile in enumerate(row)
            if tile not in ("H", "G")
        }


def rollout(policy: TabularPolicy, slippery: float, rng: random.Random) -> FrozenLakeGame:
    """
    One rollout = one episode: the agent follows its policy until it
    reaches the goal, falls in a hole or runs out of steps.
    The result is a trajectory (states + actions) and a reward.
    """
    game = create_game()
    while not is_finished(game):
        action = policy.get_action(game["position"], rng)
        if not apply_action(game, action, slippery, rng):
            break
    return game


def train_policy(policy: TabularPolicy, num_iterations: int, rollouts_per_iteration: int,
                 slippery: float, rng: random.Random) -> List[Dict]:
    """
    The RL loop:
      1. Rollout  – generate trajectories with the current policy
      2. Reward   – every trajectory gets a score
      3. Update   – make high-reward actions more likely
      4. Repeat
    """
    grid = create_game()["grid"]
    history = []

    for iteration in range(num_iterations):
        rollouts = [rollout(policy, slippery, rng) for _ in range(rollouts_per_iteration)]

        advantages = [
            policy.update_from_trajectory(g["trajectory"], g["actions_taken"], g["reward"])
            for g in rollouts
        ]

        history.append({
            "iteration": iteration + 1,
            "success_rate": sum(g["success"] for g in rollouts) / rollouts_per_iteration,
            "avg_reward": sum(g["reward"] for g in rollouts) / rollouts_per_iteration,
            "avg_steps": sum(g["steps"] for g in rollouts) / rollouts_per_iteration,
            "baseline": policy.reward_baseline,
            "rewards": [g["reward"] for g in rollouts],
            "advantages": advantages,
            "policy": policy.snapshot(grid),
        })

    return history


def evaluate(policy: TabularPolicy, episodes: int, slippery: float, rng: random.Random) -> float:
    """Success rate over `episodes` fresh rollouts."""
    return sum(rollout(policy, slippery, rng)["success"] for _ in range(episodes)) / episodes
