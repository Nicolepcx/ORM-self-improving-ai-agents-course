"""
Rock Paper Scissors Lizard Spock: the `rollout()` from
hands_on/session_03_HANDS_ON_rock_paper_lizard_spock.ipynb, run against the
notebook's base model (Qwen2.5-7B-Instruct) on OpenRouter.

Kept line-for-line where it matters (tools, normalize_move, beats table, reward,
metrics updates, stop condition, follow-up messages). Differences, all forced by
running outside ART + GPU:
  - both players call the same base model; in the notebook player 1 is the
    trainable copy, which is identical to the base model before training
  - plain message lists instead of `art.Trajectory.messages()`
  - a thread pool instead of `asyncio.gather` for the two parallel calls
  - `max_tokens` instead of `max_completion_tokens` (the notebook's own
    OpenRouter benchmark cell does the same)
Errors are not caught or patched: if the notebook's rollout would crash,
this one crashes the same way and the page shows the real exception.
"""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from labs.trajectory import Trajectory, from_openai_message

MODEL = "qwen/qwen-2.5-7b-instruct"  # = BASE_MODEL "Qwen/Qwen2.5-7B-Instruct" in the notebook
MAX_ROUNDS = 10
MAX_CALLS = 2 * MAX_ROUNDS

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "play_move",
            "description": "Play a move in rock-paper-scissors-lizard-spock",
            "parameters": {
                "type": "object",
                "properties": {
                    "move": {
                        "type": "string",
                        "enum": ["rock", "paper", "scissors", "lizard", "spock"],
                        "description": "The move to play",
                    }
                },
                "required": ["move"],
            },
        },
    }
]

# Allowed moves vocabulary
ALLOWED_MOVES = {"rock", "paper", "scissors", "lizard", "spock"}
MOVES = ["rock", "paper", "scissors", "lizard", "spock"]

# Rock beats: scissors, lizard · Paper beats: rock, spock · Scissors beats: paper, lizard
# Lizard beats: spock, paper · Spock beats: scissors, rock
BEATS = {
    "rock": {"scissors", "lizard"},
    "paper": {"rock", "spock"},
    "scissors": {"paper", "lizard"},
    "lizard": {"spock", "paper"},
    "spock": {"scissors", "rock"},
    "nothing": set(),
}


def normalize_move(move: str) -> str:
    """Normalize and validate move to ensure it's in the allowed vocabulary."""
    if not move:
        return "nothing"
    move = move.lower().strip()
    if move in ALLOWED_MOVES:
        return move
    move_mapping = {
        "papel": "paper", "piedra": "rock", "tijeras": "scissors", "roca": "rock",
        "papier": "paper", "pierre": "rock", "ciseaux": "scissors",
        "lagarto": "lizard", "lézard": "lizard",
    }
    if move in move_mapping:
        return move_mapping[move]
    return "nothing"


def get_tool_call_id_and_move(chat_completion) -> tuple[str, str]:
    tool_calls = chat_completion.choices[0].message.tool_calls
    if not tool_calls:
        return "n/a", "nothing"
    tool_call = tool_calls[0]
    try:
        raw_move = json.loads(tool_call.function.arguments)["move"]
        return tool_call.id, normalize_move(raw_move)
    except json.JSONDecodeError:
        return tool_call.id, "nothing"
    except KeyError:
        return tool_call.id, "nothing"
    except Exception:
        return tool_call.id, "nothing"


@dataclass
class Round:
    number: int
    moves: List[str]
    winner: Optional[int]  # 0, 1 or None for a draw


@dataclass
class RolloutResult:
    messages: List[List[dict]] = field(default_factory=lambda: [[], []])
    rewards: List[int] = field(default_factory=lambda: [0, 0])
    metrics: List[Dict] = field(default_factory=lambda: [{}, {}])
    rounds: List[Round] = field(default_factory=list)
    calls_used: int = 0
    error: Optional[str] = None

    def trajectory(self, player: int) -> Trajectory:
        return Trajectory(
            messages=[from_openai_message(m) for m in self.messages[player]],
            additional_histories=[],
            reward=self.rewards[player],
            metrics=self.metrics[player],
        )


def _assistant_dict(chat_completion) -> dict:
    """What ART stores as a Choice, as an OpenAI-format assistant message."""
    msg = chat_completion.choices[0].message
    out = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return out


def rollout(client, system_instruction: str, metrics_template: Dict, model: str = MODEL,
            on_round: Optional[Callable[[RolloutResult], None]] = None) -> RolloutResult:
    """The notebook's rollout(), with the attendee's system instruction and metrics dict filled in."""
    result = RolloutResult()
    # Two trajectories with the same system instruction and their own metrics dict
    result.messages = [
        [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": "What will your first move be?"},
        ]
        for _ in range(2)
    ]
    result.metrics = [copy.deepcopy(metrics_template) for _ in range(2)]

    def call(messages):
        return client.chat.completions.create(
            messages=messages, model=model, tools=TOOLS, max_tokens=100,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for round_number in range(1, MAX_ROUNDS + 1):
                chat_completions = list(pool.map(call, result.messages))
                result.calls_used += 2

                for messages, chat_completion in zip(result.messages, chat_completions):
                    messages.append(_assistant_dict(chat_completion))

                (id0, move0), (id1, move1) = list(map(get_tool_call_id_and_move, chat_completions))

                winner = None
                if move0 in BEATS and move1 in BEATS[move0]:
                    result.rewards[0] += 1
                    winner = 0
                elif move1 in BEATS and move0 in BEATS[move1]:
                    result.rewards[1] += 1
                    winner = 1
                result.rounds.append(Round(round_number, [move0, move1], winner))

                for metrics in result.metrics:
                    metrics["num_rounds"] += 1  # KeyError if "num_rounds" isn't in your metrics dict

                if move0 not in result.metrics[0]:
                    result.metrics[0][move0] = 0
                if move1 not in result.metrics[1]:
                    result.metrics[1][move1] = 0
                result.metrics[0][move0] += 1
                result.metrics[1][move1] += 1

                if on_round:
                    on_round(result)

                if max(result.rewards) > 2:
                    break

                result.messages[0].extend((
                    {"role": "tool", "tool_call_id": id0, "content": f"The other player played {move1}."},
                    {"role": "user", "content": "What will your next move be?"},
                ))
                result.messages[1].extend((
                    {"role": "tool", "tool_call_id": id1, "content": f"The other player played {move0}."},
                    {"role": "user", "content": "What will your next move be?"},
                ))
    except Exception as e:  # shown to the attendee exactly as the notebook would raise it
        result.error = f"{type(e).__name__}: {e}"
    return result
