"""
Training-input generation and RULER-style judging from
hands_on/session_02_HANDS_ON_reward_function.ipynb, with the notebook's own models
on OpenRouter (the notebook calls them via litellm as "openrouter/<model>").

Differences, all forced by running outside the notebook:
  - the OpenAI client pointed at OpenRouter instead of litellm's `acompletion`
  - synchronous calls instead of `await`
  - `print(...)` progress goes to a callback so the page can show it
  - a clearer input-generation prompt: inputs are what a user sends, never the expected output
  - `robust_score_group` takes the candidate outputs directly instead of an
    `art.TrajectoryGroup` (it only ever reads each trajectory's last message)
Errors are not swallowed beyond what the notebook itself does.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from pydantic import BaseModel, Field

from labs.s2_notebook_source import TASK_DESCRIPTIONS

# The notebook uses "openrouter/deepseek/deepseek-r1-0528" (judge) and "openrouter/moonshotai/kimi-k2"
# (input generation). In the lab both were too slow or unreliable for a live session, so the lab uses:
RULER_MODEL = "openai/gpt-6-luna-pro"
INPUT_GENERATION_MODEL = "openai/gpt-6-luna-pro"
NUM_TRAINING_INPUTS = 25  # TRAINING_CONFIG["num_training_inputs"]
MAX_GENERATION_ATTEMPTS = 8


def extract_json_object(text: str) -> str:
    if text is None:
        raise ValueError("No text to parse")

    t = text.strip()

    # Strip fenced code blocks if present
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)  # opening fence
        t = re.sub(r"\s*```$", "", t).strip()        # closing fence

    # Find first '{' or '['
    m = re.search(r"[\{\[]", t)
    if not m:
        raise ValueError(f"Could not find JSON start in: {t[:200]!r}")
    start = m.start()

    # Scan to matching closing brace/bracket while respecting strings
    stack: list[str] = []
    in_str = False
    esc = False

    for i in range(start, len(t)):
        ch = t[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            opening = stack.pop()
            if (opening == "{" and ch != "}") or (opening == "[" and ch != "]"):
                raise ValueError("Mismatched JSON brackets")
            if not stack:
                return t[start : i + 1].strip()

    raise ValueError("Could not find end of JSON object")


class TrainingInput(BaseModel):
    input: str = Field(description="The input text for the task")


class TrainingDataset(BaseModel):
    inputs: List[TrainingInput] = Field(description="List of training inputs")


def generate_training_inputs(client, task_description: str, num_examples: int = 50,
                             log: Callable[[str], None] = print) -> tuple[List[str], int]:
    """
    Generate diverse training inputs for the given task.
    Robust to models returning fewer items, wrong shape, code fences, or extra text.
    Returns the inputs and the number of model calls used.
    """
    # Clearer than the notebook's prompt: some models read the task description as
    # "do the task" and return finished outputs (e.g. text already wrapped in correction tags).
    system_prompt = f"""
You generate training INPUTS for a model that will later be trained on the task below.

The task describes what that model must do WITH each input. You do NOT perform the task.
Write only the raw input the model will receive: what a real user would send, before any processing.
Never include the expected output, a solution, corrections, tags, labels or formatting from the task's output format.

Task the model will be trained on:
{task_description}

Return STRICT JSON only. No prose. No markdown. No code fences.

Schema:
{{
  "inputs": [
    {{"input": "string"}},
    ...
  ]
}}

Rules:
- Return exactly {num_examples} items in "inputs".
- Each "input" must be realistic and different.
- Each "input" is raw user input, never the answer or the transformed result.
- No duplicates.
""".strip()

    inputs: list[str] = []
    seen: set[str] = set()

    attempt = 0
    while attempt < MAX_GENERATION_ATTEMPTS and len(inputs) < num_examples:
        attempt += 1
        remaining = num_examples - len(inputs)

        user_prompt = f"""
Generate {remaining} more items to complete the dataset.

Already have {len(inputs)} items.
Do not repeat any of these existing inputs:
{json.dumps(inputs, ensure_ascii=False, indent=2)}

Return STRICT JSON only with the same schema.
""".strip()

        log(f"Generating training inputs, attempt {attempt}, remaining {remaining}...")

        raw = ""
        try:
            response = client.chat.completions.create(
                model=INPUT_GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )

            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise ValueError("Empty model content.")

            clean = extract_json_object(raw)
            dataset = TrainingDataset.model_validate_json(clean)

            for ex in dataset.inputs:
                s = (ex.input or "").strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                inputs.append(s)
                if len(inputs) >= num_examples:
                    break

        except Exception as e:
            log(f"Attempt {attempt} failed: {type(e).__name__}: {e}")
            log(f"Raw preview: {raw[:400]!r}")

    if len(inputs) < num_examples:
        raise ValueError(f"Failed to generate {num_examples} training inputs. Got {len(inputs)}.")

    return inputs, attempt


class JudgeItem(BaseModel):
    idx: int
    score: float
    rationale: str


class JudgeResponse(BaseModel):
    items: List[JudgeItem]


@dataclass
class Judgement:
    scores: List[float]
    rationales: List[str]
    raw: str


def robust_score_group(client, candidate_outputs: List[str], judge_model: str, task_description: str,
                       temperature: float = 0.0) -> Judgement:
    """
    Robust scoring that assigns reward in [0, 1] per trajectory.
    Works even if the judge wraps JSON in markdown fences.
    """
    candidates = [{"idx": i, "assistant_output": out} for i, out in enumerate(candidate_outputs)]

    system = (
        "You are a strict evaluator.\n"
        "Return STRICT JSON only. No prose. No markdown. No code fences.\n"
        "Score each candidate output for how well it satisfies the task.\n"
        "Scores must be floats in [0, 1].\n"
    )

    user = (
        f"TASK:\n{task_description}\n\n"
        "CANDIDATES:\n"
        f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
        "Scoring rubric:\n"
        "- 1.0: Exactly matches required format and content is coherent and extracted from input\n"
        "- 0.5: Mostly matches format but missing details or minor format violations\n"
        "- 0.0: Ignores format, adds extra commentary, or does not perform task\n\n"
        'Return JSON with schema: {"items":[{"idx":0,"score":0.0,"rationale":"..."}]}\n'
    )

    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )

    raw = resp.choices[0].message.content or ""
    clean = extract_json_object(raw)
    judged = JudgeResponse.model_validate_json(clean)

    score_by_idx, rationale_by_idx = {}, {}
    for it in judged.items:
        # clamp
        score_by_idx[it.idx] = max(0.0, min(1.0, float(it.score)))
        rationale_by_idx[it.idx] = it.rationale

    return Judgement(
        scores=[score_by_idx.get(i, 0.0) for i in range(len(candidate_outputs))],
        rationales=[rationale_by_idx.get(i, "") for i in range(len(candidate_outputs))],
        raw=raw,
    )


# ------------------------------------------------------------------ judge demo (verbatim from the notebook)
JUDGE_DEMO_INPUT = ("After the last deploy the login page keeps refreshing forever on Chrome 120 on macOS. "
                    "Works on Firefox. Started yesterday.")

JUDGE_DEMO_CANDIDATES = {
    "good_trajectory": (
        "SUMMARY: Login page stuck in redirect loop on Chrome 120 after latest deploy\n"
        "PRIORITY: High\n"
        "STEPS TO REPRODUCE:\n"
        "1. Open Chrome 120 on macOS\n"
        "2. Navigate to /login\n"
        "3. Enter valid credentials and submit\n"
        "4. Observe repeated redirects/refresh and no successful login\n"
        "EXPECTED RESULT: User is authenticated and redirected to the dashboard\n"
        "ACTUAL RESULT: Page redirects/refreshes indefinitely and user cannot log in\n"
        "ENVIRONMENT: Chrome 120, macOS, regression after yesterday's deploy"
    ),
    "mediocre_trajectory": (
        "SUMMARY: Login not working on Chrome\n"
        "PRIORITY: Medium\n"
        "STEPS TO REPRODUCE:\n"
        "1. Open Chrome\n"
        "2. Try to login\n"
        "EXPECTED RESULT: Login works\n"
        "ACTUAL RESULT: Login fails\n"
        "ENVIRONMENT: Chrome"
    ),
    "bad_trajectory": "Try clearing cache.",
}


def group_relative_advantages(scores: List[float]) -> List[float]:
    """How a group-relative method (GRPO, as in ART) turns scores into a learning signal."""
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    return [(s - mean) / (std + 1e-6) for s in scores]


def task_label(constant_name: str) -> str:
    return constant_name.removesuffix("_TASK_DESCRIPTION").replace("_", " ").title()


__all__ = ["TASK_DESCRIPTIONS"]
