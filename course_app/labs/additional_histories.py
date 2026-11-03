"""
The three trajectories from hands_on/session_04_HANDS_ON_additional_histories.ipynb,
as plain Python data (no `openpipe-art` install needed on the server).

Mirrors ART's structure:
  - plain dicts in `messages_and_choices` are context (prompt) – not trained
  - `Choice` objects are what the model generated – these tokens get trained
  - `additional_histories` are extra conversations sharing the trajectory's reward
"""

from typing import Dict, List, Optional

from labs.trajectory import History, Message, Trajectory, choice, system, tool, user

# ------------------------------------------------------------------ example 1
THINK_TOKENS = Trajectory(
    messages=[
        user("What is 2+2?"),
        choice("<think>I need to add 2 and 2</think>4", branch=0),
    ],
    additional_histories=[
        History(
            label="Turn 2, as the model actually saw it",
            note="The Qwen 3 chat template strips <think> from earlier turns, so the prompt for turn 2 "
                 "contains a cleaned turn 1. This history stores that exact view.",
            messages=[
                user("What is 2+2?"),
                choice("4"),
                user("What is 3+3?"),
                choice("<think>I need to add 3 and 3</think>6"),
            ],
        )
    ],
    reward=0.9,
    metrics={"preserved_reasoning_tokens": True},
)

# ------------------------------------------------------------------ example 2
ANALYZE_ID, FIX_ID = "call_analyze_1", "call_fix_1"

SUB_AGENTS = Trajectory(
    messages=[
        user("Analyze this codebase and fix any bugs"),
        choice(tool_calls=[{"id": ANALYZE_ID, "name": "analyze_code",
                            "arguments": '{"file": "main.py", "task": "Find potential bugs"}'}], branch=0),
        tool(ANALYZE_ID, "Found 3 potential issues: null pointer on line 42, "
                         "memory leak on line 87, uninitialized variable on line 120"),
        choice(tool_calls=[{"id": FIX_ID, "name": "fix_issues",
                            "arguments": '{"issue": "null pointer on line 42", "fix": "Add null check"}'}], branch=1),
        tool(FIX_ID, "Fixed by adding null check: if (ptr != nullptr) { ... }"),
        choice("I've analyzed the codebase and fixed the critical null pointer issue on line 42. "
               "The fix adds a null check before dereferencing the pointer."),
    ],
    additional_histories=[
        History(
            label="Sub-agent 1 · code analysis",
            note="Spawned by the analyze_code call. Its findings come back to the main agent as a tool result.",
            messages=[
                system("You are a code analysis expert. Analyze code for potential bugs."),
                user("Find potential bugs in main.py"),
                choice("Found 3 potential issues:\n1. Null pointer on line 42\n"
                       "2. Memory leak on line 87\n3. Uninitialized variable on line 120"),
            ],
        ),
        History(
            label="Sub-agent 2 · bug fixing",
            note="Spawned by the fix_issues call.",
            messages=[
                system("You are a bug fixing expert. Fix code issues safely and correctly."),
                user("Fix the null pointer issue on line 42"),
                choice("Fixed by adding null check:\n```cpp\nif (ptr != nullptr) {\n    // safe to use ptr\n}\n```"),
            ],
        ),
    ],
    reward=0.85,
    metrics={"sub_agents_used": 2, "bugs_found": 3, "bugs_fixed": 1, "task_completed": True},
)

# ------------------------------------------------------------------ example 3
COMPACTION = Trajectory(
    messages=[
        system("You are a helpful assistant."),
        user("Compacted conversation history: The user asked about quantum entanglement, and I explained the "
             "basic concept of how particles can be correlated even when separated.", branch=0),
        user("Tell me more about the history of quantum entanglement"),
        choice("Quantum entanglement was first proposed by Einstein, Podolsky, and Rosen in 1935 in what's now "
               "called the EPR paradox. They were concerned that quantum mechanics seemed to violate locality, "
               "the idea that objects can only be influenced by their immediate surroundings. However, "
               "experiments by John Bell in the 1960s and subsequent tests have confirmed that entanglement is "
               "a real physical phenomenon. Today, entanglement is a key resource in quantum computing and "
               "quantum cryptography."),
        user("How is it used in quantum computing?"),
        choice("In quantum computing, entanglement is used to create quantum gates that operate on multiple "
               "qubits simultaneously. For example, the CNOT gate creates entanglement between two qubits. "
               "This enables quantum algorithms to process information in ways that classical computers cannot. "
               "Some algorithms show large speedups for specific tasks, such as factoring (Shor's algorithm) or "
               "search (Grover's algorithm)."),
    ],
    additional_histories=[
        History(
            label="The original conversation, before compaction",
            note="This is what the one-line summary replaced. Without this history, these answers would never be trained.",
            messages=[
                system("You are a helpful assistant."),
                user("Explain quantum entanglement"),
                choice("Quantum entanglement is a phenomenon where two or more particles become correlated so "
                       "that the quantum state of each particle cannot be described independently, even when the "
                       "particles are separated by large distances. Measuring one particle lets you predict "
                       "properties of the other, regardless of distance. This 'spooky action at a distance' "
                       "(as Einstein called it) has been experimentally verified and is a fundamental aspect of "
                       "quantum mechanics."),
                user("Can you give me a simple example?"),
                choice("Imagine two coins that are perfectly anti-correlated. If one is heads, the other is "
                       "tails, every time. If you separate them far apart, observing one instantly tells you "
                       "what you'd observe for the other. This is only an analogy: real entanglement is not "
                       "about hidden predetermined coin states, but it helps build intuition."),
            ],
        )
    ],
    reward=0.92,
    metrics={"history_compacted": True, "original_turns": 4, "compacted_turns": 3},
)


def approx_tokens(text: Optional[str]) -> int:
    """Rough token estimate (~4 characters per token), good enough for a comparison."""
    return max(1, round(len(text) / 4)) if text else 0


def trained_tokens(messages: List[Message]) -> int:
    """Only generated messages (Choices) carry a training signal."""
    total = 0
    for m in messages:
        if m.is_choice:
            total += approx_tokens(m.content)
            total += sum(approx_tokens(c["name"] + c["arguments"]) for c in m.tool_calls)
    return total


def simple_reward_function(final_answer: str, metrics: Dict) -> tuple[float, List[tuple[str, float, bool]]]:
    """
    The notebook's reward function: heuristics on the *final* assistant message of the
    main conversation (RULER can't score additional histories yet).
    Returns the score and which rules fired, so the page can show the breakdown.
    """
    content = (final_answer or "").lower()
    rules = [
        ('mentions "fix" / "fixed"', 0.3, "fixed" in content or "fix" in content),
        ('mentions "null" / "bug"', 0.2, "null" in content or "bug" in content),
        ("answer longer than 100 characters", 0.3, len(content) > 100),
        ('mentions "quantum" / "entanglement"', 0.2, "quantum" in content or "entanglement" in content),
        ("metrics say sub-agents were used", 0.1, metrics.get("sub_agents_used", 0) > 0),
    ]
    if not content:
        return 0.0, [(name, points, False) for name, points, _ in rules]
    score = sum(points for _, points, fired in rules if fired)
    return min(score, 1.0), rules


def final_answer(trajectory: Trajectory) -> str:
    for m in reversed(trajectory.messages):
        if m.is_choice:
            return m.content or ""
    return ""
