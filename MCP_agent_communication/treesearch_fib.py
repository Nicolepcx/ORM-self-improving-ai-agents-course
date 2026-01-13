from __future__ import annotations
import os
import re
import json
import time
import asyncio
import threading
from dataclasses import dataclass, field
from typing import TypedDict, Literal, Optional, Sequence, Dict, Any

from dotenv import load_dotenv

import treequest as tq
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from mcp_run_python import code_sandbox  # MCP server helper

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.status import Status
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.columns import Columns


# --- env & LLM setup ---------------------------------------------------------

load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
# OPENAI_API_KEY is read by langchain_openai from env

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.7)
review_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
judge = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# --- prompts -----------------------------------------------------------------

PROMPT_INITIAL = (
    "Q: Write code in Python for the Fibonacci sequence.\n"
    "Return ONLY a single fenced code block with language tag python and NO comments or prose."
)

PROMPT_REFINE_BASE = (
    "You are improving Python code. Keep it correct and idiomatic.\n"
    "Here is the current answer:\n\n{answer}\n\n"
    "Feedback:\n{feedback}\n\n"
    "Return ONLY one fenced code block with language tag python and NO comments or prose."
)

PROMPT_REVIEW_BASE = (
    "You are a senior Python reviewer. Improve API clarity, naming, typing, and make it concise.\n"
    "Do not add any top-level comments or prose. A short function docstring is allowed.\n"
    "Here is the current answer:\n\n{answer}\n\n"
    "Keep the same contract if possible. Return ONLY one fenced code block with language tag python."
)

PERF_TIPS = (
    "Performance tips:\n"
    "• Use an iterative approach for single Fibonacci(n)\n"
    "• Use a generator when you need many values in sequence\n"
    "• Memoize only if you must compute many distinct n values\n"
    "• Avoid naive recursion for large n due to exponential time"
)

# --- scoring  ----------------------------------------------------------------

class ScoreResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)

# --- state payloads ----------------------------------------------------------

@dataclass
class NodeState:
    llm_answer: str
    score: float
    tests_ok: Optional[bool] = None
    bench: Dict[str, Any] | None = None
    note: str = ""
    messages: list[str] = field(default_factory=list)  # agent-to-agent handoffs


class LGState(TypedDict, total=False):
    iterations: int
    best_answer: str
    best_score: float
    trace: str
    best_messages: list[str]

# --- helpers: extraction & sentinels ----------------------------------------

FIB10 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

def extract_python_block(text: str) -> Optional[str]:
    patterns = [
        r"```python\s*(.*?)```",
        r"```py\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith(("```", "# Q:", "# A:"))]
    code = "\n".join(lines).strip()
    return code or None

def _esc_triple_single(s: str) -> str:
    return s.replace("'''", "\\'\\'\\'")

# --- MCP sandbox (runs Pyodide in Deno) -------------------------------------

def _sb_log(level: str, message: str):
    if level.lower() in ("error", "warning"):
        print(f"{level}: {message}")

class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    def start(self):
        self.thread.start()
    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)

class SandboxClient:
    """
    Synchronous façade over the async `code_sandbox` context.
    Reuses one sandbox for the whole run.
    """
    def __init__(self, dependencies: Sequence[str] | None = None, log_handler=_sb_log):
        self.deps = list(dependencies or [])
        self.log_handler = log_handler
        self._loop_thread = _LoopThread()
        self._ctx = None
        self._sb = None

    def start(self):
        self._loop_thread.start()
        loop = self._loop_thread.loop
        self._ctx = code_sandbox(dependencies=self.deps, log_handler=self.log_handler)
        self._sb = asyncio.run_coroutine_threadsafe(self._ctx.__aenter__(), loop).result()

    def eval(self, code: str, vars: Dict[str, Any] | None = None, timeout: float = 8.0) -> Dict[str, Any]:
        assert self._sb is not None, "SandboxClient not started"
        fut = asyncio.run_coroutine_threadsafe(self._sb.eval(code, vars or {}), self._loop_thread.loop)
        try:
            return fut.result(timeout=timeout)
        except asyncio.TimeoutError:
            return {"status": "error", "stdout": "", "stderr": "", "return_value": None, "error": "timeout"}

    def close(self):
        try:
            if self._ctx is not None:
                fut = asyncio.run_coroutine_threadsafe(
                    self._ctx.__aexit__(None, None, None),
                    self._loop_thread.loop,
                )
                try:
                    fut.result(timeout=2)
                except Exception:
                    pass
        finally:
            self._loop_thread.stop()

# --- sandboxed test/bench ----------------------------------------------------

def run_unit_tests(sb: SandboxClient, code: str) -> tuple[bool, str]:
    code_src = _esc_triple_single(code)
    payload = f"""
NS = {{}}
SRC = r'''{code_src}'''
exec(compile(SRC, '<user>', 'exec'), NS, NS)

import json
expected = {FIB10!r}

try:
    fn = None
    for name in ('fib','fibonacci','Fibonacci','fib_seq'):
        obj = NS.get(name)
        if callable(obj):
            fn = obj
            break

    got = None
    if fn is not None:
        try:
            out = fn(10)
            if isinstance(out, (list, tuple)):
                got = list(out)
            else:
                try:
                    it = iter(out)
                    got = list(it)[:10]
                except TypeError:
                    if isinstance(out, int):
                        got = [fn(i) for i in range(10)]
        except TypeError:
            try:
                out = fn()
                if isinstance(out, (list, tuple)):
                    got = list(out)[:10]
                else:
                    try:
                        it = iter(out)
                        got = list(it)[:10]
                    except TypeError:
                        pass
            except Exception:
                pass

    if got is None:
        seq = NS.get('result') or NS.get('seq')
        if isinstance(seq, (list, tuple)):
            got = list(seq)[:10]

    results = []
    results.append(("fib_sequence_0_9", got == expected, f"expected {{expected!r}}, got {{got!r}}"))
    ok_type = isinstance(got, list) and all(isinstance(x, int) for x in got) if got is not None else False
    results.append(("type_ints", ok_type, "sequence not all ints"))
    ok_nonneg = bool(got) and all(x >= 0 for x in got) if got is not None else False
    results.append(("non_negative", ok_nonneg, "sequence contains negatives"))

    ___RV___ = {{"results": results}}
except Exception as e:
    ___RV___ = {{"results":[("harness_exception", False, repr(e))]}}

___RV___
"""
    res = sb.eval(payload, timeout=6.0)
    if res.get("status") == "success" and isinstance(res.get("return_value"), dict):
        data = res["return_value"]
        fails = [f"{name}: {note}" for name, ok, note in data["results"] if not ok]
        return (len(fails) == 0, "all tests passed" if not fails else " | ".join(fails))
    s = (res.get("stdout") or "") + (res.get("stderr") or "")
    try:
        for ln in reversed(s.splitlines()):
            if ln.strip().startswith("{") and ln.strip().endswith("}"):
                data = json.loads(ln)
                fails = [f"{name}: {note}" for name, ok, note in data["results"] if not ok]
                return (len(fails) == 0, "all tests passed" if not fails else " | ".join(fails))
    except Exception:
        pass
    return False, f"sandbox_error_or_empty_output: {res.get('error')!r}"

def run_benchmark(sb: SandboxClient, code: str) -> dict:
    code_src = _esc_triple_single(code)
    payload = f"""
NS = {{}}
SRC = r'''{code_src}'''
exec(compile(SRC, '<user>', 'exec'), NS, NS)

import json, time, tracemalloc

result = {{
    "contract": "missing",
    "runtime_ms": None,
    "runtime20_ms": None,
    "runtime30_ms": None,
    "growth_ratio": None,
    "bytes_used": None,
    "notes": ""
}}

try:
    fn = None
    for name in ('fib','fibonacci','Fibonacci','fib_seq'):
        obj = NS.get(name)
        if callable(obj):
            fn = obj
            break

    if fn is None:
        result["notes"] = "no fibonacci-like function found"
    else:
        contract = "unknown"
        try:
            out = fn(10)
            if isinstance(out, int):
                contract = "nth"
            else:
                try:
                    iter(out); contract = "sequence"
                except TypeError:
                    contract = "unknown"
        except TypeError:
            try:
                out = fn()
                try:
                    iter(out); contract = "sequence"
                except TypeError:
                    contract = "unknown"
            except Exception:
                contract = "unknown"

        result["contract"] = contract

        if contract == "nth":
            def _time_ms(n, reps=600):
                t0 = time.perf_counter()
                for _ in range(reps): fn(n)
                return (time.perf_counter() - t0) * 1000.0
            t20 = _time_ms(20)
            t30 = _time_ms(30)
            result["runtime20_ms"] = t20
            result["runtime30_ms"] = t30
            result["runtime_ms"] = t30
            result["growth_ratio"] = (t30 + 1e-9) / (t20 + 1e-9)
            result["notes"] = "nth: timed at n=20,30"
        elif contract == "sequence":
            reps = 300
            t0 = time.perf_counter()
            for _ in range(reps):
                try:
                    out = fn(30)
                except TypeError:
                    out = fn()
                if hasattr(out, '__iter__') and not isinstance(out, (list, tuple)):
                    list(out)
            t30 = (time.perf_counter() - t0) * 1000.0
            result["runtime30_ms"] = t30
            result["runtime_ms"] = t30
            result["growth_ratio"] = 1.0
            tracemalloc.start()
            try:
                out = fn(30)
            except TypeError:
                out = fn()
            _ = list(out)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result["bytes_used"] = int(peak)
            result["notes"] = "sequence: timed at n=30 with peak memory"
        else:
            result["notes"] = "unknown contract"
except RecursionError:
    result["runtime_ms"] = float('inf')
    result["growth_ratio"] = float('inf')
    result["notes"] = "recursion error"
except Exception as e:
    result["notes"] = f"harness_exception: {{repr(e)}}"

result
"""
    res = sb.eval(payload, timeout=6.0)
    if res.get("status") == "success" and isinstance(res.get("return_value"), dict):
        data = res["return_value"]
        if data.get("runtime_ms") is None:
            data["runtime_ms"] = float("inf")
        return data
    s = (res.get("stdout") or "") + (res.get("stderr") or "")
    try:
        for ln in reversed(s.splitlines()):
            if ln.strip().startswith("{") and ln.strip().endswith("}"):
                data = json.loads(ln)
                if data.get("runtime_ms") is None:
                    data["runtime_ms"] = float("inf")
                return data
    except Exception:
        pass
    return {"contract": "error", "runtime_ms": float("inf"),
            "runtime20_ms": None, "runtime30_ms": None,
            "growth_ratio": None, "bytes_used": None,
            "notes": res.get("error") or "no output"}

# --- scoring ----------------------------------------------------------------

def refine_prompt(
    answer: str,
    test_ok: Optional[bool],
    fail_note: str,
    bench: Dict[str, Any],
    budget_ms: float,
) -> str:
    fb = []
    if test_ok is False:
        fb.append("Unit tests failed. Fix correctness first.\n" + fail_note)
    elif test_ok is None:
        fb.append("No valid ```python fenced block``` detected. Return exactly one fenced block.")
    else:
        fb.append("Unit tests passed.")

    runtime20_ms = bench.get("runtime20_ms")
    runtime30_ms = bench.get("runtime30_ms") or bench.get("runtime_ms")
    contract = bench.get("contract")
    growth = bench.get("growth_ratio")

    if runtime20_ms is not None:
        fb.append(f"Measured runtime n=20: {runtime20_ms:.3f} ms.")
    if runtime30_ms is not None and runtime30_ms != float("inf"):
        fb.append(f"Measured runtime n=30: {float(runtime30_ms):.3f} ms.")
    fb.append(f"Target budget (n=30): {budget_ms:.3f} ms.")
    if contract:
        fb.append(f"Detected API contract: {contract}. Prefer iterative or fast doubling where applicable.")
    if growth is not None:
        fb.append(f"Growth ratio time(n=30)/time(n=20) ≈ {float(growth):.2f}.")
    fb.append(PERF_TIPS)
    return PROMPT_REFINE_BASE.format(answer=answer, feedback="\n".join(fb))

def _perf_score_dual(runtime20_ms: float | None, runtime30_ms: float | None, budget_ms: float) -> float:
    if runtime30_ms is None or runtime30_ms == float("inf"):
        return 0.0
    r = max(0.0, float(runtime30_ms) / max(1e-6, budget_ms))
    if r <= 1.0:
        return 1.0
    if r <= 2.0:
        return max(0.0, 1.0 - 0.6 * (r - 1.0))
    return max(0.0, 0.4 * (1.0 / r))

def _contract_bonus(contract: str | None, growth: float | None) -> float:
    if contract == "nth" and growth is not None and float(growth) < 1.15:
        return 0.18
    if contract == "sequence":
        return 0.10
    if contract == "nth":
        return 0.08
    return 0.0

def _growth_penalty(growth: float | None) -> float:
    if growth is None:
        return 0.25
    g = float(growth)
    if g == float("inf"):
        return 0.25
    if g >= 4.0:
        return 0.25
    if g >= 2.0:
        return 0.18
    if g >= 1.6:
        return 0.10
    if g >= 1.3:
        return 0.04
    return 0.0

def _structure_bonus(src: str) -> float:
    b = 0.0
    if "yield" in src:
        b += 0.08
    if re.search(r"def\s+fib(?:onacci)?\s*\(.*\)\s*->\s*[\w\[\], ]+", src):
        b += 0.03
    if re.search(r'"""[^"]{10,}"""', src) or re.search(r"'''[^']{10,}'''", src):
        b += 0.03
    if re.search(r"def\s+(fib|fibonacci)\s*\(", src):
        b += 0.02
    lines = [ln for ln in src.splitlines() if ln.strip()]
    if len(lines) > 60:
        b -= 0.04
    comment_lines = sum(1 for ln in lines if ln.lstrip().startswith("#"))
    if comment_lines >= 1:
        b -= 0.03
    if re.search(r"#\s*example", src, re.I):
        b -= 0.02
    return max(-0.08, min(0.12, b))

def _memory_penalty_bytes(bytes_used: int | None, contract: str | None) -> float:
    if bytes_used is None or contract != "sequence":
        return 0.0
    if bytes_used <= 6000:
        return 0.0
    return min(0.12, (bytes_used - 6000) / 50000.0)

def evaluate_answer(
    answer: str,
    tests_ok: Optional[bool],
    bench: dict,
    budget_ms: float = 5.0,
) -> float:
    if tests_ok is False:
        return 0.35

    contract = bench.get("contract")
    growth = bench.get("growth_ratio")
    rt20 = bench.get("runtime20_ms")
    rt30 = bench.get("runtime30_ms") or bench.get("runtime_ms")
    bytes_used = bench.get("bytes_used")

    perf = _perf_score_dual(rt20, rt30, budget_ms)
    src = extract_python_block(answer) or ""
    s_bonus = _structure_bonus(src)
    c_bonus = _contract_bonus(contract, growth)
    g_pen = _growth_penalty(growth)
    m_pen = _memory_penalty_bytes(bytes_used, contract)

    correctness = 1.0 if tests_ok is True else 0.8

    rubric = (
        0.55 * correctness +
        0.28 * perf +
        c_bonus + s_bonus -
        g_pen - m_pen
    )
    rubric = max(0.0, min(1.0, rubric))

    try:
        structured = judge.with_structured_output(ScoreResponse)
        j = structured.invoke([HumanMessage(
            content=("Score 0..1 as JSON {\"score\": x}. "
                     "Focus on API clarity, naming, docstring, and usability only.\n\n"
                     f"Answer:\n{answer}")
        )]).score
        judge_part = 0.25 + 0.5 * float(j)
    except Exception:
        judge_part = 0.5

    blended = 0.82 * rubric + 0.18 * judge_part
    if tests_ok is True and perf >= 0.98 and g_pen == 0.0 and m_pen == 0.0:
        return min(1.0, blended)
    return min(0.985, blended)

# --- role functions used by nodes & MCTS ------------------------------------

def role_coder(sb: SandboxClient, parent: Optional[NodeState], step_idx: int) -> NodeState:
    budgets = (4.0, 6.0, 8.0)
    budget = budgets[step_idx % len(budgets)]
    if parent is None:
        out = initial_generation(sb, budget_ms=budget)
        out.note = f"[coder] {out.note or 'initial generation'}"
        out.messages.append(f"[coder] produced initial code (score={out.score:.3f})")
        return out

    # refine using previous state as context/evidence
    code = extract_python_block(parent.llm_answer)
    if not code:
        ok, note = None, "no code block found"
        bench = {"runtime_ms": float("inf"), "contract": "missing", "growth_ratio": None,
                 "runtime20_ms": None, "runtime30_ms": None, "bytes_used": None}
    else:
        ok, note = run_unit_tests(sb, code)
        bench = run_benchmark(sb, code) if ok else {"runtime_ms": float("inf"), "contract": "missing",
                                                    "growth_ratio": None, "runtime20_ms": None,
                                                    "runtime30_ms": None, "bytes_used": None}
    out = refine_answer(sb, parent.llm_answer, parent.score, ok, note, bench, budget_ms=budget)
    out.messages = (parent.messages if parent else []) + [f"[coder] refined code (prev={parent.score:.3f} → new={out.score:.3f})"]
    out.note = f"[coder] {out.note or 'refined'}"
    return out


def role_tester(sb: SandboxClient, parent: Optional[NodeState]) -> NodeState:
    if parent is None:
        # nothing to test; fall back to coder
        out = role_coder(sb, parent, 0)
        out.messages.append("[tester] nothing to test; invoked coder")
        out.note = f"[tester] {out.note}"
        return out

    code = extract_python_block(parent.llm_answer) or ""
    if not code:
        tests_ok, note = None, "no code to test"
        bench = {"runtime_ms": float("inf"), "contract": "missing", "growth_ratio": None,
                 "runtime20_ms": None, "runtime30_ms": None, "bytes_used": None}
    else:
        tests_ok, note = run_unit_tests(sb, code)
        bench = run_benchmark(sb, code) if tests_ok else {"runtime_ms": float("inf"), "contract": "missing",
                                                          "growth_ratio": None, "runtime20_ms": None,
                                                          "runtime30_ms": None, "bytes_used": None}
    score = evaluate_answer(parent.llm_answer, tests_ok, bench, budget_ms=6.0)
    out = NodeState(llm_answer=parent.llm_answer, score=score, tests_ok=tests_ok, bench=bench, note=note)
    out.messages = parent.messages + [f"[tester] tests_ok={tests_ok} rt={(bench.get('runtime30_ms') or bench.get('runtime_ms'))}"]
    out.note = f"[tester] {out.note or 'tested'}"
    return out


def role_reviewer(sb: SandboxClient, parent: Optional[NodeState]) -> NodeState:
    if parent is None:
        out = role_coder(sb, parent, 0)
        out.messages.append("[reviewer] nothing to review; invoked coder")
        out.note = f"[reviewer] {out.note}"
        return out

    msgs: Sequence[BaseMessage] = [
        SystemMessage(content="You improve clarity and typing. Return only one ```python block."),
        HumanMessage(content=PROMPT_REVIEW_BASE.format(answer=parent.llm_answer)),
    ]
    reviewed = review_llm.invoke(msgs).content.strip()
    code = extract_python_block(reviewed)
    if not code:
        tests_ok, note = None, "no code block found"
        bench = {"runtime_ms": float("inf"), "contract": "missing", "growth_ratio": None,
                 "runtime20_ms": None, "runtime30_ms": None, "bytes_used": None}
    else:
        tests_ok, note = run_unit_tests(sb, code)
        bench = run_benchmark(sb, code) if tests_ok else {"runtime_ms": float("inf"), "contract": "missing",
                                                          "growth_ratio": None, "runtime20_ms": None,
                                                          "runtime30_ms": None, "bytes_used": None}
    score = evaluate_answer(reviewed, tests_ok, bench, budget_ms=6.0)
    out = NodeState(llm_answer=reviewed, score=score, tests_ok=tests_ok, bench=bench, note=note)
    out.messages = parent.messages + [f"[reviewer] adjusted API/readability (score={score:.3f})"]
    out.note = f"[reviewer] {out.note or 'reviewed'}"
    return out


# --- agent subgraph (coder -> tester -> reviewer) ---------------------------

class AgentState(TypedDict, total=False):
    parent: Optional[NodeState]
    step_idx: int
    out: NodeState

def coder_node_ag(state: AgentState, sb: SandboxClient, console: Console) -> Command[Literal["tester_ag"]]:
    console.print("\n[bold blue]┌─ 🤖 Coder Agent ─────────────────────────────┐[/bold blue]")
    with console.status("[bold blue]Generating code...", spinner="dots"):
        s = role_coder(sb, state.get("parent"), state.get("step_idx", 0))
    
    # Show generated code snippet
    code = extract_python_block(s.llm_answer)
    if code:
        syntax = Syntax(code[:300], "python", theme="monokai", line_numbers=False, word_wrap=True)
        console.print(Panel(
            syntax,
            title="[bold blue]Generated Code[/bold blue]",
            border_style="blue",
            expand=False
        ))
    
    if s.messages:
        console.print(f"[dim blue]  {s.messages[-1]}[/dim blue]")
    console.print("[bold blue]└───────────────────────────────────────────────┘[/bold blue]")
    console.print("[bold]  ↓ Handoff to Tester Agent[/bold]\n")
    return Command(update={"out": s}, goto="tester_ag")

def tester_node_ag(state: AgentState, sb: SandboxClient, console: Console) -> Command[Literal["reviewer_ag"]]:
    console.print("\n[bold yellow]┌─ 🧪 Tester Agent ─────────────────────────────┐[/bold yellow]")
    with console.status("[bold yellow]Running tests & benchmarks via MCP sandbox...", spinner="dots"):
        s = role_tester(sb, state.get("out"))
    
    # Show MCP execution results prominently
    test_icon = "✅" if s.tests_ok else "❌" if s.tests_ok is False else "⚠️"
    test_color = "green" if s.tests_ok else "red" if s.tests_ok is False else "yellow"
    
    rt = s.bench.get('runtime30_ms') or s.bench.get('runtime_ms') if s.bench else None
    rt_str = f"{rt:.3f} ms" if rt is not None and rt != float('inf') else "∞ (timeout/error)"
    
    bench_info = f"""{test_icon} Unit Tests: {"PASSED" if s.tests_ok else "FAILED" if s.tests_ok is False else "N/A"}
⚡ Runtime: {rt_str}
📊 Contract: {s.bench.get('contract', 'N/A') if s.bench else 'N/A'}
📈 Growth Ratio: {s.bench.get('growth_ratio', 'N/A') if s.bench else 'N/A'}
💾 Memory: {f"{s.bench.get('bytes_used', 0)} bytes" if s.bench and s.bench.get('bytes_used') else 'N/A'}"""
    
    if s.note and s.note != "all tests passed":
        bench_info += f"\n📝 Note: {s.note[:100]}"
    
    console.print(Panel(
        bench_info,
        title=f"[bold {test_color}]MCP Sandbox Execution Results[/bold {test_color}]",
        border_style=test_color,
        expand=False
    ))
    
    if s.messages:
        console.print(f"[dim yellow]  {s.messages[-1]}[/dim yellow]")
    console.print("[bold yellow]└───────────────────────────────────────────────┘[/bold yellow]")
    console.print("[bold]  ↓ Handoff to Reviewer Agent[/bold]\n")
    return Command(update={"out": s}, goto="reviewer_ag")

def reviewer_node_ag(state: AgentState, sb: SandboxClient, console: Console) -> Command[Literal["__end__"]]:
    console.print("\n[bold green]┌─ 📝 Reviewer Agent ─────────────────────────────┐[/bold green]")
    
    # Get parent code for comparison
    parent_state = state.get("out")
    parent_code = extract_python_block(parent_state.llm_answer) if parent_state else None
    
    with console.status("[bold green]Improving API & readability...", spinner="dots"):
        s = role_reviewer(sb, state.get("out"))
    
    # Show improved code
    new_code = extract_python_block(s.llm_answer)
    if new_code:
        syntax = Syntax(new_code[:300], "python", theme="monokai", line_numbers=False, word_wrap=True)
        console.print(Panel(
            syntax,
            title="[bold green]Improved Code[/bold green]",
            border_style="green",
            expand=False
        ))
        
        # Show before/after comparison if code changed
        if parent_code and parent_code != new_code:
            console.print("[dim]  Code was modified by reviewer[/dim]")
    
    # Show score information
    score_info = f"Final Score: [bold green]{s.score:.3f}[/bold green]"
    if parent_state and hasattr(parent_state, 'score'):
        score_change = s.score - parent_state.score
        if score_change != 0:
            arrow = "📈" if score_change > 0 else "📉"
            color = "green" if score_change > 0 else "red"
            score_info += f" ({arrow} {score_change:+.3f} from {parent_state.score:.3f})"
    
    console.print(Panel(score_info, border_style="green", expand=False))
    
    if s.messages:
        console.print(f"[dim green]  {s.messages[-1]}[/dim green]")
    console.print("[bold green]└───────────────────────────────────────────────┘[/bold green]\n")
    return Command(update={"out": s}, goto="__end__")

def build_agent_subgraph(sb: Optional[SandboxClient], console: Optional[Console] = None):
    # Bind sandbox and console into closures so nodes can use them at runtime.
    if console is None:
        console = Console()
    def _coder(state: AgentState): return coder_node_ag(state, sb, console)
    def _tester(state: AgentState): return tester_node_ag(state, sb, console)
    def _reviewer(state: AgentState): return reviewer_node_ag(state, sb, console)

    g = StateGraph(AgentState)
    g.add_node("coder_ag", _coder)
    g.add_node("tester_ag", _tester)
    g.add_node("reviewer_ag", _reviewer)
    g.add_edge(START, "coder_ag")
    g.add_edge("coder_ag", "tester_ag")
    g.add_edge("tester_ag", "reviewer_ag")
    return g.compile()


# --- LLM steps wrapped for sandboxed eval -----------------------------------

def initial_generation(sb: SandboxClient, budget_ms: float = 5.0) -> NodeState:
    msgs: Sequence[BaseMessage] = [
        SystemMessage(content="You write correct and fast Python. Return only one ```python fenced block and nothing else."),
        HumanMessage(content=PROMPT_INITIAL),
    ]
    answer = llm.invoke(msgs).content.strip()
    code = extract_python_block(answer)

    if not code:
        tests_ok, bench, note = None, {"runtime_ms": float("inf"), "contract": "missing", "growth_ratio": None,
                                       "runtime20_ms": None, "runtime30_ms": None, "bytes_used": None}, "no code block found"
    else:
        tests_ok, note = run_unit_tests(sb, code)
        bench = run_benchmark(sb, code) if tests_ok else {"runtime_ms": float("inf"), "contract": "missing",
                                                          "growth_ratio": None, "runtime20_ms": None,
                                                          "runtime30_ms": None, "bytes_used": None}

    score = evaluate_answer(answer, tests_ok, bench, budget_ms=budget_ms)
    return NodeState(llm_answer=answer, score=score, tests_ok=tests_ok, bench=bench, note=note)

def refine_answer(
    sb: SandboxClient,
    llm_answer: str,
    prev_score: float,
    test_ok: Optional[bool],
    fail_note: str,
    bench_prev: Dict[str, Any],
    budget_ms: float = 5.0,
) -> NodeState:
    prompt = refine_prompt(llm_answer, test_ok, fail_note, bench_prev, budget_ms)
    msgs: Sequence[BaseMessage] = [
        SystemMessage(content="Improve the code based on feedback. Correctness first, then speed. Return only one ```python block."),
        HumanMessage(content=prompt),
    ]
    refined = llm.invoke(msgs).content.strip()

    code = extract_python_block(refined)
    if not code:
        tests_ok2, note2 = None, "no code block found"
        bench2 = {"runtime_ms": float("inf"), "contract": "missing", "growth_ratio": None,
                  "runtime20_ms": None, "runtime30_ms": None, "bytes_used": None}
    else:
        tests_ok2, note2 = run_unit_tests(sb, code)
        bench2 = run_benchmark(sb, code) if tests_ok2 else {"runtime_ms": float("inf"), "contract": "missing",
                                                            "growth_ratio": None, "runtime20_ms": None,
                                                            "runtime30_ms": None, "bytes_used": None}

    score2 = evaluate_answer(refined, tests_ok2, bench2, budget_ms=budget_ms)
    return NodeState(llm_answer=refined, score=score2, tests_ok=tests_ok2, bench=bench2, note=note2)

# --- Top-level MCTS node that uses the agent subgraph -----------------------

def mcts_node(state: LGState) -> Command[Literal["__end__"]]:
    iters = int(state.get("iterations", 5))
    algo = tq.ABMCTSA()
    search_tree = algo.init_tree()

    trace_lines: list[str] = []
    sb = SandboxClient(dependencies=["numpy"], log_handler=_sb_log)
    sb.start()
    
    # Create console for interactive output
    console = Console()
    
    # Create live table for real-time updates
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Tests OK", justify="center")
    table.add_column("Runtime (ms)", justify="right")
    table.add_column("Contract", justify="center")
    table.add_column("Growth", justify="right")
    table.add_column("Note", justify="left", style="dim")
    
    try:
        # Build the agent subgraph once, reuse per action
        agent_graph = build_agent_subgraph(sb, console)

        def run_agents(parent: Optional[NodeState], step_idx: int) -> NodeState:
            # Run coder -> tester -> reviewer pipeline as a single "agent turn"
            ag_state: AgentState = {"parent": parent, "step_idx": step_idx}
            out = agent_graph.invoke(ag_state)["out"]
            return out

        console.print("\n[bold cyan]🚀 Starting MCTS search with agent collaboration...[/bold cyan]\n")
        console.print("[bold cyan]Agent Communication Flow:[/bold cyan]")
        console.print("  [blue]🤖 Coder[/blue] → [yellow]🧪 Tester[/yellow] → [green]📝 Reviewer[/green]")
        console.print("  " + "─" * 50 + "\n")
        
        prev_best_score = None
        
        with Live(table, console=console, refresh_per_second=4) as live:
            for i in range(iters):
                console.print(f"\n[bold yellow]━━━ Step {i+1}/{iters} ━━━[/bold yellow]")
                console.print("[dim]MCTS exploring multiple agent collaboration paths...[/dim]\n")
                
                with console.status("[bold]🔍 Exploring with MCTS (running 3 agent pipelines)...", spinner="dots"):
                    actions = {
                        # Three "flavors" of the pipeline, for now all use the same subgraph
                        "Coder→Tester→Reviewer#A": lambda parent, _i=i: (run_agents(parent, _i), (parent.score if parent else 0.0)),
                        "Coder→Tester→Reviewer#B": lambda parent, _i=i: (run_agents(parent, _i), (parent.score if parent else 0.0)),
                        "Coder→Tester→Reviewer#C": lambda parent, _i=i: (run_agents(parent, _i), (parent.score if parent else 0.0)),
                    }
                    # TreeQuest expects action -> callable returning (NodeState, score)
                    search_tree = algo.step(search_tree, actions)

                best, _ = tq.top_k(search_tree, algo, k=1)[0]
                
                # Show score evolution
                if prev_best_score is not None:
                    score_change = best.score - prev_best_score
                    if score_change != 0:
                        arrow = "📈" if score_change > 0 else "📉"
                        color = "green" if score_change > 0 else "red"
                        console.print(f"\n  {arrow} Score evolution: [bold {color}]{prev_best_score:.3f} → {best.score:.3f}[/bold {color}] ({score_change:+.3f})")
                    else:
                        console.print(f"\n  ➡️ Score unchanged: [bold yellow]{best.score:.3f}[/bold yellow]")
                prev_best_score = best.score
                
                with console.status("[bold]⚡ Evaluating best candidate...", spinner="dots"):
                    code = extract_python_block(best.llm_answer)
                    if code:
                        ok, note = run_unit_tests(sb, code)
                        bench = run_benchmark(sb, code) if ok else {"runtime_ms": float("inf"), "contract": "missing",
                                                                    "growth_ratio": None, "runtime20_ms": None,
                                                                    "runtime30_ms": None, "bytes_used": None}
                    else:
                        ok, note = None, "no code"
                        bench = {"runtime_ms": float("inf"), "contract": "missing", "growth_ratio": None,
                                 "runtime20_ms": None, "runtime30_ms": None, "bytes_used": None}

                rt = bench.get("runtime30_ms") or bench.get("runtime_ms")
                rt_val = float(rt) if rt is not None and rt != float('inf') else float('inf')
                
                # Format values for display
                score_str = f"[green]{best.score:.3f}[/green]" if best.score >= 0.9 else f"[yellow]{best.score:.3f}[/yellow]"
                tests_str = f"[green]{ok}[/green]" if ok is True else f"[red]{ok}[/red]" if ok is False else f"[dim]{ok}[/dim]"
                rt_str = f"{rt_val:.3f}" if rt_val != float('inf') else "[red]∞[/red]"
                contract = bench.get("contract", "?")
                growth = bench.get("growth_ratio")
                growth_str = f"{growth:.2f}" if growth is not None and growth != float('inf') else "N/A"
                note_display = (note or "")[:60]
                
                # Add row to table
                table.add_row(
                    str(i+1),
                    score_str,
                    tests_str,
                    rt_str,
                    contract,
                    growth_str,
                    note_display
                )
                
                # Also add to trace_lines for final summary
                trace_lines.append(
                    f"[Step {i+1}] score={best.score:.3f} "
                    f"tests_ok={ok} rt={rt_val:.3f} "
                    f"contract={contract} growth={growth} "
                    f"note={note_display}"
                )
                
                # Show agent handoffs for this step
                if best.messages:
                    console.print("\n[bold magenta]📋 Agent Communication Summary:[/bold magenta]")
                    for msg in best.messages[-3:]:  # Show last 3 messages (coder, tester, reviewer)
                        # Color code by agent type
                        if "[coder]" in msg:
                            console.print(f"  [blue]• {msg}[/blue]")
                        elif "[tester]" in msg:
                            console.print(f"  [yellow]• {msg}[/yellow]")
                        elif "[reviewer]" in msg:
                            console.print(f"  [green]• {msg}[/green]")
                        else:
                            console.print(f"  • {msg}")

        best_state, _ = tq.top_k(search_tree, algo, k=1)[0]
        trace_lines.append(f"Final Best Answer score={best_state.score:.3f}")
        
        console.print(f"\n[bold cyan]✅ Final Best Answer score={best_state.score:.3f}[/bold cyan]\n")

        return Command(
            update={
                "best_answer": best_state.llm_answer,
                "best_score": float(best_state.score),
                "trace": "\n".join(trace_lines),
                "best_messages": best_state.messages,
            },
            goto=END,
        )
    finally:
        sb.close()

# --- pretty trace ------------------------------------------------------------
console = Console()

def print_trace(trace_lines: list[str]):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Tests OK", justify="center")
    table.add_column("Runtime (ms)", justify="right")
    table.add_column("Contract", justify="center")
    table.add_column("Growth", justify="right")
    table.add_column("Note", justify="left")

    for line in trace_lines:
        if line.startswith("[Step"):
            parts = dict(
                item.split("=") for item in line.strip("[]").replace("Step ", "step=").split() if "=" in item
            )
            step = parts.get("step", "?")
            score = float(parts.get("score", 0))
            tests_ok = parts.get("tests_ok", "?")
            rt = parts.get("rt", "inf")
            contract = parts.get("contract", "?")
            growth = parts.get("growth", "?")
            note = " ".join(line.split("note=")[1:]) if "note=" in line else ""

            score_str = f"[green]{score:.3f}[/green]" if score >= 0.9 else f"[yellow]{score:.3f}[/yellow]"
            tests_str = f"[green]{tests_ok}[/green]" if tests_ok == "True" else f"[red]{tests_ok}[/red]"
            rt_str = f"{rt}" if rt != "inf" else "[red]∞[/red]"

            table.add_row(step, score_str, tests_str, rt_str, contract, str(growth), note)
        elif line.startswith("Final Best Answer"):
            console.print(f"\n[bold cyan]{line}[/bold cyan]")

    console.print(table)

# --- build & run graph -------------------------------------------------------

builder = StateGraph(LGState)
builder.add_node("mcts", mcts_node)
builder.add_edge(START, "mcts")
graph = builder.compile()

def _save_graph_png(filename: str, compiled_graph) -> None:
    try:
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
    except Exception as e:
        print(f"warning: could not render {filename}: {e}")
        return
    with open(filename, "wb") as f:
        f.write(png_bytes)
    print(f"saved {filename}")


if __name__ == "__main__":
    init_state: LGState = {"iterations": 12}
    final_state = graph.invoke(init_state)
    
    # Show final summary (table was already shown live during execution)
    console.print("\n[bold cyan]━━━ Final Results ━━━[/bold cyan]\n")
    
    # Extract and display code with syntax highlighting
    best_code = extract_python_block(final_state["best_answer"])
    if best_code:
        syntax = Syntax(best_code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(
            syntax,
            title="[bold green]🏆 Best Answer (Final Code)[/bold green]",
            border_style="green",
            expand=False
        ))
    else:
        console.print(Panel(final_state["best_answer"], title="[bold]Best answer[/bold]", border_style="green"))
    
    # Show score in a prominent panel
    score_panel = Panel(
        f"[bold green]{final_state['best_score']:.3f}[/bold green] / 1.000",
        title="[bold]Final Score[/bold]",
        border_style="green",
        expand=False
    )
    console.print(score_panel)
    
    # Show agent communication trace
    msgs = final_state.get("best_messages") or []
    if msgs:
        console.print("\n[bold magenta]📋 Complete Agent Communication Trace[/bold magenta]")
        console.print("[dim]Shows how agents collaborated via MCP sandbox:[/dim]\n")
        for i, m in enumerate(msgs, 1):
            # Color code by agent type
            if "[coder]" in m:
                console.print(f"  [blue]{i}. {m}[/blue]")
            elif "[tester]" in m:
                console.print(f"  [yellow]{i}. {m}[/yellow]")
            elif "[reviewer]" in m:
                console.print(f"  [green]{i}. {m}[/green]")
            else:
                console.print(f"  {i}. {m}")

    # --- Save diagrams right away ---
    agent_graph_for_viz = build_agent_subgraph(None, console)
    _save_graph_png("agent_subgraph.png", agent_graph_for_viz)

    # Top-level MCP + MCTS graph
    _save_graph_png("mcp_mcts_graph.png", graph)
