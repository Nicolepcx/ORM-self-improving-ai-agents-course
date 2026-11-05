"""
OpenRouter access for the lab pages, with spend guards.

Env vars (or course_app/.env, loaded like the notebooks' `load_dotenv()`):
  OPENROUTER_API_KEY       required for LLM pages; keep it server-side only
  LLM_MAX_CALLS_SESSION    max LLM calls per browser session (default 400)
  LLM_MAX_CALLS_TOTAL      max LLM calls per server process (default 20000)

The hard safety net is still the credit limit on the OpenRouter key itself.
"""

import os
import threading
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class BudgetExceeded(RuntimeError):
    pass


class _GlobalCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0


@st.cache_resource
def _global_counter() -> _GlobalCounter:
    return _GlobalCounter()


@st.cache_resource
def client() -> OpenAI:
    """One OpenAI-compatible client pointed at OpenRouter, shared by all sessions (thread-safe)."""
    # A hung provider request retries after 90 s instead of freezing the page for minutes.
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], timeout=90, max_retries=2)


def is_configured() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def reserve_calls(n: int) -> None:
    """
    Reserve `n` LLM calls before a run. Call from the page script (main thread):
    it touches session state, which worker threads must not.
    """
    per_session = int(os.getenv("LLM_MAX_CALLS_SESSION", "400"))
    total = int(os.getenv("LLM_MAX_CALLS_TOTAL", "20000"))
    used = st.session_state.get("llm_calls", 0)
    if used + n > per_session:
        raise BudgetExceeded("You've used this session's share of model calls. Thanks for experimenting!")
    counter = _global_counter()
    with counter.lock:
        if counter.calls + n > total:
            raise BudgetExceeded("The lab has reached its model-call limit for today.")
        counter.calls += n
    st.session_state.llm_calls = used + n


def release_calls(n: int) -> None:
    """Give back reserved calls a run didn't use (e.g. the game ended early)."""
    if n <= 0:
        return
    st.session_state.llm_calls = max(0, st.session_state.get("llm_calls", 0) - n)
    counter = _global_counter()
    with counter.lock:
        counter.calls = max(0, counter.calls - n)
