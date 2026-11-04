"""Shared building blocks so every session page looks and reads the same."""

import inspect
import textwrap
from typing import Callable, Iterable

import streamlit as st

REPO = "Nicolepcx/ORM-self-improving-ai-agents-course"


def colab_url(notebook_path: str) -> str:
    return f"https://colab.research.google.com/github/{REPO}/blob/main/{notebook_path}"


def page_header(session: int, title: str, subtitle: str, minutes: int, notebook_path: str,
                kind: str = "HANDS-ON") -> None:
    st.caption(f"SESSION {session} · {kind} · {minutes} MIN")
    st.title(title)
    st.markdown(subtitle)
    st.link_button("Open the full notebook in Colab", colab_url(notebook_path), icon=":material/open_in_new:")


def task(text: str) -> None:
    """The one thing attendees should do on this page."""
    st.markdown(
        f"""<div class="lab-task"><div class="lab-task-label">Your task</div>{text}</div>""",
        unsafe_allow_html=True,
    )


def code_panel(title: str, objs: Callable | Iterable[Callable] | str, why: str = "") -> None:
    """
    Collapsible "the code behind this" panel.

    Pass functions/classes rather than strings where possible: the panel then
    shows their real source, so what attendees read is exactly what runs.
    """
    with st.expander(f":material/code: {title}"):
        if why:
            st.markdown(why)
        if isinstance(objs, str):
            source = textwrap.dedent(objs).strip()
        else:
            objs = [objs] if callable(objs) else list(objs)
            source = "\n\n".join(textwrap.dedent(inspect.getsource(o)).strip() for o in objs)
        st.code(source, language="python")


def takeaway(text: str) -> None:
    st.markdown(
        f"""<div class="lab-takeaway"><div class="lab-task-label">Takeaway</div>{text}</div>""",
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1150px; padding-top: 2.5rem; }
        .lab-task, .lab-takeaway {
            border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.8rem 0 1.2rem;
            border: 1px solid rgba(127,127,127,0.25);
        }
        .lab-task { border-left: 5px solid #2f7de1; background: rgba(47,125,225,0.07); }
        .lab-takeaway { border-left: 5px solid #1fa37a; background: rgba(31,163,122,0.07); }
        .lab-task-label {
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; opacity: 0.7; margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
