"""
Python code editor for notebook TODOs, plus safe parsing of what attendees paste.

Attendees write what they would write in the notebook. Only *literals* are
evaluated (`ast.literal_eval`), never executed, so nothing they type runs on
the server.
"""

import ast
import re
from dataclasses import dataclass
from typing import Any, Optional

import streamlit as st
from code_editor import code_editor


@dataclass
class Parsed:
    value: Any = None
    error: Optional[str] = None
    read_as: str = ""


def editor(initial: str, key: str, ghost_text: str = "", lines: tuple[int, int] = (6, 20)) -> str:
    """A Python editor that reports edits while typing/pasting. Returns the current text."""
    response = code_editor(
        initial, lang="python", key=key, height=list(lines), ghost_text=ghost_text,
        response_mode=["debounce", "blur"], allow_reset=True,
        options={"wrap": True, "showLineNumbers": True, "tabSize": 4},
    )
    # Before the first edit the component returns an empty response
    return response["text"] if response.get("id") else initial


def _syntax_error(e: SyntaxError, line_offset: int = 0) -> str:
    line = (e.lineno or 1) - line_offset
    # The message can mention other lines ("... on line 3"); shift those too
    msg = re.sub(r"line (\d+)", lambda m: f"line {max(int(m.group(1)) - line_offset, 1)}", e.msg)
    return f"SyntaxError on line {max(line, 1)}: {msg}"


def parse_kwarg_value(text: str, name: str, expected: type) -> Parsed:
    """
    Accept `{...}`, `name={...}`, `name = {...}` and a trailing comma, i.e. exactly
    what the value looks like inside `art.Trajectory(...)` in the notebook.
    """
    text = text.strip()
    if not text:
        return Parsed(value=expected(), read_as="empty")

    # 1. A bare literal: {...}
    try:
        node = ast.parse(text, mode="eval").body
        return _literal(node, expected, read_as="literal")
    except SyntaxError:
        pass

    # 2. As a keyword argument: name={...},  (wrap it in a call so the trailing comma is valid)
    try:
        call = ast.parse(f"_(\n{text}\n)", mode="eval").body
    except SyntaxError as e:
        return Parsed(error=_syntax_error(e, line_offset=1))
    for kw in call.keywords:
        if kw.arg == name:
            return _literal(kw.value, expected, read_as=f"{name}=…")
    if len(call.args) == 1 and not call.keywords:
        return _literal(call.args[0], expected, read_as="literal")
    found = ", ".join(kw.arg for kw in call.keywords if kw.arg) or "nothing usable"
    return Parsed(error=f"Expected `{name}={{...}}` or just `{{...}}`, found {found}.")


def parse_message_content(text: str) -> Parsed:
    """
    The system message content: plain text, a Python string literal (also the
    notebook's parenthesised `( "..." "..." )` form), or `"content": (...)`.
    """
    stripped = text.strip()
    if not stripped or stripped == "()":
        return Parsed(value="", read_as="empty")
    try:
        node = ast.parse(stripped, mode="eval").body
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return Parsed(value=node.value, read_as="Python string")
    except SyntaxError:
        pass
    try:
        node = ast.parse("{" + stripped.rstrip(",") + "}", mode="eval").body
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "content":
                    return _literal(v, str, read_as='"content": …')
    except SyntaxError:
        pass
    return Parsed(value=stripped, read_as="plain text")


def _literal(node: ast.AST, expected: type, read_as: str) -> Parsed:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return Parsed(error="Only literal values are allowed here (strings, numbers, True/False, dicts, lists).")
    if not isinstance(value, expected):
        return Parsed(error=f"Expected a {expected.__name__}, got a {type(value).__name__}.")
    return Parsed(value=value, read_as=read_as)
