"""
Shared trajectory viewer: renders a trajectory as chat bubbles, tool-call chips,
a reward badge and (optionally) additional histories branching off the message
that spawned them. Used by the session 3, 4 and 5 pages.

Works with `labs.trajectory.Trajectory`.
"""

import html
import json
import re

import streamlit as st

_CSS = """
<style>
.tv { font-size: 0.92rem; line-height: 1.5; }
.tv-lane { border: 1px solid rgba(127,127,127,0.28); border-radius: 12px; padding: 0.8rem 0.9rem; }
.tv-lane-head { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin-bottom: 0.6rem; }
.tv-lane-title { font-weight: 600; margin-right: 0.3rem; }
.tv-pill { font-size: 0.72rem; padding: 0.1rem 0.5rem; border-radius: 999px; background: rgba(127,127,127,0.14); white-space: nowrap; }
.tv-reward { background: rgba(31,163,122,0.18); font-weight: 600; }
.tv-msg { border-radius: 10px; padding: 0.5rem 0.7rem; margin: 0.45rem 0; border: 1px solid rgba(127,127,127,0.18); }
.tv-msg.system { background: rgba(127,127,127,0.08); }
.tv-msg.user { background: rgba(47,125,225,0.08); }
.tv-msg.tool { background: rgba(230,160,40,0.10); }
.tv-msg.assistant { background: rgba(124,92,214,0.08); }
.tv-msg.trained { border-left: 4px solid #7c5cd6; }
.tv-meta { display: flex; gap: 0.4rem; align-items: center; font-size: 0.72rem; opacity: 0.8; margin-bottom: 0.2rem; text-transform: uppercase; letter-spacing: 0.04em; }
.tv-tag-trained { color: #7c5cd6; font-weight: 700; }
.tv-body { white-space: pre-wrap; word-wrap: break-word; }
.tv-call { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; background: rgba(127,127,127,0.12); padding: 0.15rem 0.45rem; border-radius: 6px; display: inline-block; margin: 0.1rem 0; }
.tv-think { background: rgba(230,160,40,0.22); border-radius: 4px; padding: 0 0.2rem; font-style: italic; }
.tv-branch { margin: 0.2rem 0 0.6rem 1.4rem; padding-left: 0.9rem; border-left: 2px dashed #7c5cd6; }
.tv-branch .tv-lane { border-color: rgba(124,92,214,0.45); background: rgba(124,92,214,0.03); }
.tv-note { font-size: 0.82rem; opacity: 0.8; margin-bottom: 0.4rem; }
.tv-dim { opacity: 0.35; filter: grayscale(0.8); }
.tv-off { background: rgba(214,69,69,0.16); color: inherit; }
</style>
"""

_THINK = re.compile(r"&lt;think&gt;(.*?)&lt;/think&gt;", re.S)


def _body(text):
    if not text:
        return ""
    escaped = html.escape(text)
    escaped = _THINK.sub(r'<span class="tv-think">&lt;think&gt;\1&lt;/think&gt;</span>', escaped)
    return f'<div class="tv-body">{escaped}</div>'


def _tool_call(call, branch):
    try:
        args = ", ".join(f"{k}={v!r}" for k, v in json.loads(call["arguments"]).items())
    except (ValueError, AttributeError):
        args = call["arguments"]
    spawn = f' <span class="tv-pill">→ spawns history {branch + 1}</span>' if branch is not None else ""
    return f'<div><span class="tv-call">🔧 {html.escape(call["name"])}({html.escape(args)})</span>{spawn}</div>'


def _message(m, active=True):
    generated = getattr(m, "is_choice", False)
    trained = generated and active
    role = m.role
    label = {"assistant": "assistant · generated", "tool": "tool result"}.get(role, role)
    tag = ('<span class="tv-tag-trained">● trained</span>' if trained
           else "<span>not trained</span>" if generated else "<span>context</span>")
    calls = "".join(_tool_call(c, m.branch) for c in m.tool_calls)
    ref = f'<div class="tv-note">↩ answer to {html.escape(m.tool_call_id)}</div>' if m.tool_call_id else ""
    return (f'<div class="tv-msg {role}{" trained" if trained else ""}">'
            f'<div class="tv-meta"><span>{label}</span>·{tag}</div>{ref}{calls}{_body(m.content)}</div>')


def _lane(title, messages, pills="", note="", histories=None, histories_trained=True, active=True):
    parts = [f'<div class="tv-lane"><div class="tv-lane-head"><span class="tv-lane-title">{title}</span>{pills}</div>']
    if note:
        parts.append(f'<div class="tv-note">{html.escape(note)}</div>')
    for m in messages:
        parts.append(_message(m, active))
        if histories is not None and m.branch is not None and m.branch < len(histories):
            parts.append(_history(m.branch, histories[m.branch], histories_trained))
    parts.append("</div>")
    return "".join(parts)


def _history(index, history, trained):
    status = ('<span class="tv-pill tv-reward">trained with the same reward</span>' if trained
              else '<span class="tv-pill tv-off">not trained</span>')
    inner = _lane(f"Additional history {index + 1} · {html.escape(history.label)}",
                  history.messages, pills=status, note=history.note, active=trained)
    return f'<div class="tv-branch{"" if trained else " tv-dim"}">{inner}</div>'


def render_trajectory(trajectory, histories_trained: bool = True, title: str = "Main conversation") -> None:
    """Render one trajectory; additional histories hang off the message that spawned them."""
    metrics = "".join(f'<span class="tv-pill">{html.escape(str(k))}: {html.escape(str(v))}</span>'
                      for k, v in trajectory.metrics.items())
    pills = f'<span class="tv-pill tv-reward">reward {trajectory.reward}</span>{metrics}'
    histories = list(getattr(trajectory, "additional_histories", []))
    body = _lane(title, trajectory.messages, pills=pills, histories=histories,
                 histories_trained=histories_trained)
    linked = {m.branch for m in trajectory.messages if m.branch is not None}
    body += "".join(_history(i, h, histories_trained) for i, h in enumerate(histories) if i not in linked)
    st.html(_CSS + f'<div class="tv">{body}</div>')
