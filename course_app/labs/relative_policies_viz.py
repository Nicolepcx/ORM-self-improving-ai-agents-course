"""Interactive (Plotly) versions of the demo notebook's helper_functions figures."""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots as _make_subplots


def make_subplots(*args, **kwargs):
    return _make_subplots(*args, **kwargs)

from labs.relative_policies import ACTIONS

ABS = "#2f7de1"   # notebook: blue
REL = "#f28e2b"   # notebook: orange
BASE = "#d64545"  # notebook: red dashed
POS, NEG = "#1fa37a", "#d64545"


def _style(fig: go.Figure) -> go.Figure:
    """Presentation-size fonts: these charts are shown on a projector in the demo."""
    fig.update_layout(font=dict(size=15))
    fig.update_annotations(font_size=17)
    fig.update_traces(textfont_size=14, selector=dict(type="bar"))
    return fig


def _drift(fig, drift_step: Optional[int], rows_cols):
    if drift_step is None:
        return
    for row, col in rows_cols:
        fig.add_vline(x=drift_step, line=dict(color="gray", dash="dot", width=2), row=row, col=col)


def training_summary(abs_stats, rel_stats, abs_pol, rel_pol, drift_step: Optional[int],
                     previous: Optional[Dict] = None) -> go.Figure:
    """plot_training_summary: reward, true correctness, advantage stats, final policy.
    `previous` (stats of the last run) is drawn dotted in the correctness panel for comparison."""
    steps = [s["step"] for s in abs_stats]
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.08, subplot_titles=(
        "Average reward (reward model)", "True correctness (not used in training)",
        "Advantages (the learning signal)", "Final policy"))
    fig.add_trace(go.Scatter(x=steps, y=[s["avg_reward"] for s in abs_stats], name="Absolute baseline",
                             line=dict(color=ABS, width=2), legendgroup="abs"), 1, 1)
    fig.add_trace(go.Scatter(x=steps, y=[s["avg_reward"] for s in rel_stats], name="Group relative",
                             line=dict(color=REL, width=2), legendgroup="rel"), 1, 1)
    fig.add_trace(go.Scatter(x=steps, y=[s["baseline"] for s in abs_stats], name="Baseline (absolute)",
                             line=dict(color=BASE, dash="dash", width=1.5)), 1, 1)
    fig.add_trace(go.Scatter(x=steps, y=[s["correctness_rate"] for s in abs_stats], showlegend=False,
                             line=dict(color=ABS, width=2), legendgroup="abs"), 1, 2)
    fig.add_trace(go.Scatter(x=steps, y=[s["correctness_rate"] for s in rel_stats], showlegend=False,
                             line=dict(color=REL, width=2), legendgroup="rel"), 1, 2)
    if previous:
        for key, color in (("abs_stats", ABS), ("rel_stats", REL)):
            prev = previous[key]
            fig.add_trace(go.Scatter(x=[s["step"] for s in prev], y=[s["correctness_rate"] for s in prev],
                                     line=dict(color=color, dash="dot", width=1.5), opacity=0.6,
                                     name="previous run (dotted)", showlegend=key == "abs_stats"), 1, 2)
    fig.add_trace(go.Scatter(x=steps, y=[s["mean_adv"] for s in abs_stats], name="mean advantage",
                             line=dict(color=ABS, width=2), legendgroup="abs", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=steps, y=[s["mean_adv"] for s in rel_stats],
                             line=dict(color=REL, width=2), legendgroup="rel", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=steps, y=[s["std_adv"] for s in abs_stats], name="std advantage (dashed)",
                             line=dict(color=ABS, dash="dash", width=1.5)), 2, 1)
    fig.add_trace(go.Scatter(x=steps, y=[s["std_adv"] for s in rel_stats], showlegend=False,
                             line=dict(color=REL, dash="dash", width=1.5)), 2, 1)
    fig.add_trace(go.Bar(x=ACTIONS, y=[abs_pol[-1][a] for a in ACTIONS], marker_color=ABS, showlegend=False), 2, 2)
    fig.add_trace(go.Bar(x=ACTIONS, y=[rel_pol[-1][a] for a in ACTIONS], marker_color=REL, showlegend=False), 2, 2)
    _drift(fig, drift_step, [(1, 1), (1, 2), (2, 1)])
    fig.update_yaxes(range=[-1.1, 1.1], row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%", row=1, col=2)
    fig.update_yaxes(range=[0, 1.02], row=2, col=2)
    fig.update_xaxes(title_text="training step", row=1, col=1)
    fig.update_xaxes(title_text="training step", row=1, col=2)
    fig.update_xaxes(title_text="training step", row=2, col=1)
    fig.update_layout(height=640, barmode="group", margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.08))
    return _style(fig)


def policy_at_step(abs_p: Dict[str, float], rel_p: Dict[str, float], step: int) -> go.Figure:
    """Top row of animate_training_comparison: both policies and their difference."""
    diff = [rel_p[a] - abs_p[a] for a in ACTIONS]
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        f"Absolute · step {step}", f"Relative · step {step}", "Relative − absolute"))
    fig.add_trace(go.Bar(x=ACTIONS, y=[abs_p[a] for a in ACTIONS], marker_color=ABS), 1, 1)
    fig.add_trace(go.Bar(x=ACTIONS, y=[rel_p[a] for a in ACTIONS], marker_color=REL), 1, 2)
    fig.add_trace(go.Bar(x=ACTIONS, y=diff, marker_color=[POS if d >= 0 else NEG for d in diff]), 1, 3)
    fig.update_yaxes(range=[0, 1.02], row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    fig.update_yaxes(range=[-1.02, 1.02], row=1, col=3)
    fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return _style(fig)


def step_details(step_idx: int, abs_rollouts, abs_stats, rel_rollouts, rel_advantages: List[float]) -> go.Figure:
    """visualize_step_details: sorted rewards, reward→advantage, advantage histograms, mean advantage by action."""
    abs_ro, rel_ro = abs_rollouts[step_idx], rel_rollouts[step_idx]
    baseline = abs_stats[step_idx]["baseline"]
    abs_r = [r.reward for r in abs_ro]
    rel_r = [r.reward for r in rel_ro]
    abs_a = [r - baseline for r in abs_r]
    rel_a = rel_advantages

    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.18, subplot_titles=(
        f"Abs rewards · baseline {baseline:.2f}", f"Rel rewards · mean {np.mean(rel_r):.2f}",
        "Reward → advantage",
        f"Abs adv · μ {np.mean(abs_a):.2f} σ {np.std(abs_a):.2f}",
        f"Rel adv · μ {np.mean(rel_a):.2f} σ {np.std(rel_a):.2f}",
        "Mean adv by action"))
    fig.add_trace(go.Scatter(y=sorted(abs_r), mode="markers", marker=dict(color=ABS, size=5, opacity=0.6)), 1, 1)
    fig.add_hline(y=baseline, line=dict(color=BASE, dash="dash", width=2), row=1, col=1)
    fig.add_trace(go.Scatter(y=sorted(rel_r), mode="markers", marker=dict(color=REL, size=5, opacity=0.6)), 1, 2)
    fig.add_hline(y=float(np.mean(rel_r)), line=dict(color=BASE, dash="dash", width=2), row=1, col=2)
    fig.add_trace(go.Scatter(x=abs_r, y=abs_a, mode="markers", name="absolute",
                             marker=dict(color=ABS, size=5, opacity=0.5)), 1, 3)
    fig.add_trace(go.Scatter(x=rel_r, y=rel_a, mode="markers", name="relative",
                             marker=dict(color=REL, size=5, opacity=0.5)), 1, 3)
    fig.add_trace(go.Histogram(x=abs_a, nbinsx=20, marker_color=ABS, opacity=0.8), 2, 1)
    fig.add_trace(go.Histogram(x=rel_a, nbinsx=20, marker_color=REL, opacity=0.8), 2, 2)
    for col in (1, 2):
        fig.add_vline(x=0, line=dict(color=BASE, dash="dash", width=2), row=2, col=col)

    by_abs, by_rel = defaultdict(list), defaultdict(list)
    for ro, a in zip(abs_ro, abs_a):
        by_abs[ro.action].append(a)
    for ro, a in zip(rel_ro, rel_a):
        by_rel[ro.action].append(a)
    order = sorted(ACTIONS, key=lambda a: np.mean(by_abs.get(a, [0])))
    fig.add_trace(go.Bar(y=order, x=[np.mean(by_abs.get(a, [0])) for a in order], orientation="h",
                         marker_color=ABS, name="absolute"), 2, 3)
    fig.add_trace(go.Bar(y=order, x=[np.mean(by_rel.get(a, [0])) for a in order], orientation="h",
                         marker_color=REL, name="relative"), 2, 3)
    fig.update_xaxes(title_text="rollout (sorted)", row=1, col=1)
    fig.update_xaxes(title_text="rollout (sorted)", row=1, col=2)
    fig.update_xaxes(title_text="observed reward", row=1, col=3)
    fig.update_yaxes(title_text="advantage", row=1, col=3)
    fig.update_xaxes(range=[-3.5, 3.5], row=2, col=2)
    fig.update_layout(height=620, showlegend=False, barmode="group", margin=dict(l=10, r=10, t=50, b=10))
    return _style(fig)


def noise_map_figure(prompts, before: Dict, after: Optional[Dict]) -> go.Figure:
    """Per-prompt reward scale and bias: the miscalibrated reward model, before and after drift."""
    labels = [p[1] for p in prompts]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Scale (multiplies the reward)", "Bias (added to the reward)"))
    for i, (noise, name, color) in enumerate([(before, "before drift", "#7c5cd6"), (after, "after drift", "#e4572e")]):
        if not noise:
            continue
        fig.add_trace(go.Bar(y=labels, x=[noise[p[0]][0] for p in prompts], orientation="h", name=name,
                             marker_color=color, showlegend=True), 1, 1)
        fig.add_trace(go.Bar(y=labels, x=[noise[p[0]][1] for p in prompts], orientation="h", name=name,
                             marker_color=color, showlegend=False), 1, 2)
    fig.add_vline(x=1, line=dict(color="gray", dash="dot"), row=1, col=1)
    fig.add_vline(x=0, line=dict(color="gray", dash="dot"), row=1, col=2)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=520, barmode="group", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.06))
    return _style(fig)


# ------------------------------------------------------------------ explainers
def policy_before_after(before: Dict[str, float], after: Dict[str, float], title: str) -> go.Figure:
    """One policy update: probabilities before (ghost) and after (solid)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ACTIONS, y=[before[a] for a in ACTIONS], name="before", marker_color="rgba(127,127,127,0.35)"))
    fig.add_trace(go.Bar(x=ACTIONS, y=[after[a] for a in ACTIONS], name="after", marker_color="#7c5cd6"))
    fig.update_layout(title=dict(text=title, font_size=13), height=290, barmode="group", yaxis=dict(range=[0, 1.02]),
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    return _style(fig)


def signal_bars(labels: List[str], values: List[float], title: str, y_title: str, y_range=None,
                zero_line: Optional[float] = None, colors: Optional[List[str]] = None,
                text_format: str = "{:+.2f}") -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=values,
                           marker_color=colors or [POS if v > 0 else NEG for v in values],
                           text=[text_format.format(v) for v in values], textposition="outside", cliponaxis=False))
    if zero_line is not None:
        fig.add_hline(y=zero_line, line=dict(color="gray", dash="dash"))
    fig.update_layout(title=dict(text=title, font_size=13), height=300, yaxis_title=y_title,
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    if y_range:
        fig.update_yaxes(range=y_range)
    return _style(fig)


def compare_advantages(labels: List[str], absolute: List[float], relative: List[float],
                       abs_ref: List[float], rel_ref: List[float]) -> go.Figure:
    """Absolute vs relative advantages for one group, against the well-calibrated reference (outline)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Absolute: reward − baseline", "Relative (GRPO): group mean & std"))
    for col, now, ref, color in [(1, absolute, abs_ref, ABS), (2, relative, rel_ref, REL)]:
        fig.add_trace(go.Bar(x=labels, y=ref, marker=dict(color="rgba(0,0,0,0)", line=dict(color="gray", width=1.5)),
                             name="calibrated reward model", showlegend=col == 1), 1, col)
        fig.add_trace(go.Bar(x=labels, y=now, marker_color=color, opacity=0.85, name="this reward model",
                             showlegend=col == 1), 1, col)
    fig.update_layout(height=340, barmode="overlay", margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(range=[-3, 3])
    return _style(fig)


def training_animation(r: Dict, drift_step: Optional[int], every: int = 1) -> go.Figure:
    """animate_training_comparison as a client-side Plotly animation (press ▶)."""
    abs_s, rel_s, steps = r["abs_stats"], r["rel_stats"], len(r["abs_stats"])
    from labs.relative_policies import group_relative_advantages

    def frame(i: int):
        abs_p, rel_p = r["abs_pol"][i], r["rel_pol"][i]
        diff = [rel_p[a] - abs_p[a] for a in ACTIONS]
        abs_adv = [ro.reward - abs_s[i]["baseline"] for ro in r["abs_rollouts"][i]]
        rel_adv = group_relative_advantages(r["rel_rollouts"][i])
        x = list(range(1, i + 2))
        return [
            go.Bar(x=ACTIONS, y=[abs_p[a] for a in ACTIONS], marker_color=ABS),
            go.Bar(x=ACTIONS, y=[rel_p[a] for a in ACTIONS], marker_color=REL),
            go.Bar(x=ACTIONS, y=diff, marker_color=[POS if d >= 0 else NEG for d in diff]),
            go.Scatter(x=x, y=[s["avg_reward"] for s in abs_s[: i + 1]], line=dict(color=ABS, width=2)),
            go.Scatter(x=x, y=[s["baseline"] for s in abs_s[: i + 1]], line=dict(color=BASE, dash="dash")),
            go.Scatter(x=x, y=[s["avg_reward"] for s in rel_s[: i + 1]], line=dict(color=REL, width=2)),
            go.Histogram(x=abs_adv, xbins=dict(start=-3, end=3, size=0.25), marker_color=ABS, opacity=0.55),
            go.Histogram(x=rel_adv, xbins=dict(start=-3, end=3, size=0.25), marker_color=REL, opacity=0.55),
        ]

    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.2, subplot_titles=(
        "Absolute policy", "Relative policy", "Relative − absolute",
        "Absolute: reward vs baseline", "Relative: average reward", "Advantages this step"))
    cells = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 1), (2, 2), (2, 3), (2, 3)]
    for trace, (row, col) in zip(frame(0), cells):
        fig.add_trace(trace, row, col)
    fig.frames = [go.Frame(data=frame(i), name=str(i + 1), traces=list(range(8)))
                  for i in range(0, steps, every)] + ([go.Frame(data=frame(steps - 1), name=str(steps), traces=list(range(8)))]
                                                       if (steps - 1) % every else [])
    if drift_step is not None:
        for col in (1, 2):
            fig.add_vline(x=drift_step, line=dict(color="gray", dash="dot", width=2), row=2, col=col)
    for col in (1, 2):
        fig.update_yaxes(range=[0, 1.02], row=1, col=col)
        fig.update_xaxes(range=[0, steps + 1], row=2, col=col)
        fig.update_yaxes(range=[-1.05, 1.05], row=2, col=col)
    fig.update_yaxes(range=[-1.02, 1.02], row=1, col=3)
    fig.update_xaxes(range=[-3, 3], row=2, col=3)
    fig.update_layout(
        height=620, showlegend=False, barmode="overlay", margin=dict(l=10, r=10, t=50, b=80),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.08, xanchor="left", yanchor="top", buttons=[
            dict(label="▶ Play training", method="animate",
                 args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
            dict(label="❚❚", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]),
        ])],
        sliders=[dict(x=0.2, y=-0.08, len=0.8, xanchor="left", yanchor="top", currentvalue=dict(prefix="step "),
                      steps=[dict(method="animate", label=f.name,
                                  args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for f in fig.frames])],
    )
    return _style(fig)


def rl_loop_svg() -> str:
    """The RL loop as a cycle diagram (render with st.image)."""
    boxes = [
        ("Policy", ("probabilities", "over actions"), "#7c5cd6"),
        ("Rollouts", ("sample a group", "of answers"), "#2f7de1"),
        ("Reward", ("the reward model", "scores each one"), "#e0a030"),
        ("Advantage", ("better or worse", "than expected?"), "#1fa37a"),
        ("Update", ("logits +=", "lr × advantage"), "#d64545"),
    ]
    w, h, gap, y = 176, 120, 30, 20
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1010 230" width="1010" height="230">',
             '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
             'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#8a8a8a"/></marker></defs>']
    for k, (title, sub, color) in enumerate(boxes):
        x = 8 + k * (w + gap)
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" fill-opacity="0.14" '
                     f'stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + w / 2}" y="{y + 42}" text-anchor="middle" font-family="sans-serif" '
                     f'font-size="23" font-weight="700" fill="{color}">{k + 1} · {title}</text>')
        for line, text in enumerate(sub):
            parts.append(f'<text x="{x + w / 2}" y="{y + 76 + 24 * line}" text-anchor="middle" '
                         f'font-family="sans-serif" font-size="20" fill="#8a8a8a">{text}</text>')
        if k < len(boxes) - 1:
            parts.append(f'<line x1="{x + w + 2}" y1="{y + h / 2}" x2="{x + w + gap - 3}" y2="{y + h / 2}" '
                         'stroke="#8a8a8a" stroke-width="3" marker-end="url(#a)"/>')
    last_x, first_x = 8 + 4 * (w + gap) + w / 2, 8 + w / 2
    parts.append(f'<path d="M{last_x},{y + h + 3} V{y + h + 45} H{first_x} V{y + h + 10}" fill="none" '
                 'stroke="#8a8a8a" stroke-width="3" stroke-dasharray="8 6" marker-end="url(#a)"/>')
    parts.append(f'<text x="505" y="{y + h + 76}" text-anchor="middle" font-family="sans-serif" font-size="20" '
                 'fill="#8a8a8a">repeat: no correct answers are ever given, only scores</text>')
    parts.append("</svg>")
    return "".join(parts)
