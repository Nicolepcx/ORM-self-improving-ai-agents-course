"""Plotly figures for the Frozen Lake page."""

from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from labs.frozen_lake import ACTIONS, FrozenLakeGame

TILE_STYLE = {
    "S": {"fill": "#cfe6fb", "label": "START"},
    "F": {"fill": "#e9f4fd", "label": ""},
    "H": {"fill": "#23415f", "label": "HOLE"},
    "G": {"fill": "#f6c445", "label": "GOAL"},
}
ARROW_DELTA = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
ARROW_GLYPH = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
AGENT_COLOR = "#e4572e"
PATH_COLOR = "#2f7de1"
POS = "#1fa37a"
NEG = "#d64545"


def _xy(cell: Tuple[int, int]) -> Tuple[float, float]:
    """Grid (row, col) -> plot (x, y) with row 0 at the top."""
    i, j = cell
    return j, -i


def _base_grid(grid: List[List[str]]) -> go.Figure:
    fig = go.Figure()
    for i, row in enumerate(grid):
        for j, tile in enumerate(row):
            x, y = _xy((i, j))
            style = TILE_STYLE[tile]
            fig.add_shape(
                type="rect", x0=x - 0.48, x1=x + 0.48, y0=y - 0.48, y1=y + 0.48,
                fillcolor=style["fill"], line=dict(width=0), layer="below",
            )
            if style["label"]:
                fig.add_annotation(
                    x=x, y=y - 0.36, text=style["label"], showarrow=False,
                    font=dict(size=10, color="#ffffff" if tile == "H" else "#23415f"),
                )
    rows, cols = len(grid), len(grid[0])
    fig.update_xaxes(visible=False, range=[-0.55, cols - 0.45], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-rows + 0.45, 0.55], fixedrange=True,
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=420, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def grid_figure(grid: List[List[str]]) -> go.Figure:
    return _base_grid(grid)


def policy_figure(grid: List[List[str]], policy: Dict[Tuple[int, int], Dict[str, float]]) -> go.Figure:
    """Four arrows per tile; arrow length and thickness = probability of that action."""
    fig = _base_grid(grid)
    hover_x, hover_y, hover_text = [], [], []
    for cell, probs in policy.items():
        x, y = _xy(cell)
        best = max(probs, key=probs.get)
        for a in ACTIONS:
            p = probs[a]
            if p < 0.04:
                continue
            dx, dy = ARROW_DELTA[a]
            length = 0.12 + 0.3 * p
            fig.add_annotation(
                x=x + dx * length, y=y + dy * length, ax=x, ay=y,
                xref="x", yref="y", axref="x", ayref="y", text="",
                showarrow=True, arrowhead=2, arrowsize=0.8,
                arrowwidth=1 + 5 * p,
                arrowcolor=AGENT_COLOR if a == best else "rgba(35,65,95,0.45)",
            )
        hover_x.append(x)
        hover_y.append(y)
        hover_text.append("<br>".join(f"{ARROW_GLYPH[a]} {a:<5} {probs[a]:.0%}" for a in ACTIONS))
    fig.add_trace(go.Scatter(
        x=hover_x, y=hover_y, mode="markers", marker=dict(size=40, opacity=0),
        hovertext=hover_text, hoverinfo="text",
    ))
    return fig


def rollout_figure(game: FrozenLakeGame) -> go.Figure:
    """Animated replay of one episode: the agent moves, its path is drawn behind it."""
    fig = _base_grid(game["grid"])
    points = [_xy(c) for c in game["trajectory"]]
    xs, ys = zip(*points)

    def frame_data(k: int):
        slipped = k > 0 and game["slipped"][k - 1]
        return [
            go.Scatter(x=xs[: k + 1], y=ys[: k + 1], mode="lines",
                       line=dict(color=PATH_COLOR, width=4), opacity=0.55),
            go.Scatter(x=[xs[k]], y=[ys[k]], mode="markers+text",
                       marker=dict(size=34, color=AGENT_COLOR, line=dict(width=3, color="white")),
                       text=["slip!" if slipped else ""], textposition="top center",
                       textfont=dict(color=AGENT_COLOR, size=13)),
        ]

    fig.add_traces(frame_data(0))
    fig.frames = [go.Frame(data=frame_data(k), name=str(k)) for k in range(len(points))]
    fig.update_layout(
        height=540,
        margin=dict(l=0, r=0, t=10, b=70),
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=-0.02, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=350, redraw=True), fromcurrent=True,
                                      transition=dict(duration=150))]),
                dict(label="❚❚", method="animate",
                     args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]),
            ],
        )],
        sliders=[dict(
            x=0.18, y=-0.02, len=0.82, xanchor="left", yanchor="top",
            currentvalue=dict(prefix="step ", font=dict(size=12)),
            steps=[dict(method="animate", label=str(k),
                        args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                   for k in range(len(points))],
        )],
    )
    return fig


def learning_curves(history: List[Dict], runs: Optional[List[Dict]] = None) -> go.Figure:
    """Success rate (left) and reward vs baseline (right) per training iteration."""
    it = [h["iteration"] for h in history]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Success rate", "Reward & baseline"))
    for run in runs or []:
        fig.add_trace(go.Scatter(x=run["iterations"], y=run["success"], mode="lines",
                                 name=run["label"], line=dict(width=1.5, dash="dot"),
                                 opacity=0.6), row=1, col=1)
    fig.add_trace(go.Scatter(x=it, y=[h["success_rate"] for h in history], mode="lines",
                             name="current run", line=dict(color=PATH_COLOR, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=it, y=[h["avg_reward"] for h in history], mode="lines",
                             name="avg reward", line=dict(color=AGENT_COLOR, width=2.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=it, y=[h["baseline"] for h in history], mode="lines",
                             name="baseline", line=dict(color="#888", width=2, dash="dash")), row=1, col=2)
    fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%", row=1, col=1)
    fig.update_xaxes(title_text="training iteration")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def reward_advantage_figure(rewards: List[float], advantages: List[float]) -> go.Figure:
    """One bar per rollout in the batch: what it scored and how much it pushed the policy."""
    idx = [f"#{k + 1}" for k in range(len(rewards))]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Reward per rollout", "Advantage = reward − baseline"))
    fig.add_trace(go.Bar(x=idx, y=rewards, marker_color=[POS if r > 0 else NEG for r in rewards]), row=1, col=1)
    fig.add_trace(go.Bar(x=idx, y=advantages, marker_color=[POS if a > 0 else NEG for a in advantages]), row=1, col=2)
    fig.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig
