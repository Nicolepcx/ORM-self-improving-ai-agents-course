"""Figures for the Rock Paper Scissors Lizard Spock page."""

import math
from typing import Dict, List

import plotly.graph_objects as go

from labs.rpls import BEATS, MOVES

ICON = {"rock": "🪨", "paper": "📄", "scissors": "✂️", "lizard": "🦎", "spock": "🖖", "nothing": "∅"}
PLAYER_COLORS = ["#7c5cd6", "#e4572e"]

# Around the pentagon each move beats the next one (scissors → paper → rock → lizard → spock → scissors).
_RING = ["scissors", "paper", "rock", "lizard", "spock"]


def rules_svg() -> str:
    """Pentagon with an arrow from every move to each move it beats (render with st.image)."""
    cx, cy, radius, node = 170, 165, 120, 30
    pos = {
        m: (cx + radius * math.sin(2 * math.pi * k / 5), cy - radius * math.cos(2 * math.pi * k / 5))
        for k, m in enumerate(_RING)
    }
    arrows = []
    for winner, losers in BEATS.items():
        for loser in losers:
            (x1, y1), (x2, y2) = pos[winner], pos[loser]
            d = math.hypot(x2 - x1, y2 - y1)
            ux, uy = (x2 - x1) / d, (y2 - y1) / d
            arrows.append(
                f'<line x1="{x1 + ux * node:.1f}" y1="{y1 + uy * node:.1f}" '
                f'x2="{x2 - ux * (node + 4):.1f}" y2="{y2 - uy * (node + 4):.1f}" '
                f'stroke="#8a8a8a" stroke-width="1.6" marker-end="url(#arrow)"/>'
            )
    nodes = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{node}" fill="rgba(124,92,214,0.14)" stroke="#7c5cd6"/>'
        f'<text x="{x:.1f}" y="{y - 2:.1f}" text-anchor="middle" font-size="18">{ICON[m]}</text>'
        f'<text x="{x:.1f}" y="{y + 15:.1f}" text-anchor="middle" font-size="10" fill="#8a8a8a">{m}</text>'
        for m, (x, y) in pos.items()
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 330" width="340" height="330">'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8a8a8a"/></marker></defs>'
        + "".join(arrows) + nodes + "</svg>"
    )


def move_distribution(metrics: List[Dict], labels: List[str]) -> go.Figure:
    """Radar of how often each player chose each move, straight from the metrics dicts."""
    axes = MOVES + (["nothing"] if any(m.get("nothing") for m in metrics) else [])
    fig = go.Figure()
    for player, (m, label) in enumerate(zip(metrics, labels)):
        values = [m.get(a, 0) for a in axes]
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=[f"{ICON[a]} {a}" for a in axes + axes[:1]],
            name=label, fill="toself", opacity=0.55, line=dict(color=PLAYER_COLORS[player], width=2),
        ))
    fig.update_layout(
        height=340, margin=dict(l=40, r=40, t=20, b=20),
        polar=dict(radialaxis=dict(visible=True, dtick=1, rangemode="tozero")),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig
