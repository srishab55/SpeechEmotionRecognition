"""Plotly figures: one template, static PNG export, palette enforcement.

Every figure is exported with Kaleido to `figures/<name>.png`. Inside a
notebook the PNG is also displayed, so the notebook carries static images and
no interactive widgets. A JSON copy of each figure goes to
`outputs/figure_specs/` so a test can check, after the fact, that every figure
used the shared template and only palette-derived colours.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

from unheard_lib.config import PALETTE, Config

TEMPLATE_NAME = "unheard"

#: Light end of the single-hue sequential scale built from the first palette colour.
SEQ_LOW = "#eef2f7"
SEQUENTIAL = [[0.0, SEQ_LOW], [1.0, PALETTE[0]]]
#: Diverging scale for deltas: palette blue against palette orange around white.
DIVERGING = [[0.0, PALETTE[1]], [0.5, "#ffffff"], [1.0, PALETTE[0]]]

EXPORT_WIDTH, EXPORT_HEIGHT, EXPORT_SCALE = 1000, 560, 2

_COLOR = re.compile(r"#[0-9a-fA-F]{6}\b")


def register_template() -> None:
    """Register the "unheard" template and make it the default."""
    template = go.layout.Template()
    template.layout = go.Layout(
        colorway=PALETTE,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=14, color="#222222"),
        title=dict(font=dict(size=18)),
        xaxis=dict(gridcolor="#e6e6e6", zerolinecolor="#cccccc", linecolor="#999999"),
        yaxis=dict(gridcolor="#e6e6e6", zerolinecolor="#cccccc", linecolor="#999999"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=70, r=30, t=70, b=70),
    )
    pio.templates[TEMPLATE_NAME] = template
    pio.templates.default = TEMPLATE_NAME


def in_notebook() -> bool:
    """True when running inside a Jupyter kernel (not a plain script)."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def _is_neutral(colour: str) -> bool:
    """White, black and pure greys are allowed alongside the palette."""
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return abs(r - g) <= 6 and abs(g - b) <= 6


def stray_colours(spec: Any) -> set[str]:
    """Hex colours in a figure spec that are neither palette-derived nor neutral."""
    allowed = {c.lower() for c in PALETTE} | {SEQ_LOW}
    found = {c.lower() for c in _COLOR.findall(json.dumps(spec))}
    return {c for c in found if c not in allowed and not _is_neutral(c)}


def show(
    fig: go.Figure, name: str, cfg: Config, width: int = EXPORT_WIDTH, height: int = EXPORT_HEIGHT
) -> Path | None:
    """Export `fig` as a static PNG, record its spec, display it in notebooks."""
    fig.update_layout(template=TEMPLATE_NAME, width=width, height=height)
    cfg.figures.mkdir(parents=True, exist_ok=True)
    png = cfg.figures / f"{name}.png"
    fig.write_image(str(png), scale=EXPORT_SCALE)

    specs = cfg.outputs / "figure_specs"
    specs.mkdir(parents=True, exist_ok=True)
    (specs / f"{name}.json").write_text(fig.to_json())

    if in_notebook():
        from IPython.display import Image, display

        display(Image(filename=str(png)))
        return None  # the notebook shows the image; printing a machine-specific path would add nothing
    return png


def check_export(cfg: Config) -> Path:
    """Write a tiny test PNG in the palette; raises if Kaleido cannot export."""
    fig = go.Figure(go.Bar(x=list("ABCD"), y=[3, 1, 4, 2], marker_color=PALETTE[:4]))
    fig.update_layout(title="Kaleido export check")
    show(fig, "_export_check", cfg, width=500, height=300)
    return cfg.figures / "_export_check.png"


def display_table(frame, rounding: int = 4) -> None:
    """Show a table: rich in a notebook, plain text in a terminal."""
    frame = frame.round(rounding) if hasattr(frame, "round") else frame
    if in_notebook():
        from IPython.display import display

        display(frame)
    else:
        print(frame.to_string())
