"""Result figures shared by the classical and later stages (all in the T10 palette)."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go

from unheard_lib.config import BASELINE_MACRO_F1, MODEL_COLOR, PALETTE, Config
from unheard_lib.plotting import SEQUENTIAL


def score_bars(scores: pd.DataFrame, title: str) -> go.Figure:
    """Pooled macro-F1 per model with speaker-clustered 95% intervals, coloured by model family."""
    table = scores.sort_values("macro_f1").reset_index(drop=True)
    fig = go.Figure()
    for family, group in table.groupby("family", sort=False):
        fig.add_bar(
            y=group["name"],
            x=group["macro_f1"],
            orientation="h",
            name=family,
            marker_color=MODEL_COLOR.get(family, PALETTE[9]),
            error_x=dict(
                type="data",
                symmetric=False,
                array=group["macro_f1_hi"] - group["macro_f1"],
                arrayminus=group["macro_f1"] - group["macro_f1_lo"],
                color="#555555",
            ),
        )
    fig.add_vline(
        x=BASELINE_MACRO_F1,
        line_dash="dash",
        line_color=PALETTE[9],
        annotation_text="reference baseline 0.633",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="pooled out-of-fold macro-F1 (bars: 95% interval over speakers)",
        yaxis=dict(categoryorder="array", categoryarray=list(table["name"]), automargin=True),
        xaxis_range=[max(0.0, round(float(table["macro_f1_lo"].min()) - 0.05, 1)), 1.0],
        legend_title="model family",
    )
    return fig


def layer_curve(sweep: pd.DataFrame) -> go.Figure:
    """Macro-F1 by wav2vec2 layer for a fixed logistic regression, with and without per-speaker normalisation."""
    fig = go.Figure()
    for normalised, colour, label in (
        (True, PALETTE[4], "per-speaker normalised"),
        (False, PALETTE[9], "not normalised"),
    ):
        d = sweep[sweep["per_speaker_normalised"] == normalised].sort_values("layer")
        fig.add_scatter(
            x=d["layer"], y=d["macro_f1"], mode="lines+markers", name=label, line=dict(color=colour)
        )
    fig.update_layout(
        title="Macro-F1 by wav2vec2 layer (logistic regression, mean pooling)",
        xaxis_title="layer (0 = input, 12 = last)",
        yaxis_title="pooled macro-F1",
    )
    return fig


def confusion_heatmap(cm: np.ndarray, labels: tuple[str, ...], title: str) -> go.Figure:
    """Row-normalised confusion matrix on the single-hue sequential scale."""
    norm = cm / cm.sum(1, keepdims=True)
    text = [[f"{v:.0%}" for v in row] for row in norm]
    fig = go.Figure(
        go.Heatmap(
            z=norm,
            x=list(labels),
            y=list(labels),
            colorscale=SEQUENTIAL,
            zmin=0,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="share of true class"),
        )
    )
    fig.update_layout(title=title, xaxis_title="predicted", yaxis_title="true", yaxis_autorange="reversed")
    return fig


def optuna_history(cfg: Config, models: list[str], feature: str, title: str) -> go.Figure:
    """Best inner-validation macro-F1 found so far, averaged over folds, for each tuned model."""
    fig = go.Figure()
    for i, model in enumerate(models):
        curves = []
        for db in sorted(cfg.studies.glob(f"{model}__{feature}__fold*.db")):
            study = optuna.load_study(study_name=db.stem, storage=f"sqlite:///{db}")
            values = [t.value for t in study.trials if t.value is not None]
            if values:
                curves.append(np.maximum.accumulate(values))
        if curves:
            steps = min(len(c) for c in curves)
            mean = np.mean([c[:steps] for c in curves], axis=0)
            fig.add_scatter(
                x=np.arange(1, steps + 1),
                y=mean,
                mode="lines+markers",
                name=model,
                line=dict(color=PALETTE[i % 10]),
            )
    fig.update_layout(
        title=title, xaxis_title="trial", yaxis_title="best inner macro-F1 so far (mean over folds)"
    )
    return fig


def attention_profiles(ctx, run, lengths: np.ndarray, bins: int = 10) -> go.Figure:
    """Where in a clip the model looks: mean attention mass per tenth of the clip, by true emotion.

    A model that ignored timing would sit on the dashed line (equal mass in every tenth).
    Uses out-of-fold test clips only.
    """
    from ser.evaluate import LABELS

    mass = np.zeros((len(LABELS), bins))
    count = np.zeros(len(LABELS))
    for part in run.folds.values():
        weights = part.extras.get("weights")
        if weights is None:
            continue
        for row, w in zip(part.rows, weights, strict=True):
            n = int(lengths[row])
            edges = np.minimum((np.arange(n) * bins) // n, bins - 1)
            mass[ctx.y[row]] += np.bincount(edges, weights=w[:n], minlength=bins)
            count[ctx.y[row]] += 1
    fig = go.Figure()
    for k, label in enumerate(LABELS):
        if count[k]:
            fig.add_scatter(x=np.arange(1, bins + 1), y=mass[k] / count[k], mode="lines+markers", name=label,
                            line_color=PALETTE[k % len(PALETTE)])  # fmt: skip
    fig.add_hline(y=1 / bins, line_dash="dash", line_color=PALETTE[9], annotation_text="equal attention")
    fig.update_layout(xaxis_title="tenth of the clip (start to end)", yaxis_title="share of attention",
                      legend_title="true emotion")  # fmt: skip
    return fig


def emotion_heatmap(table: pd.DataFrame, title: str) -> go.Figure:
    """Per-emotion F1: one row per model, one column per emotion."""
    fig = go.Figure(
        go.Heatmap(
            z=table.to_numpy(),
            x=list(table.columns),
            y=list(table.index),
            colorscale=SEQUENTIAL,
            zmin=0,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in table.to_numpy()],
            texttemplate="%{text}",
            colorbar=dict(title="F1"),
        )
    )
    fig.update_layout(title=title, yaxis_autorange="reversed")
    return fig


def reliability_diagram(
    before: pd.DataFrame, after: pd.DataFrame, ece_before: float, ece_after: float
) -> go.Figure:
    """Stated confidence against actual accuracy, before and after temperature scaling."""
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color=PALETTE[9], dash="dash"),
        name="perfectly calibrated",
    )
    for table, name, colour, ece in (
        (before, "raw probabilities", PALETTE[1], ece_before),
        (after, "temperature-scaled", PALETTE[0], ece_after),
    ):
        fig.add_scatter(
            x=table["confidence"], y=table["accuracy"], mode="lines+markers",
            name=f"{name} (calibration error {ece:.3f})", line=dict(color=colour),
            marker=dict(size=6 + 24 * table["clips"] / table["clips"].sum()),
        )  # fmt: skip
    fig.update_layout(
        xaxis_title="stated confidence", yaxis_title="actual accuracy", xaxis_range=[0, 1], yaxis_range=[0, 1]
    )
    return fig


def coverage_figure(curve: pd.DataFrame, overall: float) -> go.Figure:
    """Accuracy against the share of clips answered, most confident first."""
    fig = go.Figure()
    fig.add_scatter(
        x=curve["coverage"],
        y=curve["accuracy"],
        mode="lines",
        name="answer the most confident first",
        line=dict(color=PALETTE[0]),
    )
    fig.add_hline(y=overall, line_dash="dash", line_color=PALETTE[9], annotation_text="answer everything")
    fig.update_layout(
        xaxis_title="share of clips answered (coverage)",
        yaxis_title="accuracy on answered clips",
        xaxis_range=[0, 1],
    )
    return fig
