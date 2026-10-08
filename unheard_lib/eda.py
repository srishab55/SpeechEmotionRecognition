"""Exploratory analysis (node N3): figures and numbers for the notebook.

Signal analysis uses the TRAINING clips of fold 0 only, so no exploratory
decision is informed by held-out audio. Corpus-level counts (class balance,
statement balance) describe the dataset itself and use all 1,440 clips; the
captions say which is which.
"""

from __future__ import annotations

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ser.evaluate import LABELS
from ser.preprocess import load_audio
from unheard_lib.config import PALETTE
from unheard_lib.data import Context
from unheard_lib.plotting import SEQUENTIAL

EMOTION_COLOR = {label: PALETTE[i] for i, label in enumerate(LABELS)}
SYMBOLS = ("circle", "square", "diamond")


def training_rows(ctx: Context, fold: int = 0) -> np.ndarray:
    """Row indices of one fold's training clips (the only clips signal analysis may use)."""
    return ctx.folds[fold][0]


def class_balance(ctx: Context) -> go.Figure:
    """Clips per emotion and intensity (corpus-level, all 1,440 clips)."""
    counts = (
        ctx.meta.groupby(["emotion_label", "intensity_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(list(LABELS))
    )
    fig = go.Figure()
    for colour, intensity in zip((PALETTE[0], PALETTE[1]), counts.columns, strict=True):
        fig.add_bar(x=counts.index, y=counts[intensity], name=intensity, marker_color=colour)
    fig.update_layout(
        barmode="stack",
        title="Clips per emotion (all 1,440 clips)",
        yaxis_title="clips",
        legend_title="intensity",
    )
    return fig


def statement_balance(ctx: Context) -> pd.DataFrame:
    """Clips per emotion for each of the two spoken sentences (corpus-level)."""
    return pd.crosstab(ctx.meta["emotion_label"], ctx.meta["statement_label"]).reindex(list(LABELS))


def audit_boxplots(ctx: Context, audit: pd.DataFrame, column: str, title: str, axis: str) -> go.Figure:
    """Distribution of an audit measurement by emotion, on fold-0 training clips."""
    train = audit.iloc[training_rows(ctx)]
    fig = go.Figure()
    for label in LABELS:
        fig.add_box(
            y=train.loc[train["emotion_label"] == label, column],
            name=label,
            marker_color=EMOTION_COLOR[label],
            boxmean=True,
            showlegend=False,
        )
    fig.update_layout(title=f"{title} (fold-0 training clips)", yaxis_title=axis)
    return fig


def spectrogram_grid(ctx: Context) -> go.Figure:
    """One log-mel spectrogram per emotion, taken from fold-0 training clips (normal intensity, repetition 1)."""
    train = ctx.meta.iloc[training_rows(ctx)]
    fig = make_subplots(
        rows=2, cols=4, subplot_titles=list(LABELS), horizontal_spacing=0.04, vertical_spacing=0.14
    )
    for i, label in enumerate(LABELS):
        pool = train[(train["emotion_label"] == label) & (train["intensity_label"] == "normal")]
        signal, sr = load_audio(pool.iloc[0]["filepath"])
        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
        )
        fig.add_trace(
            go.Heatmap(z=mel, colorscale=SEQUENTIAL, showscale=False), row=i // 4 + 1, col=i % 4 + 1
        )
    fig.update_layout(title="Log-mel spectrograms, one clip per emotion (x: time frames, y: mel bands)")
    return fig


def pca_scatter(ctx: Context, raw: np.ndarray, normalised: np.ndarray, title: str) -> go.Figure:
    """2-D PCA (fitted on fold-0 training clips) coloured by emotion and by actor, raw vs per-speaker normalised."""
    rows = training_rows(ctx)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "raw, coloured by emotion",
            "raw, coloured by actor",
            "per-speaker normalised, by emotion",
            "per-speaker normalised, by actor",
        ),
    )
    for r, matrix in enumerate((raw, normalised), start=1):
        points = PCA(2, random_state=ctx.cfg.seed).fit_transform(StandardScaler().fit_transform(matrix[rows]))
        emotions, actors = ctx.meta["emotion_label"].to_numpy()[rows], ctx.speakers[rows]
        for label in LABELS:
            m = emotions == label
            fig.add_scatter(
                x=points[m, 0],
                y=points[m, 1],
                mode="markers",
                name=label,
                legendgroup=label,
                showlegend=(r == 1),
                marker=dict(color=EMOTION_COLOR[label], size=5, opacity=0.75),
                row=r,
                col=1,
            )
        for actor in np.unique(actors):
            m = actors == actor
            fig.add_scatter(
                x=points[m, 0],
                y=points[m, 1],
                mode="markers",
                showlegend=False,
                name=f"actor {actor}",
                marker=dict(
                    color=PALETTE[int(actor) % 10], symbol=SYMBOLS[int(actor) // 10 % 3], size=5, opacity=0.75
                ),
                row=r,
                col=2,
            )
    fig.update_layout(title=title, height=760)
    return fig


def silhouettes(ctx: Context, matrix: np.ndarray) -> dict[str, float]:
    """Silhouette of the fold-0 training clips grouped by emotion and by actor (higher = tighter groups)."""
    rows = training_rows(ctx)
    scaled = StandardScaler().fit_transform(matrix[rows])
    return {
        "by_emotion": float(silhouette_score(scaled, ctx.meta["emotion_label"].to_numpy()[rows])),
        "by_actor": float(silhouette_score(scaled, ctx.speakers[rows])),
    }


def scale_spread(matrix: np.ndarray, names: list[str] | None = None) -> pd.Series:
    """Per-feature standard deviation, to show why features need scaling."""
    return pd.Series(matrix.std(0), index=names)


def scale_figure(hand_raw: np.ndarray, hand_norm: np.ndarray) -> go.Figure:
    """Spread of the 162 hand-designed features before and after per-speaker normalisation."""
    fig = go.Figure()
    fig.add_histogram(
        x=np.log10(hand_raw.std(0) + 1e-12), name="raw", marker_color=PALETTE[9], opacity=0.8, nbinsx=30
    )
    fig.add_histogram(
        x=np.log10(hand_norm.std(0) + 1e-12),
        name="per-speaker normalised",
        marker_color=PALETTE[0],
        opacity=0.8,
        nbinsx=30,
    )
    fig.update_layout(
        barmode="overlay",
        title="Feature scale: standard deviation of each of the 162 features",
        xaxis_title="log10 of the standard deviation",
        yaxis_title="number of features",
    )
    return fig
