"""Data loading, audio audit and split checks (nodes N2 and N4).

Reuses the project's own metadata parser, audio loader, audit probe and frozen
fold file; this module adds the checks the pipeline asserts on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ser.audit import run_audit
from ser.evaluate import LABELS
from ser.features import per_speaker_normalise
from ser.metadata import load as load_metadata
from ser.splits import get_split, load_folds
from unheard_lib.config import ROOT, Config

EXPECTED_NEUTRAL, EXPECTED_OTHER, EXPECTED_ACTORS = 96, 192, 24
EXPECTED_FOLD_SIZES = (300, 300, 300, 300, 240)
DURATION_RANGE_S = (2.9, 5.3)

W2V_CACHE = ROOT / "features" / "w2v_642607c7a846.npy"
W2V_ORDER = ROOT / "features" / "w2v_642607c7a846_order.parquet"


@dataclass
class Context:
    """Everything the stages share: clip table, integer labels and fold indices."""

    cfg: Config
    meta: pd.DataFrame
    y: np.ndarray  # integer emotion id per clip, index into LABELS
    speakers: np.ndarray  # actor id per clip
    folds: dict[int, tuple[np.ndarray, np.ndarray]]  # fold -> (train rows, test rows)

    @property
    def labels(self) -> tuple[str, ...]:
        return LABELS

    def names(self, ids: np.ndarray) -> np.ndarray:
        """Integer ids back to emotion names."""
        return np.asarray(LABELS, dtype=object)[np.asarray(ids)]


def load_context(cfg: Config) -> Context:
    """Read the clip table and the frozen folds and index them by row."""
    meta = load_metadata()
    row_of = pd.Series(np.arange(len(meta)), index=meta["filename"])
    table = load_folds()
    folds = {}
    for fold in cfg.folds:
        train, test = get_split(cfg.protocol, fold, table)
        folds[fold] = (row_of.loc[train].to_numpy(), row_of.loc[test].to_numpy())
    label_id = {name: i for i, name in enumerate(LABELS)}
    y = meta["emotion_label"].map(label_id).to_numpy()
    return Context(cfg, meta, y, meta["actor"].to_numpy(), folds)


def check_folds(ctx: Context) -> pd.DataFrame:
    """Gate G1 evidence: speaker overlap, coverage and composition per fold.

    Returns:
        One row per fold with sizes, actor counts, overlap and class mix.

    Raises:
        AssertionError: on any actor overlap or if the test sides do not
            partition the clips.
    """
    rows, seen = [], []
    for fold, (train, test) in ctx.folds.items():
        train_actors, test_actors = set(ctx.speakers[train]), set(ctx.speakers[test])
        overlap = train_actors & test_actors
        assert not overlap, f"fold {fold}: actors {sorted(overlap)} appear in train and test"
        assert not set(train) & set(test), f"fold {fold}: a clip is in both train and test"
        seen.extend(test.tolist())
        rows.append(
            {
                "fold": fold,
                "train_clips": len(train),
                "test_clips": len(test),
                "train_actors": len(train_actors),
                "test_actors": len(test_actors),
                "test_actor_ids": ",".join(str(a) for a in sorted(test_actors)),
                "actor_overlap": len(overlap),
                "test_male": int((ctx.meta["gender"].to_numpy()[test] == "male").sum()),
                "test_neutral": int((ctx.y[test] == 0).sum()),
            }
        )
    if len(ctx.folds) == 5:
        assert sorted(seen) == list(range(len(ctx.meta))), "test sides must partition all clips exactly once"
        sizes = tuple(sorted((r["test_clips"] for r in rows), reverse=True))
        assert sizes == EXPECTED_FOLD_SIZES, f"unexpected fold sizes {sizes}"
    return pd.DataFrame(rows)


def audit_clips(ctx: Context, use_cache: bool = True) -> pd.DataFrame:
    """Measure every clip (rate, channels, duration, silence, clipping, DC offset).

    The audit is cached because it decodes all 1,440 files at their native rate.
    """
    cache = ctx.cfg.cache / "audit.parquet"
    if use_cache and cache.is_file():
        cached = pd.read_parquet(cache)
        if (cached["filename"].to_numpy() == ctx.meta["filename"].to_numpy()).all():
            return cached
    cache.parent.mkdir(parents=True, exist_ok=True)
    audit = run_audit(ctx.meta, cache=False)
    audit["dc_offset"] = [_dc_offset(p) for p in audit["filepath"]]
    audit.to_parquet(cache, index=False)
    return audit


def _dc_offset(path: str) -> float:
    """Mean sample value of the mono downmix; zero for a well-centred recording."""
    data, _ = sf.read(path, always_2d=True)
    return float(data.mean())


def validate_audit(audit: pd.DataFrame) -> dict[str, object]:
    """Assert the dataset facts the rest of the pipeline relies on.

    Returns:
        A summary dict for the report. Raises `AssertionError` on a violation.
    """
    assert audit["load_error"].isna().all(), "some clips failed to load"
    numeric = audit.select_dtypes("number")
    assert np.isfinite(numeric.to_numpy()).all(), "NaN or inf in the audit table"
    counts = audit["emotion_label"].value_counts()
    assert counts["neutral"] == EXPECTED_NEUTRAL, f"neutral has {counts['neutral']} clips"
    assert (counts.drop("neutral") == EXPECTED_OTHER).all(), "every other emotion must have 192 clips"
    assert audit["actor"].nunique() == EXPECTED_ACTORS
    assert (audit.groupby("actor").size() == 60).all(), "each actor must have 60 clips"
    assert audit["sr"].nunique() == 1, "sample rate is not uniform"
    low, high = DURATION_RANGE_S
    assert audit["duration"].between(low, high).all(), "a duration is outside the expected range"
    dual = audit.loc[audit["n_channels"] > 1, "filename"].tolist()
    assert len(dual) == 5, f"expected 5 dual-channel files, found {len(dual)}"
    return {
        "clips": len(audit),
        "sample_rate": int(audit["sr"].iloc[0]),
        "dual_channel_files": dual,
        "clipped_clips": int((audit["n_clipped"] > 0).sum()),
        "max_abs_dc_offset": float(audit["dc_offset"].abs().max()),
        "duration_min_max_s": (float(audit["duration"].min()), float(audit["duration"].max())),
    }


def load_w2v_cache(ctx: Context) -> np.ndarray:
    """Cached pooled wav2vec2 tensor `(clips, 13 layers, [mean, std], 768)` in clip-table order.

    Raises:
        FileNotFoundError: if the cache files are missing.
        AssertionError: if the cache does not cover exactly the clip table.
    """
    for path in (W2V_CACHE, W2V_ORDER):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; copy the wav2vec2 cache into features/")
    order = pd.read_parquet(W2V_ORDER)["filename"].tolist()
    position = {name: i for i, name in enumerate(order)}
    missing = [f for f in ctx.meta["filename"] if f not in position]
    assert not missing, f"{len(missing)} clips have no cached embedding"
    tensor = np.load(W2V_CACHE, mmap_mode="r")
    rows = [position[f] for f in ctx.meta["filename"]]
    return np.asarray(tensor[rows])


def probe_cache_alignment(ctx: Context, cache: np.ndarray, layer: int = 6) -> float:
    """Cross-validated actor-recognition accuracy from the cached embeddings.

    A row order that disagreed with the clip table would scatter each actor's
    clips across the cache and drop this to chance (1/24). Also the "actor
    recoverability" figure quoted in the results.
    """
    features = cache[:, layer, 0]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000))
    folds = StratifiedKFold(5, shuffle=True, random_state=ctx.cfg.seed)
    predicted = cross_val_predict(model, features, ctx.speakers, cv=folds)
    return float((predicted == ctx.speakers).mean())


def actor_recoverability(ctx: Context, features: np.ndarray) -> float:
    """Accuracy of recognising the actor from features (chance is 1/24)."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000))
    folds = StratifiedKFold(5, shuffle=True, random_state=ctx.cfg.seed)
    return float((cross_val_predict(model, features, ctx.speakers, cv=folds) == ctx.speakers).mean())


def actor_recoverability_within_folds(ctx: Context, run) -> tuple[float, float]:
    """Actor recognition from a run's embeddings, tested inside each outer fold separately.

    The embeddings of different folds come from different networks, so they are never mixed: within one
    fold the classifier is cross-validated over that fold's test clips and asked to name one of that
    fold's actors. Returns the mean accuracy over folds and the mean chance level (1 / actors in a fold).
    """
    scores, chances = [], []
    for part in run.folds.values():
        embedding = np.asarray(part.extras["embedding"], dtype=np.float64)
        actors = ctx.speakers[part.rows]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000))
        cv = StratifiedKFold(5, shuffle=True, random_state=ctx.cfg.seed)
        predicted = cross_val_predict(model, embedding, actors, cv=cv)
        scores.append(float((predicted == actors).mean()))
        chances.append(1.0 / len(np.unique(actors)))
    return float(np.mean(scores)), float(np.mean(chances))


def normalise_per_speaker(ctx: Context, matrix: np.ndarray) -> np.ndarray:
    """Per-speaker z-scoring, the single label-free transductive step (the one stated exception to fitting on training rows only)."""
    return per_speaker_normalise(np.asarray(matrix, dtype=np.float64), ctx.speakers)


def project_path(*parts: str) -> Path:
    """A path under the project root."""
    return ROOT.joinpath(*parts)


def check_scaler_fits_train_only(ctx: Context, matrix: np.ndarray, make_model) -> None:
    """Gate G2: the scaler inside a model pipeline sees the training rows and nothing else.

    Fits the real pipeline on each fold's training rows and compares the fitted
    scaler statistics with the training-row and all-row statistics.

    Args:
        ctx: Shared context.
        matrix: An UN-normalised feature matrix, one row per clip. Per-speaker
            centring makes every column's overall mean zero, which would hide
            the difference between training-row and all-row statistics.
        make_model: Zero-argument callable returning an unfitted pipeline whose
            first step is the scaler.
    """
    for fold, (train, test) in ctx.folds.items():
        scaler = make_model().fit(matrix[train], ctx.y[train])[0]
        from_train = np.abs(scaler.mean_ - matrix[train].mean(0)).max()
        from_all = np.abs(scaler.mean_ - matrix.mean(0)).max()
        assert from_train < 1e-9, f"fold {fold}: scaler not fitted on training rows"
        assert from_all > 1e-6, (
            f"fold {fold}: scaler statistics are indistinguishable from all-row statistics"
        )
        assert not set(train) & set(test)
