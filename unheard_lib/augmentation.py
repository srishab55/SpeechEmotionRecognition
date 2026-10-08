"""Waveform augmentation for the classical models (node N6).

Rules enforced here:

* augmentation is applied to TRAINING clips only, inside each fold, after the
  split; an augmented copy carries its source clip's name so its actor is known;
* augmented copies never appear in a test fold or in an inner-validation split;
* augmented copies are normalised with the statistics of their source speaker's
  ORIGINAL clips, so an augmentation cannot shift a speaker's baseline.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from ser.augment import DEFAULT_AUGMENTATIONS, AugmentConfig, materialise, source_name
from ser.features import extract
from unheard_lib.classical import HAS_PROBA, make_pipeline_for
from unheard_lib.config import ROOT
from unheard_lib.data import Context
from unheard_lib.features import baseline_config
from unheard_lib.store import FoldPrediction, ModelRun, RunStore

log = logging.getLogger("unheard")
AUGMENT_DIR = ROOT / "data_augmented"


def augmented_features(
    ctx: Context,
    raw_original: np.ndarray,
    configs: tuple[AugmentConfig, ...] = DEFAULT_AUGMENTATIONS,
    subset: np.ndarray | None = None,
    n_jobs: int = 2,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Write augmented copies, extract the 162 features, and normalise per source speaker.

    Args:
        ctx: Shared context.
        raw_original: Un-normalised hand-designed features of the original clips.
        configs: Augmentations to apply.
        subset: Optional row indices of source clips (used by the smoke preset).
        n_jobs: Worker processes for feature extraction.

    Returns:
        `(table, features)`: one row per augmented clip with `source_filename`
        and `source_row`, and its normalised feature matrix.
    """
    sources = ctx.meta if subset is None else ctx.meta.iloc[subset].reset_index(drop=True)
    table = materialise(sources, configs, dest=AUGMENT_DIR, progress=False).reset_index(drop=True)
    row_of = pd.Series(np.arange(len(ctx.meta)), index=ctx.meta["filename"])
    table["source_row"] = row_of.loc[table["source_filename"]].to_numpy()
    tag = "_".join(c.name for c in configs) + (f"_n{len(sources)}" if subset is not None else "")
    cache = ROOT / "features" / f"features_aug_{tag}.parquet"
    frame = extract(baseline_config(), df=table, cache_path=cache, n_jobs=n_jobs)
    columns = [c for c in frame.columns if c not in table.columns]
    raw = frame[columns].to_numpy(dtype=np.float64)
    return table, _normalise_like_source(ctx, raw, raw_original, table["source_row"].to_numpy())


def _normalise_like_source(
    ctx: Context, raw_aug: np.ndarray, raw_original: np.ndarray, source_rows: np.ndarray
) -> np.ndarray:
    """Z-score augmented rows with their source speaker's statistics from the original clips."""
    out = np.empty_like(raw_aug)
    speakers = ctx.speakers[source_rows]
    for speaker in np.unique(speakers):
        original = raw_original[ctx.speakers == speaker]
        mean, std = original.mean(0), original.std(0) + 1e-8
        rows = speakers == speaker
        out[rows] = (raw_aug[rows] - mean) / std
    return out


def training_copies(ctx: Context, fold: int, table: pd.DataFrame) -> np.ndarray:
    """Indices of augmented rows whose SOURCE clip is a training clip of `fold`."""
    train, _ = ctx.folds[fold]
    return np.flatnonzero(table["source_row"].isin(train).to_numpy())


def check_no_test_leakage(ctx: Context, table: pd.DataFrame) -> dict[str, int]:
    """Gate G3: no augmented copy used in a fold's training set derives from that fold's test actors.

    Also verifies the source names recovered from the augmented file names agree
    with the source rows, so the filename convention itself cannot leak.
    """
    recovered = table["filename"].map(source_name)
    assert (recovered == table["source_filename"]).all(), "augmented names do not map back to their sources"
    used = 0
    for fold, (_, test) in ctx.folds.items():
        picked = training_copies(ctx, fold, table)
        sources = table["source_row"].to_numpy()[picked]
        test_actors = set(ctx.speakers[test])
        assert not set(ctx.speakers[sources]) & test_actors, (
            f"fold {fold}: augmented copy of a test actor in training"
        )
        assert not set(sources) & set(test), f"fold {fold}: augmented copy of a test clip in training"
        used += len(picked)
    return {"augmented_clips": len(table), "training_copies_used_across_folds": used}


def run_with_augmentation(
    ctx: Context,
    store: RunStore,
    name: str,
    features: str,
    model: str,
    params: dict,
    x_original: np.ndarray,
    table: pd.DataFrame,
    x_aug: np.ndarray,
) -> ModelRun:
    """Fixed-hyperparameter model trained on originals plus training-fold copies; tested on originals."""
    if store.exists(name):
        return store.load(name)
    folds = {}
    for fold, (train, test) in ctx.folds.items():
        started = time.monotonic()
        copies = training_copies(ctx, fold, table)
        x_train = np.vstack([x_original[train], x_aug[copies]])
        y_train = np.concatenate([ctx.y[train], ctx.y[table["source_row"].to_numpy()[copies]]])
        fitted = make_pipeline_for(model, params, ctx.cfg).fit(x_train, y_train)
        proba = fitted.predict_proba(x_original[test]) if HAS_PROBA[model] else None
        folds[fold] = FoldPrediction(
            fold,
            test,
            fitted.predict(x_original[test]).astype(int),
            proba,
            params,
            seconds=time.monotonic() - started,
        )
    run = ModelRun(
        name,
        "Classical (hand-designed)",
        features,
        folds,
        notes=f"waveform augmentation: {sorted(table['augmentation'].unique())}",
    )
    store.save(run)
    return run
