"""Scoring and the results log (node N11).

Each finished model is scored once on its pooled out-of-fold predictions. The
score goes to `outputs/scores.csv` (with speaker-clustered intervals) and, as
the project convention requires, to the shared `results/results.csv` through
`ser.evaluate`. A model that is already in `scores.csv` is not logged twice.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ser.evaluate import evaluate_pooled
from unheard_lib import scoring
from unheard_lib.config import MIN_DETECTABLE_PP
from unheard_lib.data import Context
from unheard_lib.store import ModelRun

COLUMNS = [
    "name",
    "family",
    "features",
    "macro_f1",
    "macro_f1_lo",
    "macro_f1_hi",
    "accuracy",
    "accuracy_lo",
    "accuracy_hi",
    "n_clips",
    "n_speakers",
    "train_seconds",
    "pred_hash",
    "notes",
]


def _hash(pred: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pred, dtype=np.int64).tobytes()).hexdigest()[:12]


def scores_path(ctx: Context):
    return ctx.cfg.outputs / "scores.csv"


def load_scores(ctx: Context) -> pd.DataFrame:
    """The scores table, empty if nothing has been scored yet."""
    path = scores_path(ctx)
    return pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=COLUMNS)


def prune_scores(ctx: Context, names: list[str]) -> int:
    """Drop rows of runs that no longer exist on disk, so the table lists exactly the current runs."""
    table = load_scores(ctx)
    keep = table[table["name"].isin(set(names))].drop_duplicates("name", keep="last")
    if len(keep) != len(table):
        keep.to_csv(scores_path(ctx), index=False)
    return len(table) - len(keep)


def score_run(ctx: Context, run: ModelRun) -> dict:
    """Score one model, log it once, and return its row."""
    pooled = run.pooled(ctx)
    digest = _hash(pooled["y_pred"])
    table = load_scores(ctx)
    known = table[(table["name"] == run.name) & (table["pred_hash"] == digest)]
    if len(known):
        return known.iloc[0].to_dict()

    stats = scoring.cluster_bootstrap(
        pooled["y_true"], pooled["y_pred"], pooled["speakers"], n_boot=ctx.cfg.n_bootstrap, seed=ctx.cfg.seed
    )
    _log_to_results(ctx, run)
    row = {
        "name": run.name,
        "family": run.family,
        "features": run.features,
        "macro_f1": stats["macro_f1"],
        "macro_f1_lo": stats["macro_f1_lo"],
        "macro_f1_hi": stats["macro_f1_hi"],
        "accuracy": stats["accuracy"],
        "accuracy_lo": stats["accuracy_lo"],
        "accuracy_hi": stats["accuracy_hi"],
        "n_clips": len(pooled["y_true"]),
        "n_speakers": stats["n_speakers"],
        "train_seconds": sum(p.seconds for p in run.folds.values()),
        "pred_hash": digest,
        "notes": run.notes,
    }
    table = table[table["name"] != run.name]
    pd.concat([table, pd.DataFrame([row])], ignore_index=True)[COLUMNS].to_csv(scores_path(ctx), index=False)
    return row


def _log_to_results(ctx: Context, run: ModelRun) -> None:
    """Append the pooled score to the shared results log (a scratch file in smoke mode)."""
    per_fold = []
    for fold in sorted(run.folds):
        rows = run.folds[fold].rows
        per_fold.append((ctx.names(ctx.y[rows]), ctx.names(run.folds[fold].pred), ctx.meta.iloc[rows]))
    path = ctx.cfg.outputs / "results_smoke.csv" if ctx.cfg.preset == "smoke" else None
    evaluate_pooled(
        per_fold,
        ctx.cfg.protocol,
        model=run.name,
        feature_config=run.features,
        hyperparams={"folds": {str(f): p.params for f, p in run.folds.items()}},
        owner="final-pipeline",
        notes=f"{ctx.cfg.preset}: {run.notes}",
        results_path=path,
    )


def _seed_macro_f1(ctx: Context, run: ModelRun) -> np.ndarray | None:
    """Pooled macro-F1 of each stored training seed, or None when the run kept no seeds."""
    parts = [run.folds[f] for f in sorted(run.folds)]
    if any("seed_pred" not in p.extras for p in parts):
        return None
    n_seeds = min(p.extras["seed_pred"].shape[0] for p in parts)
    if n_seeds < 2:
        return None
    rows = np.concatenate([p.rows for p in parts])
    truth = ctx.y[rows]
    return np.array(
        [
            scoring.macro_f1(
                scoring.speaker_confusions(
                    truth,
                    np.concatenate([p.extras["seed_pred"][k] for p in parts]).astype(int),
                    ctx.speakers[rows],
                )[1].sum(0)
            )
            for k in range(n_seeds)
        ]
    )


def _pooled_macro_f1(pooled: dict) -> float:
    """Pooled macro-F1 of one set of out-of-fold predictions."""
    return float(
        scoring.macro_f1(
            scoring.speaker_confusions(pooled["y_true"], pooled["y_pred"], pooled["speakers"])[1].sum(0)
        )
    )


def compare(ctx: Context, run_a: ModelRun, run_b: ModelRun, label: str) -> dict:
    """Paired speaker-clustered comparison of two runs on the same clips (A minus B), with its verdict.

    The scored predictions are one training seed. When both runs stored several seeds, a "better" or "worse"
    verdict is kept only if the gap between the seed averages has the same sign and also exceeds the
    detectable difference; otherwise the verdict becomes "not distinguishable" and the note says why.
    """
    a, b = run_a.pooled(ctx), run_b.pooled(ctx)
    if not (a["rows"] == b["rows"]).all():
        raise AssertionError("runs must cover the same clips in the same order")
    effect = scoring.paired_bootstrap(
        a["y_true"], a["y_pred"], b["y_pred"], a["speakers"], n_boot=ctx.cfg.n_bootstrap, seed=ctx.cfg.seed
    )
    verdict = scoring.verdict(effect, MIN_DETECTABLE_PP)
    seeds_a, seeds_b = _seed_macro_f1(ctx, run_a), _seed_macro_f1(ctx, run_b)
    # A classical model has no random start, so its single score stands in for its "seed average".
    if seeds_a is None and seeds_b is not None:
        seeds_a = np.array([_pooled_macro_f1(a)])
    elif seeds_b is None and seeds_a is not None:
        seeds_b = np.array([_pooled_macro_f1(b)])
    seed_gap, note = float("nan"), ""
    if seeds_a is not None and seeds_b is not None:
        seed_gap = 100.0 * float(seeds_a.mean() - seeds_b.mean())
        if verdict != "not distinguishable" and (
            np.sign(seed_gap) != np.sign(effect["delta_pp"]) or abs(seed_gap) <= MIN_DETECTABLE_PP
        ):
            verdict = "not distinguishable"
            note = f"the average over training seeds differs by {seed_gap:+.1f} points"
    return {
        "comparison": label,
        "difference (points)": effect["delta_pp"],
        "interval low": effect["lo_pp"],
        "interval high": effect["hi_pp"],
        "verdict": verdict,
        "seed-average difference (points)": seed_gap,
        "note": note,
    }
