"""The one scoring function everyone calls.

Every experiment in this project is scored here and nowhere else. A model
that has not been through `evaluate` has not been measured: the run is not
comparable to anyone else's, and no row exists in ``results/results.csv``.

What it always reports
----------------------
Accuracy **and** macro-F1, because accuracy alone hides failure on neutral,
which has half the clips of every other class. Plus the confusion matrix and
slices by gender, by intensity and by emotion — the slices feed the fairness
and intensity-conditioning experiments.

Pooled versus per-fold
----------------------
The speaker-independent folds are unequal (300/300/300/300/**240** clips,
because 24 actors do not divide by 5). Averaging five per-fold metrics
therefore over-weights each clip in the small fold by ~25%, and per-fold
macro-F1 rests on as few as 16 neutral clips. `evaluate_pooled` concatenates
every clip's held-out prediction and scores the 1,440 of them once. That is
the headline number; per-fold rows are still logged so variance is visible.

Nothing here is fit on data. `evaluate` only scores predictions it is
handed, so it cannot leak — but note that it is the caller's job to ensure
those predictions came from a model trained on the matching training fold.
"""

from __future__ import annotations

import csv
import getpass
import json
import subprocess
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ser.metadata import EMOTIONS

#: The shared results log. Appended to, never rewritten.
RESULTS_PATH: Path = Path(__file__).resolve().parent.parent / "results" / "results.csv"

#: Fixed schema. Do not add columns -- other code writes
#: readers against this. Extra metrics go in the returned dict instead.
RESULTS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "owner",
    "protocol",
    "fold",
    "feature_config",
    "model",
    "hyperparams",
    "accuracy",
    "macro_f1",
    "notes",
)

#: Fixed label order, so confusion matrices from different runs line up.
LABELS: tuple[str, ...] = tuple(EMOTIONS.values())

#: Metadata columns the slice reports are cut by.
SLICE_COLUMNS: tuple[str, ...] = ("gender", "intensity_label", "emotion_label")


def _resolve_owner(owner: str | None = None) -> str:
    """Determine who a result belongs to.

    Five people append to one file, so an unattributed row is a row nobody
    can chase. Reads ``git config user.name`` so nobody has to remember to
    set anything.

    Args:
        owner: Explicit override. Returned as-is when given.

    Returns:
        The owner name, falling back to the OS username, then ``"unknown"``.
    """
    if owner:
        return owner
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - identity is best-effort, never fatal
        return "unknown"


def _slice_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """Accuracy and macro-F1 within each level of ``groups``."""
    rows = []
    for level in sorted(pd.unique(groups.dropna())):
        mask = (groups == level).to_numpy()
        if not mask.any():
            continue
        true_here = y_true[mask]
        # Macro-F1 is only meaningful when the slice contains more than one
        # true class. Slicing by emotion gives single-class subsets, where
        # averaging F1 over the seven absent classes produces a number that
        # looks like a score but is not one. Accuracy there equals recall.
        multiclass = len(set(true_here.tolist())) > 1
        rows.append(
            {
                "level": level,
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(true_here, y_pred[mask])),
                "macro_f1": (
                    float(
                        f1_score(
                            true_here,
                            y_pred[mask],
                            average="macro",
                            zero_division=0,
                        )
                    )
                    if multiclass
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    protocol: str,
    fold: int | str,
    metadata: pd.DataFrame,
    *,
    model: str,
    feature_config: str,
    hyperparams: dict[str, Any] | None = None,
    owner: str | None = None,
    notes: str = "",
    results_path: Path | None = None,
    append: bool = True,
) -> dict[str, Any]:
    """Score one set of predictions and log it.

    Args:
        y_true: True emotion labels, e.g. ``"angry"``. Length must match
            ``y_pred`` and ``metadata``.
        y_pred: Predicted emotion labels, same ordering.
        protocol: Split protocol name, from `ser.splits.PROTOCOLS`.
        fold: Fold number, or the string ``"pooled"`` for a pooled
            out-of-fold score.
        metadata: Rows from `ser.metadata.load` for exactly these clips, in
            the same order as ``y_true``. Supplies the slice columns.
        model: Model identifier, e.g. ``"svm-rbf"``.
        feature_config: Feature configuration identifier or hash.
        hyperparams: Serialised to sorted JSON, so identical configurations
            produce identical strings.
        owner: Defaults to ``git config user.name``.
        notes: Free text for the results log.
        results_path: Override the results file. Tests must pass a temporary
            path so the shared log is never polluted.
        append: Set False to compute metrics without logging.

    Returns:
        ``accuracy``, ``macro_f1``, ``weighted_f1``, ``n``, ``per_class``
        (DataFrame), ``confusion_matrix`` (labelled DataFrame in `LABELS`
        order), ``slices`` (dict of DataFrames), and ``row`` (the dict
        written to CSV).

    Raises:
        ValueError: If the three inputs disagree in length.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} labels but y_pred has {len(y_pred)}")
    if len(metadata) != len(y_true):
        raise ValueError(
            f"metadata has {len(metadata)} rows but y_true has {len(y_true)} "
            f"labels; they must describe the same clips in the same order"
        )

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "emotion": list(LABELS),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    # Fixed label order so matrices from different runs are comparable, and
    # so a class the model never predicts still gets a row and a column.
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(LABELS)),
        index=pd.Index(LABELS, name="true"),
        columns=pd.Index(LABELS, name="predicted"),
    )

    meta = metadata.reset_index(drop=True)
    slices = {
        column: _slice_metrics(y_true, y_pred, meta[column])
        for column in SLICE_COLUMNS
        if column in meta.columns
    }

    row = {
        "run_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "owner": _resolve_owner(owner),
        "protocol": protocol,
        "fold": fold,
        "feature_config": feature_config,
        "model": model,
        "hyperparams": json.dumps(hyperparams or {}, sort_keys=True),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "notes": notes,
    }

    if append:
        _append_row(row, results_path)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n": len(y_true),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "slices": slices,
        "row": row,
    }


def _append_row(row: dict[str, Any], results_path: Path | None = None) -> Path:
    """Append one result row, writing the header only if the file is new.

    Opened in append mode per call rather than held open, so several processes
    running experiments concurrently do not clobber each other.

    Args:
        row: Keys exactly `RESULTS_COLUMNS`.
        results_path: Destination. Defaults to `RESULTS_PATH`.

    Returns:
        The path written to.
    """
    target = Path(results_path) if results_path is not None else RESULTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists() or target.stat().st_size == 0

    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULTS_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in RESULTS_COLUMNS})
    return target


def evaluate_pooled(
    per_fold: Iterable[tuple[Sequence[str], Sequence[str], pd.DataFrame]],
    protocol: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score pooled out-of-fold predictions as a single set.

    The headline number for unequal folds. Concatenates each fold's held-out
    predictions, then delegates to `evaluate` -- the scoring logic lives in
    exactly one place.

    Args:
        per_fold: One ``(y_true, y_pred, metadata)`` triple per fold.
        protocol: Split protocol name.
        **kwargs: Forwarded to `evaluate`. ``fold`` is set to ``"pooled"``
            and must not be passed.

    Returns:
        Whatever `evaluate` returns, over the concatenated predictions.

    Raises:
        ValueError: If ``per_fold`` is empty or ``fold`` was passed.
    """
    if "fold" in kwargs:
        raise ValueError("evaluate_pooled sets fold='pooled'; do not pass fold")

    trues: list[Any] = []
    preds: list[Any] = []
    metas: list[pd.DataFrame] = []
    for y_true, y_pred, meta in per_fold:
        trues.extend(np.asarray(y_true, dtype=object).tolist())
        preds.extend(np.asarray(y_pred, dtype=object).tolist())
        metas.append(meta)

    if not metas:
        raise ValueError("per_fold is empty; nothing to pool")

    return evaluate(
        trues,
        preds,
        protocol,
        "pooled",
        pd.concat(metas, ignore_index=True),
        **kwargs,
    )


def _demo() -> None:  # pragma: no cover - gate evidence, not library code
    """Score two synthetic models to show comparable rows are produced."""
    from ser.metadata import load
    from ser.splits import get_split

    meta = load().set_index("filename")
    rng = np.random.default_rng(42)

    for name, accuracy in (("synthetic-good", 0.7), ("synthetic-poor", 0.2)):
        per_fold = []
        for fold in range(5):
            _, test = get_split("speaker_independent", fold)
            truth = meta.loc[test, "emotion_label"].to_numpy()
            keep = rng.random(len(truth)) < accuracy
            pred = np.where(keep, truth, rng.choice(LABELS, size=len(truth)))
            per_fold.append((truth, pred, meta.loc[test].reset_index()))
            evaluate(
                truth,
                pred,
                "speaker_independent",
                fold,
                meta.loc[test].reset_index(),
                model=name,
                feature_config="synthetic",
                hyperparams={"target_accuracy": accuracy},
                notes="demo row, not a real result",
            )
        pooled = evaluate_pooled(
            per_fold,
            "speaker_independent",
            model=name,
            feature_config="synthetic",
            hyperparams={"target_accuracy": accuracy},
            notes="demo row, not a real result",
        )
        print(
            f"{name:16s} pooled acc={pooled['accuracy']:.4f} "
            f"macro_f1={pooled['macro_f1']:.4f} n={pooled['n']}"
        )


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--demo" in sys.argv:
        _demo()
    else:
        print(f"Results log: {RESULTS_PATH}")
        if RESULTS_PATH.exists():
            print(pd.read_csv(RESULTS_PATH).to_string(index=False))
