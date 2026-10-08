"""Fold definitions — frozen once written to ``splits/folds.csv``.

This module defines the contract that makes every result
comparable. Once `splits/folds.csv` is committed it must never change
; `write_folds` enforces that in code rather than leaving
it to discipline.

Protocols
---------
``speaker_independent``
    5-fold ``GroupKFold`` grouped by actor. **The primary protocol.** Every
    reported result uses this.

``random_stratified``
    5-fold ``StratifiedKFold`` on emotion. Run *only* as a deliberate
    comparison to quantify leakage — the same actors appear on both sides of
    every fold, which is exactly the flaw being measured. Never report this
    number on its own.

``statement_holdout``
    Single split: train on statement 01, test on statement 02. Tests whether
    prosody generalises across lexical content. Note that all 24 actors
    appear on both sides, so a good score here is ambiguous between lexical
    generalisation and speaker familiarity.

``statement_holdout_si``
    The speaker-disjoint version of the above: train on statement 01 from
    training actors, test on statement 02 from held-out actors. The gap
    between this and ``statement_holdout`` isolates lexical generalisation
    from speaker memorisation.

Why folds are unequal
---------------------
24 actors over 5 folds forces fold sizes of 5,5,5,5,**4** actors
(300/300/300/300/240 clips). Because every actor has identical emotion
composition, emotion balance is automatically proportional, but gender runs
40/60 and cannot be improved: a 5-actor fold has odd parity and can never be
50/50. A hand-balanced assignment produces the same 20 pp gap.

Consequence: prefer pooled out-of-fold scoring over averaging per-fold
metrics, so the smaller fold is not over-weighted. `ser.evaluate` does this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from ser import SEED  # single source; do not redeclare
from ser.metadata import load as load_metadata

N_SPLITS = 5

#: The frozen fold assignments. Committed; never regenerated.
FOLDS_PATH: Path = Path(__file__).resolve().parent.parent / "splits" / "folds.csv"

PROTOCOLS: tuple[str, ...] = (
    "speaker_independent",
    "random_stratified",
    "statement_holdout",
    "statement_holdout_si",
)

FOLDS_COLUMNS: tuple[str, ...] = ("protocol", "fold", "filename", "split")

_Split = tuple[np.ndarray, np.ndarray]


def speaker_independent(df: pd.DataFrame, n_splits: int = N_SPLITS) -> list[_Split]:
    """5-fold GroupKFold grouped by actor — the primary protocol.

    No actor appears on both sides of a fold, so a model cannot score by
    recognising a voice it has already heard.

    Args:
        df: Metadata frame from `ser.metadata.load`.
        n_splits: Number of folds. Leave at 5; the committed folds use 5.

    Returns:
        ``(train_idx, test_idx)`` positional index pairs, one per fold.
    """
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(df, df["emotion_label"], groups=df["actor"]))


def random_stratified(df: pd.DataFrame, n_splits: int = N_SPLITS) -> list[_Split]:
    """5-fold StratifiedKFold on emotion — the leakage comparison only.

    Clips from the same actor land on both sides of every fold. That is the
    defect this protocol exists to quantify, by comparison against
    `speaker_independent`. Never report this number alone.

    Args:
        df: Metadata frame from `ser.metadata.load`.
        n_splits: Number of folds.

    Returns:
        ``(train_idx, test_idx)`` positional index pairs, one per fold.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(df, df["emotion_label"]))


def statement_holdout(df: pd.DataFrame) -> list[_Split]:
    """Single split: train on statement 01, test on statement 02.

    Tests lexical invariance — does prosody carry emotion across different
    spoken content? All 24 actors appear on both sides, so pair this with
    `statement_holdout_si` before drawing a conclusion.

    Args:
        df: Metadata frame from `ser.metadata.load`.

    Returns:
        A single ``(train_idx, test_idx)`` pair, in a list for interface
        consistency with the k-fold protocols.
    """
    positions = np.arange(len(df))
    statement = df["statement"].to_numpy()
    return [(positions[statement == "01"], positions[statement == "02"])]


def statement_holdout_si(df: pd.DataFrame, n_splits: int = N_SPLITS) -> list[_Split]:
    """Statement holdout with the speakers held out too.

    Reuses the `speaker_independent` actor partition, then restricts train to
    statement 01 and test to statement 02. Both the actor and the sentence
    differ between the two sides, so a good score cannot be explained by
    speaker familiarity.

    Clips that are neither (statement 01, training actor) nor (statement 02,
    held-out actor) belong to neither side of that fold. This is the reason
    ``folds.csv`` records membership explicitly rather than as a
    clip-to-test-fold mapping.

    Args:
        df: Metadata frame from `ser.metadata.load`.
        n_splits: Number of folds.

    Returns:
        ``(train_idx, test_idx)`` positional index pairs, one per fold.
    """
    statement = df["statement"].to_numpy()
    out: list[_Split] = []
    for train_idx, test_idx in speaker_independent(df, n_splits=n_splits):
        out.append(
            (
                train_idx[statement[train_idx] == "01"],
                test_idx[statement[test_idx] == "02"],
            )
        )
    return out


_BUILDERS = {
    "speaker_independent": speaker_independent,
    "random_stratified": random_stratified,
    "statement_holdout": statement_holdout,
    "statement_holdout_si": statement_holdout_si,
}


def build_folds(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the long-format fold table for every protocol.

    One row per (protocol, fold, clip, side). Long format rather than a
    compact clip-to-fold mapping because ``statement_holdout_si``'s training
    set is a subset, not the complement, of its test set — a clip can belong
    to neither side.

    Args:
        df: Metadata frame. Defaults to the full inventory.

    Returns:
        Columns ``protocol, fold, filename, split``, sorted deterministically
        so the CSV is byte-stable across machines.
    """
    if df is None:
        df = load_metadata()

    filenames = df["filename"].to_numpy()
    records: list[pd.DataFrame] = []

    for protocol in PROTOCOLS:
        for fold, (train_idx, test_idx) in enumerate(_BUILDERS[protocol](df)):
            for side, idx in (("train", train_idx), ("test", test_idx)):
                records.append(
                    pd.DataFrame(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "filename": filenames[idx],
                            "split": side,
                        }
                    )
                )

    folds = pd.concat(records, ignore_index=True)
    # Deterministic order: protocols in declared order, then fold, side, name.
    folds["_protocol_order"] = folds["protocol"].map({name: i for i, name in enumerate(PROTOCOLS)})
    folds = (
        folds.sort_values(["_protocol_order", "fold", "split", "filename"], kind="mergesort")
        .drop(columns="_protocol_order")
        .reset_index(drop=True)
    )
    return folds[list(FOLDS_COLUMNS)]


def write_folds(path: Path | None = None, force: bool = False) -> Path:
    """Write the fold assignments to CSV, once.

    Refuses to overwrite an existing file. This is the rule that folds are frozen made
    mechanical: the folds are the contract that makes every result
    comparable, so regenerating them silently would invalidate every result
    already logged against them. Nothing in this codebase passes
    ``force=True``.

    Args:
        path: Destination. Defaults to `FOLDS_PATH`.
        force: Overwrite an existing file. Do not use.

    Returns:
        The path written.

    Raises:
        FileExistsError: If the file exists and ``force`` is False.
    """
    target = Path(path) if path is not None else FOLDS_PATH
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists and folds are frozen. "
            f"Every experiment logged so far assumes these exact assignments. "
            f"If you believe a split is wrong, raise it before changing anything."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    build_folds().to_csv(target, index=False, lineterminator="\n")
    return target


def load_folds(path: Path | None = None) -> pd.DataFrame:
    """Read the frozen fold assignments.

    Args:
        path: Source. Defaults to `FOLDS_PATH`.

    Returns:
        Columns ``protocol, fold, filename, split``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    source = Path(path) if path is not None else FOLDS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"No fold assignments at {source}. Generate them once with "
            f"ser.splits.write_folds(), then commit the file."
        )
    return pd.read_csv(source, dtype={"protocol": str, "filename": str, "split": str})


def get_split(
    protocol: str,
    fold: int,
    folds: pd.DataFrame | None = None,
) -> tuple[list[str], list[str]]:
    """Return the train and test filenames for one (protocol, fold).

    Filenames rather than positional indices: an index list silently means
    something different if applied to a differently-ordered DataFrame, and
    that failure mode would be invisible.

    Args:
        protocol: One of `PROTOCOLS`.
        fold: Fold number, 0-based.
        folds: Fold table. Defaults to reading the frozen CSV.

    Returns:
        ``(train_filenames, test_filenames)``.

    Raises:
        ValueError: If the protocol is unknown or the fold does not exist.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}; expected one of {PROTOCOLS}")

    table = load_folds() if folds is None else folds
    subset = table[(table["protocol"] == protocol) & (table["fold"] == fold)]
    if subset.empty:
        available = sorted(table.loc[table["protocol"] == protocol, "fold"].unique())
        raise ValueError(f"No fold {fold} for protocol {protocol!r}; available folds: {available}")

    train = subset.loc[subset["split"] == "train", "filename"].tolist()
    test = subset.loc[subset["split"] == "test", "filename"].tolist()
    return train, test


def summarise_folds(
    folds: pd.DataFrame | None = None,
    meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-fold composition of every protocol's test side.

    This is the table to eyeball before freezing.

    Args:
        folds: Fold table. Defaults to the frozen CSV.
        meta: Metadata frame. Defaults to the full inventory.

    Returns:
        One row per (protocol, fold) with train/test sizes, actor counts,
        actor overlap, gender split and neutral-class count.
    """
    table = load_folds() if folds is None else folds
    df = load_metadata() if meta is None else meta
    lookup = df.set_index("filename")

    rows = []
    for (protocol, fold), group in table.groupby(["protocol", "fold"], sort=False):
        train_names = group.loc[group["split"] == "train", "filename"]
        test_names = group.loc[group["split"] == "test", "filename"]
        train, test = lookup.loc[train_names], lookup.loc[test_names]
        rows.append(
            {
                "protocol": protocol,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "train_actors": train["actor"].nunique(),
                "test_actors": test["actor"].nunique(),
                "actor_overlap": len(set(train["actor"]) & set(test["actor"])),
                "test_pct_male": round(100 * (test["gender"] == "male").mean(), 1),
                "test_neutral": int((test["emotion_label"] == "neutral").sum()),
                "test_emotions": test["emotion_label"].nunique(),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    if FOLDS_PATH.exists():
        print(f"{FOLDS_PATH} exists (frozen). Composition:")
    else:
        print(f"Writing {write_folds()}")
    print(summarise_folds().to_string(index=False))
