"""Cross-model comparison (node M2): verdicts, per-emotion scores, cost and error analysis.

Every number here is computed from the saved out-of-fold predictions. Formal verdicts are limited to the
comparisons declared in `config.FORMAL_COMPARISONS` before any result existed; the rest is descriptive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ser.evaluate import LABELS
from unheard_lib import scorebook, scoring
from unheard_lib.config import FORMAL_COMPARISONS, MIN_DETECTABLE_PP, Config
from unheard_lib.data import Context
from unheard_lib.stages import BASELINE_NAME
from unheard_lib.store import ModelRun, RunStore

#: Emotions expressed with high vocal energy (arousal) and low vocal energy. Disgust is ambiguous in the
#: literature; it is placed with the low group, where listeners most often rate it.
HIGH_AROUSAL = ("angry", "fearful", "happy", "surprised")
LOW_AROUSAL = ("neutral", "calm", "sad", "disgust")


# --------------------------------------------------------------------------- formal comparisons


#: Run-name markers of ablation variants. They are reported but never chosen as a family's representative.
ABLATION_MARKERS = (" (no ", "+augmented")


def is_candidate(name: str) -> bool:
    """True for a run that may represent its family: not the baseline and not an ablation variant."""
    return name != BASELINE_NAME and not any(marker in name for marker in ABLATION_MARKERS)


def inner_score(store: RunStore, name: str) -> float:
    """Mean inner-validation macro-F1 over folds (NaN when the run has none). Uses no test predictions."""
    values = [p.inner_score for p in store.load(name).folds.values() if np.isfinite(p.inner_score)]
    return float(np.mean(values)) if values else float("nan")


#: A neural run is flagged as not converged when its inner-validation macro-F1 stays this low
#: (chance for eight classes is 0.125) or its chosen epoch is this early in a typical fold.
CONVERGENCE_MIN_INNER_F1 = 0.30
CONVERGENCE_MIN_EPOCH = 5
#: Fewer completed tuning trials per fold than this counts as thin tuning.
THIN_TUNING_TRIALS = 3


def _is_neural(run: ModelRun) -> bool:
    return all("epochs" in p.params and "lr" in p.params for p in run.folds.values())


def convergence_table(store: RunStore, names: list[str]) -> pd.DataFrame:
    """Did each neural run train? Inner-validation score, typical chosen epoch and trials per fold.

    Uses only inner-validation numbers, never test predictions.
    """
    rows = []
    for name in names:
        run = store.load(name)
        if not _is_neural(run):
            continue
        folds = list(run.folds.values())
        inner = float(np.nanmean([p.inner_score for p in folds]))
        epoch = float(np.median([p.params["epochs"] for p in folds]))
        trials = [p.params["trials_completed"] for p in folds if "trials_completed" in p.params]
        rows.append(
            {
                "run": name,
                "inner-validation macro-F1": inner,
                "typical chosen epoch": epoch,
                "trials completed per fold": float(np.mean(trials)) if trials else float("nan"),
                "trained": "no"
                if inner < CONVERGENCE_MIN_INNER_F1 or epoch < CONVERGENCE_MIN_EPOCH
                else "yes",
                "tuning": "thin" if trials and float(np.mean(trials)) < THIN_TUNING_TRIALS else "ok",
            }
        )
    return pd.DataFrame(rows).sort_values("inner-validation macro-F1").reset_index(drop=True)


def overfitting_table(scores: pd.DataFrame, store: RunStore) -> pd.DataFrame:
    """Inner-validation score against pooled test score for every run that has both, largest gap first.

    The inner score is the best of several tried settings on held-back training actors, so it is optimistic
    by construction; a large gap marks a model whose tuning does not carry over to new speakers.
    """
    rows = []
    for _, row in scores.iterrows():
        inner = inner_score(store, row["name"])
        if np.isfinite(inner):
            rows.append(
                {
                    "run": row["name"],
                    "inner-validation macro-F1": inner,
                    "test macro-F1": row["macro_f1"],
                    "gap (points)": 100.0 * (inner - row["macro_f1"]),
                }
            )
    return pd.DataFrame(rows).sort_values("gap (points)", ascending=False).reset_index(drop=True)


def select_best(scores: pd.DataFrame, store: RunStore, family: str | None = None) -> str | None:
    """The run with the best inner-validation score, so the choice never looks at the test folds.

    Falls back to the pooled score only when no candidate has an inner score.
    """
    pool = scores[scores["name"].map(is_candidate)]
    if family is not None:
        pool = pool[pool["family"] == family]
    if pool.empty:
        return None
    inner = pool["name"].map(lambda n: inner_score(store, n))
    if inner.notna().any():
        return str(pool.loc[inner.idxmax(), "name"])
    return str(pool.sort_values("macro_f1").iloc[-1]["name"])


def best_by_family(scores: pd.DataFrame, store: RunStore) -> pd.DataFrame:
    """Score rows of each family's representative, chosen by inner validation, plus the baseline."""
    names = [n for n in (select_best(scores, store, f) for f in scores["family"].unique()) if n]
    return scores[scores["name"].isin(names + [BASELINE_NAME])].reset_index(drop=True)


def resolve(side: str, scores: pd.DataFrame, store: RunStore) -> str | None:
    """Turn `baseline`, `best:<family>` or a literal run name into a run name."""
    if side == "baseline":
        return BASELINE_NAME
    if side.startswith("best:"):
        return select_best(scores, store, side.removeprefix("best:"))
    return side if side in set(scores["name"]) else None


def formal_table(ctx: Context, store: RunStore, scores: pd.DataFrame) -> pd.DataFrame:
    """Verdict for every declared comparison whose two sides exist."""
    rows = []
    for a, b, label in FORMAL_COMPARISONS:
        name_a, name_b = resolve(a, scores, store), resolve(b, scores, store)
        if name_a is None or name_b is None or name_a == name_b:
            continue
        row = scorebook.compare(ctx, store.load(name_a), store.load(name_b), label)
        rows.append({**row, "A": name_a, "B": name_b})
    return pd.DataFrame(rows)


def selection_note() -> str:
    """How family representatives are chosen, stated for the report."""
    return (
        "Each family's representative is the run with the best inner-validation score (training actors held "
        "back inside each fold), not the best pooled test score, so choosing it does not use the predictions "
        "being compared. Ablation variants and the augmented baseline are excluded from that choice."
    )


# --------------------------------------------------------------------------- per-emotion and confusion


def per_emotion_f1(ctx: Context, run: ModelRun) -> np.ndarray:
    """F1 of each emotion, pooled over all out-of-fold predictions, in `LABELS` order."""
    p = run.pooled(ctx)
    return f1_score(p["y_true"], p["y_pred"], labels=list(range(len(LABELS))), average=None, zero_division=0)


def confusion_counts(ctx: Context, run: ModelRun) -> np.ndarray:
    """Row = true emotion, column = predicted emotion, over all out-of-fold predictions."""
    p = run.pooled(ctx)
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    np.add.at(cm, (p["y_true"], p["y_pred"]), 1)
    return cm


def emotion_table(ctx: Context, store: RunStore, names: list[str]) -> pd.DataFrame:
    """Per-emotion F1 for the named runs: rows are runs, columns are emotions."""
    return pd.DataFrame(
        [per_emotion_f1(ctx, store.load(n)) for n in names], index=names, columns=list(LABELS)
    )


# --------------------------------------------------------------------------- seeds and cost


def seed_spread(ctx: Context, run: ModelRun) -> dict | None:
    """Pooled macro-F1 for each training seed of a neural run, or None if seeds were not stored."""
    parts = [run.folds[f] for f in sorted(run.folds)]
    if any("seed_pred" not in p.extras for p in parts):
        return None
    n_seeds = min(p.extras["seed_pred"].shape[0] for p in parts)
    rows = np.concatenate([p.rows for p in parts])
    truth = ctx.y[rows]
    per_seed = [
        float(
            scoring.macro_f1(
                scoring.speaker_confusions(
                    truth,
                    np.concatenate([p.extras["seed_pred"][k] for p in parts]).astype(int),
                    ctx.speakers[rows],
                )[1].sum(0)
            )
        )
        for k in range(n_seeds)
    ]
    return {
        "seeds": n_seeds,
        "macro_f1 by seed": per_seed,
        "mean": float(np.mean(per_seed)),
        "sd": float(np.std(per_seed)),
    }


def cost_table(scores: pd.DataFrame, store: RunStore) -> pd.DataFrame:
    """Parameters (neural models), training time and pooled score for every run."""
    rows = []
    for _, row in scores.iterrows():
        run = store.load(row["name"])
        params = [p.params.get("n_params") for p in run.folds.values() if p.params.get("n_params")]
        rows.append(
            {
                "model": row["name"],
                "family": row["family"],
                "parameters": int(np.mean(params)) if params else None,
                "training minutes (all folds, tuning included)": row["train_seconds"] / 60.0,
                "macro-F1": row["macro_f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("macro-F1", ascending=False).reset_index(drop=True)


def seed_table(ctx: Context, store: RunStore, scores: pd.DataFrame) -> pd.DataFrame:
    """Seed-to-seed spread for every neural run that stored several seeds."""
    rows = []
    for name in scores["name"]:
        spread = seed_spread(ctx, store.load(name))
        if spread and spread["seeds"] > 1:
            rows.append(
                {
                    "model": name,
                    "seeds": spread["seeds"],
                    "mean macro-F1": spread["mean"],
                    "standard deviation": spread["sd"],
                    "range": max(spread["macro_f1 by seed"]) - min(spread["macro_f1 by seed"]),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- error analysis


def _macro_f1_on(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the classes present in `y_true` (the label-set rule)."""
    if len(y_true) == 0:
        return float("nan")
    present = np.unique(y_true)
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def slice_table(ctx: Context, run: ModelRun) -> pd.DataFrame:
    """Macro-F1 and accuracy by gender, intensity and arousal band, over classes present in each slice.

    RAVDESS has no strong-intensity neutral clips, so a fixed 8-label average would understate the strong
    slice. Each slice averages only over the emotions it contains.
    """
    p = run.pooled(ctx)
    meta = ctx.meta.iloc[p["rows"]]
    emotion = meta["emotion_label"].to_numpy()
    groups = {
        "gender": meta["gender"].to_numpy(),
        "intensity": meta["intensity_label"].to_numpy(),
        "arousal": np.where(np.isin(emotion, HIGH_AROUSAL), "high", "low"),
    }
    rows = []
    for kind, values in groups.items():
        for value in sorted(set(values)):
            mask = values == value
            yt, yp = p["y_true"][mask], p["y_pred"][mask]
            rows.append(
                {
                    "slice": kind,
                    "value": value,
                    "clips": int(mask.sum()),
                    "classes present": int(len(np.unique(yt))),
                    "macro-F1": _macro_f1_on(yt, yp),
                    "accuracy": float((yt == yp).mean()),
                }
            )
    return pd.DataFrame(rows)


def actor_table(ctx: Context, run: ModelRun) -> pd.DataFrame:
    """Accuracy per actor, sorted from worst to best."""
    p = run.pooled(ctx)
    frame = pd.DataFrame({"actor": p["speakers"], "correct": p["y_true"] == p["y_pred"]})
    gender = ctx.meta.drop_duplicates("actor").set_index("actor")["gender"]
    out = frame.groupby("actor")["correct"].agg(accuracy="mean", clips="size").reset_index()
    out["gender"] = out["actor"].map(gender)
    return out.sort_values("accuracy").reset_index(drop=True)


def confused_pairs(ctx: Context, run: ModelRun, top: int = 8) -> pd.DataFrame:
    """The most common (true, predicted) mistakes as counts and share of the true emotion's clips."""
    cm = confusion_counts(ctx, run)
    rows = [
        {
            "true": LABELS[i],
            "predicted": LABELS[j],
            "clips": int(cm[i, j]),
            "share of true emotion": cm[i, j] / cm[i].sum(),
        }
        for i in range(len(LABELS))
        for j in range(len(LABELS))
        if i != j and cm[i, j]
    ]
    return pd.DataFrame(rows).sort_values("clips", ascending=False).head(top).reset_index(drop=True)


# --------------------------------------------------------------------------- leakage control


def random_split_control(ctx: Context, hand: np.ndarray, cfg: Config) -> pd.DataFrame:
    """A leakage illustration for one model: the untuned baseline SVM under a random stratified split.

    Speakers appear on both sides of a random split, so the gap to the speaker-independent score is the
    amount by which a leaking split flatters a model. Uses the project's frozen `random_stratified` folds.
    """
    from ser.splits import get_split, load_folds
    from unheard_lib import classical

    table = load_folds()
    row_of = pd.Series(np.arange(len(ctx.meta)), index=ctx.meta["filename"])
    random_folds = {}
    for fold in cfg.folds:
        train, test = get_split("random_stratified", fold, table)
        random_folds[fold] = (row_of.loc[train].to_numpy(), row_of.loc[test].to_numpy())
    score = {}
    for protocol, folds in (("speaker-independent", ctx.folds), ("random split", random_folds)):
        pred, true = [], []
        for train, test in folds.values():
            fitted = classical.make_pipeline_for("svm_rbf", {"C": 1.0, "gamma": "scale"}, cfg).fit(
                hand[train], ctx.y[train]
            )
            pred.append(fitted.predict(hand[test]))
            true.append(ctx.y[test])
        score[protocol] = float(f1_score(np.concatenate(true), np.concatenate(pred), average="macro"))
    return pd.DataFrame(
        [
            {
                "model": "svm_rbf (defaults), 162 hand-designed features",
                "speaker-independent macro-F1": score["speaker-independent"],
                "random-split macro-F1": score["random split"],
                "inflation (points)": 100 * (score["random split"] - score["speaker-independent"]),
            }
        ]
    )


def _fmt(value: float) -> str:
    """One decimal, or two when a bound would otherwise print as zero."""
    return f"{value:+.2f}" if abs(value) < 0.05 else f"{value:+.1f}"


def verdict_sentence(row: pd.Series) -> str:
    """One sentence describing a comparison row, using only its printed numbers."""
    gap = row["difference (points)"]
    names = f" ({row['A']} against {row['B']})" if "A" in row and "B" in row else ""
    core = (
        f"{row['comparison']}{names}: {gap:+.1f} points "
        f"(interval {_fmt(row['interval low'])} to {_fmt(row['interval high'])})"
    )
    if row.get("note"):
        return f"{core}, not distinguishable ({row['note']}, so the single-seed gap is not relied on)."
    if row["verdict"] == "not distinguishable":
        return (
            f"{core}, not distinguishable (the gap is within {MIN_DETECTABLE_PP:.0f} points or the interval "
            "includes zero, so the report does not call one model better)."
        )
    edge = min(abs(row["interval low"]), abs(row["interval high"]))
    borderline = (
        f" (borderline: one edge of the interval is {edge:.2f} points from zero)" if edge < 0.1 else ""
    )
    return f"{core}, {row['verdict']}{borderline}."
