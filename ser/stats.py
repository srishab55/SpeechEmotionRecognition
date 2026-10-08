"""Uncertainty on the numbers this project reports.

Every result so far is a point estimate: 0.6311, +14.6 pp, -1.80 pp. Some of
those gaps are real and some are noise, and nothing written down says which.
This module supplies the missing half.

Why a *cluster* bootstrap
-------------------------
The obvious approach -- resample the 1,440 test clips with replacement -- is
wrong here, and wrong in the direction that flatters us. Clips are not
independent: 60 of them come from each actor, and a model that has learned a
particular voice gets that actor's clips right or wrong together. Treating them
as 1,440 independent observations understates the variance, producing
confidence intervals that are too narrow and differences that look significant
when they are not.

So the resampling unit is the **speaker**, not the clip: draw actors with
replacement, take all their clips. This is the standard cluster (block)
bootstrap, and it is the same reasoning that makes `speaker_independent` the
primary protocol. The project already accepts that the speaker is the unit of
independence for splitting; it has to be the unit of resampling too.

With 24 RAVDESS actors the resulting intervals are wide. That is the honest
width. A narrower interval here would be an artefact, not a better experiment.

Paired comparisons
------------------
When two models are scored on the *same* test clips their errors are
correlated, so comparing two independent intervals is needlessly conservative
-- overlapping intervals do not imply a non-significant difference. `paired_*`
resamples the same speakers for both models and bootstraps the *difference*,
which is the quantity actually in question.

`mcnemar` is the exact complement for accuracy specifically: it conditions on
the clips the two models disagree about and ignores the rest, which is the
sharpest test available for a paired accuracy comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ser import SEED

#: Default resamples. 2,000 is enough for a 95% interval to be stable to about
#: the third decimal, which is finer than any difference this project argues
#: about.
N_BOOT: int = 2000

Metric = Callable[[np.ndarray, np.ndarray], float]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1, with absent classes scored 0 rather than raising."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain accuracy."""
    return float(accuracy_score(y_true, y_pred))


METRICS: dict[str, Metric] = {"macro_f1": macro_f1, "accuracy": accuracy}


def _resolve(metric: str | Metric) -> Metric:
    if callable(metric):
        return metric
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRICS)}")
    return METRICS[metric]


def _speaker_blocks(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Unique speakers and the row indices belonging to each."""
    unique = np.unique(groups)
    return unique, [np.flatnonzero(groups == speaker) for speaker in unique]


def cluster_bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    groups: Sequence[str],
    metric: str | Metric = "macro_f1",
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, float]:
    """Confidence interval for one model's metric, resampling speakers.

    Args:
        y_true: True labels.
        y_pred: Predicted labels, same order.
        groups: Speaker id per clip. The resampling unit.
        metric: ``"macro_f1"``, ``"accuracy"``, or a callable.
        n_boot: Bootstrap resamples.
        alpha: 0.05 gives a 95% interval.
        seed: Seed for the resampling RNG.

    Returns:
        ``point``, ``lo``, ``hi``, ``se`` and ``n_speakers``.

    Raises:
        ValueError: If the three inputs disagree in length.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    groups = np.asarray(groups)
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError(f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, groups={len(groups)}")

    score = _resolve(metric)
    unique, blocks = _speaker_blocks(groups)
    rng = np.random.default_rng(seed)

    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([blocks[j] for j in picked])
        draws[i] = score(y_true[rows], y_pred[rows])

    return {
        "point": score(y_true, y_pred),
        "lo": float(np.quantile(draws, alpha / 2)),
        "hi": float(np.quantile(draws, 1 - alpha / 2)),
        "se": float(np.std(draws, ddof=1)),
        "n_speakers": int(len(unique)),
    }


def paired_cluster_bootstrap(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    groups: Sequence[str],
    metric: str | Metric = "macro_f1",
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, float]:
    """Confidence interval on the *difference* between two models.

    Both models are scored on the same resampled speakers each iteration, so
    the shared difficulty of a given speaker cancels. This is strictly sharper
    than comparing two separate intervals, and it is the right test when both
    models were evaluated on the same test set -- which, under the frozen
    folds, they always are.

    Args:
        y_true: True labels.
        pred_a: Predictions from model A (the candidate).
        pred_b: Predictions from model B (the baseline).
        groups: Speaker id per clip.
        metric: ``"macro_f1"``, ``"accuracy"``, or a callable.
        n_boot: Bootstrap resamples.
        alpha: 0.05 gives a 95% interval.
        seed: Seed for the resampling RNG.

    Returns:
        ``delta`` (A - B), ``lo``, ``hi``, ``p_value`` (two-sided, the
        proportion of resamples whose difference falls on the other side of
        zero, doubled), and ``significant`` at ``alpha``.
    """
    y_true = np.asarray(y_true, dtype=object)
    pred_a = np.asarray(pred_a, dtype=object)
    pred_b = np.asarray(pred_b, dtype=object)
    groups = np.asarray(groups)

    score = _resolve(metric)
    unique, blocks = _speaker_blocks(groups)
    rng = np.random.default_rng(seed)

    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([blocks[j] for j in picked])
        draws[i] = score(y_true[rows], pred_a[rows]) - score(y_true[rows], pred_b[rows])

    delta = score(y_true, pred_a) - score(y_true, pred_b)
    # Two-sided bootstrap p: how much of the distribution sits across zero.
    tail = min((draws <= 0).mean(), (draws >= 0).mean())
    p_value = float(min(1.0, 2 * tail))
    lo = float(np.quantile(draws, alpha / 2))
    hi = float(np.quantile(draws, 1 - alpha / 2))

    return {
        "delta": float(delta),
        "lo": lo,
        "hi": hi,
        "p_value": p_value,
        "significant": bool(lo > 0 or hi < 0),
        "n_speakers": int(len(unique)),
    }


def mcnemar(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
) -> dict[str, float]:
    """Exact McNemar test on a paired accuracy comparison.

    Conditions on the clips where exactly one model is right. Clips both get
    right, or both get wrong, carry no information about which is better and
    are discarded -- which is what makes this sharper than comparing two
    accuracies.

    Uses the exact binomial test rather than the chi-square approximation,
    because the discordant counts here are often small enough for the
    approximation to misbehave.

    Note this tests *accuracy*, not macro-F1. Macro-F1 is not a per-clip
    average, so it has no McNemar analogue; use `paired_cluster_bootstrap` for
    that. Note also that McNemar treats clips as independent, so it does not
    account for speaker clustering -- read it alongside the cluster bootstrap,
    not instead of it.

    Args:
        y_true: True labels.
        pred_a: Predictions from model A.
        pred_b: Predictions from model B.

    Returns:
        ``only_a_correct``, ``only_b_correct``, ``p_value`` and
        ``significant`` at 0.05.
    """
    from scipy.stats import binomtest

    y_true = np.asarray(y_true, dtype=object)
    correct_a = np.asarray(pred_a, dtype=object) == y_true
    correct_b = np.asarray(pred_b, dtype=object) == y_true

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    discordant = only_a + only_b

    if discordant == 0:
        return {
            "only_a_correct": 0,
            "only_b_correct": 0,
            "p_value": 1.0,
            "significant": False,
        }

    result = binomtest(only_a, discordant, 0.5, alternative="two-sided")
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < 0.05),
    }


def format_ci(result: dict[str, float], as_pp: bool = False) -> str:
    """Render a bootstrap result as ``point [lo, hi]`` for a report table.

    Args:
        result: Output of `cluster_bootstrap_ci` or `paired_cluster_bootstrap`.
        as_pp: Express in percentage points rather than raw units.

    Returns:
        A string like ``0.6311 [0.5842, 0.6760]`` or ``-11.11 pp [-15.02, -7.30]``.
    """
    scale = 100.0 if as_pp else 1.0
    unit = " pp" if as_pp else ""
    point = result.get("point", result.get("delta"))
    return (
        f"{scale * point:+.2f}{unit} [{scale * result['lo']:+.2f}, {scale * result['hi']:+.2f}]"
        if "delta" in result
        else f"{scale * point:.4f} [{scale * result['lo']:.4f}, {scale * result['hi']:.4f}]"
    )
