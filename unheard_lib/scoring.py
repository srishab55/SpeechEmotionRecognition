"""Fast speaker-clustered scoring (node N11).

Resampling speakers, not clips: the 60 clips of one actor succeed or fail
together, so treating clips as independent makes intervals about three times
too narrow. Each speaker's confusion matrix is computed once; a bootstrap
draw is then a weighted sum of 24 small matrices, which takes microseconds.
The point estimates are checked against `ser.stats` in the test suite.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 8


def speaker_confusions(
    y_true: np.ndarray, y_pred: np.ndarray, speakers: np.ndarray, n_classes: int = N_CLASSES
) -> tuple[np.ndarray, np.ndarray]:
    """One confusion matrix per speaker: returns `(speaker_ids, (S, C, C) counts)`."""
    ids = np.unique(speakers)
    cms = np.zeros((len(ids), n_classes, n_classes), dtype=np.float64)
    for k, speaker in enumerate(ids):
        rows = speakers == speaker
        np.add.at(cms[k], (y_true[rows], y_pred[rows]), 1)
    return ids, cms


def macro_f1(cm: np.ndarray) -> np.ndarray:
    """Macro-F1 for one matrix `(C, C)` or a batch `(B, C, C)`.

    Averages over classes that actually occur in the true labels, so a slice
    that lacks a class (RAVDESS has no strong-intensity neutral) is not
    penalised for the absent class.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(-1)
    predicted = cm.sum(-2)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    return (f1 * present).sum(-1) / present.sum(-1)


def accuracy(cm: np.ndarray) -> np.ndarray:
    """Accuracy from one matrix or a batch."""
    cm = np.asarray(cm, dtype=np.float64)
    return np.diagonal(cm, axis1=-2, axis2=-1).sum(-1) / cm.sum((-2, -1))


def _weights(n_speakers: int, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap multiplicity matrix `(n_boot, S)`: how often each speaker is drawn."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_speakers, size=(n_boot, n_speakers))
    return np.stack([np.bincount(row, minlength=n_speakers) for row in draws]).astype(np.float64)


def cluster_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    speakers: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Point estimate and speaker-clustered interval for macro-F1 and accuracy."""
    ids, cms = speaker_confusions(y_true, y_pred, speakers)
    boot_cm = np.einsum("bs,sij->bij", _weights(len(ids), n_boot, seed), cms)
    out: dict[str, float] = {"n_speakers": int(len(ids))}
    for name, fn in (("macro_f1", macro_f1), ("accuracy", accuracy)):
        draws = fn(boot_cm)
        out[name] = float(fn(cms.sum(0)))
        out[f"{name}_lo"] = float(np.quantile(draws, alpha / 2))
        out[f"{name}_hi"] = float(np.quantile(draws, 1 - alpha / 2))
    return out


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    speakers: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Difference in macro-F1 (A minus B) on the same resampled speakers, in points."""
    ids, cm_a = speaker_confusions(y_true, pred_a, speakers)
    _, cm_b = speaker_confusions(y_true, pred_b, speakers)
    w = _weights(len(ids), n_boot, seed)
    draws = macro_f1(np.einsum("bs,sij->bij", w, cm_a)) - macro_f1(np.einsum("bs,sij->bij", w, cm_b))
    delta = float(macro_f1(cm_a.sum(0)) - macro_f1(cm_b.sum(0)))
    tail = min((draws <= 0).mean(), (draws >= 0).mean())
    return {
        "delta_pp": 100 * delta,
        "lo_pp": 100 * float(np.quantile(draws, alpha / 2)),
        "hi_pp": 100 * float(np.quantile(draws, 1 - alpha / 2)),
        "p_value": float(min(1.0, 2 * tail)),
        "n_speakers": int(len(ids)),
    }


def verdict(paired: dict[str, float], min_detectable_pp: float) -> str:
    """Apply the two-part rule: gap above the floor AND an interval that excludes zero."""
    delta, lo, hi = paired["delta_pp"], paired["lo_pp"], paired["hi_pp"]
    excludes_zero = lo > 0 or hi < 0
    if abs(delta) > min_detectable_pp and excludes_zero:
        return "better" if delta > 0 else "worse"
    return "not distinguishable"
