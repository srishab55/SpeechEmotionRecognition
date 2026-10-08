"""Abstention with calibrated confidence (node N13).

A contact-centre tool can decline to label a call it is unsure about and pass it to a person. That only
works if "unsure" is measured well, so probabilities are calibrated first. Temperature scaling fits one
number, T, on clips from *inner-validation actors* (training actors held back from fitting), never on
test clips. The calibrated probability is proportional to p ** (1 / T).

Caveat: the temperature comes from a model trained on two thirds of the training actors, and is applied to
the model trained on all of them. The two are close but not identical, and the report says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from unheard_lib import classical, deep, nets
from unheard_lib.access import assert_disjoint
from unheard_lib.config import W2V_LAYERS
from unheard_lib.data import Context
from unheard_lib.store import ModelRun

TEMPERATURE_BOUNDS = (0.05, 20.0)
ECE_BINS = 10


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrated probabilities: sharpen (T < 1) or soften (T > 1) a probability matrix."""
    logits = np.log(np.clip(proba, 1e-9, 1.0)) / temperature
    logits -= logits.max(1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(1, keepdims=True)


def fit_temperature(proba: np.ndarray, truth: np.ndarray) -> float:
    """The T that minimises the negative log-likelihood of the true labels."""

    def nll(t: float) -> float:
        p = apply_temperature(proba, t)
        return float(-np.log(np.clip(p[np.arange(len(truth)), truth], 1e-9, 1.0)).mean())

    return float(minimize_scalar(nll, bounds=TEMPERATURE_BOUNDS, method="bounded").x)


def expected_calibration_error(proba: np.ndarray, truth: np.ndarray, bins: int = ECE_BINS) -> float:
    """Average gap between stated confidence and actual accuracy, weighted by how many clips fall in each bin."""
    confidence, correct = proba.max(1), proba.argmax(1) == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    return float(
        sum(
            (index == b).mean() * abs(confidence[index == b].mean() - correct[index == b].mean())
            for b in range(bins)
            if (index == b).any()
        )
    )


def reliability_bins(proba: np.ndarray, truth: np.ndarray, bins: int = ECE_BINS) -> pd.DataFrame:
    """Per confidence bin: mean stated confidence, actual accuracy and clip count."""
    confidence, correct = proba.max(1), proba.argmax(1) == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    rows = [
        {
            "confidence": confidence[index == b].mean(),
            "accuracy": correct[index == b].mean(),
            "clips": int((index == b).sum()),
        }
        for b in range(bins)
        if (index == b).any()
    ]
    return pd.DataFrame(rows)


def coverage_curve(proba: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    """Accuracy when only the most confident `coverage` share of clips is answered."""
    order = np.argsort(-proba.max(1), kind="stable")
    correct = (proba.argmax(1) == truth)[order]
    answered = np.arange(1, len(truth) + 1)
    return pd.DataFrame(
        {
            "coverage": answered / len(truth),
            "accuracy": np.cumsum(correct) / answered,
            "threshold": proba.max(1)[order],
        }
    )


def accuracy_at(curve: pd.DataFrame, coverage: float) -> float:
    """Accuracy on the most confident `coverage` share of clips."""
    return float(curve.iloc[max(0, int(round(coverage * len(curve))) - 1)]["accuracy"])


# --------------------------------------------------------------------------- validation probabilities


def validation_probabilities(
    ctx: Context, run: ModelRun, fold: int, providers: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Refit a saved model's chosen configuration on the fitting actors and score the validation actors.

    `providers` supplies the inputs: `{"w2v": PooledW2V, "hand": {name: matrix}, "frames": {name: FrameData}}`.
    Returns `(validation rows, probabilities)`. No test row is used (asserted).
    """
    train, test = ctx.folds[fold]
    fit_rows, val_rows = deep.inner_split(ctx, train, ctx.cfg)
    assert_disjoint("calibration fit", fit_rows, test)
    assert_disjoint("calibration validation", val_rows, test)
    model, feature = run.name.split("|", 1)
    params = dict(run.folds[fold].params)
    if model in nets.SEQUENCE_MODELS or model in nets.ATTENTION_MODELS:
        data = providers["frames"][feature.split(" (")[0]]
        hyper = {k: v for k, v in params.items() if k not in {"epochs", "n_params"}}
        result = deep.fit(
            model,
            data,
            ctx.y,
            ctx.speakers,
            fit_rows,
            hyper,
            ctx.cfg,
            epochs=int(params["epochs"]),
            seed=500 + fold,
        )
        return val_rows, deep.predict(result.model, data, val_rows, deep.device_for(model))["proba"]
    if feature == "w2v":
        layer = int(params.pop("layer", W2V_LAYERS[0]))
        pooling = params.pop("pooling", "mean")
        matrix = providers["w2v"].get(layer, pooling, True)
    else:
        matrix = providers["hand"][feature]
    fitted = classical.make_pipeline_for(model, params, ctx.cfg).fit(matrix[fit_rows], ctx.y[fit_rows])
    return val_rows, fitted.predict_proba(matrix[val_rows])


def calibrate_run(ctx: Context, run: ModelRun, providers: dict) -> dict:
    """Per-fold temperatures, calibrated test probabilities and the validation-based thresholds.

    Returns `{"raw", "calibrated", "truth", "temperatures", "val_threshold", "fold_thresholds",
    "row_threshold"}` with test arrays in pooled fold order. Thresholds are set per fold on that fold's own
    validation actors; `val_threshold` is their median (for display) and `row_threshold` is the threshold each
    test clip is actually held to.
    """
    raw, calibrated, truth, temps = [], [], [], []
    fold_thresholds: dict[float, list[float]] = {0.8: [], 0.9: []}
    row_thresholds: dict[float, list[np.ndarray]] = {0.8: [], 0.9: []}
    for fold in sorted(run.folds):
        part = run.folds[fold]
        val_rows, val_proba = validation_probabilities(ctx, run, fold, providers)
        temperature = fit_temperature(val_proba, ctx.y[val_rows])
        temps.append(temperature)
        calibrated_val = apply_temperature(val_proba, temperature)
        conf, correct = calibrated_val.max(1), calibrated_val.argmax(1) == ctx.y[val_rows]
        for target in fold_thresholds:
            # Each fold's threshold comes from that fold's own validation actors only, so no labelled clip of a
            # test actor can shape the threshold applied to that actor's test clips.
            ok = [t for t in np.sort(np.unique(conf)) if correct[conf >= t].mean() >= target]
            threshold = float(ok[0]) if ok else float("nan")
            fold_thresholds[target].append(threshold)
            row_thresholds[target].append(
                np.full(len(part.rows), np.inf if np.isnan(threshold) else threshold)
            )
        raw.append(part.proba)
        calibrated.append(apply_temperature(part.proba, temperature))
        truth.append(ctx.y[part.rows])
    return {
        "raw": np.concatenate(raw),
        "calibrated": np.concatenate(calibrated),
        "truth": np.concatenate(truth),
        "temperatures": temps,
        "val_threshold": {
            t: float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
            for t, v in fold_thresholds.items()
        },
        "fold_thresholds": fold_thresholds,
        "row_threshold": {t: np.concatenate(v) for t, v in row_thresholds.items()},
    }
