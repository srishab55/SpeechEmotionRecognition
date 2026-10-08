"""Stage drivers: run a list of models under a wall-clock budget.

When the budget for a stage runs out, remaining lower-priority work is skipped
and recorded in `outputs/<preset>/deferred.json` with the reason, never dropped silently.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from unheard_lib import classical as C
from unheard_lib import deep, features, scorebook, scoring
from unheard_lib.config import BASELINE_MACRO_F1, BASELINE_TOLERANCE_PP, Config
from unheard_lib.data import Context
from unheard_lib.features import PooledW2V
from unheard_lib.runtime import StageClock, record_gate
from unheard_lib.store import ModelRun, RunStore

log = logging.getLogger("unheard")

BASELINE_NAME = "svm_rbf_default|handcrafted162"
FAMILY_HAND, FAMILY_W2V = "Classical (hand-designed)", "Classical (wav2vec2)"
#: Extra hand-designed feature set, tried only with the two cheapest strong models.
MFCC40_MODELS = ("logistic", "svm_rbf")


def record_deferred(cfg: Config, name: str, reason: str) -> None:
    """Append a deferred item to `outputs/<preset>/deferred.json`."""
    path = cfg.outputs / "deferred.json"
    items = json.loads(path.read_text()) if path.is_file() else []
    if not any(i["name"] == name for i in items):
        items.append({"name": name, "reason": reason})
    cfg.outputs.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=1))


def clear_deferred(cfg: Config, name: str) -> None:
    """Remove `name` from the deferred list once it has been run."""
    path = cfg.outputs / "deferred.json"
    if path.is_file():
        items = [i for i in json.loads(path.read_text()) if i["name"] != name]
        path.write_text(json.dumps(items, indent=1))


def classical_tasks() -> list[tuple[str, str]]:
    """`(model, feature_set)` pairs in priority order: tier first, then feature set."""
    tasks = []
    for model in C.ordered_models():
        tasks.append((model, "handcrafted162"))
        tasks.append((model, "w2v"))
        if model in MFCC40_MODELS:
            tasks.append((model, "mfcc40"))
    return tasks


def run_baseline(ctx: Context, store: RunStore, hand: np.ndarray) -> ModelRun:
    """Untuned SVM-RBF on the 162 hand-designed features, and gate G6 against the reference figure."""
    run = C.run_fixed(
        ctx, store, BASELINE_NAME, "Baseline", "handcrafted162", hand, "svm_rbf", {"C": 1.0, "gamma": "scale"}
    )
    row = scorebook.score_run(ctx, run)
    gap_pp = 100 * abs(row["macro_f1"] - BASELINE_MACRO_F1)
    record_gate(
        ctx.cfg,
        "G6 baseline replication",
        gap_pp <= BASELINE_TOLERANCE_PP,
        f"macro-F1 {row['macro_f1']:.4f} vs reference {BASELINE_MACRO_F1:.4f} (gap {gap_pp:.2f} pp, "
        f"limit {BASELINE_TOLERANCE_PP:.1f} pp)",
    )
    return run


def run_classical_stage(
    ctx: Context, store: RunStore, features: dict, pooled: PooledW2V, clock: StageClock
) -> list[ModelRun]:
    """Nested-tuned classical models in priority order, within the stage budget."""
    tasks, runs = classical_tasks(), []
    for done, (model, feat) in enumerate(tasks):
        name = f"{model}|{feat}"
        if clock.expired() and not store.exists(name):
            record_deferred(
                ctx.cfg, name, f"classical stage budget of {clock.budget_s / 60:.0f} min exhausted"
            )
            continue
        share = clock.remaining() / max(1, len(tasks) - done)
        if feat == "w2v":
            run = C.run_nested(ctx, store, model, feat, C.w2v_features(pooled), FAMILY_W2V, True, share)
        else:
            run = C.run_nested(
                ctx, store, model, feat, C.fixed_features(features[feat]), FAMILY_HAND, False, share
            )
        scorebook.score_run(ctx, run)
        clear_deferred(ctx.cfg, name)
        runs.append(run)
    return runs


def layer_sweep(ctx: Context, pooled: PooledW2V, layers: tuple[int, ...]) -> pd.DataFrame:
    """Pooled macro-F1 of a fixed logistic regression per wav2vec2 layer, with and without per-speaker normalisation.

    Descriptive only: the layer used by the tuned models is chosen by inner
    validation, not read off this curve.
    """
    rows = []
    for layer in layers:
        for normalise in (True, False):
            matrix = pooled.get(layer, "mean", normalise)
            preds, truth, test_rows = [], [], []
            for train, test in ctx.folds.values():
                model = C.make_pipeline_for("logistic", {"C": 0.01}, ctx.cfg).fit(matrix[train], ctx.y[train])
                preds.append(model.predict(matrix[test]))
                truth.append(ctx.y[test])
                test_rows.append(test)
            _, cms = scoring.speaker_confusions(
                np.concatenate(truth), np.concatenate(preds), ctx.speakers[np.concatenate(test_rows)]
            )
            rows.append(
                {
                    "layer": layer,
                    "per_speaker_normalised": normalise,
                    "macro_f1": float(scoring.macro_f1(cms.sum(0))),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- neural stages

FAMILY_SEQ, FAMILY_ATTN = "Sequence (from scratch)", "Attention"
FAMILY_UNIFORM, FAMILY_ADV = "Uniform control", "Speaker-adversarial"
#: Recurrent models read MFCC frames, convolutional ones log-mel frames; the second pairing is the ablation.
SEQUENCE_TASKS: tuple[tuple[str, str], ...] = (
    ("cnn1d", "logmel64"),
    ("gru", "mfcc60"),
    ("bilstm", "mfcc60"),
    ("cnn_lstm", "logmel64"),
    ("lstm", "mfcc60"),
    ("cnn2d", "logmel64"),
    ("rnn", "mfcc60"),
    ("cnn1d", "mfcc60"),
    ("gru", "logmel64"),
    ("bilstm", "logmel64"),
    ("lstm", "logmel64"),
    ("cnn_lstm", "mfcc60"),
    ("rnn", "logmel64"),
    ("cnn2d", "mfcc60"),
)
#: Priority order for attention models: the idea with the most upside first. A plain mean-pooled network is not
#: run: it is the uniform-weights control computed a second way and gives identical predictions.
ATTENTION_TASKS: tuple[str, ...] = (
    "attn_stats",
    "uniform",
    "additive",
    "transformer",
    "adversarial",
)
FAMILY_OF_ATTENTION = {"uniform": FAMILY_UNIFORM, "adversarial": FAMILY_ADV}


def frame_inputs(sequences: dict, w2v: tuple | None) -> dict[str, deep.FrameData]:
    """Named frame inputs available to the neural models."""
    out = {
        "logmel64": deep.FrameData("logmel64", *sequences["logmel"]),
        "mfcc60": deep.FrameData("mfcc60", *sequences["mfcc"]),
    }
    if w2v is not None:
        out["w2v_frames"] = deep.FrameData("w2v_frames", *w2v)
    return out


def _suffix(specaug: bool, raw: bool) -> str:
    """Run-name suffix of an ablation variant."""
    return (NO_SPECAUG if not specaug else "") + (NO_SPEAKER_NORM if raw else "")


def _still_to_run(store: RunStore, names: list[str], done: int) -> int:
    """Tasks from position `done` onward that have no saved result, so finished runs do not dilute the time share."""
    return sum(not store.exists(n) for n in names[done:])


def _run_neural(ctx, store, clock, model, data, family, remaining_tasks, **kwargs):
    """Run one neural model if budget allows, else record it as deferred; returns the run or None."""
    name = f"{model}|{data.name}{kwargs.get('suffix', '')}"
    if clock.expired() and not store.exists(name):
        record_deferred(
            ctx.cfg, name, f"{clock.name} stage budget of {clock.budget_s / 60:.0f} min exhausted"
        )
        return None
    share = clock.remaining() / max(1, remaining_tasks)
    if share < 60.0 * ctx.cfg.min_task_min and not store.exists(name):
        record_deferred(
            ctx.cfg,
            name,
            f"its share of the {clock.name} stage budget ({share / 60:.0f} min) is under the "
            f"{ctx.cfg.min_task_min} min needed for a fair search",
        )
        return None
    run = deep.run_nested(ctx, store, model, data, family, share, **kwargs)
    scorebook.score_run(ctx, run)
    clear_deferred(ctx.cfg, name)
    return run


NO_SPEAKER_NORM = " (no speaker norm)"
NO_SPECAUG = " (no SpecAugment)"


def raw_input(ctx: Context, name: str) -> deep.FrameData:
    """The same frames without per-speaker normalisation, standardised inside each fit instead."""
    if name == "w2v_frames":
        raw = features.w2v_frames(ctx, normalise=False)
        return deep.FrameData(name, *raw, standardise=True)
    key = {"logmel64": "logmel", "mfcc60": "mfcc"}[name]
    return deep.FrameData(name, *features.sequences(ctx, normalise=False)[key], standardise=True)


def run_sequence_stage(ctx: Context, store: RunStore, inputs: dict, clock: StageClock) -> list[ModelRun]:
    """Sequence models from scratch on hand-designed frames, plus the SpecAugment and normalisation ablations."""
    runs = []
    tasks = [(m, i, True, False) for m, i in SEQUENCE_TASKS] + [
        ("cnn1d", "logmel64", False, False),
        ("cnn1d", "logmel64", True, True),
    ]
    for done, (model, name, specaug, raw) in enumerate(tasks):
        suffix = (NO_SPECAUG if not specaug else "") + (NO_SPEAKER_NORM if raw else "")
        data = raw_input(ctx, name) if raw and not store.exists(f"{model}|{name}{suffix}") else inputs[name]
        run = _run_neural(ctx, store, clock, model, data, FAMILY_SEQ,
                          _still_to_run(store, [f"{m}|{n}{_suffix(sa, r)}" for m, n, sa, r in tasks], done),
                          specaug=specaug, suffix=suffix)  # fmt: skip
        if run:
            runs.append(run)
    return runs


def run_attention_stage(
    ctx: Context, store: RunStore, data: deep.FrameData, clock: StageClock
) -> list[ModelRun]:
    """Attention-pooling models on one frame input, plus the SpecAugment and normalisation ablations."""
    runs = []
    tasks = [(m, True, False) for m in ATTENTION_TASKS] + [
        ("attn_stats", False, False),
        ("attn_stats", True, True),
    ]
    for done, (model, specaug, raw) in enumerate(tasks):
        suffix = (NO_SPECAUG if not specaug else "") + (NO_SPEAKER_NORM if raw else "")
        source = (
            raw_input(ctx, data.name) if raw and not store.exists(f"{model}|{data.name}{suffix}") else data
        )
        run = _run_neural(ctx, store, clock, model, source, FAMILY_OF_ATTENTION.get(model, FAMILY_ATTN),
                          _still_to_run(store, [f"{m}|{data.name}{_suffix(sa, r)}" for m, sa, r in tasks], done),
                          specaug=specaug, suffix=suffix)  # fmt: skip
        if run:
            runs.append(run)
    return runs


def oof_extra(ctx: Context, run: ModelRun, key: str) -> np.ndarray:
    """Out-of-fold array saved with a run (for example test-clip embeddings), in clip-table row order."""
    first = next(iter(run.folds.values())).extras[key]
    out = np.zeros((len(ctx.y), first.shape[1]), dtype=np.float32)
    for part in run.folds.values():
        out[part.rows] = part.extras[key]
    return out
