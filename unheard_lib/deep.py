"""Training, nested tuning and prediction for the neural models.

For each outer fold the training actors are split into fitting actors and
inner-validation actors (speaker-grouped). Optuna tunes on that inner split,
and early stopping picks the epoch count there. The chosen model is then
retrained on all training actors for that many epochs and predicts the outer
test fold exactly once. No test clip reaches the objective, the early-stopping
monitor, the augmenter or the normaliser.
"""

from __future__ import annotations

import copy
import logging
import os
import time
import zlib
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn

from unheard_lib import nets
from unheard_lib.access import assert_disjoint
from unheard_lib.config import Config, study_tag
from unheard_lib.data import Context
from unheard_lib.runtime import detect_device, set_all_seeds
from unheard_lib.store import FoldPrediction, ModelRun, RunStore

log = logging.getLogger("unheard")

BATCH_SIZE = 32
LEARNING_RATE_RANGE = (3e-4, 3e-3)
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
#: SpecAugment: up to this fraction of a clip's frames and of its channels is blanked, twice in time.
SPEC_TIME_FRACTION, SPEC_CHANNEL_FRACTION, SPEC_TIME_MASKS = 0.15, 0.15, 2
#: Models whose packed-sequence or masking operations are not reliable on Apple's MPS backend.
CPU_ONLY = frozenset({"rnn", "lstm", "gru", "bilstm", "cnn_lstm", "cnn_bilstm", "cnn_bilstm_w2v", "transformer"})


@dataclass(frozen=True)
class FrameData:
    """A padded frame sequence per clip: `x` is `(clips, frames, channels)`, `lengths` the real counts."""

    name: str
    x: np.ndarray
    lengths: np.ndarray
    #: Standardise inputs with statistics from the fitting clips (for frames that are not speaker-normalised).
    standardise: bool = False

    @property
    def channels(self) -> int:
        return int(self.x.shape[2])


# --------------------------------------------------------------------------- search space


def suggest_hyper(trial: optuna.Trial, model: str) -> dict:
    """Sample hyperparameters. Every model shares learning rate, width and dropout."""
    hyper = {
        "lr": trial.suggest_float("lr", *LEARNING_RATE_RANGE, log=True),
        "width": trial.suggest_categorical("width", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
    }
    if model in {"rnn", "lstm", "gru", "bilstm", "transformer"}:
        hyper["layers"] = trial.suggest_int("layers", 1, 2)
    if model == "adversarial":
        hyper["adv_weight"] = trial.suggest_float("adv_weight", 0.1, 1.0)
    return hyper


def default_hyper(model: str) -> dict:
    """Fallback used when no trial finishes inside the time budget."""
    hyper = {"lr": 1e-3, "width": 64, "dropout": 0.3}
    if model in {"rnn", "lstm", "gru", "bilstm", "transformer"}:
        hyper["layers"] = 1
    if model == "adversarial":
        hyper["adv_weight"] = 0.3
    return hyper


def search_space_table() -> pd.DataFrame:
    """The neural search space as a table for the report."""
    rows = [
        ("all", "lr", "log-uniform", "3e-4 to 3e-3"),
        ("all", "width", "choice", "32, 64, 128"),
        ("all", "dropout", "uniform", "0.1 to 0.5"),
        ("rnn, lstm, gru, bilstm, transformer", "layers", "integer", "1 to 2"),
        ("adversarial", "adv_weight", "uniform", "0.1 to 1.0"),
    ]
    fixed = [
        ("all", "optimiser", "fixed", f"AdamW, weight decay {WEIGHT_DECAY}"),
        ("all", "label smoothing", "fixed", str(LABEL_SMOOTHING)),
        ("all", "gradient clipping", "fixed", f"norm {GRAD_CLIP}"),
        ("all", "batch size", "fixed", str(BATCH_SIZE)),
    ]
    return pd.DataFrame(rows + fixed, columns=["models", "parameter", "kind", "range or value"])


# --------------------------------------------------------------------------- batches and augmentation


def spec_augment(
    x: torch.Tensor, lengths: torch.Tensor, rng: np.random.Generator, fill: torch.Tensor | None = None
) -> torch.Tensor:
    """Blank random time spans and one channel band in each clip (SpecAugment). Training batches only.

    Blanked cells take `fill`, a per-channel value. Speaker-normalised inputs use zero, which is each
    speaker's average frame. Raw inputs that are standardised inside the network pass the channel mean, so
    a blank is an average frame in both variants and the two differ only in the normalisation.
    """
    home = x.device
    x = x.detach().cpu().clone()  # the per-clip loop below is slow on an accelerator; masks are tiny, so do it on the CPU
    lengths = lengths.cpu()
    fill = torch.zeros(x.shape[2], dtype=x.dtype, device=x.device) if fill is None else fill.to(x)
    channels = x.shape[2]
    for i, length in enumerate(lengths.tolist()):
        for _ in range(SPEC_TIME_MASKS):
            width = int(rng.integers(0, int(SPEC_TIME_FRACTION * length) + 1))
            start = int(rng.integers(0, max(1, length - width + 1)))
            x[i, start : start + width] = fill
        width = int(rng.integers(0, int(SPEC_CHANNEL_FRACTION * channels) + 1))
        start = int(rng.integers(0, channels - width + 1))
        x[i, :, start : start + width] = fill[start : start + width]
    return x.to(home)


def _batch(data: FrameData, rows: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames and lengths for `rows`, cropped to the longest clip in the batch."""
    lengths = torch.as_tensor(data.lengths[rows].astype(np.int64))
    width = int(lengths.max())
    x = torch.as_tensor(np.asarray(data.x[rows][:, :width], dtype=np.float32))
    return x.to(device), lengths.to(device)


def device_for(model: str) -> torch.device:
    """CPU for architectures that MPS handles poorly, else the best available device."""
    return torch.device("cpu" if model in CPU_ONLY else detect_device())


# --------------------------------------------------------------------------- fitting


@dataclass
class FitResult:
    """A trained network with the epoch count and inner-validation score that produced it."""

    model: nn.Module
    epochs: int
    val_f1: float
    seconds: float


def _channel_stats(data: FrameData, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and standard deviation over the real frames of `rows` (fitting clips only)."""
    frames = np.concatenate([np.asarray(data.x[i, : data.lengths[i]], dtype=np.float64) for i in rows])
    return (
        torch.as_tensor(frames.mean(0), dtype=torch.float32),
        torch.as_tensor(frames.std(0) + 1e-6, dtype=torch.float32),
    )


@torch.no_grad()
def predict(
    model: nn.Module, data: FrameData, rows: np.ndarray, device: torch.device
) -> dict[str, np.ndarray]:
    """Class probabilities, embeddings and (if the model has them) frame weights for `rows`."""
    model.eval()
    probas, embeddings, weights = [], [], []
    for start in range(0, len(rows), 64):
        chunk = rows[start : start + 64]
        x, lengths = _batch(data, chunk, device)
        logits, embedding = model(x, lengths)
        probas.append(torch.softmax(logits, -1).cpu().numpy())
        embeddings.append(embedding.cpu().numpy())
        frame_weights = getattr(model, "weights", None)
        if frame_weights is not None:
            padded = np.zeros((len(chunk), data.x.shape[1]), dtype=np.float32)
            padded[:, : frame_weights.shape[1]] = frame_weights.cpu().numpy()
            weights.append(padded)
    out = {"proba": np.concatenate(probas), "embedding": np.concatenate(embeddings)}
    if weights:
        out["weights"] = np.concatenate(weights)
    return out


def fit(
    model_name: str,
    data: FrameData,
    y: np.ndarray,
    speakers: np.ndarray,
    fit_rows: np.ndarray,
    hyper: dict,
    cfg: Config,
    *,
    val_rows: np.ndarray | None = None,
    epochs: int | None = None,
    seed: int = 0,
    deadline: float | None = None,
    specaug: bool = True,
    on_epoch=None,
) -> FitResult:
    """Train one network on `fit_rows`.

    With `val_rows`, trains up to `cfg.max_epochs`, stops after `cfg.patience`
    epochs without improvement in validation macro-F1 (never before `cfg.min_epochs`, so a slow start is not
    mistaken for a plateau) and restores the best weights. Without them, trains for exactly `epochs`. `deadline` (a
    `time.monotonic` value) ends training early and returns the best so far.
    """
    started = time.monotonic()
    set_all_seeds(cfg.seed + seed)
    rng = np.random.default_rng(cfg.seed + seed)
    device = device_for(model_name)
    speaker_codes = np.unique(speakers[fit_rows], return_inverse=True)[1]
    core = nets.build(
        model_name, data.channels, int(y.max()) + 1, hyper, n_speakers=int(speaker_codes.max()) + 1
    )
    fill = None
    if data.standardise:
        mean, std = _channel_stats(data, fit_rows)
        net = nets.Standardised(core, mean, std)
        fill = mean
    else:
        net = core
    net.to(device)
    adversarial = isinstance(core, nets.SpeakerAdversarial)
    optimiser = torch.optim.AdamW(net.parameters(), lr=hyper["lr"], weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    total = epochs if val_rows is None else cfg.max_epochs
    best_f1, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, int(total) + 1):
        net.train()
        order = rng.permutation(len(fit_rows))
        for start in range(0, len(order), BATCH_SIZE):
            pick = order[start : start + BATCH_SIZE]
            if len(pick) < 2:
                continue  # a single-clip batch is not worth a gradient step
            rows = fit_rows[pick]
            x, lengths = _batch(data, rows, device)
            if specaug:
                x = spec_augment(x, lengths, rng, fill)
            target = torch.as_tensor(y[rows]).to(device)
            if adversarial:
                progress = epoch / cfg.max_epochs  # one timetable for tuning and the final fit
                core.strength = hyper["adv_weight"] * (2.0 / (1.0 + float(np.exp(-10.0 * progress))) - 1.0)
                logits, embedding = net(x, lengths)
                speaker_target = torch.as_tensor(speaker_codes[pick]).to(device)
                loss = loss_fn(logits, target) + loss_fn(core.speaker_logits(embedding), speaker_target)
            else:
                logits, _ = net(x, lengths)
                loss = loss_fn(logits, target)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimiser.step()
        if val_rows is not None:
            proba = predict(net, data, val_rows, device)["proba"]
            score = float(f1_score(y[val_rows], proba.argmax(1), average="macro"))
            if on_epoch:
                on_epoch(epoch, score)
            if score > best_f1:
                best_f1, best_epoch, waited = score, epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                waited += 1
                if waited >= cfg.patience and epoch >= cfg.min_epochs:
                    break
        if deadline is not None and time.monotonic() > deadline and epoch >= cfg.min_epochs:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    done = best_epoch if val_rows is not None else int(epoch)
    return FitResult(
        net,
        max(1, done),
        float(best_f1) if val_rows is not None else float("nan"),
        time.monotonic() - started,
    )


# --------------------------------------------------------------------------- nested tuning


def inner_split(ctx: Context, train: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """One speaker-grouped inner split of the training actors: (fit rows, validation rows)."""
    splitter = GroupKFold(n_splits=cfg.inner_folds)
    fit_idx, val_idx = next(splitter.split(train, groups=ctx.speakers[train]))
    return train[fit_idx], train[val_idx]


def tune_fold(
    ctx: Context, model: str, data: FrameData, fold: int, timeout_s: float, specaug: bool, tag: str
) -> FoldPrediction:
    """Tune on one outer fold's training actors, retrain on all of them, predict the test fold once."""
    cfg = ctx.cfg
    train, test = ctx.folds[fold]
    fit_rows, val_rows = inner_split(ctx, train, cfg)
    for role, used in (("objective fit", fit_rows), ("early stopping", val_rows), ("final fit", train)):
        assert_disjoint(role, used, test)
    cfg.studies.mkdir(parents=True, exist_ok=True)
    version = study_tag(
        search_space_table().to_csv(),
        cfg.max_epochs,
        cfg.min_epochs,
        cfg.patience,
        cfg.inner_folds,
        data.x.shape,
    )
    label = f"deep__{model}__{tag}__fold{fold}__{version}"
    study = optuna.create_study(
        study_name=label,
        storage=f"sqlite:///{cfg.studies / (label + '.db')}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed + fold + zlib.crc32(model.encode()) % 1000),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=cfg.min_epochs),
    )
    started = time.monotonic()
    tuning_deadline = started + 0.5 * timeout_s  # the rest is for the final fit and the extra seeds

    def objective(trial: optuna.Trial) -> float:
        hyper = suggest_hyper(trial, model)

        def report(epoch: int, score: float) -> None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = fit(
            model, data, ctx.y, ctx.speakers, fit_rows, hyper, cfg,
            val_rows=val_rows, seed=trial.number, deadline=tuning_deadline, specaug=specaug, on_epoch=report,
        )  # fmt: skip
        trial.set_user_attr("epochs", result.epochs)
        return result.val_f1

    finished = sum(t.state.is_finished() for t in study.trials)
    if cfg.deep_trials > finished:
        study.optimize(
            objective,
            n_trials=cfg.deep_trials - finished,
            timeout=max(1.0, 0.5 * timeout_s),
        )
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    log.info("%s|%s fold %d: %d trials completed, %d pruned", model, tag, fold, len(done), pruned)
    if done:
        best = max(done, key=lambda t: t.value)
        hyper, epochs, inner = dict(best.params), int(best.user_attrs["epochs"]), float(best.value)
    else:
        log.warning("%s|%s fold %d: no finished trial within budget; using defaults", model, tag, fold)
        hyper, epochs, inner = default_hyper(model), max(1, cfg.max_epochs // 2), float("nan")
    final = fit(
        model, data, ctx.y, ctx.speakers, train, hyper, cfg, epochs=epochs, seed=100 + fold, specaug=specaug
    )
    out = predict(final.model, data, test, device_for(model))
    # Extra seeds retrain the chosen configuration. The first seed is the model that is scored; the rest
    # show how much the score depends on the random start.
    seed_pred = [out["proba"].argmax(1)]
    for extra in range(1, cfg.deep_seeds):
        again = fit(
            model, data, ctx.y, ctx.speakers, train, hyper, cfg,
            epochs=epochs, seed=100 + fold + 1000 * extra, specaug=specaug,
        )  # fmt: skip
        seed_pred.append(predict(again.model, data, test, device_for(model))["proba"].argmax(1))
    extras = {"embedding": out["embedding"], "seed_pred": np.array(seed_pred)}
    if "weights" in out:
        extras["weights"] = out["weights"]
    params = {
        **hyper,
        "epochs": epochs,
        "n_params": nets.count_parameters(final.model),
        "trials_completed": len(done),
        "trials_pruned": pruned,
    }
    return FoldPrediction(fold, test, out["proba"].argmax(1).astype(int), out["proba"], params, inner,
                          time.monotonic() - started, extras)  # fmt: skip


def run_nested(
    ctx: Context,
    store: RunStore,
    model: str,
    data: FrameData,
    family: str,
    timeout_s: float,
    *,
    specaug: bool = True,
    suffix: str = "",
) -> ModelRun:
    """Nested-tuned neural model over all outer folds, loaded from disk when already done."""
    name = f"{model}|{data.name}{suffix}"
    if store.exists(name):
        log.info("%s: loaded from checkpoint", name)
        return store.load(name)
    log.info("%s: tuning %d folds", name, len(ctx.folds))
    torch.set_num_threads(os.cpu_count() or 1)
    per_fold = timeout_s / len(ctx.folds)
    parts = [tune_fold(ctx, model, data, f, per_fold, specaug, data.name + suffix) for f in ctx.folds]
    note = f"Optuna nested, {ctx.cfg.deep_trials} trials per fold, {'with' if specaug else 'without'} SpecAugment"
    run = ModelRun(name, family, data.name, {p.fold: p for p in parts}, notes=note)
    store.save(run)
    return run
