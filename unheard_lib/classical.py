"""Classical machine-learning models with nested Optuna tuning (nodes N7 and N10).

For every outer fold the tuning sees only that fold's training actors, split
again into speaker-grouped inner folds. The outer test clips are used once, to
predict with the tuned model. Scalers live inside the model pipeline, so they
are fitted on the rows the pipeline is fitted on and nothing else.
"""

from __future__ import annotations

import logging
import math
import os
import time
import warnings
from collections.abc import Callable

import numpy as np
import optuna
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from unheard_lib.access import assert_disjoint
from unheard_lib.config import (
    MODEL_TIERS,
    SEARCH_SPACE,
    W2V_LAYERS,
    W2V_POOLINGS,
    Config,
    study_tag,
)
from unheard_lib.data import Context
from unheard_lib.store import FoldPrediction, ModelRun, RunStore

# Tuning deliberately tries short, under-trained settings; a non-converged trial simply scores
# badly and is discarded, so the warning is noise here.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

log = logging.getLogger("unheard")
optuna.logging.set_verbosity(optuna.logging.WARNING)

#: Used when a time budget ends before a study finishes a single trial.
DEFAULTS: dict[str, dict] = {
    "logistic": {"C": 0.01},
    "svm_rbf": {"C": 1.0, "gamma": 1e-3},
    "svm_linear": {"C": 0.01},
    "random_forest": {"n_estimators": 200, "max_depth": 0, "max_features": "sqrt"},
    "xgboost": {"learning_rate": 0.1, "max_depth": 4, "n_estimators": 150, "colsample_bytree": 0.5},
    "knn": {"n_neighbors": 7, "weights": "uniform"},
    "lda": {"shrinkage": 0.5},
    "gaussian_nb": {"var_smoothing": 1e-3},
    "adaboost": {"n_estimators": 100, "learning_rate": 0.5},
    "mlp": {"hidden": 128, "alpha": 1e-2},
}
HAS_PROBA = {m: True for m in DEFAULTS} | {"svm_linear": False}


def suggest_params(trial: optuna.Trial, model: str) -> dict:
    """Draw one setting for `model` from the search-space table in `config.py`."""
    params: dict = {}
    for row in (r for r in SEARCH_SPACE if r.model == model):
        if row.kind == "float":
            params[row.param] = trial.suggest_float(row.param, row.low, row.high, log=row.log)
        elif row.kind == "int":
            params[row.param] = trial.suggest_int(row.param, int(row.low), int(row.high))
        else:
            params[row.param] = trial.suggest_categorical(row.param, list(row.choices))
    return params


def build_estimator(model: str, params: dict, cfg: Config, n_jobs: int = 1):
    """Scikit-learn estimator for `model` with the given hyperparameters."""
    seed, cap = cfg.seed, cfg.n_estimators_cap
    if model == "logistic":
        return LogisticRegression(C=params["C"], max_iter=3000, random_state=seed)
    if model == "svm_rbf":
        return SVC(C=params["C"], gamma=params["gamma"], kernel="rbf", probability=True, random_state=seed)
    if model == "svm_linear":
        return LinearSVC(C=params["C"], max_iter=2000, dual="auto", random_state=seed)
    if model == "random_forest":
        depth = params["max_depth"] or None
        return RandomForestClassifier(
            n_estimators=min(params["n_estimators"], cap),
            max_depth=depth,
            max_features=params["max_features"],
            n_jobs=n_jobs,
            random_state=seed,
        )
    if model == "xgboost":
        return XGBClassifier(
            n_estimators=min(params["n_estimators"], cap),
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            colsample_bytree=params["colsample_bytree"],
            subsample=0.8,
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=0,
        )
    if model == "knn":
        return KNeighborsClassifier(n_neighbors=params["n_neighbors"], weights=params["weights"])
    if model == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=params["shrinkage"])
    if model == "gaussian_nb":
        return GaussianNB(var_smoothing=params["var_smoothing"])
    if model == "adaboost":
        return AdaBoostClassifier(
            n_estimators=min(params["n_estimators"], cap),
            learning_rate=params["learning_rate"],
            random_state=seed,
        )
    if model == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(params["hidden"],),
            alpha=params["alpha"],
            max_iter=60 if cfg.preset == "smoke" else 400,
            random_state=seed,
        )
    raise ValueError(f"unknown model {model!r}")


def make_pipeline_for(model: str, params: dict, cfg: Config, n_jobs: int = 1) -> Pipeline:
    """Scaler plus estimator; the scaler is fitted only on the rows passed to `fit`."""
    return make_pipeline(StandardScaler(), build_estimator(model, params, cfg, n_jobs))


Featurizer = Callable[[dict], np.ndarray]


def fixed_features(matrix: np.ndarray) -> Featurizer:
    """Featurizer for a feature set with no tunable choices."""
    return lambda params: matrix


def w2v_features(pooled) -> Featurizer:
    """Featurizer whose layer and pooling come from the trial's parameters."""
    return lambda params: pooled.get(params["layer"], params["pooling"])


def _split_params(params: dict, uses_w2v: bool) -> tuple[dict, dict]:
    """Separate feature-choice parameters (layer, pooling) from model hyperparameters."""
    feature_keys = {"layer", "pooling"} if uses_w2v else set()
    return (
        {k: v for k, v in params.items() if k in feature_keys},
        {k: v for k, v in params.items() if k not in feature_keys},
    )


def _objective(
    ctx: Context, model: str, featurize: Featurizer, uses_w2v: bool, train: np.ndarray, cfg: Config
) -> Callable[[optuna.Trial], float]:
    """Inner-validation objective: mean macro-F1 over speaker-grouped inner folds."""
    inner = list(GroupKFold(n_splits=cfg.inner_folds).split(train, groups=ctx.speakers[train]))

    def objective(trial: optuna.Trial) -> float:
        hyper = suggest_params(trial, model)
        choice = {}
        if uses_w2v:
            choice = {
                "layer": trial.suggest_int("layer", *W2V_LAYERS),
                "pooling": trial.suggest_categorical("pooling", list(W2V_POOLINGS)),
            }
        matrix = featurize(choice)
        scores = []
        for step, (fit_idx, val_idx) in enumerate(inner):
            fit_rows, val_rows = train[fit_idx], train[val_idx]
            fitted = make_pipeline_for(model, hyper, cfg).fit(matrix[fit_rows], ctx.y[fit_rows])
            scores.append(f1_score(ctx.y[val_rows], fitted.predict(matrix[val_rows]), average="macro"))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def tune_fold(
    ctx: Context,
    model: str,
    feature_name: str,
    featurize: Featurizer,
    uses_w2v: bool,
    fold: int,
    timeout_s: float,
) -> FoldPrediction:
    """Tune on one outer fold's training actors, then predict its test clips once."""
    cfg = ctx.cfg
    train, test = ctx.folds[fold]
    assert_disjoint("tuning and final fit", train, test)
    cfg.studies.mkdir(parents=True, exist_ok=True)
    tag = study_tag(SEARCH_SPACE, W2V_LAYERS, W2V_POOLINGS, cfg.n_estimators_cap, cfg.inner_folds)
    label = f"{model}__{feature_name}__fold{fold}__{tag}"
    db = cfg.studies / f"{label}.db"
    study = optuna.create_study(
        study_name=label,
        storage=f"sqlite:///{db}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed + fold),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    started = time.monotonic()
    finished = sum(t.state.is_finished() for t in study.trials)
    remaining = max(0, cfg.trials_for(model) - finished)
    if remaining:
        study.optimize(
            _objective(ctx, model, featurize, uses_w2v, train, cfg),
            n_trials=remaining,
            timeout=max(1.0, timeout_s),
        )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best, inner_score = study.best_params, float(study.best_value)
    else:
        log.warning("%s fold %d: no finished trial within budget; using defaults", model, fold)
        best = dict(DEFAULTS[model]) | ({"layer": 6, "pooling": "mean_std"} if uses_w2v else {})
        inner_score = float("nan")
    choice, hyper = _split_params(best, uses_w2v)
    matrix = featurize(choice)
    fitted = make_pipeline_for(model, hyper, cfg).fit(matrix[train], ctx.y[train])
    proba = fitted.predict_proba(matrix[test]) if HAS_PROBA[model] else None
    return FoldPrediction(
        fold,
        test,
        fitted.predict(matrix[test]).astype(int),
        proba,
        best,
        inner_score,
        time.monotonic() - started,
    )


def run_nested(
    ctx: Context,
    store: RunStore,
    model: str,
    feature_name: str,
    featurize: Featurizer,
    family: str,
    uses_w2v: bool,
    timeout_s: float,
    workers: int | None = None,
) -> ModelRun:
    """Nested-tuned model over all outer folds, loaded from disk when already done."""
    name = f"{model}|{feature_name}"
    if store.exists(name):
        log.info("%s: loaded from checkpoint", name)
        return store.load(name)
    workers = workers or min(len(ctx.folds), os.cpu_count() or 1)
    rounds = math.ceil(len(ctx.folds) / workers)
    log.info("%s: tuning %d folds on %d workers", name, len(ctx.folds), workers)
    parts = Parallel(n_jobs=workers, prefer="threads")(
        delayed(tune_fold)(ctx, model, feature_name, featurize, uses_w2v, f, timeout_s / rounds)
        for f in ctx.folds
    )
    run = ModelRun(
        name,
        family,
        feature_name,
        {p.fold: p for p in parts},
        notes=f"Optuna nested, {ctx.cfg.trials_for(model)} trials per fold",
    )
    store.save(run)
    return run


def run_fixed(
    ctx: Context,
    store: RunStore,
    name: str,
    family: str,
    feature_name: str,
    matrix: np.ndarray,
    model: str,
    params: dict,
) -> ModelRun:
    """A model with fixed hyperparameters (used for the untuned baseline)."""
    if store.exists(name):
        return store.load(name)
    folds = {}
    for fold, (train, test) in ctx.folds.items():
        started = time.monotonic()
        fitted = make_pipeline_for(model, params, ctx.cfg).fit(matrix[train], ctx.y[train])
        proba = fitted.predict_proba(matrix[test]) if HAS_PROBA[model] else None
        folds[fold] = FoldPrediction(
            fold,
            test,
            fitted.predict(matrix[test]).astype(int),
            proba,
            params,
            seconds=time.monotonic() - started,
        )
    run = ModelRun(name, family, feature_name, folds, notes="fixed hyperparameters, no tuning")
    store.save(run)
    return run


def ordered_models() -> list[str]:
    """Model names in execution order, cheapest and most important tier first."""
    return sorted(MODEL_TIERS, key=lambda m: (MODEL_TIERS[m], m))
