"""Configuration: presets, palette, search spaces and statistical constants.

Everything that changes between a five-minute smoke test and the overnight
run is a field of `Config`, so the code paths are identical and only the
budgets differ.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import plotly.express as px

#: The default palette for every figure (notebook, script and PDF).
PALETTE: list[str] = list(px.colors.qualitative.T10)

#: Colour encodes model family, fixed once so a family looks the same in
#: every figure. The baseline takes the neutral grey of the palette.
MODEL_COLOR: dict[str, str] = {
    "Baseline": PALETTE[9],
    "Classical (hand-designed)": PALETTE[0],
    "Classical (wav2vec2)": PALETTE[4],
    "Sequence (from scratch)": PALETTE[1],
    "Attention": PALETTE[6],
    "Uniform control": PALETTE[8],
    "Speaker-adversarial": PALETTE[3],
}

#: A model is called better only if it wins by more than this many
#: percentage points of macro-F1 AND the paired speaker-clustered interval
#: excludes zero. With 24 test speakers, intervals run about +-2 to +-4 pp.
MIN_DETECTABLE_PP: float = 2.0

ROOT: Path = Path(__file__).resolve().parent.parent


def study_tag(*parts: object) -> str:
    """Short hash of whatever defines a search, so an Optuna study is reused only for the same search."""
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:8]


#: Reference figure from the mid-review: SVM-RBF with library defaults on 162 hand-designed
#: features, per-speaker normalised, pooled out-of-fold macro-F1.
BASELINE_MACRO_F1: float = 0.6333
BASELINE_TOLERANCE_PP: float = 2.0


@dataclass(frozen=True)
class SearchRow:
    """One line of the search-space table shown in the notebook."""

    model: str
    param: str
    kind: str  # "float", "int" or "cat"
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: tuple = ()


#: Single source of truth for tuning. `suggest_params` reads these rows and
#: the notebook prints them, so the documented ranges cannot drift from the
#: ranges actually searched.
SEARCH_SPACE: tuple[SearchRow, ...] = (
    SearchRow("logistic", "C", "float", 1e-3, 1e2, log=True),
    SearchRow("svm_rbf", "C", "float", 1e-2, 1e2, log=True),
    SearchRow("svm_rbf", "gamma", "float", 1e-4, 1e-1, log=True),
    SearchRow("svm_linear", "C", "float", 1e-4, 1e-1, log=True),
    SearchRow("random_forest", "n_estimators", "int", 100, 400),
    SearchRow("random_forest", "max_depth", "cat", choices=(4, 8, 16, 0)),
    SearchRow("random_forest", "max_features", "cat", choices=("sqrt", "log2", 0.3)),
    SearchRow("xgboost", "learning_rate", "float", 0.02, 0.3, log=True),
    SearchRow("xgboost", "max_depth", "int", 3, 8),
    SearchRow("xgboost", "n_estimators", "int", 100, 300),
    SearchRow("xgboost", "colsample_bytree", "float", 0.3, 1.0),
    SearchRow("knn", "n_neighbors", "int", 3, 25),
    SearchRow("knn", "weights", "cat", choices=("uniform", "distance")),
    SearchRow("lda", "shrinkage", "float", 0.01, 1.0, log=True),
    SearchRow("gaussian_nb", "var_smoothing", "float", 1e-10, 1e-1, log=True),
    SearchRow("adaboost", "n_estimators", "int", 50, 300),
    SearchRow("adaboost", "learning_rate", "float", 0.05, 1.5, log=True),
    SearchRow("mlp", "hidden", "cat", choices=(64, 128, 256)),
    SearchRow("mlp", "alpha", "float", 1e-5, 1e-1, log=True),
)

#: Extra search dimensions for models fed pooled wav2vec2 features. The layer
#: and pooling are chosen by inner validation, never on the test folds.
W2V_LAYERS: tuple[int, int] = (3, 9)
W2V_POOLINGS: tuple[str, ...] = ("mean", "mean_std")

#: Classical model tiers: execution order under a wall-clock budget.
MODEL_TIERS: dict[str, int] = {
    "logistic": 1,
    "svm_rbf": 1,
    "random_forest": 2,
    "xgboost": 2,
    "mlp": 2,
    "svm_linear": 2,
    "knn": 3,
    "lda": 3,
    "gaussian_nb": 3,
    "adaboost": 3,
}

#: Models that are slow enough to deserve a smaller trial budget.
HEAVY_MODELS: frozenset[str] = frozenset({"random_forest", "xgboost", "mlp", "adaboost"})

#: The only comparisons that get a verdict. Fixed here, before any result exists; everything else in the
#: report is descriptive. Each side is either a run name or `best:<family>` (the highest pooled macro-F1
#: in that family, chosen from the same predictions being compared, so the comparison is slightly
#: optimistic for the winner; the selection-bias estimate is in the conclusions).
FORMAL_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("best:Classical (hand-designed)", "baseline", "tuned hand-designed classical vs baseline"),
    ("best:Classical (wav2vec2)", "baseline", "tuned wav2vec2 classical vs baseline"),
    ("best:Sequence (from scratch)", "baseline", "best sequence model vs baseline"),
    ("best:Attention", "baseline", "best attention model vs baseline"),
    ("best:Attention", "best:Classical (wav2vec2)", "best attention model vs best pooled wav2vec2 classical"),
    (
        "best:Sequence (from scratch)",
        "best:Classical (hand-designed)",
        "best sequence model vs best hand-designed classical",
    ),
    ("best:Speaker-adversarial", "best:Attention", "speaker-adversarial vs best attention model"),
    ("best:Attention", "best:Uniform control", "best attention model vs uniform-weights control"),
)

#: Presets. Minutes are wall-clock budgets per stage (see `Config.budget`).
_PRESETS: dict[str, dict] = {
    "smoke": dict(
        n_trials_cheap=3,
        n_trials_heavy=2,
        inner_folds=3,
        n_bootstrap=200,
        n_estimators_cap=40,
        deep_trials=2,
        deep_seeds=1,
        max_epochs=6,
        min_epochs=2,
        min_task_min=0,
        patience=3,
        stage_budget_min=dict(classical=40, attention=20, sequence=20, report=10),
    ),
    "overnight": dict(
        n_trials_cheap=25,
        n_trials_heavy=10,
        inner_folds=3,
        n_bootstrap=2000,
        n_estimators_cap=400,
        deep_trials=8,
        deep_seeds=3,
        max_epochs=60,
        min_epochs=20,
        min_task_min=15,
        patience=8,
        stage_budget_min=dict(classical=90, attention=240, sequence=300, report=60),
    ),
    "full": dict(
        n_trials_cheap=50,
        n_trials_heavy=25,
        inner_folds=3,
        n_bootstrap=5000,
        n_estimators_cap=600,
        deep_trials=20,
        deep_seeds=5,
        max_epochs=100,
        min_epochs=25,
        min_task_min=30,
        patience=8,
        stage_budget_min=dict(classical=240, attention=480, sequence=480, report=120),
    ),
}


@dataclass(frozen=True)
class Config:
    """Everything a run needs to know, fixed at construction."""

    preset: str = "smoke"
    seed: int = 42
    protocol: str = "speaker_independent"
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_trials_cheap: int = 3
    n_trials_heavy: int = 2
    inner_folds: int = 3
    n_bootstrap: int = 200
    n_estimators_cap: int = 40
    deep_trials: int = 2
    deep_seeds: int = 1
    max_epochs: int = 6
    min_epochs: int = 2
    min_task_min: int = 0  # a neural task whose share of the stage budget is smaller than this is deferred
    patience: int = 3
    stage_budget_min: dict = field(default_factory=dict)
    root: Path = ROOT

    @property
    def outputs(self) -> Path:
        """Per-preset results folder, so a smoke run can never be mistaken for an overnight one."""
        return self.root / "outputs" / self.preset

    @property
    def cache(self) -> Path:
        """Preset-independent caches (the clip audit)."""
        return self.root / "outputs" / "cache"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def predictions(self) -> Path:
        return self.outputs / "predictions"

    @property
    def studies(self) -> Path:
        return self.outputs / "studies"

    def trials_for(self, model: str) -> int:
        """Trial budget for one classical model."""
        return self.n_trials_heavy if model in HEAVY_MODELS else self.n_trials_cheap

    def budget_seconds(self, stage: str) -> float:
        """Wall-clock budget for a stage, in seconds.

        The environment variable `UNHEARD_BUDGET_<STAGE>_MIN` (for example `UNHEARD_BUDGET_SEQUENCE_MIN=120`)
        overrides the preset, which is how a resumed run is given a budget that fits the work still to do.
        """
        override = os.environ.get(f"UNHEARD_BUDGET_{stage.upper()}_MIN")
        return 60.0 * float(override if override else self.stage_budget_min.get(stage, 30))


def get_config(preset: str = "smoke", root: Path | None = None) -> Config:
    """Build the `Config` for a preset name.

    An explicit `root` wins; otherwise the `UNHEARD_ROOT` environment variable,
    then the package's parent folder. Tests pass a temporary root.
    """
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(_PRESETS)}")
    base = Path(root) if root else Path(os.environ.get("UNHEARD_ROOT", ROOT))
    return Config(preset=preset, root=base, **_PRESETS[preset])
