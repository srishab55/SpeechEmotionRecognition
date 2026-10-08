"""Checkpointing: every model's out-of-fold predictions are saved to disk.

Reruns load a finished model instead of retraining it, and the comparison
stage reads only these files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from unheard_lib.data import Context


@dataclass
class FoldPrediction:
    """Predictions for one outer fold's test clips."""

    fold: int
    rows: np.ndarray  # row indices into the clip table
    pred: np.ndarray  # predicted integer labels
    proba: np.ndarray | None = None
    params: dict = field(default_factory=dict)
    inner_score: float = float("nan")
    seconds: float = 0.0
    extras: dict = field(default_factory=dict)  # named arrays, e.g. test embeddings or attention weights


@dataclass
class ModelRun:
    """All folds of one model on one feature set."""

    name: str
    family: str
    features: str
    folds: dict[int, FoldPrediction]
    notes: str = ""

    def pooled(self, ctx: Context) -> dict[str, np.ndarray]:
        """Concatenate the folds in order: rows, true labels, predictions, speakers, probabilities."""
        parts = [self.folds[f] for f in sorted(self.folds)]
        rows = np.concatenate([p.rows for p in parts])
        proba = None
        if all(p.proba is not None for p in parts):
            proba = np.concatenate([p.proba for p in parts])
        return {
            "rows": rows,
            "y_true": ctx.y[rows],
            "y_pred": np.concatenate([p.pred for p in parts]),
            "speakers": ctx.speakers[rows],
            "proba": proba,
        }


def slug(name: str) -> str:
    """File-system-safe version of a model name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


class RunStore:
    """Read and write `ModelRun` objects under `outputs/predictions/`."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, name: str) -> tuple[Path, Path]:
        base = self.dir / slug(name)
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def exists(self, name: str) -> bool:
        return all(p.is_file() for p in self._paths(name))

    def save(self, run: ModelRun) -> None:
        arrays, meta = (
            {},
            {
                "name": run.name,
                "family": run.family,
                "features": run.features,
                "notes": run.notes,
                "folds": {},
            },
        )
        for fold, p in run.folds.items():
            arrays[f"rows_{fold}"], arrays[f"pred_{fold}"] = p.rows, p.pred
            if p.proba is not None:
                arrays[f"proba_{fold}"] = p.proba.astype(np.float32)
            for key, value in p.extras.items():
                arrays[f"extra__{key}__{fold}"] = np.asarray(value, dtype=np.float32)
            meta["folds"][str(fold)] = {
                "params": p.params,
                "inner_score": p.inner_score,
                "seconds": p.seconds,
            }
        npz, js = self._paths(run.name)
        np.savez_compressed(npz, **arrays)
        js.write_text(json.dumps(meta, indent=1, default=str))

    def load(self, name: str) -> ModelRun:
        npz, js = self._paths(name)
        meta, data = json.loads(js.read_text()), np.load(npz)
        folds = {}
        for key, info in meta["folds"].items():
            fold = int(key)
            proba = data[f"proba_{fold}"] if f"proba_{fold}" in data else None  # noqa: SIM401 (NpzFile has no .get)
            folds[fold] = FoldPrediction(
                fold,
                data[f"rows_{fold}"],
                data[f"pred_{fold}"],
                proba,
                info["params"],
                info["inner_score"],
                info["seconds"],
                {
                    k.split("__")[1]: data[k]
                    for k in data.files
                    if k.startswith("extra__") and k.endswith(f"__{fold}")
                },
            )
        return ModelRun(meta["name"], meta["family"], meta["features"], folds, meta.get("notes", ""))

    def names(self) -> list[str]:
        return sorted(json.loads(p.read_text())["name"] for p in self.dir.glob("*.json"))
