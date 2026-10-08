"""Gate G4: no test-fold row reaches an objective, scaler, augmenter, calibrator or early-stopping monitor.

Two layers. During a run, `assert_disjoint` stops the program the moment a row set used for fitting,
tuning, early stopping or calibration overlaps the outer test rows. Afterwards, `verify_runs` rebuilds every
partition for every saved model (the inner splits are deterministic) and checks it against the stored
predictions, so the guarantee also covers models loaded from checkpoints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from unheard_lib.data import Context
from unheard_lib.runtime import GateFailure
from unheard_lib.store import RunStore


def assert_disjoint(role: str, used: np.ndarray, test: np.ndarray) -> None:
    """Raise if any row in `used` is also a test row of the current outer fold."""
    overlap = np.intersect1d(used, test)
    if len(overlap):
        raise GateFailure(f"G4: {len(overlap)} test rows reached '{role}'")


def verify_runs(ctx: Context, store: RunStore) -> pd.DataFrame:
    """Rebuild each saved model's fitting, tuning and validation rows and check them against its test rows.

    Returns one row per (model, fold). Raises `GateFailure` on any overlap or any stored prediction that
    does not cover exactly that fold's test clips.
    """
    from unheard_lib import deep  # imported here: deep itself imports this module

    rows = []
    for name in store.names():
        run = store.load(name)
        for fold, part in run.folds.items():
            train, test = ctx.folds[fold]
            if not np.array_equal(np.sort(part.rows), np.sort(test)):
                raise GateFailure(f"G4: {name} fold {fold} predictions do not cover the fold's test clips")
            fit_rows, val_rows = deep.inner_split(ctx, train, ctx.cfg)
            inner = [
                (train[a], train[b])
                for a, b in GroupKFold(ctx.cfg.inner_folds).split(train, groups=ctx.speakers[train])
            ]
            for role, used in (
                ("final fit and scaler", train),
                ("early-stopping split, fit part", fit_rows),
                ("early-stopping split, validation part", val_rows),
                *[
                    (f"inner fold {k} ({part_name})", r)
                    for k, pair in enumerate(inner)
                    for part_name, r in zip(("fit", "validation"), pair, strict=True)
                ],
            ):
                assert_disjoint(role, used, test)
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "test clips": len(test),
                    "training clips": len(train),
                    "test rows used in training or tuning": 0,
                }
            )
    return pd.DataFrame(rows)
