"""Train the CNN-BiLSTM on all four corpora with whole speakers held out, and score it.

Folds are the ones the SVM/logistic extension study (Part C) used: `StratifiedGroupKFold(5)` over speakers,
stratified on corpus x emotion, seeds 42-44. The clip table is in the same order as that study's, so the held-out
speakers are identical and the comparison is paired.

Per fold: the training speakers are split once more by speaker (about one sixth become validation speakers).
The network trains on the rest, early stopping watches macro-F1 on the validation speakers, the best weights are
restored, and the held-out fold is predicted exactly once. No held-out clip reaches the optimiser, the
early-stopping monitor, SpecAugment or the normaliser (frames were z-scored per speaker from the speaker's own
audio, no labels).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from multicorpus import CORPORA, LABELS, ROOT
from multicorpus import frames as frame_cache
from multicorpus.corpora import load_index
from unheard_lib import deep
from unheard_lib.config import get_config
from unheard_lib.deep import FrameData
from unheard_lib.runtime import set_all_seeds

SEEDS = (42, 43, 44)
MODEL = "cnn_bilstm_w2v"  # frame projection (768 -> width), 2 conv blocks, BiLSTM
HYPER = {"lr": 1e-3, "width": 128, "dropout": 0.3}
OUT = ROOT / "outputs" / "cnn_bilstm"


def folds_for(df: pd.DataFrame, seed: int):
    grp = df.speaker_id.to_numpy()
    strat = pd.factorize(df.source + "|" + df.label_idx.astype(str))[0]
    return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(df)), strat, grp))


def run(smoke: bool = False, device: str | None = None, max_epochs: int = 40, patience: int = 6) -> None:
    df = load_index()
    x, lengths = frame_cache.load()
    n = len(lengths)
    assert n == len(df) or smoke, f"frame cache has {n} clips, index has {len(df)}"
    y = df.label_idx.to_numpy()
    speakers = df.speaker_id.to_numpy()
    strat = pd.factorize(df.source + "|" + df.label_idx.astype(str))[0]
    data = FrameData("w2v_frames_L6", x, lengths)
    cfg = replace(get_config("smoke"), max_epochs=2 if smoke else max_epochs, min_epochs=1 if smoke else 10,
                  patience=patience)
    device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    deep.device_for = lambda model: torch.device(device)  # packed LSTM output matches the CPU to 1e-7 on MPS
    print(f"training {MODEL} on {device}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[:1] if smoke else SEEDS
    for seed in seeds:
        for fold, (train, test) in enumerate(folds_for(df, seed)):
            path = OUT / f"pred_seed{seed}_fold{fold}.npz"
            if path.exists() and not smoke:
                continue
            assert not set(speakers[train]) & set(speakers[test]), "speaker leaked across train and test"
            if smoke:  # a few hundred clips from the cache that exists, keeping the speaker structure
                train, test = train[train < n][:160], test[test < n][:60]
                if len(test) == 0 or len(train) == 0:
                    continue
            inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + fold)
            try:
                fit_i, val_i = next(inner.split(np.zeros(len(train)), strat[train], speakers[train]))
            except ValueError:
                fit_i, val_i = np.arange(len(train))[: len(train) * 5 // 6], np.arange(len(train))[len(train) * 5 // 6 :]
            fit_rows, val_rows = train[fit_i], train[val_i]
            assert not set(speakers[fit_rows]) & set(speakers[val_rows]) or smoke
            t0 = time.time()
            set_all_seeds(seed)
            res = deep.fit(MODEL, data, y, speakers, fit_rows, HYPER, cfg, val_rows=val_rows, seed=seed + fold)
            dev = deep.device_for(MODEL)
            proba = deep.predict(res.model, data, test, dev)["proba"]
            np.savez(path.with_suffix(".tmp.npz") if not smoke else OUT / "smoke.npz", rows=test, proba=proba,
                     epochs=res.epochs, val_f1=res.val_f1)
            if not smoke:
                path.with_suffix(".tmp.npz").rename(path)
            acc = float((proba.argmax(1) == y[test]).mean())
            print(f"seed {seed} fold {fold}: test clips {len(test)}, best epoch {res.epochs}, "
                  f"val macro-F1 {res.val_f1:.3f}, test acc {acc:.3f}, {time.time() - t0:.0f}s", flush=True)
    if smoke:
        print("smoke run finished")


def _macro_f1_present(conf: np.ndarray) -> float:
    """Macro-F1 over the labels that occur in the true labels (a corpus lacks some emotions)."""
    tp = np.diag(conf).astype(float)
    support, predicted = conf.sum(1), conf.sum(0)
    present = support > 0
    f1 = np.where(support + predicted > 0, 2 * tp / np.maximum(support + predicted, 1), 0.0)
    return float(f1[present].mean())


def score(n_boot: int = 2000, out=None, label: str = "cnn_bilstm|w2v_frames") -> pd.DataFrame:
    """Pool the out-of-fold predictions per seed, then report per set with a speaker-bootstrap interval.

    `out` is the folder holding `pred_seed*_fold*.npz` (default: the CNN-BiLSTM folder); `label` names the model.
    """
    out = out or OUT
    tag = "cnn_bilstm" if out == OUT else out.name
    df = load_index()
    y = df.label_idx.to_numpy()
    k = len(LABELS)
    oof = {}
    for seed in SEEDS:
        pred = np.full(len(df), -1)
        for fold in range(5):
            f = out / f"pred_seed{seed}_fold{fold}.npz"
            if f.exists():
                z = np.load(f)
                pred[z["rows"]] = z["proba"].argmax(1)
        oof[seed] = pred
    seeds = [s for s in SEEDS if (oof[s] >= 0).all()]
    if not seeds:
        raise SystemExit("no seed has all five folds yet; run training first")
    sets = {"ALL four corpora": np.ones(len(df), bool), **{c: (df.source == c).to_numpy() for c in CORPORA}}
    rng = np.random.default_rng(42)
    rows, per_emotion = [], []
    for name, mask in sets.items():
        spk = df.speaker_id.to_numpy()[mask]
        names, inv = np.unique(spk, return_inverse=True)
        conf = np.zeros((len(seeds), len(names), k, k))
        for si, s in enumerate(seeds):
            np.add.at(conf[si], (inv, y[mask], oof[s][mask]), 1)
        pooled = conf.sum(1)
        per_seed = [_macro_f1_present(c) for c in pooled]
        lo = hi = np.nan
        if len(names) >= 8:
            draws = []
            for _ in range(n_boot):
                pick = rng.integers(0, len(names), len(names))
                draws.append(np.mean([_macro_f1_present(conf[si, pick].sum(0)) for si in range(len(seeds))]))
            lo, hi = np.percentile(draws, [2.5, 97.5])
        acc = np.mean([np.trace(c) / c.sum() for c in pooled])
        rows.append({"model": label, "set": name, "clips": int(mask.sum()),
                     "speakers": len(names), "macro_f1_mean_of_seeds": np.mean(per_seed),
                     "sd_over_seeds": np.std(per_seed), "lo": lo, "hi": hi, "accuracy": acc, "seeds": len(seeds)})
        tp, sup, prd = np.diag(pooled.sum(0)), pooled.sum(0).sum(1), pooled.sum(0).sum(0)
        for ci, lab in enumerate(LABELS):
            if sup[ci] > 0:
                per_emotion.append({"set": name, "emotion": lab, "support": int(sup[ci]) // len(seeds),
                                    "f1": 2 * tp[ci] / (sup[ci] + prd[ci])})
    res = pd.DataFrame(rows)
    res.to_csv(out / f"{tag}_partC_style.csv", index=False)
    pd.DataFrame(per_emotion).to_csv(out / f"{tag}_per_emotion_f1.csv", index=False)
    ref = pd.read_csv(ROOT / "reference" / "summary_partC.csv").set_index("model")
    table = res.set_index("set")["macro_f1_mean_of_seeds"].round(3).to_frame(label).T
    table = pd.concat([table, ref[table.columns]])
    table.to_csv(out / "comparison_with_svm_logistic.csv")
    (out / "summary.json").write_text(json.dumps({"seeds_scored": seeds}, indent=1))
    print(res.round(3).to_string(index=False))
    print(f"\nMacro-F1 (mean of seeds): {label} vs the earlier pooled-feature models, same folds")
    print(table.to_string())
    return res


def ensemble(dirs, out, seeds=SEEDS) -> list[int]:
    """Average the held-out probabilities of several models, fold by fold, and save them as a new model folder.

    Only (seed, fold) pairs present in every folder are used, and their test rows must be identical. Averaging
    probabilities is label-free and fitted on nothing, so it adds no way for a test row to leak.
    Returns the seeds for which all five folds could be combined.
    """
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for seed in seeds:
        n = 0
        for fold in range(5):
            files = [d / f"pred_seed{seed}_fold{fold}.npz" for d in dirs]
            if not all(f.exists() for f in files):
                continue
            zs = [np.load(f) for f in files]
            assert all((z["rows"] == zs[0]["rows"]).all() for z in zs), "models were tested on different rows"
            np.savez(out / f"pred_seed{seed}_fold{fold}.npz", rows=zs[0]["rows"],
                     proba=np.mean([z["proba"] for z in zs], axis=0))
            n += 1
        if n == 5:
            done.append(seed)
    return done


def compare_all() -> pd.DataFrame:
    """One table: every scored model folder in `outputs/` next to the SVM and logistic regression from Part C."""
    ref = pd.read_csv(ROOT / "reference" / "summary_partC.csv").set_index("model")
    rows = {}
    for f in sorted((ROOT / "outputs").glob("*/*_partC_style.csv")):
        r = pd.read_csv(f)
        rows[r["model"].iloc[0]] = r.set_index("set")["macro_f1_mean_of_seeds"].round(3)
    table = pd.concat([pd.DataFrame(rows).T[ref.columns], ref]).sort_values("ALL four corpora", ascending=False)
    table.to_csv(ROOT / "outputs" / "comparison_all_models.csv")
    return table
