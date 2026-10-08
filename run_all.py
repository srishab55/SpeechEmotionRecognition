"""One command for the whole experiment: CNN-BiLSTM on RAVDESS + CREMA-D + TESS + SAVEE, whole speakers held out.

    python run_all.py                 # everything: index -> frames -> train (3 seeds x 5 folds) -> report
    python run_all.py --stage index   # or: frames | train | report
    python run_all.py --device mps    # force a device, e.g. cpu

Every stage is resumable: finished frames and finished (seed, fold) models are skipped on a rerun.
"""

from __future__ import annotations

import argparse

from multicorpus import frames, train
from multicorpus.corpora import prepare


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "index", "frames", "train", "report"], default="all")
    ap.add_argument("--device", default=None, help="torch device for the network (default: mps, cuda or cpu, whichever exists)")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()
    run = lambda s: args.stage in ("all", s)  # noqa: E731
    if run("index"):
        df = prepare()
        print(df.groupby("source").agg(clips=("filename", "size"), speakers=("speaker_id", "nunique")))
    if run("frames"):
        frames.build()
    if run("train"):
        train.run(device=args.device, max_epochs=args.max_epochs, patience=args.patience)
    if run("report"):
        train.score()


if __name__ == "__main__":
    main()
