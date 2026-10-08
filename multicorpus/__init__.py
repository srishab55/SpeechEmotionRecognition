"""CNN-BiLSTM on RAVDESS + CREMA-D + TESS + SAVEE with whole-speaker hold-out (self-contained)."""

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
SEED = 42
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
CORPORA = ["RAVDESS", "CREMA-D", "TESS", "SAVEE"]
