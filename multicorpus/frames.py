"""wav2vec2-base layer-6 frame states for all 12,162 clips, speaker-normalised, as a padded float16 memmap.

Pass 1 encodes every clip on its own (no padding: batched padding shifts wav2vec2-base activations) and
appends the raw frames to a ragged file; it can be interrupted and resumed. Pass 2 z-scores each channel with
the speaker's own frame statistics (audio only, no labels; the same per-speaker normalisation as the main
study) and writes the padded array the trainer reads. Clips longer than MAX_FRAMES (6.4 s) are truncated.
"""

from __future__ import annotations

import time

import numpy as np

from multicorpus import ROOT
from multicorpus.corpora import load_index

LAYER, DIM, MAX_FRAMES = 6, 768, 320
FEATURES = ROOT / "features"
RAW, RAW_LEN = FEATURES / "mc_raw_frames_L6.bin", FEATURES / "mc_raw_lengths_L6.npy"
OUT, OUT_LEN = FEATURES / "mc_frames_L6_spknorm.npy", FEATURES / "mc_frames_L6_spknorm_lengths.npy"


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_raw(limit: int | None = None) -> np.ndarray:
    """Pass 1. Returns the frame count of every clip. Resumes from `RAW_LEN` if present."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    from ser.preprocess import load_audio

    FEATURES.mkdir(exist_ok=True)
    df = load_index()
    n = len(df) if limit is None else min(limit, len(df))
    done = np.load(RAW_LEN) if RAW_LEN.exists() else np.zeros(0, dtype=np.int32)
    if len(done) >= n:
        return done[:n]
    # drop any frames written after the last saved length checkpoint
    keep_bytes = int(done.sum()) * DIM * 2
    mode = "r+b" if RAW.exists() else "w+b"
    device = _device()
    extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = AutoModel.from_pretrained("facebook/wav2vec2-base").eval().to(device)
    lengths = list(done)
    t0 = time.time()
    with open(RAW, mode) as fh:
        fh.truncate(keep_bytes)
        fh.seek(keep_bytes)
        for i in range(len(done), n):
            signal, _ = load_audio(ROOT / df.filepath[i])
            inputs = extractor(signal, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                states = model(inputs.input_values.to(device), output_hidden_states=True).hidden_states[LAYER]
            frames = states[0, :MAX_FRAMES].float().cpu().numpy().astype(np.float16)
            fh.write(frames.tobytes())
            lengths.append(len(frames))
            if (i + 1) % 250 == 0 or i + 1 == n:
                fh.flush()
                np.save(RAW_LEN, np.asarray(lengths, dtype=np.int32))
                rate = (i + 1 - len(done)) / (time.time() - t0)
                print(f"  encoded {i + 1}/{n} clips ({rate:.1f} clips/s, ~{(n - i - 1) / rate / 60:.0f} min left)", flush=True)
    return np.asarray(lengths, dtype=np.int32)


def normalise_and_pad(lengths: np.ndarray) -> None:
    """Pass 2. Per-speaker z-scoring over real frames, written as `(clips, MAX_FRAMES, 768)` float16."""
    df = load_index().iloc[: len(lengths)]
    offsets = np.concatenate([[0], np.cumsum(lengths.astype(np.int64))])
    raw = np.memmap(RAW, dtype=np.float16, mode="r", shape=(int(offsets[-1]), DIM))
    out = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float16, shape=(len(lengths), MAX_FRAMES, DIM))
    speakers = df.speaker_id.to_numpy()
    for k, speaker in enumerate(np.unique(speakers)):
        idx = np.flatnonzero(speakers == speaker)
        total, square, count = np.zeros(DIM), np.zeros(DIM), 0
        for i in idx:
            f = np.asarray(raw[offsets[i] : offsets[i + 1]], dtype=np.float64)
            total, square, count = total + f.sum(0), square + (f**2).sum(0), count + len(f)
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean**2, 0.0)) + 1e-8
        for i in idx:
            f = np.asarray(raw[offsets[i] : offsets[i + 1]], dtype=np.float64)
            out[i, : len(f)] = ((f - mean) / std).astype(np.float16)
        if (k + 1) % 20 == 0:
            print(f"  normalised {k + 1}/{len(np.unique(speakers))} speakers", flush=True)
    out.flush()
    np.save(OUT_LEN, lengths.astype(np.int32))


def build(limit: int | None = None) -> None:
    """Build the speaker-normalised frame cache if it does not exist yet."""
    if OUT.exists() and OUT_LEN.exists() and limit is None:
        print("frame cache already built:", OUT.name)
        return
    lengths = encode_raw(limit)
    normalise_and_pad(lengths)
    print(f"frame cache ready: {OUT.name}, {len(lengths)} clips, frames {lengths.min()}-{lengths.max()}")


def load():
    """(frames memmap, lengths) of the speaker-normalised cache."""
    return np.load(OUT, mmap_mode="r"), np.load(OUT_LEN)
