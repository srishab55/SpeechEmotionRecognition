"""Feature construction (node N5).

Three families feed the models:

* hand-designed vectors: the 162-number baseline set and a richer MFCC set;
* pooled wav2vec2 embeddings (mean and standard deviation over time, per layer);
* frame sequences: a padded log-mel or MFCC matrix for the CNN/RNN family, and
  frame-level wav2vec2 states for the attention models.

Per-speaker z-scoring is applied here because it is a property of the feature
(audio only, no labels). Any scaler that is *fitted* lives inside the model
pipeline, so it only ever sees training rows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from ser.features import FeatureConfig, extract
from ser.preprocess import load_audio
from unheard_lib.config import ROOT
from unheard_lib.data import Context, load_w2v_cache, normalise_per_speaker

log = logging.getLogger("unheard")

BASELINE_EXTRAS = ("chroma", "spectral_contrast", "spectral_centroid", "zcr")
BASELINE_DIM = 162

# Sequence input: 64 log-mel bins, 64 ms window, 16 ms hop at 16 kHz.
SEQ_SR, SEQ_N_FFT, SEQ_HOP, SEQ_N_MELS, SEQ_TOP_DB = 16_000, 1024, 256, 64, 30
SEQ_N_MFCC = 20
SEQ_CACHE = ROOT / "features" / "sequences_logmel64_hop256.npz"


def baseline_config() -> FeatureConfig:
    """The 162-feature set used for the reference baseline."""
    return FeatureConfig(n_mfcc=20, extras=BASELINE_EXTRAS, normalisation="per_speaker")


def rich_mfcc_config() -> FeatureConfig:
    """40 MFCCs with deltas and delta-deltas, no extra descriptors."""
    return FeatureConfig(n_mfcc=40, normalisation="per_speaker")


def hand_designed(ctx: Context, kind: str, n_jobs: int = 2, normalise: bool = True) -> np.ndarray:
    """One feature row per clip, per-speaker normalised unless `normalise` is False.

    Args:
        ctx: Shared context.
        kind: "handcrafted162" or "mfcc40".
        n_jobs: Worker processes for the first (uncached) extraction.
        normalise: Apply per-speaker z-scoring (audio only, no labels).
    """
    configs = {"handcrafted162": baseline_config, "mfcc40": rich_mfcc_config}
    if kind not in configs:
        raise ValueError(f"kind must be one of {sorted(configs)}")
    frame = extract(configs[kind](), df=ctx.meta, n_jobs=n_jobs)
    columns = [c for c in frame.columns if c not in ctx.meta.columns]
    matrix = frame[columns].to_numpy(dtype=np.float64)
    if kind == "handcrafted162":
        assert matrix.shape[1] == BASELINE_DIM, f"expected {BASELINE_DIM} features, got {matrix.shape[1]}"
    assert np.isfinite(matrix).all(), "non-finite value in hand-designed features"
    return normalise_per_speaker(ctx, matrix) if normalise else matrix


class PooledW2V:
    """Cached pooled wav2vec2 embeddings with per-speaker normalisation."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.tensor = load_w2v_cache(ctx)
        self._cache: dict[tuple[int, str, bool], np.ndarray] = {}

    def get(self, layer: int, pooling: str, normalise: bool = True) -> np.ndarray:
        """Feature matrix for one layer; "mean" is 768 wide, "mean_std" 1,536."""
        if pooling not in ("mean", "mean_std"):
            raise ValueError("pooling must be 'mean' or 'mean_std'")
        key = (layer, pooling, normalise)
        if key not in self._cache:
            block = self.tensor[:, layer].astype(np.float64)
            raw = block[:, 0] if pooling == "mean" else np.concatenate([block[:, 0], block[:, 1]], axis=1)
            self._cache[key] = normalise_per_speaker(self.ctx, raw) if normalise else raw
        return self._cache[key]


def _clip_sequences(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Log-mel and MFCC frame matrices `(frames, channels)` for one trimmed clip."""
    signal, sr = load_audio(path, sr=SEQ_SR)
    signal, _ = librosa.effects.trim(signal, top_db=SEQ_TOP_DB)
    power = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=SEQ_N_FFT, hop_length=SEQ_HOP, n_mels=SEQ_N_MELS, fmin=0.0, fmax=sr / 2
    )
    log_mel = librosa.power_to_db(power, ref=1.0, top_db=80.0)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=SEQ_N_MFCC)
    stacked = np.concatenate([mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)])
    return log_mel.T.astype(np.float32), stacked.T.astype(np.float32)


def _speaker_normalise_frames(clips: list[np.ndarray], speakers: np.ndarray) -> list[np.ndarray]:
    """Z-score each channel with the speaker's own frame statistics (audio only, no labels)."""
    out: list[np.ndarray] = [None] * len(clips)  # type: ignore[list-item]
    for speaker in np.unique(speakers):
        idx = np.flatnonzero(speakers == speaker)
        frames = np.concatenate([clips[i] for i in idx])
        mean, std = frames.mean(0), frames.std(0) + 1e-8
        for i in idx:
            out[i] = ((clips[i] - mean) / std).astype(np.float32)
    return out


def _pad(clips: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad every clip with zeros to a common length and keep the true lengths."""
    lengths = np.array([len(c) for c in clips], dtype=np.int32)
    out = np.zeros((len(clips), width, clips[0].shape[1]), dtype=np.float32)
    for i, clip in enumerate(clips):
        out[i, : len(clip)] = clip
    return out, lengths


def sequences(
    ctx: Context, n_jobs: int = 2, cache: Path = SEQ_CACHE, normalise: bool = True
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Padded, speaker-normalised frame matrices for the CNN/RNN family.

    Args:
        normalise: Centre and scale each channel per speaker (default). False gives the raw frames,
            used only for the normalisation ablation.

    Returns:
        `{"logmel": (X, lengths), "mfcc": (X, lengths)}` with `X` shaped
        `(clips, MAX_FRAMES, channels)`. Padding is zeros and `lengths` holds
        each clip's real frame count; every model must mask by it.
    """
    names = ctx.meta["filename"].to_numpy().astype(str)
    if not normalise:
        cache = cache.with_name(cache.stem + "_unnormalised.npz")
    if cache.is_file():
        stored = np.load(cache, allow_pickle=False)
        if (stored["filenames"] == names).all():
            return {k: (stored[f"{k}_x"].astype(np.float32), stored[f"{k}_len"]) for k in ("logmel", "mfcc")}
    from joblib import Parallel, delayed

    log.info("extracting frame sequences for %d clips", len(names))
    pairs = Parallel(n_jobs=n_jobs)(delayed(_clip_sequences)(p) for p in ctx.meta["filepath"])
    width = max(len(p[0]) for p in pairs)
    out = {}
    for i, key in enumerate(("logmel", "mfcc")):
        clips = [p[i] for p in pairs]
        normed = _speaker_normalise_frames(clips, ctx.speakers) if normalise else clips
        out[key] = _pad(normed, width)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        filenames=names,
        **{f"{k}_x": v[0].astype(np.float16) for k, v in out.items()},
        **{f"{k}_len": v[1] for k, v in out.items()},
    )
    return out


def verify_frame_mean(
    frames: np.ndarray, lengths: np.ndarray, pooled_mean: np.ndarray, tolerance: float = 0.02
) -> float:
    """Check that the mean over REAL frames matches the cached pooled mean.

    The cache stores mean and std over unmasked frames only, so averaging over
    padding as well would not match. Returns the relative error; raises if it
    exceeds `tolerance`.
    """
    mask = np.arange(frames.shape[1])[None, :] < lengths[:, None]
    mean = (frames * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
    error = float(np.abs(mean - pooled_mean).mean() / np.abs(pooled_mean).mean())
    if error > tolerance:
        raise AssertionError(
            f"frame mean differs from the pooled cache by {error:.3%} (limit {tolerance:.0%})"
        )
    return error


W2V_FRAME_LAYER = 6  # read off the descriptive layer sweep (Section 8.2); disclosed as a post-hoc choice


def w2v_frames(
    ctx: Context, layer: int = W2V_FRAME_LAYER, normalise: bool = True
) -> tuple[np.ndarray, np.ndarray] | None:
    """wav2vec2 frame states `(clips, frames, 768)` for one layer, speaker-normalised, as float16.

    Returns None when the frame cache has not been built. Building it needs the
    pretrained model download, so it is a separate step:
    `python scripts/extract_w2v_frames.py`.

    Each speaker's frames are centred and scaled with that speaker's own mean
    and standard deviation over real frames (audio only, no labels), the frame
    version of the per-speaker normalisation used for pooled features.
    """
    from ser.embeddings import CACHE_DIR, EmbeddingConfig

    path = CACHE_DIR / f"frames_{EmbeddingConfig().hash}_L{layer}.npy"
    lengths_path = path.with_name(path.stem + "_lengths.npy")
    if not (path.is_file() and lengths_path.is_file()):
        return None
    raw, lengths = np.load(path, mmap_mode="r"), np.load(lengths_path)
    if len(raw) != len(ctx.meta):
        raise AssertionError(f"frame cache has {len(raw)} clips, the clip table has {len(ctx.meta)}")
    _check_frame_alignment(ctx, raw, lengths)
    if not normalise:
        return np.asarray(raw, dtype=np.float16), lengths.astype(np.int32)
    out = np.zeros(raw.shape, dtype=np.float16)
    for speaker in np.unique(ctx.speakers):
        idx = np.flatnonzero(ctx.speakers == speaker)
        total, square, count = 0.0, 0.0, 0
        for i in idx:
            frames = np.asarray(raw[i, : lengths[i]], dtype=np.float64)
            total, square, count = total + frames.sum(0), square + (frames**2).sum(0), count + len(frames)
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean**2, 0.0)) + 1e-8
        for i in idx:
            out[i, : lengths[i]] = ((np.asarray(raw[i, : lengths[i]], dtype=np.float64) - mean) / std).astype(
                np.float16
            )
    return out, lengths.astype(np.int32)


def _check_frame_alignment(ctx: Context, raw: np.ndarray, lengths: np.ndarray, floor: float = 0.99) -> float:
    """Confirm frame-cache row k belongs to clip k.

    The frame cache has no filename list, so each clip's mean frame is compared with the pooled cache's
    layer-6 mean: the nearest pooled row (by cosine) must be the clip's own row for at least `floor` of clips.
    Small extraction differences do not matter for this test; a shifted or shuffled cache fails it.
    """
    pooled = load_w2v_cache(ctx)[:, W2V_FRAME_LAYER, 0]
    means = np.stack([np.asarray(raw[i, : lengths[i]], dtype=np.float64).mean(0) for i in range(len(raw))])
    unit = lambda m: (m - m.mean(0)) / (np.linalg.norm(m - m.mean(0), axis=1, keepdims=True) + 1e-12)  # noqa: E731
    hit = float((np.argmax(unit(means) @ unit(pooled).T, axis=1) == np.arange(len(raw))).mean())
    if hit < floor:
        raise AssertionError(f"frame cache rows do not line up with the clip table (own-row match {hit:.1%})")
    return hit
