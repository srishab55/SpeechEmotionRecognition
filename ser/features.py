"""Feature extraction — parameterised, because comparing configs is the experiment.

Every axis below is an argument, not a constant: MFCC count, deltas,
aggregation strategy, trimming, and normalisation. They can be varied one at a time. This
module deliberately does not privilege one setting as "correct".

Caching
-------
Each configuration is hashed and cached to ``features/{hash}.parquet``. The
cache is gitignored and regenerable; an existing config is
never recomputed.

Normalisation and leakage
-------------------------
`extract` applies only **per-clip** transforms — nothing that requires
statistics pooled across clips. Global and per-speaker z-scoring are fit on
training folds only and therefore live in `fit_scaler` / `apply_scaler`,
which take an explicit train/test split. Doing it any other way would fit on
test data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from ser.metadata import load as load_metadata
from ser.preprocess import TARGET_SR, load_audio

#: Gitignored, regenerable.
CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "features"

AGGREGATIONS: tuple[str, ...] = ("mean", "mean_std", "mean_std_min_max", "percentiles")


#: Delta context we hold constant when the hop changes. ``librosa`` measures
#: delta width in FRAMES, so a fixed width silently means a different amount
#: of time at a different hop. See `FeatureConfig.resolved_delta_width`.
DEFAULT_DELTA_CONTEXT_MS: float = 288.0


@dataclass(frozen=True)
class FeatureConfig:
    """One point in the feature-design space.

    Every parameter that affects the numbers is named here. None is left to a
    library default: `librosa` is a *music* library and several of its
    defaults are wrong for speech, but the real problem with an unstated
    default is that nobody can see it, question it, or sweep it.

    Attributes:
        n_mfcc: Number of MFCC coefficients.
        deltas: Append first-order deltas.
        delta_deltas: Append second-order deltas.
        aggregation: How frame-level features collapse to a fixed vector.
        trim: Remove leading/trailing silence before extraction.
        top_db: Silence threshold for trimming, in dB below peak.
        sr: Sample rate to load at.
        n_fft: Analysis window in samples. The default 2048 is **128 ms** at
            16 kHz -- roughly 5x the 20-40 ms conventional for speech. It is
            librosa's default, inherited rather than chosen, and it
            tests whether it costs us.
        hop_length: Step between frames in samples. 512 is **32 ms** at
            16 kHz, against a ~10 ms speech convention.
        delta_width: Frames spanned when computing deltas. ``None`` derives
            it from ``hop_length`` so the *time* context stays
            `DEFAULT_DELTA_CONTEXT_MS` regardless of framing -- otherwise
            changing the hop would silently change two things at once and no
            ablation result could be attributed.
        n_mels: Mel filterbank bands. 128 is dense for 16 kHz speech, where
            40-80 is typical.
        fmin: Lowest mel frequency. 0 Hz includes rumble below the speech
            range.
        fmax: Highest mel frequency; ``None`` means ``sr / 2``.
        extras: Optional descriptors — any of ``chroma``,
            ``spectral_contrast``, ``spectral_centroid``, ``rolloff``,
            ``zcr``, ``rms``.
        normalisation: ``none`` | ``global`` | ``per_speaker``. Applied by
            `fit_scaler`, never inside `extract`, because it needs
            train-fold statistics.
    """

    n_mfcc: int = 40
    deltas: bool = True
    delta_deltas: bool = True
    aggregation: str = "mean_std"
    trim: bool = True
    top_db: int = 30
    sr: int = TARGET_SR
    # --- framing: librosa defaults, made explicit. Changing these changes
    # --- the numbers, so they can be varied rather than edited here.
    n_fft: int = 2048
    hop_length: int = 512
    delta_width: int | None = None
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float | None = None
    extras: tuple[str, ...] = field(default_factory=tuple)
    normalisation: str = "none"

    def __post_init__(self) -> None:
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {self.aggregation!r}")
        if self.normalisation not in ("none", "global", "per_speaker"):
            raise ValueError(f"unknown normalisation {self.normalisation!r}")
        if self.delta_width is not None and (self.delta_width < 3 or self.delta_width % 2 == 0):
            raise ValueError(f"delta_width must be an odd integer >= 3, got {self.delta_width}")

    # --- human-facing views on the framing -------------------------------

    @property
    def window_ms(self) -> float:
        """Analysis window length in milliseconds."""
        return 1000.0 * self.n_fft / self.sr

    @property
    def hop_ms(self) -> float:
        """Step between consecutive frames, in milliseconds."""
        return 1000.0 * self.hop_length / self.sr

    @property
    def resolved_delta_width(self) -> int:
        """Delta width in frames, derived from the hop when not set.

        Holds the delta's *time* context near
        `DEFAULT_DELTA_CONTEXT_MS`: width 9 at a 32 ms hop, width 29 at a
        10 ms hop. Always odd and at least 3, as librosa requires.
        """
        if self.delta_width is not None:
            return self.delta_width
        width = round(DEFAULT_DELTA_CONTEXT_MS / self.hop_ms)
        width = max(3, width)
        return width if width % 2 else width + 1

    @property
    def delta_context_ms(self) -> float:
        """Time span the delta computation actually sees."""
        return self.resolved_delta_width * self.hop_ms

    @property
    def hash(self) -> str:
        """Stable 12-char hash of everything that affects the stored values.

        ``normalisation`` is excluded: it is applied per-fold after loading,
        so configs differing only in normalisation share one cache entry.
        """
        payload = {k: v for k, v in asdict(self).items() if k != "normalisation"}
        payload["extras"] = sorted(payload["extras"])
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def name(self) -> str:
        """Short human-readable identifier, for ``results.csv``.

        Framing, mel and normalisation settings appear **only when they
        differ from the defaults**. That keeps the baseline name
        ``mfcc40-d-dd-mean_std-trim`` byte-stable -- it is already written
        into results.csv rows, and changing it would break comparability
        with every result logged so far.
        """
        default = _DEFAULTS
        parts = [f"mfcc{self.n_mfcc}"]
        if self.deltas:
            parts.append("d")
        if self.delta_deltas:
            parts.append("dd")
        parts.append(self.aggregation)
        parts.append("trim" if self.trim else "notrim")
        if self.extras:
            parts.append("+".join(sorted(self.extras)))
        if (self.n_fft, self.hop_length) != (default["n_fft"], default["hop_length"]):
            parts.append(f"win{self.window_ms:g}ms-hop{self.hop_ms:g}ms")
        if self.delta_width is not None:
            parts.append(f"dw{self.delta_width}")
        if self.n_mels != default["n_mels"]:
            parts.append(f"mel{self.n_mels}")
        if self.fmin != default["fmin"]:
            parts.append(f"fmin{self.fmin:g}")
        if self.fmax != default["fmax"]:
            parts.append(f"fmax{self.fmax:g}")
        if self.normalisation != "none":
            parts.append(self.normalisation)
        return "-".join(parts)

    @property
    def cache_path(self) -> Path:
        return CACHE_DIR / f"features_{self.hash}.parquet"

    def describe(self) -> str:
        """Multi-line summary of every parameter in effect, for notebooks."""
        return "\n".join(
            [
                f"name            {self.name}",
                f"hash            {self.hash}",
                f"sample rate     {self.sr} Hz",
                f"window          {self.n_fft} samples = {self.window_ms:.1f} ms",
                f"hop             {self.hop_length} samples = {self.hop_ms:.1f} ms",
                f"overlap         {100 * (1 - self.hop_length / self.n_fft):.0f}%",
                f"frame rate      {self.sr / self.hop_length:.1f} frames/s",
                f"mel bands       {self.n_mels}  ({self.fmin:g} Hz to "
                f"{self.fmax if self.fmax is not None else self.sr // 2:g} Hz)",
                f"MFCCs           {self.n_mfcc}"
                f"{' +delta' if self.deltas else ''}"
                f"{' +delta2' if self.delta_deltas else ''}",
                f"delta width     {self.resolved_delta_width} frames = "
                f"{self.delta_context_ms:.0f} ms of context",
                f"aggregation     {self.aggregation}",
                f"trim silence    {self.trim} (top_db={self.top_db})",
                f"normalisation   {self.normalisation}",
            ]
        )


#: Default field values, used by `FeatureConfig.name` to decide what to omit.
_DEFAULTS: dict[str, object] = {
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 0.0,
    "fmax": None,
}


def _aggregate(matrix: np.ndarray, strategy: str) -> tuple[np.ndarray, list[str]]:
    """Collapse a (n_features, n_frames) matrix to a fixed-length vector.

    Args:
        matrix: Frame-level features.
        strategy: One of `AGGREGATIONS`.

    Returns:
        ``(vector, suffixes)`` where ``suffixes`` names each statistic in the
        order it was concatenated.
    """
    if strategy == "mean":
        return matrix.mean(axis=1), ["mean"]
    if strategy == "mean_std":
        return (
            np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)]),
            ["mean", "std"],
        )
    if strategy == "mean_std_min_max":
        return (
            np.concatenate(
                [
                    matrix.mean(axis=1),
                    matrix.std(axis=1),
                    matrix.min(axis=1),
                    matrix.max(axis=1),
                ]
            ),
            ["mean", "std", "min", "max"],
        )
    percentiles = np.percentile(matrix, [10, 25, 50, 75, 90], axis=1)
    return percentiles.reshape(-1), ["p10", "p25", "p50", "p75", "p90"]


def _frame_features(y: np.ndarray, sr: int, config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """Build the frame-level feature matrix for one clip."""
    blocks: list[np.ndarray] = []
    names: list[str] = []

    # Every framing parameter is passed explicitly. Relying on librosa's
    # defaults here is what made the window length invisible in the first
    # place -- and its defaults are tuned for music, not speech.
    spec_kw = {
        "n_fft": config.n_fft,
        "hop_length": config.hop_length,
    }
    mel_kw = {
        **spec_kw,
        "n_mels": config.n_mels,
        "fmin": config.fmin,
        "fmax": config.fmax,
    }

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.n_mfcc, **mel_kw)
    blocks.append(mfcc)
    names += [f"mfcc{i:02d}" for i in range(config.n_mfcc)]

    width = config.resolved_delta_width
    # librosa requires width <= number of frames; short clips can fall under.
    width = min(width, mfcc.shape[1] - (1 - mfcc.shape[1] % 2))
    width = max(3, width if width % 2 else width - 1)

    if config.deltas:
        blocks.append(librosa.feature.delta(mfcc, width=width))
        names += [f"d_mfcc{i:02d}" for i in range(config.n_mfcc)]
    if config.delta_deltas:
        blocks.append(librosa.feature.delta(mfcc, order=2, width=width))
        names += [f"dd_mfcc{i:02d}" for i in range(config.n_mfcc)]

    extras = set(config.extras)
    if "chroma" in extras:
        block = librosa.feature.chroma_stft(y=y, sr=sr, **spec_kw)
        blocks.append(block)
        names += [f"chroma{i:02d}" for i in range(block.shape[0])]
    if "spectral_contrast" in extras:
        block = librosa.feature.spectral_contrast(y=y, sr=sr, **spec_kw)
        blocks.append(block)
        names += [f"contrast{i:02d}" for i in range(block.shape[0])]
    if "spectral_centroid" in extras:
        blocks.append(librosa.feature.spectral_centroid(y=y, sr=sr, **spec_kw))
        names.append("centroid")
    if "rolloff" in extras:
        blocks.append(librosa.feature.spectral_rolloff(y=y, sr=sr, **spec_kw))
        names.append("rolloff")
    if "zcr" in extras:
        blocks.append(
            librosa.feature.zero_crossing_rate(y, frame_length=config.n_fft, hop_length=config.hop_length)
        )
        names.append("zcr")
    if "rms" in extras:
        blocks.append(librosa.feature.rms(y=y, frame_length=config.n_fft, hop_length=config.hop_length))
        names.append("rms")

    return np.vstack(blocks), names


def extract_one(path: str | Path, config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """Extract the feature vector for a single clip.

    Only per-clip transforms are applied. Nothing here uses statistics
    pooled across clips, so calling this on a test clip cannot leak.

    Args:
        path: Path to a ``.wav`` file.
        config: The feature configuration.

    Returns:
        ``(vector, column_names)``.
    """
    y, sr = load_audio(path, sr=config.sr)
    if config.trim:
        # Silence is judged over these same frames, so the trim resolution
        # tracks the analysis resolution instead of librosa's fixed default.
        y, _ = librosa.effects.trim(
            y,
            top_db=config.top_db,
            frame_length=config.n_fft,
            hop_length=config.hop_length,
        )
    matrix, base_names = _frame_features(y, sr, config)
    vector, suffixes = _aggregate(matrix, config.aggregation)
    columns = [f"{name}_{suffix}" for suffix in suffixes for name in base_names]
    return vector, columns


def extract(
    config: FeatureConfig | None = None,
    df: pd.DataFrame | None = None,
    use_cache: bool = True,
    cache_path: Path | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Extract features for every clip, caching by config hash.

    Args:
        config: Feature configuration. Defaults to `FeatureConfig()`.
        df: Metadata frame. Defaults to the full inventory.
        use_cache: Read from and write to the parquet cache.
        cache_path: Override the cache file. Required when extracting over a
            clip set other than the RAVDESS inventory -- a different
            clip set would share a `FeatureConfig`, and therefore a hash,
            with RAVDESS, so without a distinct path the two would evict each
            other on every call.
        n_jobs: Worker processes for extraction. ``1`` keeps the original
            serial path exactly. Extraction is CPU-bound and per-clip
            independent, so this scales close to linearly -- which matters
            once the clip count goes from 1,440 to 12,162.

    Returns:
        Metadata columns plus one column per feature, in metadata row order.
    """
    config = config or FeatureConfig()
    if df is None:
        df = load_metadata()

    target = cache_path if cache_path is not None else config.cache_path

    if use_cache and target.is_file():
        cached = pd.read_parquet(target)
        if len(cached) == len(df) and (cached["filename"].to_numpy() == df["filename"].to_numpy()).all():
            return cached

    paths = df["filepath"].tolist()
    if n_jobs == 1:
        results = [extract_one(path, config) for path in paths]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(extract_one)(path, config) for path in paths
        )

    vectors = [vector for vector, _ in results]
    columns = results[0][1] if results else []

    features = pd.DataFrame(np.vstack(vectors), columns=columns)
    out = pd.concat([df.reset_index(drop=True), features], axis=1)

    if use_cache:
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False)
    return out


def feature_columns(frame: pd.DataFrame, meta: pd.DataFrame | None = None) -> list[str]:
    """Names of the numeric feature columns, excluding metadata."""
    meta_cols = set((meta if meta is not None else load_metadata()).columns)
    return [c for c in frame.columns if c not in meta_cols]


def fit_scaler(train: np.ndarray, normalisation: str = "global"):
    """Fit a scaler on the TRAINING FOLD ONLY.

    Args:
        train: Training-fold feature matrix.
        normalisation: ``global`` fits one `StandardScaler`; ``none``
            returns None.

    Returns:
        A fitted scaler, or None.

    Raises:
        ValueError: For ``per_speaker``, which is not a fold-level fit at all
            -- use `per_speaker_normalise`.
    """
    from sklearn.preprocessing import StandardScaler

    if normalisation == "none":
        return None
    if normalisation == "global":
        return StandardScaler().fit(train)
    raise ValueError(
        "per-speaker normalisation is not a train-fold fit; call "
        "per_speaker_normalise(matrix, speakers) instead"
    )


def apply_scaler(scaler, matrix: np.ndarray) -> np.ndarray:
    """Apply a scaler fitted on the training fold. Returns input if None."""
    return matrix if scaler is None else scaler.transform(matrix)


def per_speaker_normalise(matrix: np.ndarray, speakers: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score each speaker's features against that speaker's own statistics.

    Also known as cepstral mean and variance normalisation (CMVN). The intent
    is to strip out what is constant about a voice -- its timbre and the
    recording gain -- leaving what varies within it, which is the emotion.

    **A stated exception to the rule that nothing is fitted on test data.** That rule says per-speaker
    statistics are fit on the training fold only. Under
    ``speaker_independent`` that is not merely inconvenient, it is impossible:
    a held-out speaker has no training clips by construction. So a test
    speaker's statistics are computed from that speaker's own held-out clips.

    Why this is defensible: it consumes **no labels**, only audio, which is
    equally available at deployment time -- you would have a caller's other
    utterances before deciding anything about them. It is standard practice
    in speech.

    Why it is still worth flagging: it is transductive. The method sees the
    test audio, if not the test labels, so a strict reviewer may discount it.
    It is stated here plainly rather than buried.

    Args:
        matrix: Feature matrix, one row per clip.
        speakers: Speaker (actor) id per row. Same length as ``matrix``.
        eps: Floor on the divisor, so a speaker with a constant feature does
            not produce a division by zero.

    Returns:
        A new matrix, z-scored within each speaker.

    Raises:
        ValueError: If ``speakers`` does not align with ``matrix``.
    """
    if len(speakers) != len(matrix):
        raise ValueError(f"speakers has {len(speakers)} entries but matrix has {len(matrix)} rows")

    out = np.asarray(matrix, dtype=float).copy()
    for speaker in np.unique(speakers):
        mask = speakers == speaker
        block = out[mask]
        out[mask] = (block - block.mean(axis=0)) / (block.std(axis=0) + eps)
    return out


if __name__ == "__main__":  # pragma: no cover
    cfg = FeatureConfig()
    frame = extract(cfg)
    cols = feature_columns(frame)
    print(f"{cfg.name}  hash={cfg.hash}")
    print(f"{len(frame)} clips x {len(cols)} features -> {cfg.cache_path.name}")
