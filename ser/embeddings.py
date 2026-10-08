"""Frozen self-supervised speech embeddings .

The counterpart to `ser.features`: same discipline, different feature source.
`ser.features` computes MFCCs we specify; this module reads representations a
pretrained transformer learned. Both end in a fixed-length vector per clip
that `ser.models.classical` can consume unchanged.

Why one pass caches every layer
-------------------------------
A wav2vec2 forward pass is the expensive step (minutes per thousand clips on
CPU). *Which* transformer layer to read from is an open question -- for
emotion, middle layers routinely beat the last, which specialises toward the
phonetic content an ASR objective rewards. Extracting once per candidate layer
would multiply the cost by the number of candidates for no reason, because a
single pass with ``output_hidden_states=True`` already returns all of them.

So `extract` runs the model once and caches the pooled statistics of **every**
layer to a ``.npy`` array of shape ``(clips, layers, 2, dim)`` -- index 0 of
axis 2 is the per-clip mean over frames, index 1 the standard deviation.
Selecting a layer, or a pooling strategy, or a learned weighting across
layers, is then a cheap slice of a memory-mapped file rather than another
forward pass.

``.npy`` rather than parquet -- the only place this module departs from
`ser.features`' storage -- because the natural shape here is a 4-D tensor, and
memory-mapping lets a caller read one layer (~37 MB) without loading the whole
cache (~110 MB for the 1,440 RAVDESS clips).

Padding, and why there is none
------------------------------
Clips are variable length and wav2vec2 accepts them that way, with an
attention mask marking the real frames. Pooling is computed over masked
positions only, so padding contributes nothing. A fixed-length window would
have meant either discarding real speech or averaging silence into every
statistic -- both would change the numbers, invisibly.

Leakage
-------
Nothing here is fit on data: a frozen pretrained model applies the same
transformation to every clip regardless of fold. Normalisation is applied
afterwards by the caller through `ser.features.fit_scaler` /
`per_speaker_normalise`, which is where the train-only discipline lives.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ser.preprocess import TARGET_SR, load_audio

#: Shared with `ser.features` so every cached representation lives together.
CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "features"

#: Backbones this module has been exercised with. Any HF model exposing
#: ``hidden_states`` works; these are the ones with recorded results.
KNOWN_MODELS: dict[str, str] = {
    "facebook/wav2vec2-base": "wav2vec2-base, 12 layers, 768-d, 95M params",
    "microsoft/wavlm-base-plus": "WavLM-base+, 12 layers, 768-d -- usually "
    "stronger on emotion and speaker tasks",
    "facebook/hubert-base-ls960": "HuBERT-base, 12 layers, 768-d",
}

POOLINGS: tuple[str, ...] = ("mean", "mean_std", "std")


@dataclass(frozen=True)
class EmbeddingConfig:
    """One point in the embedding-design space.

    Every parameter that changes the stored numbers is named here, for the
    same reason `ser.features.FeatureConfig` does it: an unstated default is
    one nobody can see, question, or sweep.

    Attributes:
        model_id: HuggingFace model id. See `KNOWN_MODELS`.
        sr: Sample rate fed to the model. Asserted against the model's own
            feature extractor -- wav2vec2 is trained at 16 kHz and silently
            feeding it another rate produces plausible, wrong numbers.
        layer: Which hidden state to read. ``"last"``, an int index into
            ``hidden_states`` (0 is the CNN feature projection, 1..12 the
            transformer blocks), or ``"weighted"`` for a learned combination.
            A *selection*, not an extraction parameter -- excluded from the
            cache hash because one pass stores all layers.
        pooling: How frame-level states collapse to a fixed vector. Also a
            selection, also excluded from the hash.
        max_duration_s: Truncate clips longer than this. ``None`` keeps every
            clip whole, which is the default because the corpora's own
            durations are short and trimming would discard real speech.
        peak_norm: Divide each clip by its peak amplitude. Off by default:
            Absolute loudness carries emotion, so discarding it is a
            trade-off to be measured, not assumed. It exists for corpora
            recorded at different gains.
        trim: Remove leading and trailing silence before encoding.
        top_db: Silence threshold for trimming, in dB below peak.
        normalisation: ``none`` | ``global`` | ``per_speaker``. Applied by the
            caller per fold, never here. Excluded from the hash.
        batch_size: Clips per forward pass. Affects speed only.
    """

    model_id: str = "facebook/wav2vec2-base"
    sr: int = TARGET_SR
    layer: int | str = "last"
    pooling: str = "mean"
    max_duration_s: float | None = None
    peak_norm: bool = False
    trim: bool = False
    top_db: int = 30
    normalisation: str = "none"
    batch_size: int = 8

    #: Parameters that describe a *selection* from the cache rather than the
    #: extraction itself. Excluded from `hash`, so every layer/pooling/
    #: normalisation choice reuses one cached forward pass.
    _SELECTION: tuple[str, ...] = field(
        default=("layer", "pooling", "normalisation", "batch_size"),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        if self.normalisation not in ("none", "global", "per_speaker"):
            raise ValueError(f"unknown normalisation {self.normalisation!r}")
        if isinstance(self.layer, str) and self.layer not in ("last", "weighted"):
            raise ValueError(f"layer must be an int, 'last' or 'weighted', got {self.layer!r}")
        if self.max_duration_s is not None and self.max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive or None")

    @property
    def hash(self) -> str:
        """Stable 12-char hash of everything that affects the stored tensor."""
        payload = {k: v for k, v in asdict(self).items() if k not in self._SELECTION}
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def name(self) -> str:
        """Short human-readable identifier, for ``results.csv``."""
        parts = [self.model_id.split("/")[-1]]
        parts.append(f"L{self.layer}")
        parts.append(self.pooling)
        if self.trim:
            parts.append("trim")
        if self.peak_norm:
            parts.append("peaknorm")
        if self.max_duration_s is not None:
            parts.append(f"max{self.max_duration_s:g}s")
        if self.normalisation != "none":
            parts.append(self.normalisation)
        return "-".join(parts)

    @property
    def cache_path(self) -> Path:
        """The pooled per-layer tensor, ``(clips, layers, 2, dim)`` float32."""
        return CACHE_DIR / f"w2v_{self.hash}.npy"

    @property
    def order_path(self) -> Path:
        """Filenames in cache row order, so a cache can be checked not assumed."""
        return CACHE_DIR / f"w2v_{self.hash}_order.parquet"

    def describe(self) -> str:
        """Multi-line summary of every parameter in effect, for notebooks."""
        return "\n".join(
            [
                f"name            {self.name}",
                f"hash            {self.hash}",
                f"model           {self.model_id}",
                f"                {KNOWN_MODELS.get(self.model_id, 'not in KNOWN_MODELS')}",
                f"sample rate     {self.sr} Hz",
                f"layer           {self.layer}",
                f"pooling         {self.pooling}",
                f"max duration    {self.max_duration_s or 'whole clip'}",
                f"peak norm       {self.peak_norm}",
                f"trim silence    {self.trim} (top_db={self.top_db})",
                f"normalisation   {self.normalisation} (applied per fold by the caller)",
                f"cache           {self.cache_path.name}",
            ]
        )


def load_signal(path: str | Path, config: EmbeddingConfig) -> np.ndarray:
    """Load one clip exactly as the encoder will see it.

    Goes through `ser.preprocess.load_audio`, the project's single audio entry
    point, so the mono downmix and target rate hold here as everywhere else.

    Args:
        path: Path to a ``.wav``.
        config: Supplies rate, trimming and normalisation choices.

    Returns:
        A 1-D float32 signal.
    """
    signal, _ = load_audio(path, sr=config.sr)
    signal = np.asarray(signal, dtype=np.float32)

    if config.trim:
        import librosa

        signal, _ = librosa.effects.trim(signal, top_db=config.top_db)

    if config.peak_norm:
        peak = float(np.abs(signal).max()) if signal.size else 0.0
        if peak > 0:
            signal = signal / peak

    if config.max_duration_s is not None:
        limit = int(config.max_duration_s * config.sr)
        signal = signal[:limit]

    return signal


def extract(
    config: EmbeddingConfig | None = None,
    df: pd.DataFrame | None = None,
    use_cache: bool = True,
    progress: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Encode every clip once, caching the pooled statistics of every layer.

    The expensive call. On CPU this is minutes per thousand clips; the cache
    exists so it happens once per ``(model, audio-preparation)`` pair and
    every later layer or pooling question is answered from disk.

    Args:
        config: Embedding configuration. Defaults to `EmbeddingConfig()`.
        df: Frame with ``filepath`` and ``filename`` columns, in the order the
            rows should be cached. Defaults to the RAVDESS inventory.
        use_cache: Read from and write to `EmbeddingConfig.cache_path`.
        progress: Show a progress bar.

    Returns:
        ``(tensor, df)`` where ``tensor`` has shape
        ``(len(df), n_layers, 2, dim)`` -- axis 2 is ``[mean, std]`` over the
        clip's unmasked frames -- and ``df`` is the frame the rows correspond
        to.

    Raises:
        ValueError: If the model's expected sample rate differs from
            ``config.sr``, which would otherwise silently produce wrong
            numbers.
    """
    config = config or EmbeddingConfig()
    if df is None:
        from ser.metadata import load as load_metadata

        df = load_metadata()

    # The cache check comes before torch is imported, deliberately. Reading a
    # cached tensor is pure numpy, and requiring a 2 GB deep-learning stack to
    # hand back an array that is already on disk makes every downstream driver
    # fail on a machine that cannot load torch -- which can happen on a
    # locked-down machine.
    if use_cache and config.cache_path.is_file() and config.order_path.is_file():
        order = pd.read_parquet(config.order_path)
        if len(order) == len(df) and (order["filename"].to_numpy() == df["filename"].to_numpy()).all():
            return np.load(config.cache_path, mmap_mode="r"), df

    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(config.model_id)
    if int(extractor.sampling_rate) != int(config.sr):
        raise ValueError(
            f"{config.model_id} expects {extractor.sampling_rate} Hz but the "
            f"config asks for {config.sr} Hz. Feeding a model the wrong rate "
            f"produces plausible, wrong embeddings."
        )

    model = AutoModel.from_pretrained(config.model_id)
    model.eval()

    paths = df["filepath"].tolist()
    batches = range(0, len(paths), config.batch_size)
    if progress:
        from tqdm.auto import tqdm

        batches = tqdm(batches, desc=config.model_id.split("/")[-1], unit="batch")

    out: list[np.ndarray] = []
    for start in batches:
        chunk = paths[start : start + config.batch_size]
        signals = [load_signal(p, config) for p in chunk]
        inputs = extractor(
            signals,
            sampling_rate=config.sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        with torch.inference_mode():
            result = model(**inputs, output_hidden_states=True)

        # (layers, batch, frames, dim)
        states = torch.stack(result.hidden_states, dim=0)

        mask = _frame_mask(model, inputs, states.shape[2])
        # (1, batch, frames, 1) so it broadcasts across layers and dims
        weights = mask[None, :, :, None].to(states.dtype)
        counts = weights.sum(dim=2).clamp(min=1.0)

        mean = (states * weights).sum(dim=2) / counts
        variance = (((states - mean.unsqueeze(2)) ** 2 * weights).sum(dim=2) / counts).clamp(min=0.0)
        std = variance.sqrt()

        # -> (batch, layers, 2, dim)
        pooled = torch.stack([mean, std], dim=2).permute(1, 0, 2, 3).contiguous()
        out.append(pooled.to(torch.float32).numpy())

    tensor = np.concatenate(out, axis=0)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(config.cache_path, tensor)
        df[["filename"]].reset_index(drop=True).to_parquet(config.order_path, index=False)
    return tensor, df


def _frame_mask(model, inputs, n_frames: int):
    """Per-frame validity mask for a padded batch.

    The encoder's convolutional front end downsamples, so the sample-level
    attention mask the tokenizer produces does not line up with the frame
    axis of ``hidden_states``. Transformers exposes the length arithmetic;
    where it does not, every frame is treated as valid, which is exact for a
    batch of equal-length clips.
    """
    import torch

    attention = inputs.get("attention_mask")
    if attention is None:
        return torch.ones(inputs["input_values"].shape[0], n_frames)

    getter = getattr(model, "_get_feat_extract_output_lengths", None)
    if getter is None:
        return torch.ones(attention.shape[0], n_frames)

    lengths = getter(attention.sum(dim=-1)).to(torch.long)
    mask = torch.arange(n_frames)[None, :] < lengths[:, None]
    return mask.to(torch.float32)


def select(
    tensor: np.ndarray,
    config: EmbeddingConfig,
    layer_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Slice a cached tensor down to one feature matrix.

    Args:
        tensor: ``(clips, layers, 2, dim)`` from `extract`.
        config: Supplies ``layer`` and ``pooling``.
        layer_weights: Required when ``config.layer == "weighted"``; one
            weight per layer. They are applied to the already-pooled
            statistics.

            For the **mean** channel this is *exact*, because a weighted sum
            and a mean commute:
            ``sum_i w_i * mean(h_i) == mean(sum_i w_i * h_i)``. That identity
            is what lets the cache store per-layer means instead of per-layer
            frame sequences, and it is asserted in
            ``tests/test_embeddings.py``.

            For the **std** channel it is an *approximation*: the result is a
            weighted average of each layer's dispersion, not the dispersion of
            the weighted combination. Those differ whenever layers are
            correlated, which they are. The approximation is usable -- it is
            still a monotone measure of within-clip variability -- but a
            ``weighted`` + ``std``/``mean_std`` result must be reported as
            such, not as an exact layer-weighted statistic. Use
            ``pooling="mean"`` when exactness matters.

    Returns:
        ``(clips, dim)`` for mean or std pooling, ``(clips, 2 * dim)`` for
        ``mean_std``.

    Raises:
        ValueError: On a layer index out of range, or missing weights.
    """
    n_layers = tensor.shape[1]

    if config.layer == "weighted":
        if layer_weights is None:
            raise ValueError("layer='weighted' requires layer_weights")
        weights = np.asarray(layer_weights, dtype=np.float32)
        if weights.shape != (n_layers,):
            raise ValueError(f"layer_weights must have shape ({n_layers},), got {weights.shape}")
        # Sums over the layer axis, leaving (clips, [mean,std], dim).
        block = np.tensordot(weights, np.asarray(tensor), axes=([0], [1]))
    else:
        index = n_layers - 1 if config.layer == "last" else int(config.layer)
        if not -n_layers <= index < n_layers:
            raise ValueError(f"layer {config.layer} out of range for {n_layers} cached layers")
        block = np.asarray(tensor[:, index])

    mean, std = block[:, 0, :], block[:, 1, :]
    if config.pooling == "mean":
        return np.ascontiguousarray(mean)
    if config.pooling == "std":
        return np.ascontiguousarray(std)
    return np.ascontiguousarray(np.concatenate([mean, std], axis=1))


def extract_frames(
    df: pd.DataFrame,
    config: EmbeddingConfig | None = None,
    layer: int = 6,
    progress: bool = True,
):
    """Yield **per-frame** hidden states, for models that attend over time.

    `extract` pools each clip to a mean and a standard deviation, which is what
    a fixed-length classifier needs and is useless to anything with attention:
    pooling is precisely the step attention is meant to replace. This is the
    escape hatch.

    Streaming rather than cached, deliberately. One layer of frame-level states
    for a corpus of 12,000 clips is several GB, and every layer would be 13 times
    that. RAVDESS alone at one layer is ~820 MB, which
    `cache_frames` will materialise if you want random access; for a larger corpus,
    iterate this instead and keep nothing.

    Args:
        df: Frame with ``filepath``, in the order batches should arrive.
        config: Embedding configuration. ``config.layer`` is ignored -- pass
            ``layer`` explicitly, since the point here is one layer at a time.
        layer: Which hidden state to return. Defaults to 6, a middle layer.
        progress: Show a progress bar.

    Yields:
        ``(states, lengths, filenames)`` per batch, where ``states`` is a
        float32 array ``(batch, max_frames, dim)`` **zero-padded on the right**
        and ``lengths`` gives each clip's real frame count.

        **Use the lengths.** Clips differ in duration, so the padding is real
        and an attention layer that does not mask it will happily attend to
        silence that is not in the audio. Build a mask as
        ``torch.arange(max_frames)[None, :] < lengths[:, None]`` and add
        ``-inf`` to the scores where it is False.
    """
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    config = config or EmbeddingConfig()
    extractor = AutoFeatureExtractor.from_pretrained(config.model_id)
    if int(extractor.sampling_rate) != int(config.sr):
        raise ValueError(f"{config.model_id} expects {extractor.sampling_rate} Hz, config says {config.sr}")
    model = AutoModel.from_pretrained(config.model_id)
    model.eval()

    paths = df["filepath"].tolist()
    names = df["filename"].tolist()
    batches = range(0, len(paths), config.batch_size)
    if progress:
        from tqdm.auto import tqdm

        batches = tqdm(batches, desc=f"frames L{layer}", unit="batch")

    for start in batches:
        chunk = paths[start : start + config.batch_size]
        signals = [load_signal(p, config) for p in chunk]
        inputs = extractor(
            signals,
            sampling_rate=config.sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        with torch.inference_mode():
            result = model(**inputs, output_hidden_states=True)

        states = result.hidden_states[layer]
        mask = _frame_mask(model, inputs, states.shape[1])
        yield (
            states.to(torch.float32).numpy(),
            mask.sum(dim=1).to(torch.int32).numpy(),
            names[start : start + config.batch_size],
        )


def cache_frames(
    df: pd.DataFrame,
    config: EmbeddingConfig | None = None,
    layer: int = 6,
    path: Path | None = None,
    max_frames: int = 320,
):
    """Materialise one layer of frame-level states to a ``.npy`` for random access.

    Worth it for RAVDESS (~820 MB at layer 6) when a training loop needs to
    shuffle clips across epochs; not worth it for a much larger corpus. Written as
    a memmap so a caller can slice a batch without loading the whole file.

    Args:
        df: Frame with ``filepath`` and ``filename``.
        config: Embedding configuration.
        layer: Which hidden state to store.
        path: Destination ``.npy``. Defaults to a hash-derived name beside the
            pooled caches.
        max_frames: Clips are padded or truncated to this many frames. 320 is
            ~6.4 s at wav2vec2's 50 frames/s, comfortably above RAVDESS's 5.3 s
            maximum; anything longer is truncated and reported.

    Returns:
        ``(array, lengths, filenames)`` where ``array`` is a memmap of shape
        ``(len(df), max_frames, dim)`` and ``lengths`` the true frame counts.
    """
    config = config or EmbeddingConfig()
    target = Path(path) if path is not None else CACHE_DIR / f"frames_{config.hash}_L{layer}.npy"
    lengths_path = target.with_name(target.stem + "_lengths.npy")

    if target.is_file() and lengths_path.is_file():
        return (
            np.load(target, mmap_mode="r"),
            np.load(lengths_path),
            df["filename"].tolist(),
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    array, lengths, cursor, truncated = None, [], 0, 0

    for states, frame_counts, _ in extract_frames(df, config, layer):
        if array is None:
            array = np.lib.format.open_memmap(
                target,
                mode="w+",
                dtype=np.float32,
                shape=(len(df), max_frames, states.shape[-1]),
            )
        take = min(states.shape[1], max_frames)
        array[cursor : cursor + len(states), :take] = states[:, :take]
        truncated += int((frame_counts > max_frames).sum())
        lengths.extend(np.minimum(frame_counts, max_frames).tolist())
        cursor += len(states)

    array.flush()
    lengths = np.asarray(lengths, dtype=np.int32)
    np.save(lengths_path, lengths)
    if truncated:
        print(f"note: {truncated} clips exceeded {max_frames} frames and were truncated")
    return np.load(target, mmap_mode="r"), lengths, df["filename"].tolist()


def feature_names(dim: int, pooling: str) -> list[str]:
    """Column names for a selected matrix, for frames and feature reports."""
    if pooling == "mean":
        return [f"emb_mean_{i:03d}" for i in range(dim)]
    if pooling == "std":
        return [f"emb_std_{i:03d}" for i in range(dim)]
    half = dim // 2
    return [f"emb_mean_{i:03d}" for i in range(half)] + [f"emb_std_{i:03d}" for i in range(half)]


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    print(EmbeddingConfig().describe())
