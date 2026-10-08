"""Waveform augmentation, applied to training folds only.

Augmentation is a way to give the model more to learn from without acquiring more data.

Why this is not obviously a good idea
-------------------------------------
Augmentation for emotion recognition is riskier than for speech recognition,
and the risk is specific: **the transformations most commonly used perturb the
very cues that carry the label.** The exploratory analysis found
pitch and energy separating the emotion classes. Shifting pitch by two
semitones may therefore turn an angry clip into something that is acoustically
no longer angry, while keeping the `angry` label — which is label noise, not
augmentation.

So the three transforms are run **separately as well as together**, and each is
chosen to stress a different assumption:

``pitch``
    Shifts F0 while preserving duration. Directly perturbs an emotion cue, and
    is the one most likely to hurt. Included precisely because the expectation
    is testable.
``stretch``
    Changes speaking rate while preserving pitch. Speaking rate differs by
    emotion, so this has the same risk in a milder form.
``noise``
    Additive white Gaussian noise at a fixed SNR. Perturbs no prosodic cue,
    only the channel. Should be the safe one, and is the closest analogue of
    the contact-centre deployment the business case describes.

Every parameter is explicit here rather than left to a librosa default, for the
same reason `FeatureConfig` names all of its: an unstated default cannot be
questioned or swept.

Leakage
-------
Augmented clips derive from a source clip and inherit its filename stem, so
the caller can include only the augmentations of clips in the current
training fold. Nothing derived from a held-out clip ever enters training, and
no augmented clip is ever scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ser import SEED
from ser.preprocess import TARGET_SR

#: Augmented audio lands here. Gitignored and regenerable, like `features/`.
AUGMENTED_DIR: Path = Path(__file__).resolve().parent.parent / "data_augmented"

#: Separator between a source stem and its augmentation tag. Chosen so the
#: source filename is recoverable by a single split.
TAG = "__"


@dataclass(frozen=True)
class AugmentConfig:
    """One augmentation, fully specified.

    Attributes:
        name: Short tag used in filenames and results rows.
        kind: ``pitch`` | ``stretch`` | ``noise``.
        amount: Semitones for ``pitch``, rate multiplier for ``stretch``,
            signal-to-noise ratio in dB for ``noise``.
        sr: Sample rate the transform operates at.
    """

    name: str
    kind: str
    amount: float
    sr: int = TARGET_SR

    def __post_init__(self) -> None:
        if self.kind not in ("pitch", "stretch", "noise"):
            raise ValueError(f"unknown augmentation kind {self.kind!r}")
        if self.kind == "stretch" and self.amount <= 0:
            raise ValueError("stretch rate must be positive")


#: The three tried. Deliberately small: each costs a full extraction pass over
#: 1,440 clips, and the point is to attribute an effect, not to maximise count.
DEFAULT_AUGMENTATIONS: tuple[AugmentConfig, ...] = (
    AugmentConfig("pitch+2", "pitch", 2.0),
    AugmentConfig("stretch1.1", "stretch", 1.1),
    AugmentConfig("noise20db", "noise", 20.0),
)


def apply(signal: np.ndarray, config: AugmentConfig, seed: int = SEED) -> np.ndarray:
    """Apply one augmentation to a waveform.

    Args:
        signal: 1-D float32 waveform at ``config.sr``.
        config: The augmentation to apply.
        seed: Seeds the noise generator, so a given clip always receives the
            same noise and the experiment is reproducible.

    Returns:
        A new 1-D float32 waveform. Length is preserved except for
        ``stretch``, which changes duration by design.
    """
    import librosa

    signal = np.asarray(signal, dtype=np.float32)

    if config.kind == "pitch":
        return librosa.effects.pitch_shift(y=signal, sr=config.sr, n_steps=config.amount).astype(np.float32)

    if config.kind == "stretch":
        return librosa.effects.time_stretch(y=signal, rate=config.amount).astype(np.float32)

    # noise: scale white noise to hit the requested SNR for THIS clip, so a
    # quiet clip and a loud one are degraded equally in relative terms.
    power = float(np.mean(signal**2))
    if power <= 0:
        return signal
    noise_power = power / (10 ** (config.amount / 10))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return (signal + noise).astype(np.float32)


def augmented_name(filename: str, config: AugmentConfig) -> str:
    """Filename for an augmented copy, with the source stem recoverable."""
    return f"{Path(filename).stem}{TAG}{config.name}.wav"


def source_name(augmented: str) -> str:
    """Recover the source filename from an augmented one.

    Returns the input unchanged if it carries no augmentation tag, so callers
    can pass a mixed list.
    """
    stem = Path(augmented).stem
    if TAG not in stem:
        return augmented
    return f"{stem.split(TAG)[0]}.wav"


def materialise(
    df,
    configs: tuple[AugmentConfig, ...] = DEFAULT_AUGMENTATIONS,
    dest: Path | None = None,
    progress: bool = True,
):
    """Write augmented copies of every clip in ``df`` to disk.

    Written rather than generated on the fly so the augmented set goes through
    exactly the same `ser.embeddings.extract` path as the originals -- one code
    path, one cache mechanism, and the files are inspectable if a result looks
    strange.

    Idempotent: a copy already on disk is not rewritten.

    Args:
        df: Frame with ``filepath`` and ``filename``, e.g. the RAVDESS index.
        configs: Augmentations to apply.
        dest: Destination root. Defaults to `AUGMENTED_DIR`.
        progress: Show a progress bar.

    Returns:
        A frame in the same shape as ``df`` with one row per augmented clip,
        plus ``augmentation`` and ``source_filename`` columns.
    """
    import pandas as pd
    import soundfile as sf

    from ser.preprocess import load_audio

    dest = Path(dest) if dest is not None else AUGMENTED_DIR
    dest.mkdir(parents=True, exist_ok=True)

    rows = []
    iterator = df.itertuples()
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, total=len(df), desc="augment", unit="clip")

    for row in iterator:
        signal = None
        for config in configs:
            out_dir = dest / config.name
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / augmented_name(row.filename, config)

            if not target.exists():
                if signal is None:
                    signal, _ = load_audio(row.filepath)
                sf.write(target, apply(signal, config), config.sr, subtype="PCM_16")

            record = {c: getattr(row, c) for c in df.columns}
            record["filename"] = target.name
            record["filepath"] = str(target)
            record["augmentation"] = config.name
            record["source_filename"] = row.filename
            rows.append(record)

    return pd.DataFrame(rows)
