"""The single audio-loading path for the whole project.

Every module that opens a `.wav` goes through `load_audio`. Having exactly
one entry point is what makes the channel and sample-rate decisions in
the data audit actually hold everywhere, rather than being re-decided
(or forgotten) at each call site.

Two decisions are enforced here:

**Channels.** Five of the 1,440 files are dual-channel with bit-identical
channels. `load_audio` always returns a 1-D
signal, downmixing by averaging. For those five files the average of two
identical channels is the original signal exactly, so this is a no-op that
loses nothing; for the other 1,435 it is already 1-D. One code path, no
special-casing, and the raw files are never modified.

**Sample rate.** Files are 48 kHz natively. The project default is 16 kHz
(`TARGET_SR`) -- Nyquist-sufficient for speech, ~3x cheaper, and the rate the
pretrained models require. `sr` is an explicit argument with an
explicit default precisely because `librosa.load` silently defaults to
22 050 Hz, which matches neither the source nor our target.

Nothing here is fit on data, so there is no train/test distinction to
respect. Trimming and normalisation are deliberately *not* applied here --
they are experiment axes owned by `ser/features.py`, not fixed
preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

#: Project-wide target rate.
TARGET_SR: int = 16_000

#: The rate the files are stored at.
NATIVE_SR: int = 48_000


def load_audio(
    path: str | Path,
    sr: int | None = TARGET_SR,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load one clip as a 1-D float32 signal at a known sample rate.

    Args:
        path: Path to a ``.wav`` file.
        sr: Target sample rate. Defaults to `TARGET_SR` (16 kHz). Pass
            ``None`` to keep the file's native rate -- used by `ser.audit`,
            which must describe the true files rather than a resampled
            version of them.
        mono: Downmix to a single channel. Leave ``True``; the argument
            exists so the audit can assert the downmix is a no-op.

    Returns:
        ``(signal, sample_rate)``. The signal is 1-D float32 when
        ``mono=True``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No audio file at {path}")

    y, out_sr = librosa.load(str(path), sr=sr, mono=mono)
    return y, int(out_sr)


def channel_report(path: str | Path) -> dict[str, object]:
    """Describe a file's channel layout without downmixing it.

    Reads the header for the stored channel count, then, for multi-channel
    files, measures how far apart the channels actually are. This is what
    justifies treating the five dual-channel files as duplicated mono rather
    than as genuine stereo.

    Args:
        path: Path to a ``.wav`` file.

    Returns:
        ``n_channels``, ``is_multichannel``, and ``max_channel_diff`` -- the
        largest absolute difference between any two channels at the same
        sample, or ``0.0`` for a mono file. A value of exactly ``0.0`` on a
        multi-channel file means the channels are bit-identical and the
        downmix is lossless.
    """
    path = Path(path)
    info = sf.info(str(path))
    n_channels = int(info.channels)

    max_diff = 0.0
    if n_channels > 1:
        data, _ = sf.read(str(path), always_2d=True)
        first = data[:, :1]
        max_diff = float(np.abs(data - first).max())

    return {
        "n_channels": n_channels,
        "is_multichannel": n_channels > 1,
        "max_channel_diff": max_diff,
    }
