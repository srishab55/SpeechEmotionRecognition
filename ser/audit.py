"""Integrity and property audit of the raw audio.

Every measurement here is taken **at the native sample rate** with
``librosa.load(..., sr=None)``. This matters: `librosa.load` defaults to
``sr=22050`` and would silently resample, so an audit run at the default
would describe a resampled artifact rather than the files on disk.

Nothing in this module is fit on data -- these are per-file measurements, so
there is no train/test distinction to respect. Plotting lives in `ser.eda`;
these functions return frames, never figures.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from ser.metadata import load as load_metadata
from ser.preprocess import load_audio

#: Silence threshold in dB below peak, passed to ``librosa.effects.trim``.
#: A parameter rather than a constant so it can be varied.
DEFAULT_TOP_DB: int = 30

#: A sample at or above this magnitude is treated as clipped. float32 audio
#: decoded from PCM_16 tops out just below 1.0.
CLIP_THRESHOLD: float = 0.999

#: Regenerable, gitignored. Rebuilt in ~60 s by `run_audit`.
AUDIT_CACHE: Path = Path(__file__).resolve().parent.parent / "features" / "audit.parquet"


def probe_file(path: str | Path, top_db: int = DEFAULT_TOP_DB) -> dict[str, object]:
    """Measure the acoustic properties of one clip.

    Container metadata (sample rate, channel count, subtype) is read from the
    header via `soundfile.info` **before** the mono downmix, so a stereo file
    is still reported as stereo. The signal is then loaded at its native rate
    and collapsed to mono for the amplitude and silence measurements.

    A file that fails to load is *recorded* with a populated ``load_error``
    rather than raising, so one bad file cannot abort a full-dataset pass.

    Args:
        path: Path to a ``.wav`` file.
        top_db: Threshold in dB below peak under which audio counts as
            silence, for the leading/trailing trim measurement.

    Returns:
        One flat dict of measurements. ``load_error`` is ``None`` on success
        and every numeric field is ``nan`` on failure.
    """
    path = Path(path)
    row: dict[str, object] = {
        "filename": path.name,
        "filepath": str(path),
        "sr": np.nan,
        "n_channels": np.nan,
        "subtype": None,
        "sf_format": None,
        "n_frames": np.nan,
        "duration": np.nan,
        "peak_amp": np.nan,
        "rms": np.nan,
        "dbfs_peak": np.nan,
        "n_clipped": np.nan,
        "trimmed_duration": np.nan,
        "lead_silence": np.nan,
        "trail_silence": np.nan,
        "silence_fraction": np.nan,
        "load_error": None,
    }

    try:
        info = sf.info(str(path))
        row.update(
            sr=int(info.samplerate),
            n_channels=int(info.channels),
            subtype=info.subtype,
            sf_format=info.format,
            n_frames=int(info.frames),
        )

        # sr=None keeps the native rate: the audit must describe the true
        # files, not a resampled version. The mono downmix is the shared
        # project-wide one from ser.preprocess.
        y, sr = load_audio(path, sr=None, mono=True)
        duration = len(y) / sr if sr else np.nan
        peak = float(np.abs(y).max()) if y.size else 0.0
        rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0

        if y.size:
            trimmed, (start, end) = librosa.effects.trim(y, top_db=top_db)
            trimmed_duration = len(trimmed) / sr
            lead = start / sr
            trail = (len(y) - end) / sr
            silence_fraction = 1.0 - (trimmed_duration / duration) if duration else 0.0
        else:
            trimmed_duration = lead = trail = silence_fraction = 0.0

        row.update(
            duration=duration,
            peak_amp=peak,
            rms=rms,
            dbfs_peak=20 * np.log10(peak) if peak > 0 else -np.inf,
            n_clipped=int((np.abs(y) >= CLIP_THRESHOLD).sum()),
            trimmed_duration=trimmed_duration,
            lead_silence=lead,
            trail_silence=trail,
            silence_fraction=silence_fraction,
        )
    except Exception as exc:  # noqa: BLE001 - a bad file is data, not a crash
        row["load_error"] = f"{type(exc).__name__}: {exc}"

    return row


def run_audit(
    df: pd.DataFrame | None = None,
    top_db: int = DEFAULT_TOP_DB,
    cache: Path | None = None,
) -> pd.DataFrame:
    """Probe every clip and join the measurements onto the metadata.

    Args:
        df: Metadata frame from `ser.metadata.load`. Defaults to loading the
            full inventory. Pass a subset to audit a sample.
        top_db: Silence threshold, forwarded to `probe_file`.
        cache: Parquet path to write. Defaults to `AUDIT_CACHE`; pass
            ``False``-y to skip writing.

    Returns:
        The metadata frame with the `probe_file` measurement columns joined
        on, one row per clip, in the input row order.
    """
    if df is None:
        df = load_metadata()

    probes = pd.DataFrame([probe_file(p, top_db=top_db) for p in df["filepath"]])
    merged = df.merge(probes.drop(columns=["filename"]), on="filepath", how="left")

    if len(merged) != len(df):
        raise ValueError(f"audit join changed row count: {len(df)} in, {len(merged)} out")

    target = AUDIT_CACHE if cache is None else cache
    if target:
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(target, index=False)

    return merged


def summarise(audit: pd.DataFrame) -> dict[str, object]:
    """Build the summary tables the audit document reports.

    Args:
        audit: Frame returned by `run_audit`.

    Returns:
        Named tables and scalars: format consistency, per-actor counts,
        duration and silence distributions overall and sliced by emotion,
        gender and intensity, plus the integrity headlines.
    """
    ok = audit[audit["load_error"].isna()]

    def _dist(series: pd.Series) -> dict[str, float]:
        return {
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "std": float(series.std()),
        }

    return {
        "n_files": len(audit),
        "n_failed": int(audit["load_error"].notna().sum()),
        "failed_files": audit.loc[audit["load_error"].notna(), "filename"].tolist(),
        "sample_rates": ok["sr"].value_counts().to_dict(),
        "subtypes": ok["subtype"].value_counts().to_dict(),
        "channels": ok["n_channels"].value_counts().to_dict(),
        "stereo_files": ok.loc[ok["n_channels"] > 1, "filepath"].tolist(),
        "n_zero_length": int((ok["duration"] == 0).sum()),
        "n_files_with_clipping": int((ok["n_clipped"] > 0).sum()),
        "total_clipped_samples": int(ok["n_clipped"].sum()),
        "clips_per_actor": ok["actor"].value_counts().sort_index().to_dict(),
        "duration": _dist(ok["duration"]),
        "trimmed_duration": _dist(ok["trimmed_duration"]),
        "silence_fraction": _dist(ok["silence_fraction"]),
        "lead_silence": _dist(ok["lead_silence"]),
        "trail_silence": _dist(ok["trail_silence"]),
        "peak_amp": _dist(ok["peak_amp"]),
        "rms": _dist(ok["rms"]),
        "duration_by_emotion": ok.groupby("emotion_label")["duration"].agg(["mean", "std", "min", "max"]),
        "trimmed_by_emotion": ok.groupby("emotion_label")["trimmed_duration"].agg(
            ["mean", "std", "min", "max"]
        ),
        "silence_by_emotion": ok.groupby("emotion_label")["silence_fraction"].mean(),
        "duration_by_gender": ok.groupby("gender")["duration"].agg(["mean", "std"]),
        "duration_by_intensity": ok.groupby("intensity_label")["duration"].agg(["mean", "std"]),
        "rms_by_actor": ok.groupby("actor")["rms"].mean(),
        "rms_by_emotion": ok.groupby("emotion_label")["rms"].mean(),
        "peak_by_actor": ok.groupby("actor")["peak_amp"].mean(),
    }


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    result = run_audit()
    stats = summarise(result)
    print(f"{stats['n_files']} files, {stats['n_failed']} failed to load")
    print("sample rates:", stats["sample_rates"])
    print("channels:", stats["channels"])
    print("silence fraction:", stats["silence_fraction"])
