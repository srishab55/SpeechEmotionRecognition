"""Parse RAVDESS filenames into a DataFrame.

This module is the single source of truth for *which clips exist*. It reads
filenames and directory names only -- it never opens an audio file. Nothing
here is fit on data, so there is no train/test distinction to respect.

Filename schema in this repository
----------------------------------
The dataset as packaged here uses **five** hyphen-separated fields::

    emotion-intensity-statement-repetition-actor.wav
    e.g. data/data/Person01/06-02-01-02-01.wav

Canonical RAVDESS filenames carry seven fields, the first two being
``modality`` and ``vocal-channel``. Both are constant for the speech subset
(``03`` audio-only, ``01`` speech) and were dropped when this copy was
repackaged. `load` re-adds them as constant columns so the resulting frame
matches the canonical RAVDESS shape.

The actor is encoded twice -- as field 5 and as the ``PersonNN`` directory.
`parse_filename` checks the two agree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

#: Root holding the per-actor directories. Note the doubled ``data`` segment.
DATA_ROOT: Path = Path(__file__).resolve().parent.parent / "data" / "data"

#: Constant fields stripped from these filenames: audio-only, speech.
MODALITY: str = "03"
VOCAL_CHANNEL: str = "01"

#: Ordered field names of the five-field filename schema.
FIELDS: tuple[str, ...] = (
    "emotion",
    "intensity",
    "statement",
    "repetition",
    "actor_code",
)

EMOTIONS: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITIES: dict[str, str] = {"01": "normal", "02": "strong"}

#: Statement 01 is "Kids are talking by the door", 02 is "Dogs are sitting by
#: the door". Held as short keys because they double as plot labels.
STATEMENTS: dict[str, str] = {"01": "kids", "02": "dogs"}

REPETITIONS: dict[str, str] = {"01": "1st", "02": "2nd"}

#: Actor directories are named ``Person01`` .. ``Person24``.
_ACTOR_DIR_RE = re.compile(r"^Person(\d{2})$")

EXPECTED_ROWS = 1440
EXPECTED_ACTORS = 24
CLIPS_PER_ACTOR = 60
NEUTRAL_CLIPS = 96
OTHER_EMOTION_CLIPS = 192


def parse_filename(path: str | Path) -> dict[str, object]:
    """Parse one RAVDESS filename into its fields.

    Strict by design: a filename that is not exactly five known-valued fields
    raises rather than being coerced. A canonical seven-field RAVDESS file
    dropped into this tree is therefore a loud failure, not a silent
    mis-parse that would shift every field by two positions.

    Args:
        path: Path to a ``.wav`` file. Only the stem and the parent directory
            name are read; the file is never opened and need not exist.

    Returns:
        Field codes, their human-readable labels, ``actor`` as an int,
        ``gender``, and the originating ``filename`` / ``filepath``.

    Raises:
        ValueError: If the stem does not split into exactly five fields, if
            any field holds an unknown code, or if field 5 disagrees with the
            ``PersonNN`` directory the file sits in.
    """
    path = Path(path)
    parts = path.stem.split("-")
    if len(parts) != len(FIELDS):
        raise ValueError(
            f"{path.name!r}: expected {len(FIELDS)} hyphen-separated fields "
            f"({'-'.join(FIELDS)}), got {len(parts)}. Canonical seven-field "
            f"RAVDESS names are not accepted here -- see module docstring."
        )

    emotion, intensity, statement, repetition, actor_code = parts

    for value, valid, name in (
        (emotion, EMOTIONS, "emotion"),
        (intensity, INTENSITIES, "intensity"),
        (statement, STATEMENTS, "statement"),
        (repetition, REPETITIONS, "repetition"),
    ):
        if value not in valid:
            raise ValueError(f"{path.name!r}: unknown {name} code {value!r}; expected one of {sorted(valid)}")

    if not re.fullmatch(r"\d{2}", actor_code):
        raise ValueError(f"{path.name!r}: actor code {actor_code!r} is not two digits")
    actor = int(actor_code)
    if not 1 <= actor <= EXPECTED_ACTORS:
        raise ValueError(f"{path.name!r}: actor {actor} outside 1..{EXPECTED_ACTORS}")

    dir_match = _ACTOR_DIR_RE.match(path.parent.name)
    if dir_match is not None and int(dir_match.group(1)) != actor:
        raise ValueError(f"{path.name!r}: actor field {actor} disagrees with directory {path.parent.name!r}")

    return {
        "filename": path.name,
        "filepath": str(path),
        "modality": MODALITY,
        "vocal_channel": VOCAL_CHANNEL,
        "emotion": emotion,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
        "emotion_label": EMOTIONS[emotion],
        "intensity_label": INTENSITIES[intensity],
        "statement_label": STATEMENTS[statement],
        "repetition_label": REPETITIONS[repetition],
        # RAVDESS convention: odd-numbered actors are male, even are female.
        "gender": "male" if actor % 2 else "female",
    }


def load(root: Path | None = None, validate: bool = True) -> pd.DataFrame:
    """Build the clip inventory for the whole dataset.

    Rows are sorted by ``(actor, emotion, intensity, statement, repetition)``
    and the index reset, so row order is identical on every machine
    regardless of filesystem glob order. Downstream code that indexes by
    position depends on this.

    Args:
        root: Directory holding the ``PersonNN`` folders. Defaults to
            `DATA_ROOT`.
        validate: Run the dataset invariants and raise on violation. Pass
            ``False`` only to inspect a deliberately incomplete tree.

    Returns:
        One row per ``.wav`` file, with the columns listed in
        `parse_filename`.

    Raises:
        FileNotFoundError: If ``root`` does not exist or holds no ``.wav``
            files.
        ValueError: From `parse_filename` on a malformed name, or from
            `validate_frame` on a dataset-level invariant breach.
    """
    root = Path(root) if root is not None else DATA_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"Audio root {root} not found. Audio is gitignored -- place the "
            f"RAVDESS clips in data/data/PersonNN/ before running."
        )

    paths = sorted(root.glob("Person*/*.wav"))
    if not paths:
        raise FileNotFoundError(f"No .wav files under {root}/Person*/")

    df = pd.DataFrame([parse_filename(p) for p in paths])
    df = df.sort_values(
        ["actor", "emotion", "intensity", "statement", "repetition"],
        kind="mergesort",
    ).reset_index(drop=True)

    if validate:
        validate_frame(df)
    return df


def validate_frame(df: pd.DataFrame) -> None:
    """Assert the documented RAVDESS invariants hold on ``df``.

    These are the facts the whole project rests on. They are checked, not
    assumed, because a partial download or a stray file would otherwise
    surface much later as an inexplicable model result. Every failure is
    collected before raising, so one run reports the full picture.

    Args:
        df: A frame in the shape `load` returns.

    Raises:
        ValueError: Listing every invariant that failed, with expected and
            observed values.
    """
    problems: list[str] = []

    if len(df) != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS} clips, found {len(df)}")

    if df["filename"].duplicated().any():
        dupes = sorted(df.loc[df["filename"].duplicated(), "filename"].unique())
        problems.append(f"duplicate filenames: {dupes[:5]}")

    actors = sorted(df["actor"].unique())
    if actors != list(range(1, EXPECTED_ACTORS + 1)):
        problems.append(f"expected actors 1..{EXPECTED_ACTORS}, found {actors}")

    per_actor = df["actor"].value_counts()
    bad_actors = per_actor[per_actor != CLIPS_PER_ACTOR]
    if not bad_actors.empty:
        problems.append(f"expected {CLIPS_PER_ACTOR} clips per actor; off: {bad_actors.to_dict()}")

    counts = df["emotion_label"].value_counts()
    for label in EMOTIONS.values():
        expected = NEUTRAL_CLIPS if label == "neutral" else OTHER_EMOTION_CLIPS
        found = int(counts.get(label, 0))
        if found != expected:
            problems.append(f"emotion {label}: expected {expected}, found {found}")

    # Neutral was never recorded at strong intensity -- that is why it has
    # half the clips of every other emotion.
    strong_neutral = int(((df["emotion"] == "01") & (df["intensity"] == "02")).sum())
    if strong_neutral:
        problems.append(f"found {strong_neutral} strong-intensity neutral clips; RAVDESS has none")

    genders = df.groupby("gender")["actor"].nunique().to_dict()
    if genders != {"female": 12, "male": 12}:
        problems.append(f"expected 12 actors per gender, found {genders}")

    expected_gender = df["actor"].map(lambda a: "male" if a % 2 else "female")
    parity_ok = df["gender"] == expected_gender
    if not parity_ok.all():
        problems.append(f"{int((~parity_ok).sum())} rows where gender != actor parity")

    if df.isna().any().any():
        null_cols = sorted(df.columns[df.isna().any()])
        problems.append(f"null values in columns {null_cols}")

    if problems:
        raise ValueError("Dataset invariants failed:\n  - " + "\n  - ".join(problems))


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    frame = load()
    print(frame.shape)
    print(frame["emotion_label"].value_counts())
