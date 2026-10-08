"""Build one clip table (`data/dataset_index.csv`) from the four corpora on disk.

Expected layout under the project folder (see README, "Get the audio"):

    data/data/Person01 ... Person24/*.wav      RAVDESS, 5-field names (06-02-01-02-12.wav)
    data_external/cremad/AudioWAV/*.wav        CREMA-D  (1044_MTI_ANG_XX.wav)
    data_external/toronto-emotional-speech-set-tess/**/*.wav   TESS (OAF_back_angry.wav)
    data_external/surrey-audiovisual-expressed-emotion-savee/ALL/*.wav   SAVEE (DC_sa01.wav)

Every clip gets one of the eight RAVDESS labels. A corpus simply never uses the labels it lacks
(CREMA-D has no calm or surprised; TESS and SAVEE have no calm). Paths are stored relative to the project.
"""

from __future__ import annotations

import pandas as pd

from multicorpus import CORPORA, LABELS, ROOT

EXT = ROOT / "data_external"
RAVDESS_CODES = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry",
                 "06": "fearful", "07": "disgust", "08": "surprised"}
CREMA_CODES = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}
TESS_CODES = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry", "fear": "fearful",
              "disgust": "disgust", "ps": "surprised"}
SAVEE_CODES = {"n": "neutral", "h": "happy", "sa": "sad", "a": "angry", "f": "fearful", "d": "disgust",
               "su": "surprised"}


def _rel(p) -> str:
    return p.relative_to(ROOT).as_posix()


def _ravdess() -> list[dict]:
    rows = []
    for p in sorted((ROOT / "data" / "data").glob("Person*/*.wav")):
        f = p.stem.split("-")
        assert len(f) == 5, f"unexpected RAVDESS name {p.name}"
        actor = int(f[4])
        rows.append(dict(filename=p.name, filepath=_rel(p), source="RAVDESS", speaker_id=f"RAVDESS_{actor:02d}",
                         gender="male" if actor % 2 else "female", emotion_label=RAVDESS_CODES[f[0]]))
    return rows


def _crema() -> list[dict]:
    rows = []
    for p in (EXT / "cremad").rglob("*.wav"):
        actor, _sentence, emo, _level = p.stem.split("_")
        name = p.name.replace("_SAD_X.wav", "_SAD_XX.wav")  # one Kaggle file lacks an X; canonical name used
        rows.append(dict(filename=name, filepath=_rel(p), source="CREMA-D", speaker_id=f"CREMA-D_{actor}",
                         gender="unknown", emotion_label=CREMA_CODES[emo]))
    return sorted(rows, key=lambda r: r["filename"])


def _tess() -> list[dict]:
    rows = []
    for p in (EXT / "toronto-emotional-speech-set-tess").rglob("*.wav"):
        speaker = p.parent.name.split("_")[0]  # folder, not file name: one file is misspelt "OA_..."
        emo = p.stem.split("_")[-1]
        name = "OAF_" + p.name[3:] if p.name.startswith("OA_") else p.name  # canonical name for the misspelt file
        rows.append(dict(filename=name, filepath=_rel(p), source="TESS", speaker_id=f"TESS_{speaker}",
                         gender="female", emotion_label=TESS_CODES[emo]))
    return sorted(rows, key=lambda r: r["filename"])


def _savee() -> list[dict]:
    rows = []
    for p in sorted((EXT / "surrey-audiovisual-expressed-emotion-savee").rglob("*.wav")):
        speaker, rest = p.stem.split("_")
        code = "".join(c for c in rest if c.isalpha())
        rows.append(dict(filename=p.name, filepath=_rel(p), source="SAVEE", speaker_id=f"SAVEE_{speaker}",
                         gender="male", emotion_label=SAVEE_CODES[code]))
    return rows


def build_index() -> pd.DataFrame:
    """Scan the four corpora and return the clip table in a fixed order (corpus, then file name)."""
    df = pd.DataFrame(_ravdess() + _crema() + _tess() + _savee())
    df["label_idx"] = df["emotion_label"].map({l: i for i, l in enumerate(LABELS)})
    assert df["filename"].is_unique, "file names must be unique across corpora"
    return df.reset_index(drop=True)


EXPECTED = {"RAVDESS": (1440, 24), "CREMA-D": (7442, 91), "TESS": (2800, 2), "SAVEE": (480, 4)}


def check_index(df: pd.DataFrame) -> None:
    """Assert corpus sizes, speaker counts and (when present) agreement with the reference study's table."""
    for corpus in CORPORA:
        sub = df[df.source == corpus]
        want = EXPECTED[corpus]
        got = (len(sub), sub.speaker_id.nunique())
        assert got == want, f"{corpus}: found {got[0]} clips / {got[1]} speakers, expected {want[0]} / {want[1]}"
    ref_path = ROOT / "reference" / "dataset_index_reference.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        same = (len(ref) == len(df) and (ref.filename.values == df.filename.values).all()
                and (ref.speaker_id.values == df.speaker_id.values).all()
                and (ref.emotion_label.values == df.emotion_label.values).all())
        assert same, "clip order, speakers or labels differ from reference/dataset_index_reference.csv"
        gender = dict(zip(ref.speaker_id, ref.gender))
        df["gender"] = df.speaker_id.map(gender)  # CREMA-D genders come from the reference table


def prepare(path=None) -> pd.DataFrame:
    path = path or ROOT / "data" / "dataset_index.csv"
    df = build_index()
    check_index(df)
    df.to_csv(path, index=False)
    return df


def load_index() -> pd.DataFrame:
    path = ROOT / "data" / "dataset_index.csv"
    if not path.exists():
        return prepare(path)
    return pd.read_csv(path)
