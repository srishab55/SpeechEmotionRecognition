"""Arrange the RAVDESS speech recordings into the layout the pipeline reads.

The official download (Zenodo record 1188976, file `Audio_Speech_Actors_01-24.zip`) unpacks to
`Actor_01/ ... Actor_24/` with seven-field names such as `03-01-06-01-02-01-12.wav`. The pipeline reads
`data/data/Person01/ ... Person24/` with five-field names (`06-01-02-01-12.wav`: emotion, intensity, statement,
repetition, actor), because the first two fields, modality and vocal channel, are constant for this subset.

    python scripts/prepare_data.py PATH/TO/unzipped/folder

Files are copied, never moved, and the unzipped download is left untouched.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

N_ACTORS, CLIPS_PER_ACTOR = 24, 60
CONSTANT_FIELDS = ("03", "01")  # audio-only, speech


def prepare(source: Path, destination: Path) -> int:
    """Copy every clip into `destination/PersonNN/` under its five-field name; returns the clip count."""
    copied = 0
    for actor_dir in sorted(source.rglob("Actor_*")):
        if not actor_dir.is_dir():
            continue
        number = int(actor_dir.name.split("_")[1])
        target = destination / f"Person{number:02d}"
        target.mkdir(parents=True, exist_ok=True)
        for clip in sorted(actor_dir.glob("*.wav")):
            fields = clip.stem.split("-")
            if len(fields) != 7 or tuple(fields[:2]) != CONSTANT_FIELDS:
                raise ValueError(
                    f"{clip.name} is not a RAVDESS speech clip (expected 03-01-... with 7 fields)"
                )
            if int(fields[6]) != number:
                raise ValueError(f"{clip.name} sits in {actor_dir.name} but names actor {fields[6]}")
            shutil.copyfile(clip, target / ("-".join(fields[2:]) + ".wav"))
            copied += 1
    return copied


def check(destination: Path) -> None:
    """Fail loudly unless the layout holds 24 actors with 60 clips each."""
    folders = sorted(destination.glob("Person*"))
    if len(folders) != N_ACTORS:
        raise SystemExit(f"expected {N_ACTORS} actor folders in {destination}, found {len(folders)}")
    for folder in folders:
        n = len(list(folder.glob("*.wav")))
        if n != CLIPS_PER_ACTOR:
            raise SystemExit(f"{folder.name} has {n} clips, expected {CLIPS_PER_ACTOR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path, help="folder that contains the Actor_01 ... Actor_24 folders")
    parser.add_argument(
        "--destination", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "data"
    )
    args = parser.parse_args()
    if not args.source.is_dir():
        sys.exit(f"{args.source} is not a folder")
    count = prepare(args.source, args.destination)
    check(args.destination)
    print(f"copied {count} clips into {args.destination}")


if __name__ == "__main__":
    main()
