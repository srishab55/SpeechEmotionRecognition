"""Run-time plumbing: logging, seeds, device, wall-clock budgets and gates.

A gate is a check that must pass before the next stage runs. A failed gate
raises `GateFailure`, which makes the script exit non-zero.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unheard_lib.config import Config

log = logging.getLogger("unheard")


class GateFailure(RuntimeError):
    """Raised when a gate check fails; carries the gate name."""


def setup_logging(path: Path | None = None, level: int = logging.INFO) -> None:
    """Timestamped logging to the console and, optionally, a file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def detect_device() -> str:
    """ "cuda", "mps" or "cpu", whichever torch can use; "cpu" without torch.

    The environment variable `UNHEARD_DEVICE` (cpu, mps or cuda) overrides the choice, for example
    `UNHEARD_DEVICE=cpu` to rule the GPU out when something misbehaves.
    """
    forced = os.environ.get("UNHEARD_DEVICE")
    if forced:
        return forced
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class StageClock:
    """Wall-clock budget for one stage; lower-priority work is skipped when spent."""

    name: str
    budget_s: float
    started: float = 0.0

    def __post_init__(self) -> None:
        self.started = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        return max(0.0, self.budget_s - self.elapsed())

    def expired(self) -> bool:
        return self.remaining() <= 0.0


def record_gate(cfg: Config, name: str, passed: bool, detail: str) -> None:
    """Append a gate result to `outputs/gates.jsonl` and raise if it failed."""
    cfg.outputs.mkdir(parents=True, exist_ok=True)
    row = {
        "gate": name,
        "passed": bool(passed),
        "detail": detail,
        "preset": cfg.preset,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (cfg.outputs / "gates.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    (log.info if passed else log.error)("gate %s %s: %s", name, "PASSED" if passed else "FAILED", detail)
    if not passed:
        raise GateFailure(f"{name}: {detail}")


STAGES = ("data", "features", "classical", "attention", "sequence", "compare", "report")


def parse_cli(argv: list[str] | None = None):
    """Command-line options shared by the script and the notebook.

    `--preset` picks the budget (smoke, overnight, full). `--stage` selects
    which heavy stage to compute; sections of other stages render from their
    saved checkpoints. Unknown arguments (a notebook kernel passes some) are
    ignored. Defaults come from `UNHEARD_PRESET` and `UNHEARD_STAGE`.
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Unheard speech-emotion pipeline")
    parser.add_argument(
        "--preset", default=os.environ.get("UNHEARD_PRESET", "smoke"), choices=("smoke", "overnight", "full")
    )
    parser.add_argument("--stage", default=os.environ.get("UNHEARD_STAGE", "all"), choices=("all", *STAGES))
    args, _ = parser.parse_known_args(argv)
    return args


def gate(cfg: Config, name: str, check: Callable[[], Any], detail: str = "passed") -> Any:
    """Run `check`, record the gate as passed or failed, and return the check's result.

    A check signals failure by raising `AssertionError`; the message becomes the
    recorded detail and the gate failure stops the run.
    """
    try:
        result = check()
    except AssertionError as exc:
        record_gate(cfg, name, False, str(exc))
        raise
    record_gate(cfg, name, True, detail)
    return result
