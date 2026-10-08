"""Helper library for the Unheard speech-emotion pipeline.

The pipeline script (`unheard_pipeline.py`) orchestrates and reports; every
piece of logic lives here so it can be unit-tested. The existing `ser` package
supplies the audio loader, the frozen folds, the scorer and the per-speaker
normaliser, and is reused rather than re-implemented.
"""

# torch is imported before scikit-learn, XGBoost and Optuna on purpose. On macOS each of them can bring its own
# copy of the OpenMP threading library, and loading torch after them can crash the process (segmentation fault)
# on the first parallel tensor operation. Loading it first avoids the clash. Silent when torch is not installed.
import contextlib

with contextlib.suppress(ImportError):
    import torch  # noqa: F401

from unheard_lib.config import (
    MIN_DETECTABLE_PP,
    MODEL_COLOR,
    PALETTE,
    Config,
    get_config,
)

__all__ = ["MIN_DETECTABLE_PP", "MODEL_COLOR", "PALETTE", "Config", "get_config"]
