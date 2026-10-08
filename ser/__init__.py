"""Speech emotion recognition on RAVDESS (ISB AMPBA foundation project).

`SEED` is defined here and imported everywhere else. It used to be redeclared
in four separate modules, which meant nothing stopped them drifting apart.
"""

from __future__ import annotations

import os
import random

#: The one seed. Import from here; do not redeclare.
SEED: int = 42


def set_seeds(seed: int = SEED) -> None:
    """Fix every random source we can reach.

    Passing ``random_state=`` to an estimator only covers that estimator.
    This covers the global generators that otherwise make a rerun
    irreproducible -- numpy's legacy global RNG, the stdlib ``random``
    module, and torch for the neural models.

    Args:
        seed: The value to seed with. Defaults to `SEED`.

    Note:
        ``PYTHONHASHSEED`` is set here for completeness, but CPython reads it
        at interpreter startup. To make string hashing deterministic it must
        be exported *before* Python launches::

            PYTHONHASHSEED=42 python -m pytest

        Nothing we currently do depends on string hash order, so this is
        belt-and-braces rather than load-bearing.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    # ImportError covers torch simply not being installed -- it lives in
    # requirements-deep.txt, so a classical-only environment is expected and
    # fine. OSError covers torch being *present but unloadable*: on Windows,
    # Smart App Control blocks its unsigned DLLs with WinError 4551, which is
    # an OSError, not an ImportError. Catching only the latter turned a seeding
    # call into a hard crash for every driver in the project, including the
    # ones that never touch torch.
    try:  # pragma: no cover - depends on the local environment
        import torch
    except (ImportError, OSError):
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
