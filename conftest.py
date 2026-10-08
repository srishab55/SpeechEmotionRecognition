"""Test setup: import torch before scikit-learn and XGBoost load their own OpenMP (see unheard_lib/__init__.py)."""

import contextlib

with contextlib.suppress(ImportError):
    import torch  # noqa: F401
