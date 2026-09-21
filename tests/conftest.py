"""Shared fixtures.

The deterministic tests pin exact fallback behaviour (OTHER, 0.55 short-name
transfers, ...). Those contracts must not move every time the ONNX model is
retrained, so stage 6 is off for every test unless it opts in with the
`onnx_model` fixture from `test_ml.py`.
"""
from __future__ import annotations

import pytest

from phonepe_categorizer import ml


@pytest.fixture(autouse=True)
def _no_ml_by_default():
    with ml.disabled():
        yield
