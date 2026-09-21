"""Shared fixtures.

The deterministic tests pin exact fallback behaviour (OTHER, 0.55 short-name
transfers, ...). Those contracts must not move every time the ONNX model is
retrained, so both model layers (stage 6 n-gram, stage 7 MiniLM) are off for
every test unless it opts in — see `test_ml.py` and `test_semantic.py`.
"""
from __future__ import annotations

import pytest

from phonepe_categorizer import ml, semantic


@pytest.fixture(autouse=True)
def _no_models_by_default():
    with ml.disabled(), semantic.disabled():
        yield
