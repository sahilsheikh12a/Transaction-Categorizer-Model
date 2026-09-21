"""Stage 6 — ONNX model, for strings every curated layer missed.

Runs only where the deterministic pipeline would otherwise fall back below
`REVIEW_THRESHOLD`: no override, no directory hit, no rule, not a person-name
shape, no fuzzy match. It never outranks a curated layer and its answers are
never learned into the merchant directory (see `db.repository`), so a wrong
prediction cannot spread into exact or fuzzy matching.

The model is a linear classifier over hashed character n-grams — it reads
spelling patterns ("... kirana bhandar", "... medicos"), not meaning. It is
trained by `train.py` and shipped as `models/categorizer.onnx`.

Featurization happens here in plain Python rather than inside the ONNX graph,
so the graph is one matrix multiply and any runtime (including onnxruntime on
Android) can execute it. The featurizer is ~20 lines and deliberately simple
to port; `FEATURE_SPEC` is written into the model's metadata and checked at
load, so a model trained against a different featurizer is refused instead of
silently producing garbage.

Optional by construction: if onnxruntime/numpy are not installed, or the model
file is missing, `predict` returns None and the pipeline behaves exactly as it
did without this layer.
"""
from __future__ import annotations

import json
import math
import os
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "categorizer.onnx"

# Bumped whenever `features` changes. A model whose metadata disagrees is not
# loaded.
FEATURE_DIM = 1 << 14
NGRAM_RANGE = (2, 4)
FEATURE_SPEC = f"crc32 char{NGRAM_RANGE[0]}-{NGRAM_RANGE[1]}+word l2 dim={FEATURE_DIM} v1"

# A model answer sits below every curated layer (RULE is 0.90), whatever
# probability the model reports.
MAX_CONFIDENCE = 0.89

# Set to "0"/"off" to disable the layer without uninstalling anything.
ENV_SWITCH = "PHONEPE_CATEGORIZER_ML"


def features(text: str) -> dict[int, float]:
    """Sparse feature vector for a normalized merchant string.

    Each word contributes itself ("w:kirana") and its character n-grams with
    space padding (" ki", "kir", ..., "na "). Grams are hashed with CRC32 —
    stable across processes and platforms, unlike Python's `hash()` — and the
    vector is L2-normalized.
    """
    counts: dict[int, float] = {}
    lo, hi = NGRAM_RANGE
    for word in text.split():
        grams = [f"w:{word}"]
        padded = f" {word} "
        for n in range(lo, hi + 1):
            grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
        for g in grams:
            idx = zlib.crc32(g.encode("utf-8")) % FEATURE_DIM
            counts[idx] = counts.get(idx, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values()))
    return {k: v / norm for k, v in counts.items()} if norm else {}


@dataclass(frozen=True, slots=True)
class Prediction:
    category: str
    probability: float


class Model:
    """A loaded ONNX categorizer. Construct with `Model.load`."""

    def __init__(self, session, classes: list[str]):
        self._session = session
        self._input = session.get_inputs()[0].name
        self.classes = classes

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> Model:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # one row at a time; threads only add overhead
        session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"],
        )
        meta = session.get_modelmeta().custom_metadata_map
        spec = meta.get("feature_spec")
        if spec != FEATURE_SPEC:
            raise ValueError(
                f"model was trained with features {spec!r}, "
                f"this code produces {FEATURE_SPEC!r} — retrain it"
            )
        return cls(session, json.loads(meta["classes"]))

    def predict(self, text: str) -> Prediction | None:
        feats = features(text)
        if not feats:
            return None
        import numpy as np

        x = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        for idx, val in feats.items():
            x[0, idx] = val
        _, probs = self._session.run(None, {self._input: x})
        best = int(np.argmax(probs[0]))
        return Prediction(self.classes[best], float(probs[0][best]))


# ── process-wide default ─────────────────────────────────────────────────
# Loaded once, lazily, on the first row that reaches this layer.

_UNSET = object()
_default: object = _UNSET


def default_model() -> Model | None:
    """The shipped model, or None if the layer is disabled or unavailable."""
    global _default
    if _default is _UNSET:
        _default = None
        if os.environ.get(ENV_SWITCH, "").strip().lower() not in ("0", "off", "false"):
            try:
                if MODEL_PATH.exists():
                    _default = Model.load(MODEL_PATH)
            except (ImportError, ValueError):
                _default = None
    return _default  # type: ignore[return-value]


def set_default_model(model: Model | None) -> None:
    """Replace the process-wide model. `None` disables the layer."""
    global _default
    _default = model


def reset_default_model() -> None:
    """Forget the cached model so the next call reloads from disk."""
    global _default
    _default = _UNSET


@contextmanager
def disabled() -> Iterator[None]:
    """Run the pipeline without this layer, restoring the previous state after.

    Training labels its data this way: a model must never learn from its own
    earlier answers.
    """
    global _default
    saved = _default
    _default = None
    try:
        yield
    finally:
        _default = saved


def predict(text: str) -> Prediction | None:
    model = default_model()
    return model.predict(text) if model is not None else None


__all__ = [
    "Model", "Prediction", "features", "predict", "default_model",
    "set_default_model", "reset_default_model", "disabled",
    "MODEL_PATH", "FEATURE_DIM", "FEATURE_SPEC", "MAX_CONFIDENCE",
]
