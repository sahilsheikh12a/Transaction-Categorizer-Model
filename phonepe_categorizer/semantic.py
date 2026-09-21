"""MiniLM semantic neighbours: a hint for rows that end as OTHER.

Everything that reaches this layer has been missed by every curated layer and
by the stage-6 n-gram model. It embeds the merchant with all-MiniLM-L6-v2
(ONNX), lets the 5 most similar known merchants vote, and the engine writes
the winner into the evidence as a suggestion. The row stays OTHER and
flagged; the reviewer decides.

It used to set the category itself. Evaluated against a hand-labelled
statement it got 1 of 40 rows right and cost ~10 points of accuracy, almost
all by replacing a correct OTHER with a wrong guess, so it was demoted.

Files (not in git — `python -m phonepe_categorizer setup-minilm` fetches and
builds them, ~90 MB):

  models/minilm/model.onnx       sentence-transformers/all-MiniLM-L6-v2, pinned
  models/minilm/tokenizer.json   its WordPiece tokenizer
  models/minilm/index.npz        embeddings of every labelled training merchant,
                                 rebuilt by `train` so corrections flow in

Needs onnxruntime, numpy and tokenizers. If any piece is missing the layer
switches itself off and the pipeline falls back to OTHER as before.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models" / "minilm"

REPO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FILES = {
    # name: (path in repo, sha256)
    "model.onnx": ("onnx/model.onnx",
                   "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"),
    "tokenizer.json": ("tokenizer.json",
                       "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"),
}
INDEX_FILE = "index.npz"

K = 5                 # neighbours that vote
MAX_TOKENS = 64       # merchant strings are short; this is generous

ENV_SWITCH = "PHONEPE_CATEGORIZER_MINILM"


@dataclass(frozen=True, slots=True)
class Neighbours:
    category: str
    share: float          # winner's share of the similarity-weighted vote
    similarity: float     # cosine similarity of the nearest neighbour
    nearest: str          # the nearest neighbour's text
    ranked: tuple[tuple[str, float], ...]  # every voted category, best first


class Embedder:
    """MiniLM sentence embeddings: mean-pooled, L2-normalized, 384-d."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_truncation(MAX_TOKENS)
        self._tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_dir / "model.onnx"), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def embed(self, texts: list[str], batch: int = 64):
        import numpy as np

        out = []
        for i in range(0, len(texts), batch):
            enc = self._tok.encode_batch(texts[i:i + batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            hidden = self._session.run(None, {
                "input_ids": ids, "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            })[0]
            vec = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
            out.append(vec / np.linalg.norm(vec, axis=1, keepdims=True))
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, 384), np.float32)


class SemanticModel:
    def __init__(self, embedder: Embedder, vectors, labels: list[str], texts: list[str]):
        self.embedder = embedder
        self.vectors = vectors
        self.labels = labels
        self.texts = texts

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR) -> SemanticModel:
        import numpy as np

        data = np.load(model_dir / INDEX_FILE)
        return cls(Embedder(model_dir), data["vectors"],
                   [str(x) for x in data["labels"]], [str(x) for x in data["texts"]])

    def neighbours(self, text: str) -> Neighbours | None:
        import numpy as np

        if not text or not self.labels:
            return None
        sims = self.vectors @ self.embedder.embed([text])[0]
        top = np.argsort(-sims)[:K]
        votes: dict[str, float] = {}
        for i in top:
            votes[self.labels[i]] = votes.get(self.labels[i], 0.0) + max(float(sims[i]), 0.0)
        total = sum(votes.values()) or 1.0
        ranked = tuple(sorted(((c, v / total) for c, v in votes.items()), key=lambda cv: -cv[1]))
        return Neighbours(ranked[0][0], ranked[0][1], float(sims[top[0]]),
                          self.texts[top[0]], ranked)


# ── setup ────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(model_dir: Path = MODEL_DIR, *, log=print) -> None:
    """Download the pinned model files, verifying each checksum."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, (remote, sha) in FILES.items():
        dest = model_dir / name
        if dest.exists() and _sha256(dest) == sha:
            log(f"  {name}: present, checksum ok")
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}"
        log(f"  {name}: downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        got = _sha256(tmp)
        if got != sha:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: checksum mismatch ({got}), download discarded")
        tmp.replace(dest)
        log(f"  {name}: checksum ok")


def is_installed(model_dir: Path = MODEL_DIR) -> bool:
    return all((model_dir / n).exists() for n in FILES)


def build_index(texts: list[str], labels: list[str], model_dir: Path = MODEL_DIR) -> int:
    """Embed the labelled merchants and write the neighbour index."""
    import numpy as np

    vectors = Embedder(model_dir).embed(texts)
    np.savez_compressed(model_dir / INDEX_FILE, vectors=vectors,
                        labels=np.array(labels), texts=np.array(texts))
    return len(texts)


# ── process-wide default ─────────────────────────────────────────────────

_UNSET = object()
_default: object = _UNSET


def default_model() -> SemanticModel | None:
    global _default
    if _default is _UNSET:
        _default = None
        off = os.environ.get(ENV_SWITCH, "").strip().lower() in ("0", "off", "false")
        if not off and is_installed() and (MODEL_DIR / INDEX_FILE).exists():
            try:
                _default = SemanticModel.load()
            except (ImportError, OSError, KeyError, ValueError):
                _default = None
    return _default  # type: ignore[return-value]


def set_default_model(model: SemanticModel | None) -> None:
    global _default
    _default = model


def reset_default_model() -> None:
    global _default
    _default = _UNSET


@contextmanager
def disabled() -> Iterator[None]:
    global _default
    saved = _default
    _default = None
    try:
        yield
    finally:
        _default = saved


def neighbours(text: str) -> Neighbours | None:
    model = default_model()
    return model.neighbours(text) if model is not None else None


__all__ = [
    "Embedder", "SemanticModel", "Neighbours", "neighbours", "fetch", "build_index",
    "is_installed", "default_model", "set_default_model", "reset_default_model",
    "disabled", "MODEL_DIR",
]
