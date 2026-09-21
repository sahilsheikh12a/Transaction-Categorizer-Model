"""Train the stage-6 ONNX model.

    python -m phonepe_categorizer train transactions2.csv [--db categorizer.db]

Needs the training extras (scikit-learn, skl2onnx); the runtime needs only
onnxruntime and numpy.

WHERE THE LABELS COME FROM
--------------------------
There is no hand-labelled ground truth, so the model learns from what the
curated layers already know, one example per normalized merchant:

  override   the user's corrections                       weight 3, highest priority
  learned    merchants the DB directory already holds
  statement  outgoing rows an identity layer (exact / rule / fuzzy) decided,
             plus person-to-person payments as PERSONAL_TRANSFER
  built-in   the shipped brand directory
  rule       every rule pattern, as a tiny example of its category

Statement rows are labelled with the model switched off, so it never learns
from its own earlier answers. INCOME and OTHER are never targets: income is a
property of a payment's direction, not of a merchant's name, and OTHER is the
absence of an answer.

This is distillation, and it is circular on purpose: the value is not that the
model knows more than the rules, but that it generalizes their spelling
patterns to strings no rule matches exactly ("xyz kirana bhandar" when only
"kirana" is a rule, or a typo the fuzzy guards reject). The cross-validation
numbers therefore measure agreement with the rules on unseen merchants, not
accuracy against reality.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import categories as C
from . import merchants as _merchants
from . import ml as _ml
from . import rules as _rules
from . import semantic as _semantic
from .engine import REVIEW_THRESHOLD, classify_core
from .importer import parse_file
from .normalize import normalize_merchant

# Later origins overwrite earlier ones for the same merchant.
_PRIORITY = ("rule", "built-in", "statement", "learned", "override")
_WEIGHT = {"override": 3.0}

# Inverse regularization strength, swept 1-100 on the shipped statement. At 1
# almost nothing clears the review threshold; from 30 up, one-word names start
# getting 0.87+ on a few shared n-grams. 10 is the cautious end of what works.
DEFAULT_C = 10.0


@dataclass
class Example:
    text: str
    label: str
    origin: str

    @property
    def weight(self) -> float:
        return _WEIGHT.get(self.origin, 1.0)


@dataclass
class TrainReport:
    examples: int
    by_origin: dict[str, int]
    by_label: dict[str, int]
    cv_accuracy: float
    cv_gated_coverage: float
    cv_gated_accuracy: float
    unresolved_rows: int = 0
    resolved_rows: int = 0
    resolved_by_label: dict[str, int] = field(default_factory=dict)
    samples: list[tuple[str, str, float]] = field(default_factory=list)
    output: str = ""
    semantic_index: int = 0

    def as_text(self) -> str:
        lines = [
            f"trained on {self.examples} merchants  "
            + "  ".join(f"{k} {v}" for k, v in self.by_origin.items()),
            "  labels   " + ", ".join(f"{k} {v}" for k, v in self.by_label.items()),
            "",
            "5-fold cross-validation on statement merchants "
            "(agreement with the curated layers, not ground truth):",
            f"  accuracy                    {self.cv_accuracy:.1%}",
            f"  p >= {REVIEW_THRESHOLD:.2f} coverage          {self.cv_gated_coverage:.1%}",
            f"  p >= {REVIEW_THRESHOLD:.2f} accuracy          {self.cv_gated_accuracy:.1%}",
        ]
        if self.unresolved_rows:
            lines += [
                "",
                f"on the {self.unresolved_rows} statement rows the curated layers leave "
                f"under the threshold, the model now resolves {self.resolved_rows}:",
                "  " + ", ".join(f"{k} {v}" for k, v in self.resolved_by_label.items()),
            ]
            lines += [f"    {p:.2f}  {lab:<20} {text}" for text, lab, p in self.samples]
        if self.output:
            lines += ["", f"model written to {self.output}"]
        if self.semantic_index:
            lines += [f"MiniLM index rebuilt with {self.semantic_index} merchants"]
        return "\n".join(lines)


# ── dataset ──────────────────────────────────────────────────────────────

@contextmanager
def _curated_only() -> Iterator[None]:
    """Both model layers off: labels must come from the curated layers alone."""
    with _ml.disabled(), _semantic.disabled():
        yield


def _statement_label(res) -> str | None:
    if res.needs_review:
        return None
    if res.source in C.IDENTITY_SOURCES and res.txn_type == "PAYMENT_TO_MERCHANT":
        return res.category
    if res.source == C.SRC_TXN_TYPE and res.txn_type == "PAYMENT_TO_PERSON":
        return C.PERSONAL_TRANSFER
    return None


def build_dataset(
    statements: list[Path],
    *,
    overrides: dict[str, str] | None = None,
    learned: dict[str, str] | None = None,
) -> list[Example]:
    """One labelled example per normalized merchant."""
    by_origin: dict[str, dict[str, str]] = {o: {} for o in _PRIORITY}

    for rule in _rules.RULES:
        by_origin["rule"][normalize_merchant(rule.pattern)] = rule.category
    by_origin["built-in"].update(_merchants.BUILT_IN)
    by_origin["learned"].update(learned or {})
    by_origin["override"].update(overrides or {})

    with _curated_only():
        for path in statements:
            for txn in parse_file(path).rows:
                if txn.is_credit:
                    continue
                res = classify_core(txn.counterparty, is_credit=False,
                                    direction=txn.direction, learned=learned)
                label = _statement_label(res)
                if label and res.normalized_merchant:
                    by_origin["statement"][res.normalized_merchant] = label

    merged: dict[str, Example] = {}
    for origin in _PRIORITY:
        for text, label in by_origin[origin].items():
            label = C.coerce(label)
            if text and label not in (C.INCOME, C.OTHER):
                merged[text] = Example(text, label, origin)
    return sorted(merged.values(), key=lambda e: e.text)


def _fallback_rows(statements: list[Path], learned: dict[str, str] | None = None):
    """Outgoing rows the curated layers leave to the fallback — stage 6's input."""
    out = []
    with _curated_only():
        for path in statements:
            for txn in parse_file(path).rows:
                if txn.is_credit:
                    continue
                res = classify_core(txn.counterparty, direction=txn.direction, learned=learned)
                if res.source == C.SRC_FALLBACK:
                    out.append(txn)
    return out


# ── training ─────────────────────────────────────────────────────────────

def _matrix(texts: list[str]):
    from scipy.sparse import csr_matrix

    rows, cols, vals = [], [], []
    for i, text in enumerate(texts):
        for j, v in _ml.features(text).items():
            rows.append(i)
            cols.append(j)
            vals.append(v)
    return csr_matrix((vals, (rows, cols)), shape=(len(texts), _ml.FEATURE_DIM),
                      dtype="float32")


def _fit(X, y, w, c: float):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=c, max_iter=5000)
    clf.fit(X, y, sample_weight=w)
    return clf


def cross_validate(examples: list[Example], c: float = DEFAULT_C, folds: int = 5):
    """(accuracy, gated coverage, gated accuracy) on held-out statement merchants."""
    import numpy as np
    from sklearn.model_selection import KFold

    X = _matrix([e.text for e in examples])
    y = np.array([e.label for e in examples])
    w = np.array([e.weight for e in examples])
    scored = np.array([e.origin == "statement" for e in examples])

    hits = total = gated = gated_hits = 0
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        clf = _fit(X[tr], y[tr], w[tr], c)
        proba = clf.predict_proba(X[te])
        pred = clf.classes_[proba.argmax(1)]
        conf = proba.max(1)
        for k, idx in enumerate(te):
            if not scored[idx]:
                continue
            ok = pred[k] == y[idx]
            total += 1
            hits += ok
            if conf[k] >= REVIEW_THRESHOLD:
                gated += 1
                gated_hits += ok
    return (hits / total if total else 0.0,
            gated / total if total else 0.0,
            gated_hits / gated if gated else 0.0)


def export_onnx(clf, path: Path, *, examples: int) -> None:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    model = convert_sklearn(
        clf,
        initial_types=[("features", FloatTensorType([None, _ml.FEATURE_DIM]))],
        options={id(clf): {"zipmap": False}},
        target_opset={"": 17, "ai.onnx.ml": 3},
    )
    meta = {
        "feature_spec": _ml.FEATURE_SPEC,
        "classes": json.dumps([str(c) for c in clf.classes_]),
        "examples": str(examples),
        "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def train(
    statements: list[Path],
    *,
    overrides: dict[str, str] | None = None,
    learned: dict[str, str] | None = None,
    output: Path = _ml.MODEL_PATH,
    c: float = DEFAULT_C,
) -> TrainReport:
    import numpy as np

    examples = build_dataset(statements, overrides=overrides, learned=learned)
    labels = {e.label for e in examples}
    if len(labels) < 2:
        raise ValueError("need at least two categories to train")

    acc, cov, gacc = cross_validate(examples, c)

    X = _matrix([e.text for e in examples])
    y = np.array([e.label for e in examples])
    clf = _fit(X, y, np.array([e.weight for e in examples]), c)
    export_onnx(clf, output, examples=len(examples))

    # The exported graph must reproduce sklearn exactly, or the CV numbers
    # above describe a different model from the one shipped.
    model = _ml.Model.load(output)
    probe = X[: min(200, X.shape[0])].toarray()
    _, onnx_proba = model._session.run(None, {model._input: probe})
    if not np.allclose(onnx_proba, clf.predict_proba(probe), atol=1e-4):
        raise RuntimeError("exported ONNX model disagrees with the trained classifier")

    report = TrainReport(
        examples=len(examples),
        by_origin={o: sum(e.origin == o for e in examples) for o in _PRIORITY},
        by_label={lab: sum(e.label == lab for e in examples) for lab in sorted(labels)},
        cv_accuracy=acc, cv_gated_coverage=cov, cv_gated_accuracy=gacc,
        output=str(output),
    )

    # Re-run the fallback rows through the real engine with the new model, so
    # its guards (masked handles, ambiguous phrases, commercial keywords) apply.
    rows = _fallback_rows(statements, learned)
    report.unresolved_rows = len(rows)
    seen: dict[str, tuple[str, float]] = {}
    with _curated_only():
        _ml.set_default_model(model)
        for txn in rows:
            res = classify_core(txn.counterparty, direction=txn.direction, learned=learned)
            if res.source != C.SRC_ML_MODEL:
                continue
            report.resolved_rows += 1
            report.resolved_by_label[res.category] = (
                report.resolved_by_label.get(res.category, 0) + 1
            )
            seen[res.normalized_merchant] = (res.category, res.confidence)
    report.samples = sorted(((t, lab, p) for t, (lab, p) in seen.items()),
                            key=lambda s: -s[2])[:25]

    if _semantic.is_installed():
        report.semantic_index = build_semantic_index(examples)
    return report


def build_semantic_index(examples: list[Example]) -> int:
    """Rebuild stage 7's neighbour index from the same labelled merchants."""
    n = _semantic.build_index([e.text for e in examples], [e.label for e in examples])
    _semantic.reset_default_model()
    return n


__all__ = [
    "train", "build_dataset", "build_semantic_index", "cross_validate",
    "Example", "TrainReport", "DEFAULT_C",
]
