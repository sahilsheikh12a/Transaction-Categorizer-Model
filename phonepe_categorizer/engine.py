"""Classification engine — the layered V1 pipeline.

    RAW TRANSACTION
      -> merchant normalization          (normalize.py)
      -> transaction-type detection      (txn_type.py)      Stage 1
      -> user merchant overrides         (overrides.py)     Stage 2
      -> exact merchant lookup           (merchants.py)     Stage 3
      -> keyword / rule classification   (rules.py)         Stage 4
      -> fuzzy merchant matching         (fuzzy.py)         Stage 5
      -> ONNX model, below-threshold only (ml.py)           Stage 6
      -> fallback OTHER + needs_review                      Stage 7

Two design decisions worth stating explicitly, because they are the ones a
reader will want to argue with:

1. **Type detection runs before categorization.** A person-to-person transfer
   is not a spending category, so asking "which category?" first is the wrong
   question. This is what stops "Paid to Ritik" becoming SHOPPING.

2. **User overrides outrank type detection.** If the user says the person-named
   vendor "CHETMANI DWARKAPRASAD SAHU" is GROCERIES, that is the answer — even
   though the string is a perfect person name. Small Indian vendors routinely
   collect on personal UPI accounts, and only the user can resolve that.

No LLM and no network. Stages 1–5 are hand-curated; stage 6 is a small
ONNX classifier that only sees strings those stages could not place, and only
clears the review flag when it is at least `REVIEW_THRESHOLD` sure. Inference
is deterministic too, so the same input still always yields the same output,
which is what makes the review queue and the evaluation report trustworthy.

`classify_core` is the whole engine and takes plain dicts for the two pieces of
learned state (user overrides, known merchants). Persistence lives in
`phonepe_categorizer.db` and is entirely optional — the engine never touches a
database, which is what lets it run inside a test, a CLI, or an Android app.
"""
from __future__ import annotations

import re

from . import categories as C
from . import fuzzy as _fuzzy
from . import merchants as _merchants
from . import ml as _ml
from . import rules as _rules
from . import triage as _triage
from . import txn_type as _txn_type
from .normalize import normalize_merchant
from .schema import ClassifyResult
from .txn_type import TxnType

# Confidence floor below which a result is queued for review even though a
# layer produced an answer.
REVIEW_THRESHOLD = 0.75

# Words that make an incoming transaction income rather than a transfer from a
# friend. Checked before the person-name heuristic, because "Salary Credit" is
# two alphabetic tokens and reads as a name to any shape test.
_INCOME_MARKER_RE = re.compile(
    r"\b(salary|wages|stipend|payroll|interest|dividend|bonus|"
    r"reimbursement|reimburse|cashback|cash\s*back|refund|payout|"
    r"settlement|maturity|redemption)\b",
    re.IGNORECASE,
)



def classify_core(
    merchant: str | None,
    *,
    is_credit: bool = False,
    direction: str | None = None,
    overrides: dict[str, str] | None = None,
    learned: dict[str, str] | None = None,
) -> ClassifyResult:
    """Classify one transaction. Deterministic and side-effect free.

    `overrides` and `learned` are both keyed by `normalize_merchant()` output.
    `overrides` are the user's explicit corrections; `learned` are merchants
    resolved previously (used for exact lookup and as the fuzzy corpus).
    """
    raw = (merchant or "").strip()
    norm = normalize_merchant(raw)
    tr = _txn_type.detect(raw, is_credit=is_credit, direction=direction)
    ttype = tr.type.value

    def _r(category, source, confidence, *, review=False, evidence=""):
        return ClassifyResult.make(
            category, source, confidence,
            needs_review=review or confidence < REVIEW_THRESHOLD,
            txn_type=ttype, normalized_merchant=norm, evidence=evidence,
        )

    # ── Stage 2 — user overrides. Highest priority, full stop. ──────────
    if overrides and norm and norm in overrides:
        return _r(C.coerce(overrides[norm]), C.SRC_USER_OVERRIDE, 1.0,
                  evidence=f"user override on '{norm}'")

    # ── Stage 1 outcomes that end the pipeline ──────────────────────────
    if tr.type is TxnType.REFUND:
        return _r(C.INCOME, C.SRC_TXN_TYPE, 0.95, evidence=tr.reason)

    if tr.type is TxnType.MONEY_RECEIVED:
        # Money in from a person is a transfer, not income. Money in from a
        # business or institution is income. This split matters: counting
        # every friend's repayment as income would wreck the budget view.
        if _INCOME_MARKER_RE.search(raw):
            return _r(C.INCOME, C.SRC_TXN_TYPE, 0.95, evidence="income keyword")
        if _triage.has_commercial_keyword(raw):
            return _r(C.INCOME, C.SRC_TXN_TYPE, 0.90, evidence="credit from a business")
        if _merchants.lookup(raw, learned) or _rules.lookup(raw):
            return _r(C.INCOME, C.SRC_TXN_TYPE, 0.90, evidence="credit from a known merchant")
        if _triage.looks_like_person(raw) or _triage.detect_opaque(raw):
            return _r(C.PERSONAL_TRANSFER, C.SRC_TXN_TYPE, 0.90,
                      evidence="credit from an individual")
        # Indeterminate counterparty on an incoming row.
        return _r(C.INCOME, C.SRC_TXN_TYPE, 0.60, review=True,
                  evidence="credit from an unrecognized counterparty")

    if not raw:
        return _r(C.OTHER, C.SRC_FALLBACK, 0.0, review=True, evidence="empty counterparty")

    # ── Stage 3 — exact merchant directory ──────────────────────────────
    # Runs BEFORE the person short-circuit. A directory hit is identity-level
    # evidence and must beat a name-shape guess: plenty of real merchant
    # strings ("Swiggy Order", "Zoya Chat") are two alphabetic tokens and look
    # exactly like an Indian person name to any shape heuristic.
    hit = _merchants.lookup(raw, learned)
    if hit is not None:
        category, conf = hit
        return _r(category, C.SRC_EXACT_MERCHANT, conf, evidence=f"directory: '{norm}'")

    # ── Stage 4 — keyword / brand rules ─────────────────────────────────
    # Also ahead of the person short-circuit, for the same reason: a curated
    # keyword is stronger evidence than "it has two words and no digits".
    rule_hit = _rules.lookup(raw)
    if rule_hit is not None:
        category, conf, pattern = rule_hit
        return _r(category, C.SRC_RULE, conf, evidence=f"rule: '{pattern}'")

    # ── Stage 1 (continued) — person payment, once merchant evidence is out
    if tr.type is TxnType.PAYMENT_TO_PERSON:
        return _r(C.PERSONAL_TRANSFER, C.SRC_TXN_TYPE, 0.90, evidence=tr.reason)

    # ── Stage 5 — fuzzy match against known merchants ───────────────────
    corpus = _merchants.corpus(learned)
    fz = _fuzzy.match(raw, corpus)
    if fz is not None:
        category, conf, key = fz
        return _r(category, C.SRC_FUZZY_MATCH, conf, evidence=f"~ '{key}'")

    # ── Stage 6 — ONNX model ────────────────────────────────────────────
    # Everything below would land under REVIEW_THRESHOLD, so this is where
    # the model gets its say. It abstains on masked handles (there is no
    # merchant text to read) and on phrases the rules deliberately refuse to
    # guess. A confident answer is taken; an unsure one is kept as a hint in
    # the evidence of whatever the fallback decides.
    hint = ""
    if not _triage.detect_opaque(raw) and not _rules.is_ambiguous(raw):
        pred = _ml.predict(norm)
        if pred is not None:
            confident = pred.probability >= REVIEW_THRESHOLD
            # A commercial keyword rules out a person, however the name reads.
            if pred.category == C.PERSONAL_TRANSFER and _triage.has_commercial_keyword(raw):
                confident = False
            if confident:
                return _r(pred.category, C.SRC_ML_MODEL,
                          min(pred.probability, _ml.MAX_CONFIDENCE),
                          evidence=f"model: p={pred.probability:.2f}")
            hint = f"; model suggests {pred.category} (p={pred.probability:.2f})"

    # ── Stage 7 — fallback ──────────────────────────────────────────────
    # A masked account number is a UPI handle, i.e. almost certainly a person.
    # Useful enough to surface, uncertain enough to always review.
    if _triage.detect_opaque(raw):
        return _r(C.PERSONAL_TRANSFER, C.SRC_FALLBACK, 0.70, review=True,
                  evidence="masked account/UPI handle")

    # Outgoing to a short, non-commercial, alphabetic name ("tannu", "Nirmala")
    # — too few tokens for the person heuristic, but no merchant signal either.
    tokens = norm.split()
    if (
        not is_credit
        and 1 <= len(tokens) <= 2
        and all(t.isalpha() for t in tokens)
        and not _triage.has_commercial_keyword(raw)
    ):
        return _r(C.PERSONAL_TRANSFER, C.SRC_FALLBACK, 0.55, review=True,
                  evidence="short non-commercial name, likely a person" + hint)

    return _r(C.OTHER, C.SRC_FALLBACK, 0.0, review=True, evidence="no layer matched" + hint)



__all__ = ["classify_core", "ClassifyResult", "REVIEW_THRESHOLD"]
