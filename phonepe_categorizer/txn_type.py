"""Stage 1 — what KIND of transaction is this?

Runs BEFORE any merchant categorization. The V1 spec's core requirement:
"Paid to Ritik" must never become SHOPPING just because a keyword matched.
Deciding person-vs-merchant first makes that structurally impossible.

Signals, strongest first:
  1. `direction` — the statement's own verb. PhonePe emits "Paid to",
     "Received from", "Paid" (non-UPI / recharge), "Bill paid", and the PDF
     parser also produces "Sent to" / "Refund from". This is ground truth
     about flow direction and is never guessed.
  2. amount sign / is_credit — fallback when direction is absent (SMS, PDF).
  3. counterparty shape — person-name heuristic vs commercial keywords vs
     opaque masked strings (see triage.py).

Everything here is pure and DB-free so it is trivially testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from . import triage
from .normalize import normalize_merchant


class TxnType(StrEnum):
    PAYMENT_TO_PERSON = "PAYMENT_TO_PERSON"
    PAYMENT_TO_MERCHANT = "PAYMENT_TO_MERCHANT"
    MONEY_RECEIVED = "MONEY_RECEIVED"
    REFUND = "REFUND"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class TypeResult:
    type: TxnType
    confidence: float
    reason: str


# --- direction vocabulary -------------------------------------------------
# Matched against the lowercased, whitespace-collapsed direction cell.

_DIR_OUT = {"paid to", "paid", "sent to", "payment to", "transfer to", "debited", "bill paid"}
_DIR_IN = {"received from", "credited", "money received", "received"}
_DIR_REFUND = {"refund from", "refund", "refunded", "reversal", "reversed"}
_DIR_BILL = {"bill paid", "bill payment"}

# Merchant strings that are themselves a refund/cashback marker.
_REFUND_TEXT_RE = re.compile(
    r"\b(refund|reversal|reversed|cashback|cash\s*back|chargeback)\b", re.IGNORECASE
)

# Non-UPI "Paid" rows in PhonePe are recharges / bill payments — the
# counterparty is a service label ("Mobile Recharge", "Electricity"), never a
# person. Treated as merchant payments regardless of name shape.
_SERVICE_LABELS = {
    "mobile recharge", "recharge", "electricity", "dth recharge", "dth",
    "gas", "water", "broadband", "datacard", "landline", "fastag",
    "credit card", "loan repayment", "insurance",
}


def _norm_direction(direction: str | None) -> str:
    if not direction:
        return ""
    return " ".join(direction.strip().lower().split())


def detect(
    merchant: str | None,
    *,
    is_credit: bool = False,
    direction: str | None = None,
) -> TypeResult:
    """Classify the transaction's kind. Never raises."""
    raw = (merchant or "").strip()
    d = _norm_direction(direction)
    norm = normalize_merchant(raw)

    # --- refunds (highest specificity: an incoming flow with a refund marker)
    if d in _DIR_REFUND or any(d.startswith(x) for x in _DIR_REFUND):
        return TypeResult(TxnType.REFUND, 0.99, "direction says refund")
    if is_credit and _REFUND_TEXT_RE.search(raw):
        return TypeResult(TxnType.REFUND, 0.90, "credit with refund keyword")

    # --- incoming --------------------------------------------------------
    if d in _DIR_IN or any(d.startswith(x) for x in _DIR_IN):
        return TypeResult(TxnType.MONEY_RECEIVED, 0.99, "direction says received")
    if not d and is_credit:
        return TypeResult(TxnType.MONEY_RECEIVED, 0.90, "positive amount")

    # --- outgoing --------------------------------------------------------
    # A service label ("Mobile Recharge", "Electricity") is a merchant payment
    # whatever its name shape. Checked before the person heuristic.
    if norm in _SERVICE_LABELS or d in _DIR_BILL:
        return TypeResult(TxnType.PAYMENT_TO_MERCHANT, 0.99, "service/bill label")

    if not raw:
        return TypeResult(TxnType.UNKNOWN, 0.0, "empty counterparty")

    # Masked account numbers and bare UTRs carry no merchant information. They
    # are almost always P2P UPI transfers to a phone number, but we cannot
    # prove it, so UNKNOWN keeps them out of the spending categories.
    if triage.detect_opaque(raw):
        return TypeResult(TxnType.UNKNOWN, 0.60, "opaque/masked counterparty")

    # Shape heuristics run on the NORMALIZED string, not the raw one. Raw
    # statement text carries punctuation and separators ("RAHUL_DANDARE_") that
    # make a plain person name fail an isalpha() token test, and the merchant
    # layers downstream already work on the normalized form — running both on
    # the same string is what keeps the per-row and batch paths in agreement.
    shape = norm or raw

    if triage.has_commercial_keyword(shape):
        return TypeResult(TxnType.PAYMENT_TO_MERCHANT, 0.95, "commercial keyword")

    if triage.looks_like_person(shape):
        return TypeResult(TxnType.PAYMENT_TO_PERSON, 0.85, "person-name shape")

    if d in _DIR_OUT or any(d.startswith(x) for x in _DIR_OUT) or (not d and not is_credit):
        # Outgoing, but the counterparty is neither clearly a person nor
        # clearly a business (single-token names like "Sweety", "Google").
        # Abstain rather than guess — the merchant layers get a chance next.
        return TypeResult(TxnType.UNKNOWN, 0.30, "outgoing, indeterminate counterparty")

    return TypeResult(TxnType.UNKNOWN, 0.0, "no usable signal")


def is_incoming(t: TxnType) -> bool:
    return t in (TxnType.MONEY_RECEIVED, TxnType.REFUND)


__all__ = ["TxnType", "TypeResult", "detect", "is_incoming"]
