"""Result type emitted by the classifier.

Every classification carries not just an answer but *why* and *how sure*, so
the UI can explain itself, the review queue can be built from data rather than
guesswork, and bad rules can be found without re-running the pipeline.

`needs_review` is the single most important field: it is the difference
between "this is OTHER because we know it is miscellaneous" and "this is
OTHER because we could not tell". Only the latter should reach the user as a
question.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import categories as C


@dataclass(frozen=True, slots=True)
class ClassifyResult:
    category: str
    source: str
    confidence: float
    needs_review: bool = False
    txn_type: str = "UNKNOWN"
    normalized_merchant: str = ""
    # What actually fired — a rule pattern, a directory key, a fuzzy match.
    # Free-form and for humans; never branch on it.
    evidence: str = ""

    @classmethod
    def make(
        cls,
        category: str,
        source: str,
        confidence: float,
        *,
        needs_review: bool = False,
        txn_type: str = "UNKNOWN",
        normalized_merchant: str = "",
        evidence: str = "",
    ) -> ClassifyResult:
        return cls(
            category=category,
            source=source,
            confidence=max(0.0, min(1.0, float(confidence))),
            needs_review=needs_review,
            txn_type=txn_type,
            normalized_merchant=normalized_merchant,
            evidence=evidence,
        )

    @property
    def legacy_category(self) -> str:
        """Category in the vocabulary the existing DB/UI speaks."""
        return C.to_legacy(self.category)

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source,
            "needsReview": self.needs_review,
            "txnType": self.txn_type,
            "normalizedMerchant": self.normalized_merchant,
            "evidence": self.evidence,
        }


__all__ = ["ClassifyResult"]
