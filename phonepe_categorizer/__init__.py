"""Offline transaction categorization for PhonePe statements.

    from phonepe_categorizer import classify, importer

    stmt = importer.parse_file("transactions2.csv")
    for txn in stmt.rows:
        result = classify(txn.counterparty,
                          is_credit=txn.is_credit,
                          direction=txn.direction)
        print(result.category, result.confidence, result.source)

The pipeline, in order:

    raw counterparty
      -> merchant normalization
      -> transaction-type detection      (person / merchant / received / refund)
      -> user overrides
      -> exact merchant lookup
      -> keyword + brand rules
      -> fuzzy merchant matching
      -> ONNX n-gram model               (only below the review threshold)
      -> MiniLM neighbours               (instead of OTHER; always answers)
      -> OTHER + needs_review            (only when there is no text at all)

No LLM and no network: the same input always produces the same output.
Persistence and the user-correction loop are optional and live in
`phonepe_categorizer.db`.
"""
from . import (
    categories,
    fuzzy,
    importer,
    merchants,
    ml,
    normalize,
    rules,
    semantic,
    triage,
    txn_type,
)
from .categories import ALL as CATEGORIES
from .engine import REVIEW_THRESHOLD, classify_core
from .normalize import normalize_merchant
from .schema import ClassifyResult
from .txn_type import TxnType

__version__ = "1.0.0"

# `classify` is the friendly name; `classify_core` stays as the explicit one.
classify = classify_core

__all__ = [
    "classify", "classify_core", "ClassifyResult", "TxnType",
    "CATEGORIES", "REVIEW_THRESHOLD", "normalize_merchant",
    "categories", "normalize", "triage", "txn_type", "rules",
    "merchants", "fuzzy", "ml", "semantic", "importer",
    "__version__",
]
