"""Optional persistence for the categorization engine.

The engine itself never imports this package — it takes learned state as plain
dicts. That separation is deliberate: it keeps `classify_core` testable without
a database and portable to a host app with entirely different storage (Room on
Android, for instance).
"""
from .models import (
    Base,
    ClassificationRule,
    Merchant,
    MerchantOverride,
    Transaction,
)
from .repository import CategorizerRepository, open_session

__all__ = [
    "Base", "Transaction", "Merchant", "MerchantOverride", "ClassificationRule",
    "CategorizerRepository", "open_session",
]
