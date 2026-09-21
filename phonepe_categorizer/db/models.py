"""SQLAlchemy models — transactions, merchants, overrides, learned rules.

Two invariants drive this schema:

1. **The original statement text is never overwritten.** `raw_counterparty`
   holds exactly what PhonePe printed, and `raw_row` keeps the entire source
   row as JSON. Normalization, categorization and user corrections all write to
   *other* columns, so a bad rule or a bad correction is always recoverable and
   a re-import is never lossy.

2. **Every classification is auditable.** category, confidence, source,
   evidence and needs_review are all persisted, so "why is this GROCERIES?" is
   answerable from a row alone, without re-running the pipeline.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    """Timezone-aware UTC now.

    `datetime.utcnow()` is deprecated and scheduled for removal, and it returns
    a naive datetime that silently compares wrong against aware ones.
    """
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    """One imported statement row plus its classification."""

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # ── original data — write once, never mutate ─────────────────────────
    raw_counterparty: Mapped[str] = mapped_column(String(300))
    raw_direction: Mapped[str] = mapped_column(String(40), default="")
    raw_row: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── derived, recomputable ────────────────────────────────────────────
    normalized_merchant: Mapped[str] = mapped_column(String(300), default="", index=True)

    amount: Mapped[float] = mapped_column(Float)          # always positive
    is_credit: Mapped[bool] = mapped_column(Boolean, default=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)

    transaction_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    utr: Mapped[str | None] = mapped_column(String(40), nullable=True)
    account_last4: Mapped[str | None] = mapped_column(String(8), nullable=True)

    # ── classification ───────────────────────────────────────────────────
    txn_type: Mapped[str] = mapped_column(String(24), default="UNKNOWN", index=True)
    category: Mapped[str] = mapped_column(String(32), default="OTHER", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(20), default="FALLBACK", index=True)
    evidence: Mapped[str] = mapped_column(String(200), default="")
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    imported_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        # A statement can be re-uploaded; the transaction id makes rows unique.
        UniqueConstraint("transaction_id", name="uq_transactions_txn_id"),
        Index("ix_transactions_review_queue", "needs_review", "occurred_at"),
    )

    @property
    def signed_amount(self) -> float:
        return self.amount if self.is_credit else -self.amount

    def raw_dict(self) -> dict:
        try:
            return json.loads(self.raw_row) if self.raw_row else {}
        except (TypeError, ValueError):
            return {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<Transaction {self.occurred_at:%Y-%m-%d} "
                f"{self.raw_counterparty!r} {self.amount} {self.category}>")


class Merchant(Base):
    """A known merchant and its category — the exact-lookup directory.

    Rows arrive from two places: the shipped built-in directory, and merchants
    the pipeline resolved confidently during an import. Both are also the
    corpus the fuzzy matcher compares against, which is why `times_seen` is
    tracked: a merchant seen fifty times is a safer fuzzy target than one seen
    once.
    """

    __tablename__ = "merchants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    normalized_name: Mapped[str] = mapped_column(String(300), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(300), default="")
    category: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(20), default="RULE")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    times_seen: Mapped[int] = mapped_column(Integer, default=0)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class MerchantOverride(Base):
    """A user correction. Outranks every other layer, forever.

    Keyed by `normalize_merchant()` output so one correction covers every
    spelling variant that normalizes the same way — "XYZ General Store",
    "Paid to XYZ General Store" and "XYZ General Store 2" share a key.
    """

    __tablename__ = "merchant_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    normalized_merchant: Mapped[str] = mapped_column(String(300), unique=True, index=True)
    category: Mapped[str] = mapped_column(String(32))
    # What the classifier had said — kept so we can measure which rules the
    # user corrects most, and fix those rules rather than accumulate overrides.
    previous_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    previous_source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    example_raw: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


class ClassificationRule(Base):
    """A user-defined keyword rule, evaluated ahead of the built-in table.

    The built-in rules live in `rules.py` under version control, which is where
    rules that apply to *everyone* belong. This table is for rules a particular
    user adds for their own statements without editing code.
    """

    __tablename__ = "classification_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pattern: Mapped[str] = mapped_column(String(120), index=True)
    match_kind: Mapped[str] = mapped_column(String(10), default="phrase")  # phrase | token
    category: Mapped[str] = mapped_column(String(32))
    priority: Mapped[int] = mapped_column(Integer, default=100)  # lower runs first
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    note: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("pattern", "match_kind", name="uq_rule_pattern_kind"),
        Index("ix_rules_active", "enabled", "priority"),
    )


__all__ = ["Base", "Transaction", "Merchant", "MerchantOverride", "ClassificationRule"]
