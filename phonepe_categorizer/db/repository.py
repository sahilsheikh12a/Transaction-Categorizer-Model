"""Persistence and the learning loop.

`CategorizerRepository` is the only place that knows both the engine and the
database. It does three things:

  * loads learned state (overrides, known merchants, user rules) and hands the
    engine plain dicts;
  * persists imported transactions with their full classification provenance;
  * records user corrections so the same merchant is never asked about twice.

The correction loop is the point of the whole design. `record_correction`
writes an override, back-applies it to every past transaction with the same
normalized merchant, and clears their review flags — so one answer from the
user fixes the past and the future at once.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from .. import categories as C
from ..engine import classify_core
from ..importer import ImportedTransaction
from ..normalize import normalize_merchant
from ..schema import ClassifyResult
from .models import (
    Base,
    ClassificationRule,
    Merchant,
    MerchantOverride,
    Transaction,
    _utcnow,
)

# Confidence at or above which an auto-resolved merchant is worth remembering
# as a directory entry (and therefore as a fuzzy-match target).
LEARN_THRESHOLD = 0.90

# Only these layers identify *what a merchant is*. TXN_TYPE and FALLBACK derive
# their category from the direction of one payment instead, which is not a
# property of the merchant: a refund from a garage is INCOME for that row, but
# the garage is still TRANSPORT. Learning from them poisons the directory —
# and, because the directory is also the fuzzy corpus, poisons fuzzy matching.
_IDENTITY_SOURCES = frozenset({
    C.SRC_EXACT_MERCHANT, C.SRC_RULE, C.SRC_FUZZY_MATCH,
})


@contextmanager
def open_session(
    url: str = "sqlite:///categorizer.db", *, create: bool = True,
) -> Iterator[Session]:
    """Yield a session against `url`, creating tables on first use."""
    engine = create_engine(url, future=True)
    if create:
        Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.close()


class CategorizerRepository:
    def __init__(self, session: Session):
        self.session = session

    # ── learned state ────────────────────────────────────────────────────

    def overrides(self) -> dict[str, str]:
        rows = self.session.execute(
            select(MerchantOverride.normalized_merchant, MerchantOverride.category)
        ).all()
        return {k: v for k, v in rows if k}

    def learned_merchants(self) -> dict[str, str]:
        rows = self.session.execute(
            select(Merchant.normalized_name, Merchant.category)
        ).all()
        return {k: v for k, v in rows if k}

    def user_rules(self) -> list[ClassificationRule]:
        return list(
            self.session.execute(
                select(ClassificationRule)
                .where(ClassificationRule.enabled.is_(True))
                .order_by(ClassificationRule.priority, ClassificationRule.id)
            ).scalars()
        )

    # ── classification ───────────────────────────────────────────────────

    def classify(self, counterparty: str, *, is_credit: bool = False,
                 direction: str = "") -> ClassifyResult:
        """Classify one string using all currently-learned state."""
        return self._classify(
            counterparty, is_credit, direction,
            self.overrides(), self.learned_merchants(), self.user_rules(),
        )

    def _classify(self, counterparty, is_credit, direction,
                  overrides, learned, user_rules) -> ClassifyResult:
        # A user-defined rule sits between overrides and the built-in table:
        # more specific than a shipped keyword, less specific than "this exact
        # merchant is X". Applied by pre-seeding `overrides` only on a match.
        norm = normalize_merchant(counterparty)
        if norm and norm not in overrides:
            for rule in user_rules:
                pattern = (rule.pattern or "").lower().strip()
                if not pattern:
                    continue
                hit = (pattern in norm.split()) if rule.match_kind == "token" else (pattern in norm)
                if hit:
                    return ClassifyResult.make(
                        C.coerce(rule.category), C.SRC_RULE, 0.92,
                        txn_type="PAYMENT_TO_MERCHANT",
                        normalized_merchant=norm,
                        evidence=f"user rule: '{rule.pattern}'",
                    )
        return classify_core(
            counterparty, is_credit=is_credit, direction=direction,
            overrides=overrides, learned=learned,
        )

    # ── import ───────────────────────────────────────────────────────────

    def import_transactions(
        self, rows: Iterable[ImportedTransaction], *, learn: bool = True,
    ) -> dict[str, int]:
        """Classify and persist statement rows. Idempotent on transaction_id."""
        overrides = self.overrides()
        learned = self.learned_merchants()
        user_rules = self.user_rules()

        existing = {
            t for (t,) in self.session.execute(
                select(Transaction.transaction_id).where(Transaction.transaction_id.is_not(None))
            ).all()
        }

        # Memoize per (normalized merchant, is_credit, direction) — a 2000-row
        # statement has a few hundred unique merchants.
        memo: dict[tuple, ClassifyResult] = {}
        added = skipped = flagged = 0

        for row in rows:
            if row.transaction_id and row.transaction_id in existing:
                skipped += 1
                continue

            norm = normalize_merchant(row.counterparty)
            key = (norm or f"\x00raw:{row.counterparty}", row.is_credit, row.direction)
            result = memo.get(key)
            if result is None:
                result = self._classify(
                    row.counterparty, row.is_credit, row.direction,
                    overrides, learned, user_rules,
                )
                memo[key] = result

            self.session.add(Transaction(
                raw_counterparty=row.counterparty,
                raw_direction=row.direction,
                raw_row=json.dumps(row.raw) if row.raw else None,
                normalized_merchant=result.normalized_merchant,
                amount=row.amount,
                is_credit=row.is_credit,
                occurred_at=row.date,
                transaction_id=row.transaction_id or None,
                utr=row.utr or None,
                account_last4=row.account_last4 or None,
                txn_type=result.txn_type,
                category=result.category,
                confidence=result.confidence,
                source=result.source,
                evidence=result.evidence[:200],
                needs_review=result.needs_review,
            ))
            if row.transaction_id:
                existing.add(row.transaction_id)
            added += 1
            if result.needs_review:
                flagged += 1

            if learn and self._is_learnable(result):
                self._remember(norm, row.counterparty, result)

        self.session.commit()
        return {"added": added, "skipped_existing": skipped, "needs_review": flagged}

    @staticmethod
    def _is_learnable(result: ClassifyResult) -> bool:
        """Is this result evidence about the merchant, or just about one row?"""
        return (
            not result.needs_review
            and result.confidence >= LEARN_THRESHOLD
            and result.source in _IDENTITY_SOURCES
            and result.txn_type == "PAYMENT_TO_MERCHANT"
        )

    def _remember(self, norm: str, display: str, result: ClassifyResult) -> None:
        """Record a confidently-resolved merchant in the directory."""
        if not norm:
            return
        row = self.session.execute(
            select(Merchant).where(Merchant.normalized_name == norm)
        ).scalar_one_or_none()
        now = _utcnow()
        if row is None:
            self.session.add(Merchant(
                normalized_name=norm, display_name=display[:300],
                category=result.category, source=result.source,
                confidence=result.confidence, times_seen=1,
                first_seen=now, last_seen=now,
            ))
        else:
            row.times_seen += 1
            row.last_seen = now
            # A user override is authoritative; never let an auto-resolution
            # quietly overwrite what the user decided.
            if row.source != C.SRC_USER_OVERRIDE:
                row.category = result.category
                row.confidence = result.confidence
                row.source = result.source

    # ── the learning loop ────────────────────────────────────────────────

    def record_correction(self, counterparty: str, category: str) -> dict[str, int]:
        """Apply a user's correction: store it, back-apply it, clear review.

        Returns how many past transactions were updated.
        """
        canonical = C.coerce(category)
        norm = normalize_merchant(counterparty)
        if not norm:
            return {"updated": 0}

        row = self.session.execute(
            select(MerchantOverride).where(MerchantOverride.normalized_merchant == norm)
        ).scalar_one_or_none()

        previous = self.session.execute(
            select(Transaction.category, Transaction.source)
            .where(Transaction.normalized_merchant == norm).limit(1)
        ).first()

        if row is None:
            self.session.add(MerchantOverride(
                normalized_merchant=norm,
                category=canonical,
                previous_category=previous[0] if previous else None,
                previous_source=previous[1] if previous else None,
                example_raw=counterparty[:300],
            ))
        else:
            row.category = canonical

        # Back-apply to history so the user sees the correction take effect.
        updated = 0
        for txn in self.session.execute(
            select(Transaction).where(Transaction.normalized_merchant == norm)
        ).scalars():
            txn.category = canonical
            txn.source = C.SRC_USER_OVERRIDE
            txn.confidence = 1.0
            txn.evidence = f"user override on '{norm}'"
            txn.needs_review = False
            updated += 1

        # Keep the merchant directory in step, so future imports and fuzzy
        # matches inherit the corrected category too.
        merchant = self.session.execute(
            select(Merchant).where(Merchant.normalized_name == norm)
        ).scalar_one_or_none()
        if merchant is None:
            self.session.add(Merchant(
                normalized_name=norm, display_name=counterparty[:300],
                category=canonical, source=C.SRC_USER_OVERRIDE,
                confidence=1.0, times_seen=updated,
            ))
        else:
            merchant.category = canonical
            merchant.source = C.SRC_USER_OVERRIDE
            merchant.confidence = 1.0

        self.session.commit()
        return {"updated": updated}

    def add_rule(self, pattern: str, category: str, *, match_kind: str = "phrase",
                 priority: int = 100, note: str = "") -> ClassificationRule:
        rule = ClassificationRule(
            pattern=pattern.lower().strip(), match_kind=match_kind,
            category=C.coerce(category), priority=priority, note=note,
        )
        self.session.add(rule)
        self.session.commit()
        return rule

    # ── queries ──────────────────────────────────────────────────────────

    def review_queue(self, limit: int = 50) -> list[Transaction]:
        """Unresolved merchants worth asking about, highest spend first."""
        return list(self.session.execute(
            select(Transaction)
            .where(Transaction.needs_review.is_(True))
            .order_by(Transaction.amount.desc())
            .limit(limit)
        ).scalars())

    def category_totals(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for cat, amt in self.session.execute(
            select(Transaction.category, Transaction.amount)
        ).all():
            out[cat] = out.get(cat, 0.0) + amt
        return out

    def reclassify_all(self) -> dict[str, int]:
        """Re-run the pipeline over stored rows after rules or overrides change.

        Reads `raw_counterparty`, which is why that column is never mutated.
        """
        overrides, learned, user_rules = (
            self.overrides(), self.learned_merchants(), self.user_rules()
        )
        # A category change and a source change mean different things. After a
        # fresh import most rows legitimately move RULE -> EXACT_MERCHANT
        # (their merchant is now in the directory) without changing category;
        # reporting that as "changed" would hide the changes that matter.
        recategorized = resourced = 0
        rows = list(self.session.execute(select(Transaction)).scalars())
        for txn in rows:
            result = self._classify(
                txn.raw_counterparty, txn.is_credit, txn.raw_direction,
                overrides, learned, user_rules,
            )
            if result.category != txn.category:
                recategorized += 1
            elif result.source != txn.source:
                resourced += 1
            txn.normalized_merchant = result.normalized_merchant
            txn.txn_type = result.txn_type
            txn.category = result.category
            txn.confidence = result.confidence
            txn.source = result.source
            txn.evidence = result.evidence[:200]
            txn.needs_review = result.needs_review
        self.session.commit()
        return {
            "total": len(rows),
            "recategorized": recategorized,
            "source_changed": resourced,
        }


__all__ = ["CategorizerRepository", "open_session", "LEARN_THRESHOLD"]
