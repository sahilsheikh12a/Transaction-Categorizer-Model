"""Persistence and the user-correction loop.

These tests exist mainly to pin one behaviour: a correction must fix the past
as well as the future. A system that only applies a correction going forward
leaves the user staring at the rows they just fixed.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from phonepe_categorizer import categories as C
from phonepe_categorizer.db import (
    Base,
    CategorizerRepository,
    Merchant,
    MerchantOverride,
    Transaction,
)
from phonepe_categorizer.importer import parse

PHONEPE = ("date,time,direction,counterparty,transaction_id,utr,"
           "account_flow,account_last4,type,amount\n")


@pytest.fixture
def repo():
    engine = create_engine("sqlite://", future=True)   # in-memory
    Base.metadata.create_all(engine)
    session = sessionmaker(engine, expire_on_commit=False, future=True)()
    yield CategorizerRepository(session)
    session.close()


def _rows(*lines: str):
    return parse(PHONEPE + "".join(lines)).rows


def _row(name, amount=50, txn_id="T1", direction="Paid to", flow="Debited From", typ="DEBIT"):
    return f"2026-01-01,10:00,{direction},{name},{txn_id},1,{flow},4950,{typ},{amount}\n"


class TestImport:
    def test_persists_classification_provenance(self, repo):
        repo.import_transactions(_rows(_row("NEW MATRUCHAYA DAILY AND GENERAL STORE")))
        t = repo.session.execute(select(Transaction)).scalar_one()
        assert t.category == C.GROCERIES
        assert t.source == C.SRC_RULE
        assert t.confidence == pytest.approx(0.90)
        assert t.txn_type == "PAYMENT_TO_MERCHANT"
        assert t.evidence and t.needs_review is False

    def test_original_text_is_never_overwritten(self, repo):
        repo.import_transactions(_rows(_row("  Maturchaya  Kirana 2 ")))
        t = repo.session.execute(select(Transaction)).scalar_one()
        assert t.raw_counterparty == "Maturchaya  Kirana 2"     # verbatim
        assert t.normalized_merchant == "maturchaya kirana"     # derived
        assert t.raw_dict()["utr"] == "1"

    def test_reimport_is_idempotent(self, repo):
        rows = _rows(_row("EBENEZER BAKERY"))
        assert repo.import_transactions(rows)["added"] == 1
        second = repo.import_transactions(rows)
        assert second["added"] == 0 and second["skipped_existing"] == 1

    def test_confident_merchants_are_learned(self, repo):
        repo.import_transactions(_rows(_row("EBENEZER BAKERY")))
        m = repo.session.execute(select(Merchant)).scalar_one()
        assert m.normalized_name == "ebenezer bakery"
        assert m.category == C.FOOD_AND_DINING

    def test_unresolved_merchants_are_not_learned(self, repo):
        repo.import_transactions(_rows(_row("Vbnmqwe Asdzxc Poiuyt Lkjhgf Mnbvcx Qazwsx")))
        assert repo.session.execute(select(Merchant)).all() == []

    def test_batch_import_matches_single_classification(self, repo):
        names = ["Sajid Sheikh", "Maturchaya Kirana 2", "******6258", "EBENEZER BAKERY"]
        repo.import_transactions(_rows(
            *[_row(n, txn_id=f"T{i}") for i, n in enumerate(names)]
        ))
        stored = {t.raw_counterparty: t.category
                  for t in repo.session.execute(select(Transaction)).scalars()}
        for n in names:
            assert stored[n] == repo.classify(n, direction="Paid to").category


class TestCorrectionLoop:
    def test_correction_back_applies_to_history(self, repo):
        repo.import_transactions(_rows(
            _row("Smart Point NAGPUR U178", txn_id="T1"),
            _row("Smart Point NAGPUR U178", txn_id="T2"),
            _row("Smart Point NAGPUR U179", txn_id="T3"),   # same shop, other outlet
        ))
        assert repo.record_correction("Smart Point NAGPUR U178", C.GROCERIES)["updated"] == 3
        for t in repo.session.execute(select(Transaction)).scalars():
            assert t.category == C.GROCERIES
            assert t.source == C.SRC_USER_OVERRIDE
            assert t.needs_review is False

    def test_correction_drives_future_classification(self, repo):
        repo.record_correction("XYZ General Store", C.HEALTH)
        res = repo.classify("Paid to XYZ General Store 2", direction="Paid to")
        assert res.category == C.HEALTH
        assert res.source == C.SRC_USER_OVERRIDE

    def test_correction_beats_person_detection(self, repo):
        repo.import_transactions(_rows(_row("CHETMANI DWARKAPRASAD SAHU")))
        assert repo.classify("CHETMANI DWARKAPRASAD SAHU",
                             direction="Paid to").category == C.PERSONAL_TRANSFER
        repo.record_correction("CHETMANI DWARKAPRASAD SAHU", C.GROCERIES)
        assert repo.classify("CHETMANI DWARKAPRASAD SAHU",
                             direction="Paid to").category == C.GROCERIES

    def test_correction_records_what_it_replaced(self, repo):
        repo.import_transactions(_rows(_row("Maturchaya Kirana 2")))
        repo.record_correction("Maturchaya Kirana 2", C.FOOD_AND_DINING)
        ov = repo.session.execute(select(MerchantOverride)).scalar_one()
        assert ov.previous_category == C.GROCERIES
        assert ov.previous_source == C.SRC_RULE

    def test_correction_is_idempotent_and_updatable(self, repo):
        repo.record_correction("A Shop Name", C.GROCERIES)
        repo.record_correction("A Shop Name", C.SHOPPING)
        rows = repo.session.execute(select(MerchantOverride)).scalars().all()
        assert len(rows) == 1 and rows[0].category == C.SHOPPING

    def test_legacy_category_names_are_coerced(self, repo):
        repo.record_correction("Some Shop", "food")
        assert repo.classify("Some Shop", direction="Paid to").category == C.FOOD_AND_DINING


class TestUserRules:
    def test_user_rule_applies(self, repo):
        repo.add_rule("gotivada", C.FOOD_AND_DINING, match_kind="token")
        res = repo.classify("SWAPNA LAVA GOTIVADA", direction="Paid to")
        assert res.category == C.FOOD_AND_DINING
        assert "user rule" in res.evidence

    def test_override_still_beats_a_user_rule(self, repo):
        repo.add_rule("gotivada", C.FOOD_AND_DINING, match_kind="token")
        repo.record_correction("SWAPNA LAVA GOTIVADA", C.GROCERIES)
        assert repo.classify("SWAPNA LAVA GOTIVADA",
                             direction="Paid to").category == C.GROCERIES

    def test_disabled_rules_are_ignored(self, repo):
        rule = repo.add_rule("gotivada", C.FOOD_AND_DINING, match_kind="token")
        rule.enabled = False
        repo.session.commit()
        assert repo.classify("SWAPNA LAVA GOTIVADA",
                             direction="Paid to").category == C.PERSONAL_TRANSFER


class TestQueries:
    def test_review_queue_is_ordered_by_amount(self, repo):
        repo.import_transactions(_rows(
            _row("Qwertzu Asdfgh Poiuyt Lkjhgf Mnbvcx Qazwsx", amount=100, txn_id="T1"),
            _row("Zxcvbnm Qwerlkj Poiuyt Lkjhgf Mnbvcx Qazwsx", amount=900, txn_id="T2"),
        ))
        queue = repo.review_queue()
        assert [t.amount for t in queue] == [900.0, 100.0]

    def test_reclassify_after_adding_a_rule(self, repo):
        repo.import_transactions(_rows(_row("SWAPNA LAVA GOTIVADA")))
        assert repo.session.execute(select(Transaction)).scalar_one().category \
            == C.PERSONAL_TRANSFER
        repo.add_rule("gotivada", C.FOOD_AND_DINING, match_kind="token")
        assert repo.reclassify_all()["recategorized"] == 1
        assert repo.session.execute(select(Transaction)).scalar_one().category \
            == C.FOOD_AND_DINING

    def test_reclassify_reads_the_untouched_original(self, repo):
        repo.import_transactions(_rows(_row("Maturchaya Kirana 2")))
        repo.reclassify_all()
        t = repo.session.execute(select(Transaction)).scalar_one()
        assert t.raw_counterparty == "Maturchaya Kirana 2"

    def test_category_totals(self, repo):
        repo.import_transactions(_rows(
            _row("Maturchaya Kirana 2", amount=100, txn_id="T1"),
            _row("EBENEZER BAKERY", amount=250, txn_id="T2"),
        ))
        totals = repo.category_totals()
        assert totals[C.GROCERIES] == 100.0
        assert totals[C.FOOD_AND_DINING] == 250.0


class TestDirectoryHygiene:
    """The merchant directory records what a merchant *is*.

    It must never absorb a category that came from the direction of a single
    payment, because the directory is consulted for payments in both
    directions — and doubles as the fuzzy matcher's corpus.
    """

    def test_people_are_not_learned_as_merchants(self, repo):
        repo.import_transactions(_rows(
            _row("Sajid Sheikh", txn_id="T1"),
            _row("Ritik Pandey", txn_id="T2"),
        ))
        assert repo.session.execute(select(Merchant)).all() == []

    def test_a_refund_does_not_make_the_payee_income(self, repo):
        # A credit from a garage is INCOME for that row; the garage is still
        # TRANSPORT. Learning INCOME here would misfile every future payment.
        repo.import_transactions(_rows(
            _row("Simran Auto Mobile", txn_id="T1",
                 direction="Received from", flow="Credited To", typ="CREDIT"),
        ))
        assert repo.session.execute(select(Merchant)).all() == []
        assert repo.classify("Simran Auto Mobile",
                             direction="Paid to").category == C.TRANSPORT

    def test_merchants_are_still_learned_from_payments(self, repo):
        repo.import_transactions(_rows(_row("EBENEZER BAKERY")))
        m = repo.session.execute(select(Merchant)).scalar_one()
        assert m.category == C.FOOD_AND_DINING

    def test_reclassify_is_stable_immediately_after_import(self, repo):
        repo.import_transactions(_rows(
            _row("EBENEZER BAKERY", txn_id="T1"),
            _row("Sajid Sheikh", txn_id="T2"),
            _row("Maturchaya Kirana 2", txn_id="T3"),
            _row("A U K HOTELS PRIVATE LIMITED", txn_id="T4",
                 direction="Received from", flow="Credited To", typ="CREDIT"),
        ))
        assert repo.reclassify_all()["recategorized"] == 0
