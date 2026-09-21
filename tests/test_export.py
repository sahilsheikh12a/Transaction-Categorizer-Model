"""CSV-in / CSV-out tests.

The contract worth defending here is row conservation: every data row in the
input must appear in the output. A batch export that silently drops the rows it
could not handle is impossible to reconcile against the source statement.
"""
from __future__ import annotations

import csv

import pytest

from phonepe_categorizer import categories as C
from phonepe_categorizer.export import (
    OUTPUT_COLUMNS,
    categorize_csv,
    merchant_map_csv,
)

HEADER = ("date,time,direction,counterparty,transaction_id,utr,"
          "account_flow,account_last4,type,amount\n")


def _row(name, amount=50, txn_id="T1", direction="Paid to",
         flow="Debited From", typ="DEBIT"):
    return f"2026-01-01,10:00,{direction},{name},{txn_id},1,{flow},4950,{typ},{amount}\n"


@pytest.fixture
def run(tmp_path):
    def _run(body: str, **kw):
        src = tmp_path / "in.csv"
        dst = tmp_path / "out.csv"
        src.write_text(HEADER + body)
        summary = categorize_csv(src, dst, **kw)
        with dst.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        return summary, rows
    return _run


class TestShape:
    def test_original_columns_are_preserved_verbatim(self, run):
        _, rows = run(_row("EBENEZER BAKERY"))
        r = rows[0]
        assert r["date"] == "2026-01-01"
        assert r["counterparty"] == "EBENEZER BAKERY"
        assert r["utr"] == "1"
        assert r["amount"] == "50"

    def test_classification_columns_are_appended(self, run):
        _, rows = run(_row("EBENEZER BAKERY"))
        assert all(c in rows[0] for c in OUTPUT_COLUMNS)
        assert rows[0]["category"] == C.FOOD_AND_DINING
        assert rows[0]["category_source"] == C.SRC_RULE
        assert rows[0]["category_needs_review"] == "no"
        assert rows[0]["status"] == "ok"

    def test_appended_names_never_shadow_a_source_column(self, tmp_path):
        # Re-running this tool on its own output hits exactly this case: the
        # input already has `category` and `status`. Without renaming, the
        # original values are silently lost to the duplicate field name.
        src, dst = tmp_path / "a.csv", tmp_path / "b.csv"
        src.write_text("date,direction,counterparty,type,amount,category,status\n"
                       "2026-01-01,Paid to,EBENEZER BAKERY,DEBIT,50,my-old-label,fine\n")
        categorize_csv(src, dst)
        with dst.open(newline="") as fh:
            row = next(csv.DictReader(fh))
        assert row["category"] == "my-old-label"    # source value survives
        assert row["status"] == "fine"
        assert row["category_2"] == C.FOOD_AND_DINING
        assert row["status_2"] == "ok"

    def test_export_is_reentrant(self, tmp_path):
        """Categorizing an already-categorized file must not lose anything."""
        a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
        a.write_text(HEADER + _row("EBENEZER BAKERY"))
        categorize_csv(a, b)
        categorize_csv(b, c)
        with c.open(newline="") as fh:
            row = next(csv.DictReader(fh))
        assert row["counterparty"] == "EBENEZER BAKERY"
        assert row["category"] == C.FOOD_AND_DINING      # first pass
        assert row["category_2"] == C.FOOD_AND_DINING    # second pass

    def test_blank_headers_are_named_not_dropped(self, tmp_path):
        src, dst = tmp_path / "a.csv", tmp_path / "b.csv"
        src.write_text("date,counterparty,amount,\n2026-01-01,A Shop,50,x\n")
        categorize_csv(src, dst)
        assert "column_4" in dst.read_text().splitlines()[0]


class TestRowConservation:
    def test_every_input_row_appears_in_the_output(self, run):
        body = (_row("EBENEZER BAKERY", txn_id="T1")
                + "garbage,,,,,,,,,\n"
                + _row("Maturchaya Kirana 2", txn_id="T2")
                + "2026-01-01,10:00,Paid to,,T3,1,Debited From,4950,DEBIT,5\n")
        summary, rows = run(body)
        assert summary.input_rows == 4
        assert len(rows) == 4

    def test_unparseable_rows_are_marked_not_dropped(self, run):
        _, rows = run("nope,10:00,Paid to,A Shop,T9,1,Debited From,4950,DEBIT,5\n")
        assert rows[0]["status"] == "skipped: unparseable date"
        assert rows[0]["category"] == ""
        assert rows[0]["counterparty"] == "A Shop"   # original data still there

    def test_blank_counterparty_is_marked(self, run):
        _, rows = run("2026-01-01,10:00,Paid,,T1,1,Debited From,4950,DEBIT,5\n")
        assert rows[0]["status"] == "skipped: blank counterparty"

    def test_output_order_matches_input_order(self, run):
        names = ["EBENEZER BAKERY", "Ritik Pandey", "Maturchaya Kirana 2"]
        body = "".join(_row(n, txn_id=f"T{i}") for i, n in enumerate(names))
        _, rows = run(body)
        assert [r["counterparty"] for r in rows] == names

    def test_short_rows_are_padded_not_misaligned(self, run):
        _, rows = run("2026-01-01,10:00,Paid to\n" + _row("EBENEZER BAKERY", txn_id="T2"))
        assert len(rows) == 2
        assert rows[1]["category"] == C.FOOD_AND_DINING   # columns still line up


class TestDuplicates:
    def test_duplicates_are_kept_by_default(self, run):
        summary, rows = run(_row("EBENEZER BAKERY") + _row("EBENEZER BAKERY"))
        assert len(rows) == 2
        assert all(r["status"] == "ok" for r in rows)

    def test_dedupe_flag_marks_the_second_copy(self, run):
        summary, rows = run(_row("EBENEZER BAKERY") + _row("EBENEZER BAKERY"),
                            dedupe=True)
        assert len(rows) == 2                      # still conserved
        assert rows[0]["status"] == "ok"
        assert "duplicate" in rows[1]["status"]
        assert summary.duplicates == 1


class TestLearnedState:
    def test_overrides_are_applied_to_the_export(self, run):
        _, rows = run(_row("Maturchaya Kirana 2"),
                      overrides={"maturchaya kirana": C.HEALTH})
        assert rows[0]["category"] == C.HEALTH
        assert rows[0]["category_source"] == C.SRC_USER_OVERRIDE
        assert rows[0]["category_needs_review"] == "no"

    def test_learned_merchants_resolve_previously_unknown_names(self, run):
        unknown = "Smart Point NAGPUR U178"
        _, plain = run(_row(unknown))
        assert plain[0]["category"] == C.OTHER
        _, taught = run(_row(unknown), learned={"smart point nagpur": C.GROCERIES})
        assert taught[0]["category"] == C.GROCERIES


class TestRealStatement:
    def test_full_statement_round_trip(self, tmp_path):
        import pathlib
        src = pathlib.Path(__file__).resolve().parents[1] / "transactions2.csv"
        if not src.exists():
            pytest.skip("statement not present")
        dst = tmp_path / "out.csv"
        summary = categorize_csv(src, dst)

        with dst.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == summary.input_rows
        assert summary.classified + summary.skipped == summary.input_rows
        # Every classified row carries a valid category and a source.
        for r in rows:
            if r["status"] == "ok":
                assert r["category"] in C.VALID
                assert r["category_source"] in C.SOURCES
                assert r["category_needs_review"] in ("yes", "no")


class TestMerchantMap:
    """The two-column merchant -> category lookup table."""

    def _map(self, tmp_path, body, **kw):
        src, dst = tmp_path / "in.csv", tmp_path / "map.csv"
        src.write_text(HEADER + body)
        summary = merchant_map_csv(src, dst, **kw)
        with dst.open(newline="") as fh:
            rows = list(csv.reader(fh))
        return summary, rows

    def test_exactly_two_columns(self, tmp_path):
        _, rows = self._map(tmp_path, _row("EBENEZER BAKERY"))
        assert rows[0] == ["merchant", "category"]
        assert all(len(r) == 2 for r in rows)

    def test_one_row_per_merchant_not_per_transaction(self, tmp_path):
        body = "".join(_row("EBENEZER BAKERY", txn_id=f"T{i}") for i in range(5))
        summary, rows = self._map(tmp_path, body)
        assert summary.transactions == 5
        assert summary.merchants == 1
        assert len(rows) == 2                      # header + one merchant

    def test_spelling_variants_collapse_to_one_row(self, tmp_path):
        body = (_row("Maturchaya Kirana 2", txn_id="T1")
                + _row("MATURCHAYA KIRANA", txn_id="T2")
                + _row("Maturchaya Kirana 2", txn_id="T3"))
        summary, rows = self._map(tmp_path, body)
        assert summary.merchants == 1
        # The most frequent spelling is the one a human will recognize.
        assert rows[1][0] == "Maturchaya Kirana 2"

    def test_sorted_alphabetically(self, tmp_path):
        body = (_row("Zoya Chat Center", txn_id="T1")
                + _row("EBENEZER BAKERY", txn_id="T2")
                + _row("Maturchaya Kirana 2", txn_id="T3"))
        _, rows = self._map(tmp_path, body)
        names = [r[0] for r in rows[1:]]
        assert names == sorted(names, key=str.lower)

    def test_a_credit_only_merchant_keeps_its_own_category(self, tmp_path):
        # This garage appears solely as a refund. Labelling it INCOME would be
        # true of that one payment and wrong for the merchant.
        _, rows = self._map(tmp_path, _row(
            "Simran Auto Mobile", direction="Received from",
            flow="Credited To", typ="CREDIT",
        ))
        assert rows[1] == ["Simran Auto Mobile", C.TRANSPORT]

    def test_people_are_still_personal_transfer(self, tmp_path):
        _, rows = self._map(tmp_path, _row("Ritik Pandey"))
        assert rows[1] == ["Ritik Pandey", C.PERSONAL_TRANSFER]

    def test_masked_accounts_do_not_collapse_together(self, tmp_path):
        # They all normalize to "" — keying on that would merge every masked
        # account into a single nameless row.
        body = (_row("******6258", txn_id="T1") + _row("******1971", txn_id="T2"))
        summary, rows = self._map(tmp_path, body)
        assert summary.merchants == 2
        assert {r[0] for r in rows[1:]} == {"******6258", "******1971"}

    def test_overrides_are_applied(self, tmp_path):
        _, rows = self._map(tmp_path, _row("Maturchaya Kirana 2"),
                            overrides={"maturchaya kirana": C.HEALTH})
        assert rows[1][1] == C.HEALTH

    def test_names_with_commas_survive_a_round_trip(self, tmp_path):
        # The name is quoted in the input and must be quoted again on the way
        # out, or the two-column file silently becomes three columns.
        _, rows = self._map(
            tmp_path,
            '2026-01-01,10:00,Paid to,"Sharma Bakery, Nagpur",T1,1,'
            "Debited From,4950,DEBIT,50\n",
        )
        assert rows[1] == ["Sharma Bakery, Nagpur", C.FOOD_AND_DINING]

    def test_real_statement(self, tmp_path):
        import pathlib
        src = pathlib.Path(__file__).resolve().parents[1] / "transactions2.csv"
        if not src.exists():
            pytest.skip("statement not present")
        dst = tmp_path / "m.csv"
        summary = merchant_map_csv(src, dst)
        with dst.open(newline="") as fh:
            rows = list(csv.reader(fh))
        assert len(rows) == summary.merchants + 1
        assert all(len(r) == 2 for r in rows)
        assert all(r[1] in C.VALID for r in rows[1:])
        # No duplicate merchant names.
        names = [r[0] for r in rows[1:]]
        assert len(names) == len(set(names))
