"""Importer tests — schema resolution and per-row resilience.

The importer's contract is that a bad row is data, not an exception. Every
test here that feeds it garbage asserts it kept going.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from phonepe_categorizer.importer import (
    UnresolvedHeaders,
    parse,
    parse_amount,
    parse_date,
    resolve_headers,
)

PHONEPE = ("date,time,direction,counterparty,transaction_id,utr,"
           "account_flow,account_last4,type,amount\n")
ROW = "2026-02-28,10:00,Paid to,EBENEZER BAKERY,T1,455710966211,Debited From,4950,DEBIT,50\n"


class TestHeaderResolution:
    def test_resolves_the_phonepe_schema(self):
        m = resolve_headers(PHONEPE.strip().split(","))
        assert m["date"] == 0 and m["time"] == 1 and m["direction"] == 2
        assert m["counterparty"] == 3 and m["amount"] == 9 and m["type"] == 8

    def test_column_order_is_not_assumed(self):
        m = resolve_headers(["Amount", "Narration", "Txn Date"])
        assert m["amount"] == 0 and m["counterparty"] == 1 and m["date"] == 2

    def test_alternative_bank_vocabulary(self):
        m = resolve_headers(["Value Date", "Particulars", "Withdrawal", "Deposit"])
        assert m["date"] == 0 and m["counterparty"] == 1
        assert m["debit"] == 2 and m["credit"] == 3

    def test_a_column_is_claimed_by_only_one_role(self):
        m = resolve_headers(["Transaction Date", "Transaction ID", "Name", "Amount"])
        assert m["date"] == 0 and m["transaction_id"] == 1
        assert m["counterparty"] == 2 and m["amount"] == 3

    def test_missing_required_columns_raises(self):
        with pytest.raises(UnresolvedHeaders) as exc:
            parse("colour,size\nred,large\n")
        assert "date" in exc.value.missing

    def test_prelude_lines_before_the_table_are_skipped(self):
        res = parse("Statement for account XX4950\nGenerated 2026-01-01\n\n" + PHONEPE + ROW)
        assert len(res.rows) == 1


class TestCellParsers:
    @pytest.mark.parametrize("raw", ["2026-02-28", "28-02-2026", "28/02/2026", "28-Feb-2026"])
    def test_date_formats(self, raw):
        assert parse_date(raw) == datetime(2026, 2, 28)

    def test_unparseable_date_returns_none(self):
        assert parse_date("not-a-date") is None
        assert parse_date("") is None

    @pytest.mark.parametrize("raw,expected", [
        ("1,234.50", 1234.50), ("₹500", 500.0), ("50.00 Dr", 50.0), ("-40", -40.0),
    ])
    def test_amount_formats(self, raw, expected):
        assert parse_amount(raw) == expected

    def test_unparseable_amount_returns_none(self):
        assert parse_amount("abc") is None


class TestSignResolution:
    def test_type_column_drives_the_sign(self):
        res = parse(PHONEPE +
                    "2026-01-01,10:00,Paid to,X Store,T1,1,Debited From,4950,DEBIT,50\n"
                    "2026-01-02,10:00,Received from,Y,T2,2,Credited To,4950,CREDIT,80\n")
        assert [r.is_credit for r in res.rows] == [False, True]
        assert [r.signed_amount for r in res.rows] == [-50.0, 80.0]

    def test_separate_debit_credit_columns(self):
        res = parse("Date,Narration,Withdrawal,Deposit\n"
                    "2026-01-01,Shop A,150,\n"
                    "2026-01-02,Salary,,5000\n")
        assert [(r.amount, r.is_credit) for r in res.rows] == [(150.0, False), (5000.0, True)]

    def test_signed_amount_column_only(self):
        res = parse("Date,Description,Amount\n2026-01-01,Shop,-99\n2026-01-02,Refund,45\n")
        assert [r.is_credit for r in res.rows] == [False, True]


class TestResilience:
    def test_one_bad_row_never_stops_the_import(self):
        res = parse(PHONEPE + "garbage,,,,,,,,,\n" + ROW + ",,,,,,,,,\n" + ROW.replace("T1", "T2"))
        assert len(res.rows) == 2
        assert res.problems  # the garbage row was recorded, not raised

    def test_problems_carry_row_numbers(self):
        res = parse(PHONEPE + "nope,10:00,Paid to,A Shop,T9,1,Debited From,4950,DEBIT,5\n")
        assert res.problems[0].row_number == 2
        assert res.problems[0].reason == "unparseable date"

    def test_short_row_does_not_crash(self):
        assert parse(PHONEPE + "2026-02-28,10:00,Paid to\n").skipped == 1

    def test_original_counterparty_is_preserved_verbatim(self):
        res = parse(PHONEPE + ROW.replace("EBENEZER BAKERY", "  new MATRUCHAYA  Store 2 "))
        assert res.rows[0].counterparty == "new MATRUCHAYA  Store 2"

    def test_whole_source_row_is_retained(self):
        res = parse(PHONEPE + ROW)
        assert res.rows[0].raw["utr"] == "455710966211"
        assert res.rows[0].raw["account_flow"] == "Debited From"

    def test_semicolon_delimiter(self):
        res = parse("date;counterparty;amount;type\n2026-01-01;A Shop;50;DEBIT\n")
        assert len(res.rows) == 1


class TestDeduplication:
    def test_duplicate_transaction_ids_collapse(self):
        res = parse(PHONEPE + ROW + ROW)
        assert len(res.rows) == 1 and res.duplicates == 1

    def test_rows_without_ids_dedupe_on_content(self):
        no_id = "2026-01-01,10:00,Paid to,A Shop,,1,Debited From,4950,DEBIT,50\n"
        res = parse(PHONEPE + no_id + no_id)
        assert len(res.rows) == 1 and res.duplicates == 1

    def test_distinct_payments_are_not_collapsed(self):
        res = parse(PHONEPE +
                    "2026-01-01,10:00,Paid to,A Shop,T1,1,Debited From,4950,DEBIT,50\n"
                    "2026-01-01,10:00,Paid to,A Shop,T2,2,Debited From,4950,DEBIT,50\n")
        assert len(res.rows) == 2

    def test_dedupe_can_be_disabled(self):
        assert len(parse(PHONEPE + ROW + ROW, dedupe=False).rows) == 2


class TestRealStatement:
    def test_parses_the_shipped_statement(self):
        import pathlib
        p = pathlib.Path(__file__).resolve().parents[1] / "transactions2.csv"
        if not p.exists():
            pytest.skip("statement not present")
        res = parse(p.read_bytes())
        assert len(res.rows) > 1900
        assert res.problem_counts() == {"blank counterparty": 2}
        assert all(r.amount > 0 for r in res.rows)
        assert {r.direction for r in res.rows} >= {"Paid to", "Received from", "Paid", "Bill paid"}
