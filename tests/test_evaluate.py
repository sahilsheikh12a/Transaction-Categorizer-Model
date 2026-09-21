"""Evaluation tooling: the blind label sheet and the scoring.

Uses a small invented statement in the parser's 10-column schema, so none
of this depends on real data.
"""
from __future__ import annotations

import csv

import pytest

from phonepe_categorizer import categories as C
from phonepe_categorizer.evaluate import (
    SHEET_COLUMNS,
    LabelError,
    as_text,
    evaluate,
    label_sheet,
    read_labels,
)

HEADER = ("date,time,direction,counterparty,transaction_id,utr,"
          "account_flow,account_last4,type,amount")


def _statement(path, rows):
    lines = [HEADER]
    for i, (direction, who, typ, amount) in enumerate(rows):
        flow = "Credited To" if typ == "CREDIT" else "Debited From"
        lines.append(f"2026-07-{1 + i % 28:02d},10:00,{direction},{who},T{i:020d},"
                     f"{100000000000 + i},{flow},4950,{typ},{amount}")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def test_csv(tmp_path):
    return _statement(tmp_path / "test.csv", [
        ("Paid to", "Swiggy", "DEBIT", 400),
        ("Paid to", "Swiggy Order", "DEBIT", 300),        # own key; left unlabelled below
        ("Paid to", "Ritik Pandey", "DEBIT", 50),
        ("Paid to", "Ritik Pandey", "DEBIT", 70),
        ("Received from", "Ritik Pandey", "CREDIT", 500),
        ("Paid to", "Zzq Unrecognizable Enterprises Holdings", "DEBIT", 900),
    ])


def _fill(sheet, labels: dict[tuple[str, str], str]):
    rows = list(csv.DictReader(sheet.open()))
    for r in rows:
        r["true_category"] = labels.get((r["key"], r["flow"]), "")
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)


class TestLabelSheet:
    def test_one_line_per_merchant_and_flow(self, test_csv, tmp_path):
        out = tmp_path / "labels.csv"
        assert label_sheet(test_csv, out) == 5
        rows = {(r["key"], r["flow"]): r for r in csv.DictReader(out.open())}
        assert rows[("ritik pandey", "OUT")]["count"] == "2"
        assert rows[("ritik pandey", "OUT")]["total_amount"] == "120.00"
        assert ("ritik pandey", "IN") in rows   # money in is labelled separately

    def test_sheet_is_blind(self, test_csv, tmp_path):
        out = tmp_path / "labels.csv"
        label_sheet(test_csv, out)
        rows = list(csv.DictReader(out.open()))
        assert tuple(rows[0]) == SHEET_COLUMNS
        assert all(r["true_category"] == "" for r in rows)
        text = out.read_text()
        assert not any(cat in text for cat in C.ALL)   # no prediction leaks in

    def test_biggest_spend_first(self, test_csv, tmp_path):
        out = tmp_path / "labels.csv"
        label_sheet(test_csv, out)
        amounts = [float(r["total_amount"]) for r in csv.DictReader(out.open())]
        assert amounts == sorted(amounts, reverse=True)


class TestReadLabels:
    def test_canonical_legacy_and_blank(self, tmp_path):
        p = tmp_path / "l.csv"
        p.write_text("key,flow,true_category\na,OUT,groceries\nb,OUT,food\nc,OUT,\n"
                     "d,OUT,Personal Transfer\ne,OUT,other\n")
        assert read_labels(p) == {
            ("a", "OUT"): C.GROCERIES, ("b", "OUT"): C.FOOD_AND_DINING,
            ("d", "OUT"): C.PERSONAL_TRANSFER, ("e", "OUT"): C.OTHER,
        }

    def test_unknown_category_is_an_error_not_a_silent_other(self, tmp_path):
        p = tmp_path / "l.csv"
        p.write_text("key,flow,true_category\na,OUT,snacks\n")
        with pytest.raises(LabelError, match="snacks"):
            read_labels(p)


class TestEvaluate:
    @pytest.fixture
    def report(self, test_csv, tmp_path):
        sheet = tmp_path / "labels.csv"
        label_sheet(test_csv, sheet)
        _fill(sheet, {
            ("swiggy", "OUT"): C.FOOD_AND_DINING,
            ("ritik pandey", "OUT"): C.GROCERIES,           # a vendor on a personal account
            ("ritik pandey", "IN"): C.PERSONAL_TRANSFER,
            ("zzq unrecognizable enterprises holdings", "OUT"): C.SHOPPING,
            # "swiggy order" left blank on purpose
        })
        train = _statement(tmp_path / "train.csv", [("Paid to", "Swiggy", "DEBIT", 10)])
        return evaluate(test_csv, sheet, train_csv=train)

    def test_counts_only_labelled_rows(self, report):
        assert report["test_rows"] == 6
        assert report["labelled_rows"] == 5
        assert report["unlabelled_rows"] == 1

    def test_scores(self, report):
        s = report["configs"]["curated only"]
        # Swiggy right (auto), Ritik out x2 wrong (auto: person shape),
        # Ritik in right (auto), Zzq wrong but flagged (OTHER).
        assert s["accuracy"] == pytest.approx(2 / 5)
        assert s["coverage"] == pytest.approx(4 / 5)
        assert s["auto_accuracy"] == pytest.approx(2 / 4)
        assert s["silent_errors"] == 2
        assert s["error_catch_rate"] == pytest.approx(1 / 3)

    def test_silent_errors_are_listed(self, report):
        (e,) = report["silent_errors"]
        assert e["truth"] == C.GROCERIES and e["predicted"] == C.PERSONAL_TRANSFER
        assert e["rows"] == 2 and e["amount"] == 120

    def test_seen_vs_new(self, report):
        assert report["seen_vs_new"]["seen"]["rows"] == 1      # swiggy
        assert report["seen_vs_new"]["new"]["rows"] == 4

    def test_every_config_is_reported(self, report):
        assert list(report["configs"]) == ["curated only", "+ n-gram model", "+ MiniLM (full)"]

    def test_text_report_renders(self, report):
        text = as_text(report)
        assert "silent errors" in text and "curated only" in text
