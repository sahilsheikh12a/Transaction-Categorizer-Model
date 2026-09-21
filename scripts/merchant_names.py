"""Write the unique merchants of a statement to a CSV: name and DEBIT/CREDIT.

    .venv/bin/python scripts/merchant_names.py test.csv -o reports/merchants.csv

Names are the counterparty text as printed on the statement, one line per
distinct merchant (by matching key) and type, in first-seen order. A merchant
you both paid and received money from gets two lines.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phonepe_categorizer.evaluate import merchant_key  # noqa: E402
from phonepe_categorizer.importer import parse_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="statement in the 10-column schema")
    ap.add_argument("-o", "--output", default="reports/merchant_names.csv")
    args = ap.parse_args()

    names: dict[tuple[str, str], str] = {}
    for txn in parse_file(args.csv).rows:
        kind = "CREDIT" if txn.is_credit else "DEBIT"
        names.setdefault((merchant_key(txn.counterparty), kind), txn.counterparty.strip())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["merchant", "type"])
        w.writerows([name, kind] for (_, kind), name in names.items())
    print(f"{len(names)} merchants -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
