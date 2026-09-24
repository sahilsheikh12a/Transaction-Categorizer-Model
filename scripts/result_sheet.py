"""Write a result sheet: one line per merchant with what the system predicted.

    .venv/bin/python scripts/result_sheet.py statement.csv -o reports/results.csv

Three columns: input (the merchant as printed), actual (yours to fill in) and
predicted. Fill in `actual`, re-run the same command, and it keeps what you
typed and prints how many match.

NOTE: a sheet that shows the prediction anchors the labeller — people agree
with what is already written. For a trustworthy accuracy figure label the
blind sheet instead (`python -m phonepe_categorizer label-sheet`).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phonepe_categorizer import categories as C  # noqa: E402
from phonepe_categorizer.engine import classify_core  # noqa: E402
from phonepe_categorizer.evaluate import merchant_key  # noqa: E402
from phonepe_categorizer.importer import parse_file  # noqa: E402

COLUMNS = ("input", "actual", "predicted")


def build(statement: Path, out: Path, previous: dict[tuple[str, str], str]) -> int:
    groups: dict[tuple[str, str], dict] = {}
    for txn in parse_file(statement).rows:
        kind = "CREDIT" if txn.is_credit else "DEBIT"
        key = (merchant_key(txn.counterparty), kind)
        g = groups.setdefault(key, {"example": txn.counterparty.strip(),
                                    "total": 0.0, "txn": txn})
        g["total"] += txn.amount

    rows = []
    for (key, kind), g in groups.items():
        t = g["txn"]
        res = classify_core(t.counterparty, is_credit=t.is_credit, direction=t.direction)
        # Money in and money out are separate answers for the same name, so the
        # direction stays with the input text rather than in its own column.
        label = g["example"] if kind == "DEBIT" else f"{g['example']} (money in)"
        rows.append([label, previous.get((key, kind), ""), res.category, g["total"]])
    rows.sort(key=lambda r: -r[3])
    rows = [r[:3] for r in rows]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)

    labelled = [r for r in rows if r[1]]
    if labelled:
        right = sum(r[1] == r[2] for r in labelled)
        print(f"labelled {len(labelled)} of {len(rows)} merchants — {right} match "
              f"({right / len(labelled):.1%})")
    return len(rows)


def read_previous(path: Path) -> dict[tuple[str, str], str]:
    """Keep any `actual` values already filled in, so re-running is safe."""
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            actual = (row.get("actual") or "").strip()
            if actual:
                text = row["input"]
                kind = "CREDIT" if text.endswith("(money in)") else "DEBIT"
                text = text.removesuffix(" (money in)")
                out[(merchant_key(text), kind)] = C.coerce(actual)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="statement in the 10-column schema")
    ap.add_argument("-o", "--output", default="reports/results.csv")
    args = ap.parse_args()

    out = Path(args.output)
    n = build(Path(args.csv), out, read_previous(out))
    print(f"{n} merchants -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
