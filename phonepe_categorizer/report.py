"""Run the categorizer over a real statement and report what it did.

    python -m phonepe_categorizer.report <statement.csv> [--json out.json]

This is a *review* tool, not a test. It never asserts the classifier is right —
it surfaces the evidence needed to judge that by hand, which is why the last
two sections (ambiguous classifications, weak-evidence merchants) matter more
than the headline percentages.

Everything runs against the pure `classify_core`, so no database, no network,
and no user context is involved. Same statement in, same report out.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from . import categories as C
from .engine import classify_core
from .importer import parse_file
from .normalize import normalize_merchant

BAR = "═" * 78
SUB = "─" * 78


def _fmt_money(v: float) -> str:
    return f"₹{v:,.0f}"


def evaluate(path: Path, *, overrides: dict[str, str] | None = None) -> dict:
    parsed = parse_file(path)

    records = []
    for txn in parsed.rows:
        res = classify_core(
            txn.counterparty,
            is_credit=txn.is_credit,
            direction=txn.direction,
            overrides=overrides,
        )
        records.append({
            "raw": txn.counterparty,
            "normalized": res.normalized_merchant or normalize_merchant(txn.counterparty),
            "amount": txn.amount,
            "is_credit": txn.is_credit,
            "direction": txn.direction,
            "category": res.category,
            "source": res.source,
            "confidence": res.confidence,
            "needs_review": res.needs_review,
            "txn_type": res.txn_type,
            "evidence": res.evidence,
        })

    return {"parsed": parsed, "records": records}


def report(path: Path, out_json: Path | None = None) -> None:
    data = evaluate(path)
    parsed, recs = data["parsed"], data["records"]
    n = len(recs)
    if not n:
        print("No rows parsed.")
        return

    debits = [r for r in recs if not r["is_credit"]]
    credits = [r for r in recs if r["is_credit"]]
    review = [r for r in recs if r["needs_review"]]
    other = [r for r in recs if r["category"] == C.OTHER]
    auto = [r for r in recs if not r["needs_review"]]

    print(BAR)
    print(f" CATEGORIZATION REPORT — {path.name}")
    print(BAR)

    # ── 1-6. headline counts ────────────────────────────────────────────
    problems = parsed.problem_counts()
    print(f"\n1.  Total transactions parsed      {n:>6}   (of {parsed.total_data_rows} data rows)")
    print(f"    Duplicate rows collapsed       {parsed.duplicates:>6}")
    print(f"    Rows skipped by the importer   {parsed.skipped:>6}   "
          f"{problems if problems else ''}")
    debit_total = _fmt_money(sum(r["amount"] for r in debits))
    credit_total = _fmt_money(sum(r["amount"] for r in credits))
    print(f"2.  Debit transactions             {len(debits):>6}   {debit_total}")
    print(f"3.  Credit transactions            {len(credits):>6}   {credit_total}")
    print(f"4.  Classified automatically       {len(auto):>6}   {100*len(auto)/n:5.1f}%")
    print(f"5.  Requiring review               {len(review):>6}   {100*len(review)/n:5.1f}%")
    print(f"6.  Classified as OTHER            {len(other):>6}   {100*len(other)/n:5.1f}%")

    # ── 7. category + source breakdown ──────────────────────────────────
    print(f"\n{SUB}\n7.  BREAKDOWN BY CATEGORY\n{SUB}")
    cat_n = Counter(r["category"] for r in recs)
    cat_amt: dict[str, float] = defaultdict(float)
    for r in recs:
        cat_amt[r["category"]] += r["amount"]
    print(f"    {'CATEGORY':<22}{'TXNS':>6}{'SHARE':>8}{'VALUE':>14}")
    for cat in C.ALL:
        if not cat_n.get(cat):
            continue
        share = 100 * cat_n[cat] / n
        print(f"    {cat:<22}{cat_n[cat]:>6}{share:>7.1f}%"
              f"{_fmt_money(cat_amt[cat]):>14}")

    print("\n    CLASSIFICATION SOURCE")
    src_n = Counter(r["source"] for r in recs)
    for src in C.SOURCES:
        if src_n.get(src):
            print(f"    {src:<22}{src_n[src]:>6}{100*src_n[src]/n:>7.1f}%")

    print("\n    TRANSACTION TYPE")
    for t, c in Counter(r["txn_type"] for r in recs).most_common():
        print(f"    {t:<22}{c:>6}{100*c/n:>7.1f}%")

    # ── 8/9. unique merchants and their verdicts ────────────────────────
    merch: dict[str, dict] = {}
    for r in recs:
        # Masked/opaque counterparties normalize to "", which would collapse
        # every distinct masked account into one meaningless report row.
        key = r["normalized"] or f"(opaque) {r['raw'][:24]}"
        m = merch.setdefault(key, {
            "raws": Counter(), "n": 0, "amounts": [], "category": r["category"],
            "source": r["source"], "confidence": r["confidence"],
            "needs_review": r["needs_review"], "evidence": r["evidence"],
            "txn_types": Counter(),
        })
        m["raws"][r["raw"]] += 1
        m["n"] += 1
        m["amounts"].append(r["amount"])
        m["txn_types"][r["txn_type"]] += 1

    print(f"\n{SUB}\n8/9. UNIQUE MERCHANTS ({len(merch)}) AND ASSIGNED CATEGORY\n{SUB}")
    print(f"    {'N':>4}  {'MEDIAN':>8}  {'CATEGORY':<20}{'SOURCE':<16}{'CONF':>5}  MERCHANT")
    for key, m in sorted(merch.items(), key=lambda kv: -kv[1]["n"]):
        flag = "!" if m["needs_review"] else " "
        med = statistics.median(m["amounts"])
        print(f"  {flag} {m['n']:>4}  {med:>8.0f}  {m['category']:<20}{m['source']:<16}"
              f"{m['confidence']:>5.2f}  {key[:44]}")

    # ── 10. ambiguous / suspicious ──────────────────────────────────────
    print(f"\n{SUB}\n10. SUSPICIOUS & AMBIGUOUS CLASSIFICATIONS\n{SUB}")

    # (a) Person-named counterparties that behave like shops: many small
    #     payments. A friend does not take 30 payments of ₹40.
    vendor_like = [
        (k, m) for k, m in merch.items()
        if m["category"] == C.PERSONAL_TRANSFER
        and m["n"] >= 8
        and statistics.median(m["amounts"]) <= 150
        and m["txn_types"].get("MONEY_RECEIVED", 0) == 0
    ]
    print(f"\n  (a) Person-named counterparties behaving like merchants  [{len(vendor_like)}]")
    print("      Many small outgoing payments — likely a vendor collecting on a")
    print("      personal UPI account. Classified PERSONAL_TRANSFER; probably wrong.")
    for k, m in sorted(vendor_like, key=lambda kv: -kv[1]["n"]):
        median = _fmt_money(statistics.median(m["amounts"]))
        print(f"      {m['n']:>4} txns  median {median:>7}  {k}")

    # (b) Categories resting on a single generic keyword.
    weak_patterns = {"general", "tea", "point", "hotel", "collection", "novelty",
                     "mess", "sweet", "fashion", "electric", "metro", "ola"}
    weak = [
        (k, m) for k, m in merch.items()
        if m["source"] == C.SRC_RULE
        and any(f"'{w}'" == m["evidence"].split("rule: ")[-1] for w in weak_patterns)
    ]
    print(f"\n  (b) Category rests on one generic keyword  [{len(weak)}]")
    print("      A single common word decided these. Highest risk of a wrong rule.")
    for k, m in sorted(weak, key=lambda kv: -kv[1]["n"]):
        print(f"      {m['n']:>4} txns  {m['category']:<20}"
              f"{m['evidence']:<22}{k}")

    # (c) Fuzzy matches — always worth an eyeball.
    fz = [(k, m) for k, m in merch.items() if m["source"] == C.SRC_FUZZY_MATCH]
    print(f"\n  (c) Fuzzy matches  [{len(fz)}]")
    for k, m in sorted(fz, key=lambda kv: -kv[1]["n"]):
        print(f"      {m['n']:>4} txns  {m['category']:<20}{m['confidence']:.2f}  "
              f"{k}  {m['evidence']}")

    # (d) Highest-impact unresolved merchants, by money and by volume.
    unresolved = [(k, m) for k, m in merch.items() if m["needs_review"]]
    by_value = sorted(unresolved, key=lambda kv: -sum(kv[1]["amounts"]))[:20]
    print(f"\n  (d) Unresolved merchants by value  [{len(unresolved)} total need review]")
    print("      Fixing the top of this list buys the most accuracy per correction.")
    for k, m in by_value:
        print(f"      {_fmt_money(sum(m['amounts'])):>12}  {m['n']:>4} txns  "
              f"{m['category']:<20}{k[:40]}")

    # (e) Amount profile contradicts the category.
    odd = []
    for k, m in merch.items():
        mx = max(m["amounts"])
        if m["category"] in (C.GROCERIES, C.FOOD_AND_DINING) and mx >= 10000:
            odd.append((k, m, mx))
    print(f"\n  (e) Amount profile contradicts the category  [{len(odd)}]")
    for k, m, mx in sorted(odd, key=lambda x: -x[2]):
        print(f"      max {_fmt_money(mx):>10}  {m['category']:<20}{k[:44]}")

    print(f"\n{BAR}")

    if out_json:
        payload = {
            "totals": {
                "transactions": n, "debits": len(debits), "credits": len(credits),
                "auto_classified": len(auto), "needs_review": len(review),
                "other": len(other), "unique_merchants": len(merch),
            },
            "by_category": {c: cat_n[c] for c in C.ALL if cat_n.get(c)},
            "by_source": {s: src_n[s] for s in C.SOURCES if src_n.get(s)},
            "merchants": [
                {
                    "normalized": k, "n": m["n"], "category": m["category"],
                    "source": m["source"], "confidence": m["confidence"],
                    "needs_review": m["needs_review"], "evidence": m["evidence"],
                    "median_amount": statistics.median(m["amounts"]),
                    "total_amount": sum(m["amounts"]),
                    "raw_variants": list(m["raws"]),
                }
                for k, m in sorted(merch.items(), key=lambda kv: -kv[1]["n"])
            ],
        }
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"JSON written to {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    report(args.csv, args.json)


if __name__ == "__main__":
    main()
