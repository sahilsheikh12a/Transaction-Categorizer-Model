"""Command-line interface.

    python -m phonepe_categorizer serve                        # browser UI
    python -m phonepe_categorizer categorize transactions2.csv -o categorized.csv
    python -m phonepe_categorizer merchants  transactions2.csv -o merchants.csv
    python -m phonepe_categorizer report    transactions2.csv
    python -m phonepe_categorizer import    transactions2.csv --db categorizer.db
    python -m phonepe_categorizer classify  "Paid to NEW MATRUCHAYA GENERAL STORE"
    python -m phonepe_categorizer review    --limit 20
    python -m phonepe_categorizer correct   "Smart Point NAGPUR U178" GROCERIES
    python -m phonepe_categorizer rule      kirana GROCERIES --kind token
    python -m phonepe_categorizer reclassify
    python -m phonepe_categorizer train     transactions2.csv

`report` and `classify` are stateless. Everything else reads and writes the
SQLite file given by --db, which is where overrides and learned merchants live.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import categories as C
from .db import CategorizerRepository, open_session
from .engine import classify_core
from .export import categorize_csv, merchant_map_csv
from .importer import UnresolvedHeaders, parse_file
from .report import report as run_report

DEFAULT_DB = "categorizer.db"


def _url(path: str) -> str:
    return f"sqlite:///{path}"


def _print_result(label: str, res) -> None:
    flag = "  ⚠ needs review" if res.needs_review else ""
    print(f"{label}\n"
          f"  category    {res.category}\n"
          f"  confidence  {res.confidence:.2f}{flag}\n"
          f"  source      {res.source}\n"
          f"  txn type    {res.txn_type}\n"
          f"  normalized  {res.normalized_merchant!r}\n"
          f"  evidence    {res.evidence}")


def cmd_categorize(args) -> int:
    """CSV in, CSV out — the batch path."""
    overrides = learned = None
    if args.db and Path(args.db).exists():
        # Use whatever the system has already been taught.
        with open_session(_url(args.db)) as s:
            repo = CategorizerRepository(s)
            overrides, learned = repo.overrides(), repo.learned_merchants()
        if overrides or learned:
            print(f"using {len(overrides)} override(s) and "
                  f"{len(learned)} learned merchant(s) from {args.db}")

    out = args.output or str(Path(args.csv).with_name(Path(args.csv).stem + "_categorized.csv"))
    try:
        summary = categorize_csv(
            args.csv, out, overrides=overrides, learned=learned, dedupe=args.dedupe,
        )
    except UnresolvedHeaders as exc:
        print(f"error: {exc}")
        return 2
    print(summary.as_text())
    return 0


def cmd_merchants(args) -> int:
    """Two-column merchant -> category lookup table."""
    overrides = learned = None
    if args.db and Path(args.db).exists():
        with open_session(_url(args.db)) as s:
            repo = CategorizerRepository(s)
            overrides, learned = repo.overrides(), repo.learned_merchants()
    out = args.output or str(
        Path(args.csv).with_name(Path(args.csv).stem + "_merchants.csv")
    )
    try:
        summary = merchant_map_csv(args.csv, out, overrides=overrides, learned=learned)
    except UnresolvedHeaders as exc:
        print(f"error: {exc}")
        return 2
    print(summary.as_text())
    return 0


def cmd_report(args) -> int:
    run_report(Path(args.csv), Path(args.json) if args.json else None)
    return 0


def cmd_classify(args) -> int:
    text = " ".join(args.text)
    if args.db and Path(args.db).exists():
        with open_session(_url(args.db)) as s:
            res = CategorizerRepository(s).classify(
                text, is_credit=args.credit, direction=args.direction,
            )
    else:
        res = classify_core(text, is_credit=args.credit, direction=args.direction)
    _print_result(f"{text!r}", res)
    return 0


def cmd_import(args) -> int:
    try:
        parsed = parse_file(args.csv)
    except UnresolvedHeaders as exc:
        print(f"error: {exc}")
        return 2

    print(f"parsed {len(parsed.rows)} transactions "
          f"({parsed.duplicates} duplicates collapsed, {parsed.skipped} rows skipped)")
    for reason, count in parsed.problem_counts().items():
        print(f"  skipped: {reason} × {count}")

    with open_session(_url(args.db)) as s:
        stats = CategorizerRepository(s).import_transactions(parsed.rows)
    print(f"imported {stats['added']}, already present {stats['skipped_existing']}, "
          f"flagged for review {stats['needs_review']}")
    print(f"database: {args.db}")
    return 0


def cmd_review(args) -> int:
    with open_session(_url(args.db)) as s:
        rows = CategorizerRepository(s).review_queue(limit=args.limit)
        if not rows:
            print("review queue is empty")
            return 0
        print(f"{'AMOUNT':>10}  {'CATEGORY':<20}{'CONF':>5}  MERCHANT")
        for t in rows:
            print(f"{t.amount:>10,.0f}  {t.category:<20}{t.confidence:>5.2f}  "
                  f"{t.raw_counterparty[:44]}")
        print("\nCorrect one with:\n"
              "  python -m phonepe_categorizer correct \"<merchant>\" <CATEGORY>")
    return 0


def cmd_correct(args) -> int:
    if args.category not in C.VALID:
        print(f"error: unknown category {args.category!r}\nvalid: {', '.join(C.ALL)}")
        return 2
    with open_session(_url(args.db)) as s:
        stats = CategorizerRepository(s).record_correction(args.merchant, args.category)
    print(f"override saved; {stats['updated']} past transaction(s) updated")
    return 0


def cmd_rule(args) -> int:
    if args.category not in C.VALID:
        print(f"error: unknown category {args.category!r}\nvalid: {', '.join(C.ALL)}")
        return 2
    with open_session(_url(args.db)) as s:
        CategorizerRepository(s).add_rule(
            args.pattern, args.category, match_kind=args.kind, priority=args.priority,
        )
    print(f"rule added: {args.pattern!r} ({args.kind}) -> {args.category}")
    return 0


def cmd_reclassify(args) -> int:
    with open_session(_url(args.db)) as s:
        stats = CategorizerRepository(s).reclassify_all()
    print(f"reclassified {stats['total']} transactions: "
          f"{stats['recategorized']} recategorized, "
          f"{stats['source_changed']} kept their category via a different layer")
    return 0


def cmd_train(args) -> int:
    try:
        from .train import train
    except ImportError as exc:
        print(f"error: {exc}\ninstall the training extras: "
              f".venv/bin/pip install -r requirements-train.txt")
        return 2
    overrides = learned = None
    if args.db and Path(args.db).exists():
        with open_session(_url(args.db)) as s:
            repo = CategorizerRepository(s)
            overrides, learned = repo.overrides(), repo.learned_merchants()
    kwargs = {"output": Path(args.output)} if args.output else {}
    report = train([Path(p) for p in args.csv], overrides=overrides, learned=learned,
                   c=args.c, **kwargs)
    print(report.as_text())
    return 0


def cmd_serve(args) -> int:
    from .web import serve
    return serve(host=args.host, port=args.port, db=None if args.no_db else args.db)


def cmd_categories(args) -> int:
    for cat in C.ALL:
        print(f"{cat:<22}{C.DEFINITIONS[cat]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="phonepe_categorizer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "categorize",
        help="classify a CSV and write a categorized CSV (input columns preserved)",
    )
    p.add_argument("csv")
    p.add_argument("-o", "--output", help="defaults to <input>_categorized.csv")
    p.add_argument("--db", default=DEFAULT_DB,
                   help="apply overrides/learned merchants from this DB if it exists")
    p.add_argument("--dedupe", action="store_true",
                   help="collapse duplicate transactions instead of marking them")
    p.set_defaults(func=cmd_categorize)

    p = sub.add_parser(
        "merchants",
        help="write a two-column merchant,category CSV (one row per merchant)",
    )
    p.add_argument("csv")
    p.add_argument("-o", "--output", help="defaults to <input>_merchants.csv")
    p.add_argument("--db", default=DEFAULT_DB)
    p.set_defaults(func=cmd_merchants)

    p = sub.add_parser("report", help="classify a statement and print a full report")
    p.add_argument("csv")
    p.add_argument("--json", help="also write the report as JSON")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("classify", help="classify a single merchant string")
    p.add_argument("text", nargs="+")
    p.add_argument("--credit", action="store_true", help="treat as money received")
    p.add_argument("--direction", default="", help='e.g. "Paid to", "Received from"')
    p.add_argument("--db", default=DEFAULT_DB, help="use learned state from this DB")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("import", help="import a statement into the database")
    p.add_argument("csv")
    p.add_argument("--db", default=DEFAULT_DB)
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("review", help="list transactions needing review")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("correct", help="teach the system a merchant's category")
    p.add_argument("merchant")
    p.add_argument("category")
    p.add_argument("--db", default=DEFAULT_DB)
    p.set_defaults(func=cmd_correct)

    p = sub.add_parser("rule", help="add a user-defined keyword rule")
    p.add_argument("pattern")
    p.add_argument("category")
    p.add_argument("--kind", choices=("phrase", "token"), default="phrase")
    p.add_argument("--priority", type=int, default=100)
    p.add_argument("--db", default=DEFAULT_DB)
    p.set_defaults(func=cmd_rule)

    p = sub.add_parser("reclassify", help="re-run the pipeline over stored rows")
    p.add_argument("--db", default=DEFAULT_DB)
    p.set_defaults(func=cmd_reclassify)

    p = sub.add_parser("train", help="retrain the stage-6 ONNX model")
    p.add_argument("csv", nargs="+", help="one or more statements to learn from")
    p.add_argument("--db", default=DEFAULT_DB,
                   help="also learn from this DB's corrections and merchants if it exists")
    p.add_argument("-o", "--output", help="defaults to the shipped model path")
    p.add_argument("--c", type=float, default=10.0, help="inverse regularization strength")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("serve", help="run the local browser UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--no-db", action="store_true",
                   help="do not persist corrections (scratch session)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("categories", help="list the category taxonomy")
    p.set_defaults(func=cmd_categories)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
