"""Evaluate the categorizer against a hand-labelled test statement.

    python -m phonepe_categorizer label-sheet test.csv -o labels.csv
    # ... fill in the true_category column ...
    python -m phonepe_categorizer evaluate test.csv labels.csv --train transactions2.csv

WHY LABEL MERCHANTS, NOT ROWS
-----------------------------
The category belongs to the merchant, not to the single payment. Labelling
one line per (merchant, money in/out) is roughly a third of the work of
labelling every row, and every row inherits its merchant's label.

WHY THE SHEET IS BLIND
----------------------
The sheet shows the raw statement text, the count and the amount, and never
the system's prediction. A reviewer who sees "GROCERIES 0.90" next to a
name tends to agree with it, and then the evaluation grades the system
against its own answers.

WHAT GETS REPORTED
------------------
Every number is computed over labelled rows only:

  accuracy              all rows, and weighted by amount
  coverage              rows decided without asking (needs_review False)
  auto accuracy         of those, how many are right: the headline number
  silent errors         auto-decided AND wrong; nobody will ever look at these
  error catch rate      of all wrong rows, the share the review flag caught
  per layer             which layer decided, and how often it was right
  per category          precision / recall, plus the most common confusions
  seen vs new           merchants also present in the training statement vs
                        never seen, which is the real test of generalization
  ablation              the same metrics with the model layers switched off,
                        to show what each one adds
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from . import categories as C
from . import ml as _ml
from . import semantic as _semantic
from .engine import classify_core
from .importer import ImportedTransaction, parse_file
from .normalize import canon, normalize_merchant

SHEET_COLUMNS = ("key", "flow", "example", "count", "total_amount", "direction",
                 "true_category", "notes")

# Pipeline variants for the ablation, in the order they are reported.
CONFIGS = (
    ("curated only", {"ml": False, "semantic": False}),
    ("+ n-gram model", {"ml": True, "semantic": False}),
    ("+ MiniLM (full)", {"ml": True, "semantic": True}),
)


def merchant_key(counterparty: str) -> str:
    """The label key: the matching key, or the raw text when that is empty.

    Masked handles and bare numbers normalize to "", but are still distinct
    counterparties that deserve their own label.
    """
    return normalize_merchant(counterparty) or canon(counterparty) or counterparty.strip()


def _flow(txn: ImportedTransaction) -> str:
    return "IN" if txn.is_credit else "OUT"


# ── labelling sheet ──────────────────────────────────────────────────────

def label_sheet(test_csv: Path, out: Path) -> int:
    """Write one blind row per (merchant, flow). Returns the number of rows."""
    groups: dict[tuple[str, str], dict] = {}
    for txn in parse_file(test_csv).rows:
        k = (merchant_key(txn.counterparty), _flow(txn))
        g = groups.setdefault(k, {"example": txn.counterparty, "count": 0,
                                  "total": 0.0, "direction": txn.direction})
        g["count"] += 1
        g["total"] += txn.amount
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SHEET_COLUMNS)
        # Biggest money first, so a partial labelling still covers most spend.
        for (key, flow), g in sorted(groups.items(), key=lambda kv: -kv[1]["total"]):
            w.writerow([key, flow, g["example"], g["count"], f"{g['total']:.2f}",
                        g["direction"], "", ""])
    return len(groups)


class LabelError(ValueError):
    pass


def read_labels(path: Path) -> dict[tuple[str, str], str]:
    """(key, flow) -> canonical category. Blank labels are skipped."""
    labels, bad = {}, []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            raw = (row.get("true_category") or "").strip()
            if not raw:
                continue
            cat = raw.upper().replace(" ", "_").replace("-", "_")
            if cat not in C.VALID:
                legacy = C.coerce(raw)
                if legacy == C.OTHER and raw.lower() != "other":
                    bad.append(f"line {i}: {raw!r}")
                    continue
                cat = legacy
            labels[(row["key"], row["flow"])] = cat
    if bad:
        raise LabelError("unknown categories in the label sheet:\n  " + "\n  ".join(bad)
                         + f"\nvalid: {', '.join(C.ALL)}")
    return labels


# ── scoring ──────────────────────────────────────────────────────────────

@dataclass
class Row:
    key: str
    flow: str
    raw: str
    amount: float
    truth: str
    pred: str
    source: str
    review: bool
    seen: bool

    @property
    def ok(self) -> bool:
        return self.pred == self.truth


def _predict(rows: list[ImportedTransaction], config: dict, learned=None, overrides=None):
    with ExitStack() as stack:
        if not config["ml"]:
            stack.enter_context(_ml.disabled())
        if not config["semantic"]:
            stack.enter_context(_semantic.disabled())
        return [classify_core(t.counterparty, is_credit=t.is_credit,
                              direction=t.direction, learned=learned, overrides=overrides)
                for t in rows]


def _pct(a: float, b: float) -> float:
    return a / b if b else 0.0


def score(rows: list[Row]) -> dict:
    n = len(rows)
    auto = [r for r in rows if not r.review]
    wrong = [r for r in rows if not r.ok]
    silent = [r for r in auto if not r.ok]
    amount = sum(r.amount for r in rows)
    return {
        "rows": n,
        "accuracy": _pct(sum(r.ok for r in rows), n),
        "accuracy_by_amount": _pct(sum(r.amount for r in rows if r.ok), amount),
        "coverage": _pct(len(auto), n),
        "auto_accuracy": _pct(sum(r.ok for r in auto), len(auto)),
        "flagged_accuracy": _pct(sum(r.ok for r in rows if r.review), n - len(auto)),
        "errors": len(wrong),
        "silent_errors": len(silent),
        "error_catch_rate": _pct(len(wrong) - len(silent), len(wrong)),
    }


def _per_layer(rows: list[Row]) -> dict:
    out = {}
    for src in C.SOURCES:
        rs = [r for r in rows if r.source == src]
        if rs:
            out[src] = {"rows": len(rs), "accuracy": _pct(sum(r.ok for r in rs), len(rs)),
                        "auto_share": _pct(sum(not r.review for r in rs), len(rs))}
    return out


def _per_category(rows: list[Row]) -> dict:
    out = {}
    for cat in C.ALL:
        tp = sum(r.ok and r.truth == cat for r in rows)
        predicted = sum(r.pred == cat for r in rows)
        actual = sum(r.truth == cat for r in rows)
        if predicted or actual:
            out[cat] = {"support": actual, "precision": _pct(tp, predicted),
                        "recall": _pct(tp, actual)}
    return out


def evaluate(
    test_csv: Path,
    labels_csv: Path,
    *,
    train_csv: Path | None = None,
    learned: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
) -> dict:
    txns = parse_file(test_csv).rows
    labels = read_labels(labels_csv)
    train_keys = ({merchant_key(t.counterparty) for t in parse_file(train_csv).rows}
                  if train_csv else set())

    labelled = [(t, labels[(merchant_key(t.counterparty), _flow(t))]) for t in txns
                if (merchant_key(t.counterparty), _flow(t)) in labels]

    report: dict = {
        "frozen": _frozen_state(),
        "database": {"learned": len(learned or {}), "overrides": len(overrides or {})},
        "test_rows": len(txns),
        "labelled_rows": len(labelled),
        "unlabelled_rows": len(txns) - len(labelled),
        "configs": {},
    }
    full_rows: list[Row] = []
    for name, config in CONFIGS:
        preds = _predict([t for t, _ in labelled], config, learned, overrides)
        rows = [Row(merchant_key(t.counterparty), _flow(t), t.counterparty, t.amount, truth,
                    p.category, p.source, p.needs_review,
                    merchant_key(t.counterparty) in train_keys)
                for (t, truth), p in zip(labelled, preds, strict=True)]
        report["configs"][name] = score(rows)
        full_rows = rows

    report["per_layer"] = _per_layer(full_rows)
    report["per_category"] = _per_category(full_rows)
    report["confusions"] = [
        {"truth": t, "predicted": p, "rows": k}
        for (t, p), k in Counter((r.truth, r.pred) for r in full_rows if not r.ok).most_common(10)
    ]
    if train_keys:
        report["seen_vs_new"] = {
            "seen": score([r for r in full_rows if r.seen]),
            "new": score([r for r in full_rows if not r.seen]),
        }
    silent: dict[tuple, list[Row]] = defaultdict(list)
    for r in full_rows:
        if not r.review and not r.ok:
            silent[(r.key, r.truth, r.pred, r.source)].append(r)
    report["silent_errors"] = [
        {"merchant": rs[0].raw, "truth": t, "predicted": p, "source": s,
         "rows": len(rs), "amount": round(sum(r.amount for r in rs), 2)}
        for (_, t, p, s), rs in sorted(silent.items(),
                                       key=lambda kv: -sum(r.amount for r in kv[1]))
    ]
    return report


def _frozen_state() -> dict:
    """Exactly what was evaluated, so the numbers can be tied to a version."""
    here = Path(__file__).resolve().parent

    def sha(p: Path) -> str | None:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else None

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=here, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        "ngram_model": sha(_ml.MODEL_PATH),
        "minilm_index": sha(_semantic.MODEL_DIR / _semantic.INDEX_FILE),
    }


# ── text report ──────────────────────────────────────────────────────────

def as_text(r: dict) -> str:
    f = r["frozen"]
    lines = [
        f"evaluated commit {f['commit'] or '?'}"
        + (" (with uncommitted changes)" if f["uncommitted_changes"] else "")
        + f"  n-gram {f['ngram_model'] or 'absent'}  minilm {f['minilm_index'] or 'absent'}",
        f"database state: {r['database']['learned']} learned merchants, "
        f"{r['database']['overrides']} overrides",
        f"test rows {r['test_rows']}, labelled {r['labelled_rows']}, "
        f"unlabelled {r['unlabelled_rows']} (excluded)",
        "",
        f"{'':<17}{'accuracy':>9}{'by ₹':>7}{'coverage':>10}{'auto acc':>10}"
        f"{'silent err':>12}{'caught':>8}",
    ]
    for name, s in r["configs"].items():
        lines.append(
            f"{name:<17}{s['accuracy']:>9.1%}{s['accuracy_by_amount']:>7.0%}"
            f"{s['coverage']:>10.1%}{s['auto_accuracy']:>10.1%}"
            f"{s['silent_errors']:>12}{s['error_catch_rate']:>8.0%}")
    if "seen_vs_new" in r:
        lines += ["", "merchants seen in training vs new (full system):"]
        for name, s in r["seen_vs_new"].items():
            lines.append(f"  {name:<6}{s['rows']:>5} rows  accuracy {s['accuracy']:.1%}  "
                         f"auto accuracy {s['auto_accuracy']:.1%}  coverage {s['coverage']:.1%}")
    lines += ["", "by deciding layer (full system):"]
    for src, s in r["per_layer"].items():
        lines.append(f"  {src:<16}{s['rows']:>5} rows  accuracy {s['accuracy']:>6.1%}  "
                     f"auto {s['auto_share']:>5.0%}")
    lines += ["", f"  {'category':<22}{'support':>8}{'precision':>11}{'recall':>8}"]
    for cat, s in r["per_category"].items():
        lines.append(f"  {cat:<22}{s['support']:>8}{s['precision']:>11.1%}{s['recall']:>8.1%}")
    if r["confusions"]:
        lines += ["", "most common mistakes (truth -> predicted):"]
        lines += [f"  {c['rows']:>4}  {c['truth']} -> {c['predicted']}" for c in r["confusions"]]
    if r["silent_errors"]:
        lines += ["", "silent errors: wrong and NOT flagged for review (biggest first):"]
        lines += [f"  ₹{e['amount']:>9,.0f}  x{e['rows']:<3} {e['merchant'][:40]:<40} "
                  f"{e['truth']} -> {e['predicted']} [{e['source']}]"
                  for e in r["silent_errors"][:20]]
    return "\n".join(lines)


def write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


__all__ = ["label_sheet", "read_labels", "evaluate", "score", "as_text", "write_json",
           "merchant_key", "LabelError", "SHEET_COLUMNS"]
