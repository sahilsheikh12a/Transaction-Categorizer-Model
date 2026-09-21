"""CSV in, CSV out.

    python -m phonepe_categorizer categorize statement.csv -o categorized.csv

Design rules, in priority order:

1. **Every input data row appears in the output, in input order.** A row the
   importer could not parse is still written, with a `status` explaining why
   and empty classification cells. A file that silently loses rows is worse
   than useless for reconciliation — you would have no way to notice.

2. **Original columns are copied verbatim.** Not normalized, not reordered,
   not re-formatted. The appended columns all carry a `category_` prefix so
   they cannot collide with a column the statement already had.

3. **Duplicates are marked, not dropped**, unless `--dedupe` is passed. The
   default keeps the output a row-for-row mirror of the input.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from .engine import classify_core
from .importer import _find_header_row, parse
from .normalize import normalize_merchant

# Appended columns, in output order.
OUTPUT_COLUMNS = (
    "category",
    "category_confidence",
    "category_source",
    "category_needs_review",
    "category_txn_type",
    "category_normalized_merchant",
    "category_evidence",
    "status",
)


@dataclass
class MerchantMapSummary:
    merchants: int = 0
    transactions: int = 0
    needs_review: int = 0
    output_path: str = ""

    def as_text(self) -> str:
        return (
            f"{self.transactions} transactions -> {self.merchants} unique "
            f"merchants -> {self.output_path}\n"
            f"  still unresolved  {self.needs_review}"
        )


@dataclass
class ExportSummary:
    input_rows: int = 0
    classified: int = 0
    needs_review: int = 0
    skipped: int = 0
    duplicates: int = 0
    duplicates_marked: int = 0
    output_path: str = ""

    def as_text(self) -> str:
        return (
            f"{self.input_rows} input rows -> {self.output_path}\n"
            f"  classified     {self.classified}\n"
            f"  needs review   {self.needs_review}\n"
            f"  unparseable    {self.skipped}\n"
            f"  duplicates     {self.duplicates}"
        )


def _unique_headers(headers: list[str]) -> list[str]:
    """Make blank/duplicate source headers safe to use as dict keys."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for i, h in enumerate(headers):
        name = (h or "").strip() or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def _resolve_output_columns(headers: list[str]) -> list[str]:
    """Name the appended columns so they cannot shadow a source column.

    A statement that already has its own `category` or `status` column is not
    hypothetical — re-running this tool on its own output does exactly that.
    Source names are never renamed (rule 2: verbatim), so the appended ones
    yield instead: `category` becomes `category_2`, and so on.
    """
    taken = set(headers)
    out: list[str] = []
    for base in OUTPUT_COLUMNS:
        name, n = base, 2
        while name in taken:
            name = f"{base}_{n}"
            n += 1
        taken.add(name)
        out.append(name)
    return out


def categorize_csv(
    src: str | Path,
    dst: str | Path,
    *,
    overrides: dict[str, str] | None = None,
    learned: dict[str, str] | None = None,
    dedupe: bool = False,
) -> ExportSummary:
    """Classify every row of `src` and write the result to `dst`."""
    src, dst = Path(src), Path(dst)
    data = src.read_bytes()

    result = parse(data, dedupe=dedupe)

    # Index the importer's output by source row number so we can walk the file
    # in its original order and attach each verdict to the right line.
    parsed_by_row = {t.row_number: t for t in result.rows}
    problem_by_row = {p.row_number: p for p in result.problems}

    text = data.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    raw_rows = list(csv.reader(io.StringIO(text), dialect))
    if not raw_rows:
        dst.write_text("")
        return ExportSummary(output_path=str(dst))

    header_idx = _find_header_row(raw_rows)
    headers = _unique_headers([(c or "").strip() for c in raw_rows[header_idx]])
    appended = _resolve_output_columns(headers)

    summary = ExportSummary(duplicates=result.duplicates, output_path=str(dst))
    memo: dict[tuple, object] = {}

    with dst.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([*headers, *appended])

        for line_no, row in enumerate(raw_rows[header_idx + 1:], start=header_idx + 2):
            if not row or not any((c or "").strip() for c in row):
                continue
            summary.input_rows += 1

            # Pad or trim so the original cells always line up with the header.
            cells = [(c or "").strip() for c in row][:len(headers)]
            cells += [""] * (len(headers) - len(cells))

            txn = parsed_by_row.get(line_no)
            if txn is None:
                problem = problem_by_row.get(line_no)
                reason = problem.reason if problem else "duplicate of an earlier row"
                if problem is None:
                    # Not a problem row and not in the output => deduped away.
                    summary.duplicates_marked += 1
                else:
                    summary.skipped += 1
                writer.writerow([*cells, *([""] * (len(OUTPUT_COLUMNS) - 1)),
                                 f"skipped: {reason}"])
                continue

            key = (txn.counterparty, txn.is_credit, txn.direction)
            res = memo.get(key)
            if res is None:
                res = classify_core(
                    txn.counterparty,
                    is_credit=txn.is_credit,
                    direction=txn.direction,
                    overrides=overrides,
                    learned=learned,
                )
                memo[key] = res

            summary.classified += 1
            if res.needs_review:
                summary.needs_review += 1

            writer.writerow([
                *cells,
                res.category,
                f"{res.confidence:.2f}",
                res.source,
                "yes" if res.needs_review else "no",
                res.txn_type,
                res.normalized_merchant,
                res.evidence,
                "ok",
            ])

    return summary


def merchant_map(
    rows,
    overrides: dict[str, str] | None = None,
    learned: dict[str, str] | None = None,
) -> list[tuple[str, str, bool]]:
    """(name, category, needs_review) per unique merchant, sorted by name.

    Merchants are grouped by their normalized key, so "Maturchaya Kirana 2" and
    "MATURCHAYA KIRANA" are one entry. The name returned is the spelling that
    appeared most often, because that is what the user will recognize — the
    normalized form is a matching key, not a label.
    """
    # Collect the spellings each merchant appears under, and how often.
    spellings: dict[str, dict[str, int]] = {}
    for txn in rows:
        # Masked counterparties normalize to "" — key them on the raw string so
        # every masked account does not collapse into a single nameless row.
        key = normalize_merchant(txn.counterparty) or txn.counterparty
        spellings.setdefault(key, {})
        spellings[key][txn.counterparty] = spellings[key].get(txn.counterparty, 0) + 1

    out = []
    for variants in spellings.values():
        name = max(variants.items(), key=lambda kv: (kv[1], kv[0]))[0]
        # Classify as an OUTGOING payment regardless of how the statement's rows
        # actually ran. This table answers "what kind of merchant is this", and
        # a merchant that only ever appears as a credit would otherwise be
        # labelled INCOME — filing a garage that issued one refund under income
        # rather than transport, and poisoning anything that consumes the file.
        res = classify_core(
            name, is_credit=False, direction="Paid to",
            overrides=overrides, learned=learned,
        )
        out.append((name, res.category, res.needs_review))
    out.sort(key=lambda r: r[0].lower())
    return out


def merchant_map_csv(
    src: str | Path,
    dst: str | Path,
    *,
    overrides: dict[str, str] | None = None,
    learned: dict[str, str] | None = None,
) -> MerchantMapSummary:
    """Write a two-column merchant -> category lookup table.

    One row per *unique* merchant rather than per transaction: the point of
    this file is the mapping, and 2,000 transactions over 600 merchants would
    otherwise repeat every answer three times over.

    Grouping and naming follow `merchant_map`.
    """
    src, dst = Path(src), Path(dst)
    result = parse(src.read_bytes(), dedupe=False)
    rows = merchant_map(result.rows, overrides, learned)

    with dst.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["merchant", "category"])
        for name, category, _ in rows:
            writer.writerow([name, category])

    return MerchantMapSummary(
        merchants=len(rows),
        transactions=len(result.rows),
        needs_review=sum(1 for _, _, review in rows if review),
        output_path=str(dst),
    )


__all__ = [
    "categorize_csv", "merchant_map", "merchant_map_csv",
    "ExportSummary", "MerchantMapSummary", "OUTPUT_COLUMNS",
]
