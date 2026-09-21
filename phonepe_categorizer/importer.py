"""PhonePe statement CSV importer.

Resolves columns by *role* rather than by position, so a statement whose
columns were reordered, renamed slightly, or exported with extra fields still
imports. Nothing about the schema is assumed: `resolve_headers` reports what it
found and `ImportResult.mapping` echoes it back so a caller can show the user
which column became which field.

Resilience is the point. One malformed row must never abort an import of two
thousand good ones, so every row is parsed defensively and anything
unparseable is counted and recorded in `ImportResult.problems` rather than
raised. The only fatal condition is being unable to locate a date or a
counterparty column at all — without those there is no transaction to build.

Reference schema (the PhonePe PDF parser's locked 10-column output):

    date, time, direction, counterparty, transaction_id, utr,
    account_flow, account_last4, type, amount
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime

# ── column roles ──────────────────────────────────────────────────────────

ROLE_DATE = "date"
ROLE_TIME = "time"
ROLE_DIRECTION = "direction"
ROLE_COUNTERPARTY = "counterparty"
ROLE_TXN_ID = "transaction_id"
ROLE_UTR = "utr"
ROLE_ACCOUNT_FLOW = "account_flow"
ROLE_ACCOUNT_LAST4 = "account_last4"
ROLE_TYPE = "type"
ROLE_AMOUNT = "amount"
ROLE_DEBIT = "debit"
ROLE_CREDIT = "credit"

# Header synonyms, checked as normalized substrings. Order within a role does
# not matter; the first *column* that matches an unclaimed role wins.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    ROLE_DATE: ("date", "txn date", "transaction date", "value date", "posted on"),
    ROLE_TIME: ("time", "txn time", "transaction time"),
    ROLE_DIRECTION: ("direction", "txn direction", "flow", "narration type"),
    ROLE_COUNTERPARTY: (
        "counterparty", "payee", "beneficiary", "merchant", "name",
        "narration", "description", "particulars", "remarks", "details",
    ),
    ROLE_TXN_ID: ("transaction id", "txn id", "transaction_id", "reference no"),
    ROLE_UTR: ("utr", "rrn", "utr no"),
    ROLE_ACCOUNT_FLOW: ("account flow", "account_flow"),
    ROLE_ACCOUNT_LAST4: ("account last4", "account_last4", "account no", "account"),
    ROLE_TYPE: ("type", "dr cr", "cr dr", "drcr", "txn type", "transaction type"),
    ROLE_AMOUNT: ("amount", "txn amount", "transaction amount", "value"),
    ROLE_DEBIT: ("debit", "withdrawal", "withdrawn", "paid out"),
    ROLE_CREDIT: ("credit", "deposit", "paid in"),
}

# Roles without which we cannot build a transaction at all.
REQUIRED_ROLES = (ROLE_DATE, ROLE_COUNTERPARTY)

_DATE_FORMATS = (
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y",
    "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d", "%m/%d/%Y",
)
_TIME_FORMATS = ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p")

_NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")
_WS_RE = re.compile(r"[^a-z0-9]+")

_CREDIT_TOKENS = ("credit", "cr", "received", "deposit", "refund", "in")
_DEBIT_TOKENS = ("debit", "dr", "paid", "sent", "withdrawal", "withdrawn", "out")


class UnresolvedHeaders(Exception):
    """Raised when a required column role cannot be located."""

    def __init__(self, missing: list[str], headers: list[str]):
        self.missing = missing
        self.headers = headers
        super().__init__(
            f"could not locate required column(s) {missing} in headers {headers}"
        )


@dataclass(frozen=True, slots=True)
class ImportedTransaction:
    """One statement row, normalized. The raw counterparty is never altered."""

    date: datetime
    counterparty: str            # verbatim from the statement
    amount: float                # always positive; direction carries the sign
    is_credit: bool
    direction: str = ""
    transaction_id: str = ""
    utr: str = ""
    account_last4: str = ""
    row_number: int = 0
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def signed_amount(self) -> float:
        return self.amount if self.is_credit else -self.amount


@dataclass
class RowProblem:
    row_number: int
    reason: str
    raw: list[str]


@dataclass
class ImportResult:
    rows: list[ImportedTransaction]
    mapping: dict[str, int | None]
    headers: list[str]
    total_data_rows: int = 0
    problems: list[RowProblem] = field(default_factory=list)
    duplicates: int = 0

    @property
    def skipped(self) -> int:
        return len(self.problems)

    def problem_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.problems:
            out[p.reason] = out.get(p.reason, 0) + 1
        return out


# ── header resolution ─────────────────────────────────────────────────────

def _norm_header(h: str) -> str:
    return _WS_RE.sub(" ", (h or "").strip().lower()).strip()


def resolve_headers(headers: list[str]) -> dict[str, int | None]:
    """Map each role to a column index, or None.

    A column is claimed by at most one role, and exact matches are preferred
    over substring matches so a "transaction date" column does not steal the
    "transaction id" role.
    """
    normed = [_norm_header(h) for h in headers]
    mapping: dict[str, int | None] = dict.fromkeys(_SYNONYMS)
    claimed: set[int] = set()

    for exact_pass in (True, False):
        for role, synonyms in _SYNONYMS.items():
            if mapping[role] is not None:
                continue
            for idx, h in enumerate(normed):
                if idx in claimed or not h:
                    continue
                hit = (h in synonyms) if exact_pass else any(s in h for s in synonyms)
                if hit:
                    mapping[role] = idx
                    claimed.add(idx)
                    break
    return mapping


def _find_header_row(rows: list[list[str]]) -> int:
    """Statements often start with account-summary lines before the table."""
    vocab = ("date", "amount", "counterparty", "narration", "description",
             "debit", "credit", "type", "direction", "particulars")
    for i, row in enumerate(rows[:30]):
        cells = [_norm_header(c) for c in row]
        if sum(1 for c in cells if any(v in c for v in vocab)) >= 2:
            return i
    return 0


# ── cell parsers (never raise) ────────────────────────────────────────────

def _cell(row: list[str], idx: int | None) -> str:
    if idx is None or idx < 0 or idx >= len(row):
        return ""
    return (row[idx] or "").strip()


def parse_date(s: str) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(s)
    except ValueError:
        return None
    return parsed


def parse_time(s: str) -> tuple[int, int] | None:
    if not s:
        return None
    for fmt in _TIME_FORMATS:
        try:
            t = datetime.strptime(s.strip(), fmt)
            return t.hour, t.minute
        except ValueError:
            continue
    return None


def parse_amount(s: str) -> float | None:
    """Tolerates currency symbols, thousands separators and trailing Dr/Cr."""
    if not s:
        return None
    m = _NUMBER_RE.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _resolve_sign(row: list[str], mapping: dict[str, int | None]) -> tuple[float, bool] | None:
    """Return (magnitude, is_credit) using whichever shape the file uses."""
    # Shape 1: separate debit / credit columns.
    if mapping.get(ROLE_DEBIT) is not None or mapping.get(ROLE_CREDIT) is not None:
        debit = parse_amount(_cell(row, mapping.get(ROLE_DEBIT))) or 0.0
        credit = parse_amount(_cell(row, mapping.get(ROLE_CREDIT))) or 0.0
        if credit and not debit:
            return abs(credit), True
        if debit:
            return abs(debit), False
        return None

    amount = parse_amount(_cell(row, mapping.get(ROLE_AMOUNT)))
    if amount is None or amount == 0:
        return None

    # Shape 2: a DEBIT/CREDIT marker column, amount kept positive.
    for role in (ROLE_TYPE, ROLE_ACCOUNT_FLOW, ROLE_DIRECTION):
        marker = _cell(row, mapping.get(role)).lower()
        if not marker:
            continue
        if any(marker.startswith(t) or t == marker for t in _CREDIT_TOKENS):
            return abs(amount), True
        if any(marker.startswith(t) or t == marker for t in _DEBIT_TOKENS):
            return abs(amount), False

    # Shape 3: the sign lives in the number itself.
    return abs(amount), amount > 0


# ── entry point ───────────────────────────────────────────────────────────

def parse(data: bytes | str, *, dedupe: bool = True) -> ImportResult:
    """Parse a statement. Raises UnresolvedHeaders only if date/counterparty
    cannot be located; every other failure is per-row and non-fatal."""
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    raw_rows = list(csv.reader(io.StringIO(text), dialect))
    if not raw_rows:
        return ImportResult(rows=[], mapping={}, headers=[])

    header_idx = _find_header_row(raw_rows)
    headers = [(c or "").strip() for c in raw_rows[header_idx]]
    mapping = resolve_headers(headers)

    missing = [r for r in REQUIRED_ROLES if mapping.get(r) is None]
    if missing:
        raise UnresolvedHeaders(missing, headers)

    out: list[ImportedTransaction] = []
    problems: list[RowProblem] = []
    seen: set[tuple] = set()
    total = 0
    duplicates = 0

    for offset, row in enumerate(raw_rows[header_idx + 1:], start=header_idx + 2):
        if not row or not any((c or "").strip() for c in row):
            continue
        total += 1

        counterparty = _cell(row, mapping[ROLE_COUNTERPARTY])
        if not counterparty:
            problems.append(RowProblem(offset, "blank counterparty", row))
            continue

        when = parse_date(_cell(row, mapping[ROLE_DATE]))
        if when is None:
            problems.append(RowProblem(offset, "unparseable date", row))
            continue

        hm = parse_time(_cell(row, mapping.get(ROLE_TIME)))
        if hm:
            when = when.replace(hour=hm[0], minute=hm[1])

        sign = _resolve_sign(row, mapping)
        if sign is None:
            problems.append(RowProblem(offset, "missing or zero amount", row))
            continue
        amount, is_credit = sign

        txn = ImportedTransaction(
            date=when,
            counterparty=counterparty,
            amount=amount,
            is_credit=is_credit,
            direction=_cell(row, mapping.get(ROLE_DIRECTION)),
            transaction_id=_cell(row, mapping.get(ROLE_TXN_ID)),
            utr=_cell(row, mapping.get(ROLE_UTR)),
            account_last4=_cell(row, mapping.get(ROLE_ACCOUNT_LAST4)),
            row_number=offset,
            raw={h: _cell(row, i) for i, h in enumerate(headers)},
        )

        if dedupe:
            # A transaction id is unique when present. Without one, fall back to
            # the tuple that identifies a payment in a statement.
            key = (
                ("id", txn.transaction_id)
                if txn.transaction_id
                else ("row", when, counterparty.lower(), round(amount, 2), is_credit)
            )
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)

        out.append(txn)

    return ImportResult(
        rows=out,
        mapping=mapping,
        headers=headers,
        total_data_rows=total,
        problems=problems,
        duplicates=duplicates,
    )


def parse_file(path) -> ImportResult:
    from pathlib import Path
    return parse(Path(path).read_bytes())


__all__ = [
    "parse", "parse_file", "resolve_headers", "ImportedTransaction",
    "ImportResult", "RowProblem", "UnresolvedHeaders",
    "parse_date", "parse_time", "parse_amount",
]
