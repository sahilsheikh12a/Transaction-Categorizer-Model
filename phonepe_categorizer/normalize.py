"""Merchant-string normalization.

Two functions with deliberately different jobs:

`canon()`  — the STORAGE key. Lowercase + strip punctuation, nothing else.
             Overrides, the cache and the merchant directory are all keyed by
             it, so its output must never change: a drift here silently misses
             every cached row and orphans every user override.

`normalize_merchant()` — the MATCHING key. Aggressively strips transaction
             noise (prefixes, reference IDs, branch codes, corporate suffixes)
             so that rules and fuzzy matching see the merchant and not the
             plumbing around it.

They are separate on purpose. `canon()` is conservative because it is
persisted; `normalize_merchant()` can evolve because it is recomputed.

Guiding rule for `normalize_merchant`: **never destroy brand information.**
It is always better to leave noise in (the rules layer will still find the
brand substring) than to strip a token that WAS the merchant name. Every
removal below is anchored to a shape that cannot be a brand on its own.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def canon(text: str | None) -> str:
    """Lowercase, drop punctuation, collapse whitespace to single spaces.

    Example: "  YUNUSKHA  PATHAN-KIRANA!! " -> "yunuskha pathan kirana"

    STABLE CONTRACT — see module docstring. Do not "improve" this.
    """
    if not text:
        return ""
    return " ".join(_TOKEN_RE.findall(text.lower()))


# --- normalize_merchant ---------------------------------------------------

# Statement verbs PhonePe (and our PDF/SMS adapters) put in front of the name.
# Anchored to the start so a merchant genuinely called "Paid Man" survives.
_PREFIX_RE = re.compile(
    r"^(?:"
    r"paid\s+to|received\s+from|refund\s+from|sent\s+to|bill\s+paid|paid|"
    r"payment\s+to|transfer\s+to|upi\s*/?|vpa|to\s+)"
    r"\s*",
    re.IGNORECASE,
)

# "M S SHARMA STORES" / "M/S" / "Ms." — the Indian "Messrs" business prefix.
_MS_PREFIX_RE = re.compile(r"^m\s*[/.]?\s*s\s+", re.IGNORECASE)

# Corporate form suffixes. Removed only as whole tokens.
_CORPORATE_TOKENS = frozenset({
    "pvt", "pvt.", "private", "ltd", "ltd.", "limited", "limtied",  # sic: real typo in data
    "llp", "inc", "incorporated", "corp", "corporation", "company",
})

# A token is a reference/transaction ID if it mixes letters and digits and is
# long, or is a long run of digits. Short codes like "5" (BIG 5) or "8"
# (Hotel Cafe 8) are handled by the trailing-token rule instead, so that
# embedded brand numbers survive.
_REF_TOKEN_RE = re.compile(r"^(?=.*\d)[a-z0-9]{8,}$")
_LONG_DIGITS_RE = re.compile(r"^\d{5,}$")

# Masked account fragments: "******6258", "xxxxxx3159". Matched against a
# whole token, and also stripped from the RAW string before punctuation is
# flattened — once "*" becomes a space, "******6258" would otherwise survive
# as the bare digits "6258" and pollute the merchant key.
_MASKED_RE = re.compile(r"^[*x]{2,}\d*$", re.IGNORECASE)
_MASKED_RAW_RE = re.compile(r"[*x\u2022]{2,}\s*\d*", re.IGNORECASE)

# Branch/outlet codes that trail a merchant name: "U178", "MS2", "NTP".
# Only stripped in TRAILING position (see below) so "U178 Traders" is safe.
_OUTLET_CODE_RE = re.compile(r"^(?=.*\d)[a-z]{0,3}\d{1,5}$")

_SEPARATORS_RE = re.compile(r"[_\-/\\|.,;:()\[\]{}'\"!?*#@+~`^<>=&]+")
_WS_RE = re.compile(r"\s+")


def _dedupe_repeated_phrase(tokens: list[str]) -> list[str]:
    """Collapse "sajid sheikh sajid sheikh" -> "sajid sheikh".

    Adapters that set both `merchant` and `text` to the same string produce an
    exactly-doubled signal, which breaks token-count heuristics (a 3-token
    person name becomes 6 tokens and stops looking like a name). Fixing it at
    the source is preferred; this is the belt-and-braces half.
    """
    n = len(tokens)
    if n >= 2 and n % 2 == 0 and tokens[: n // 2] == tokens[n // 2:]:
        return tokens[: n // 2]
    return tokens


def normalize_merchant(text: str | None) -> str:
    """Strip transaction noise, keep the merchant.

    >>> normalize_merchant("Paid to NEW MATRUCHAYA DAILY AND GENERAL STORE")
    'new matruchaya daily and general store'
    >>> normalize_merchant("MAH NAG Mate Square KFC 6453")
    'mah nag mate square kfc'
    >>> normalize_merchant("Shabana Bakery and Foods Pvt Ltd-Shabana Bakery 2")
    'shabana bakery and foods shabana bakery'
    """
    if not text:
        return ""

    s = text.strip()
    s = _PREFIX_RE.sub("", s, count=1)
    s = _MS_PREFIX_RE.sub("", s, count=1)
    s = _MASKED_RAW_RE.sub(" ", s)
    s = _SEPARATORS_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    if not s:
        return ""

    tokens = [t for t in s.split(" ") if t]

    # Drop reference IDs / masked fragments anywhere in the string — these
    # shapes are never brand names.
    tokens = [
        t for t in tokens
        if not (_REF_TOKEN_RE.match(t) or _LONG_DIGITS_RE.match(t) or _MASKED_RE.match(t))
    ]

    # Drop corporate-form tokens ("pvt", "ltd", "private limited").
    tokens = [t for t in tokens if t not in _CORPORATE_TOKENS]

    # Trailing outlet/branch codes and bare numbers: "Kirana Store 2",
    # "Smart Point NAGPUR U178". Only from the end, and never if it would
    # empty the string.
    while len(tokens) > 1 and (
        _OUTLET_CODE_RE.match(tokens[-1]) or tokens[-1].isdigit()
    ):
        tokens.pop()

    tokens = _dedupe_repeated_phrase(tokens)

    # Re-run canon so the output charset matches the storage key exactly.
    return canon(" ".join(tokens))


def normalize_person(text: str | None) -> str:
    """Normalized key for a person name — same as merchant minus outlet codes.

    Kept separate so contact matching does not accidentally inherit
    merchant-specific stripping rules later.
    """
    if not text:
        return ""
    s = _PREFIX_RE.sub("", text.strip(), count=1)
    s = _MASKED_RAW_RE.sub(" ", s)
    s = _SEPARATORS_RE.sub(" ", s)
    tokens = _dedupe_repeated_phrase(canon(s).split())
    return " ".join(tokens)


__all__ = ["canon", "normalize_merchant", "normalize_person"]
