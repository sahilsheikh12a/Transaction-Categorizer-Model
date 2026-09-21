"""Counterparty shape heuristics — is this string a person or a business?

Pure functions over a merchant string. No database, no I/O, no configuration.
`txn_type.detect` composes them into a transaction-type decision.

The person heuristic is deliberately conservative: single-word strings and
anything carrying a commercial keyword abstain rather than guess, because a
false "this is a person" silently removes a real expense from every spending
category. The orchestrator gives merchant evidence (directory, rules) a chance
first, so abstaining here is cheap and guessing here is not.
"""
from __future__ import annotations

import re

# Strings like "******1234", "XXXXXX3159", or bare UTRs (long alnum, no spaces).
_MASKED_RE = re.compile(r"^[\*xX]{2,}\d+$")
_BARE_UTR_RE = re.compile(r"^[A-Z0-9]{12,}$")  # all caps, no spaces, long

_GLUED_DIGITS_RE = re.compile(r"(?<=[A-Za-z])\d+$")

# Honorifics that prefix Indian names — stripped before counting tokens.
_HONORIFICS = re.compile(r"^(mr|mrs|miss|ms|dr|shri|smt|sri|md)\.?\s+", re.IGNORECASE)

# Upper bound on tokens in a person name. Widened 4 -> 5 after finding real
# 5-token names in the PhonePe statement ("ABDUL FAIZAL ABDUL SALIM SHEIKH",
# "ARVINDER KOUR SARANJEET SINGH NAGI"). Businesses of that length are caught
# by the commercial-keyword filter instead, so this does not cost precision.
_MAX_NAME_TOKENS = 5

# Anything with one of these substrings in any token is clearly a merchant,
# not a person. Curated from the actual dev DB. Expand as new patterns appear.
_COMMERCIAL_TOKENS = frozenset({
    # retail / general
    "kirana", "store", "stores", "shop", "shops", "mart", "supermart",
    "supermarket", "supermarkets", "megamart", "bazaar", "bazar", "bhandar",
    "emporium", "collection", "collections", "novelty", "general", "retail",
    "wholesale", "outlet", "depot", "palace", "house", "needs",
    # food & drink
    "cafe", "coffee", "restaurant", "restaurants", "resturant", "restorent",
    "hotel", "hotels", "biryani", "dhaba", "mess", "sweets", "sweet",
    "bakery", "bakers", "dairy", "milk", "tiffin", "caterers", "catering",
    "chinese", "chinescenter", "chat", "nasta", "canteen", "juice", "sharbat",
    "panipuri", "panipoori", "pav", "bhaji", "gola", "icecream",
    "parlour", "parlor", "tea", "stall", "stoll", "farsan", "namkeen",
    "pan", "paan", "eating", "foods", "food", "kitchen", "pizza", "burger",
    "fruit", "fruits", "vegetable", "vegetables", "sabzi", "chicken",
    "mutton", "fish", "egg", "eggs", "flour", "besan", "mill", "coconut",
    "dining", "shawarma", "chaha", "popcorn",
    # health
    "medical", "medicals", "medicine", "pharmacy", "pharma", "chemist",
    "chemists", "clinic", "clinics", "hospital", "hospitals", "lab", "labs",
    "diagnostics", "pathology", "path", "scan", "dental", "optical",
    "optician", "homoeopathic", "homeopathic", "ayurvedic", "nursing",
    "madicals", "medicos", "medicose",
    # entertainment / leisure
    "films", "film", "cinema", "cinemas", "theater", "theatre", "mall",
    "multiplex", "recreation", "entertainment", "gaming", "games",
    # apparel / lifestyle
    "fashion", "garment", "garments", "textiles", "textile", "saree", "sadi",
    "footwear", "shoes", "boutique", "tailor", "tailors", "laundry",
    "drycleaning", "cleaners", "salon", "spa", "studio", "shoppe", "uphar",
    # electronics / hardware / auto
    "electronics", "electronic", "electric", "electrical", "electricals", "mobile",
    "mobiles", "recharge", "telecom", "repairing", "repairs", "hardware",
    "furniture", "furnitures", "cycle", "cycles", "tyres", "tyre",
    "automobiles", "automobile", "motors", "autos", "auto", "parts",
    "petroleum", "petrol", "diesel", "fuel", "lube", "garage",
    # services / corporate
    "services", "service", "enterprise", "enterprises", "traders", "trading",
    "trade", "corp", "corporation", "pvt", "ltd", "llp", "inc", "company",
    "private", "limited", "pte",
    "agencies", "agency", "suppliers", "supplier", "distributors",
    "distribution", "industries", "industrial", "solutions", "technologies",
    "holdings", "sales", "promotions", "logistics", "carriers", "transport",
    "travels", "tours", "packers", "movers",
    # utilities / infra
    "gas", "station", "energy", "solar", "broadband", "network", "cable",
    "dish", "news", "electricity", "water", "municipal", "railway",
    "railways", "metro", "rail",
    # education / stationery
    "education", "school", "college", "academy", "institute", "training",
    "tuition", "coaching", "classes", "library", "books", "book",
    "stationery", "stationary", "printing", "printer", "xerox", "photocopy",
    # civic / misc
    "sports", "gym", "fitness", "club", "society", "association",
    "foundation", "trust", "ngo", "point", "centre", "center", "hub",
    "plaza", "complex", "tower", "market", "bank", "finance", "insurance",
})


def detect_opaque(signal: str) -> bool:
    """True if the row is a masked account / bare UTR / "no merchant info" string."""
    if not signal:
        return False
    s = signal.strip()
    if _MASKED_RE.match(s):
        return True
    return bool(_BARE_UTR_RE.match(s) and " " not in s)


def looks_like_person(signal: str) -> bool:
    """Conservative person-name heuristic.

    Fires when ALL of these hold:
      - 2 to 5 tokens after honorific stripping
      - every token is pure alpha (no digits, no punctuation embedded)
      - no token matches a commercial keyword
      - every token is at least 1 character (single-letter initials allowed)

    Single-word strings ("Sweety") and anything ambiguous fall through to the
    LLM — better to abstain than auto-tag a small business as a person.
    """
    if not signal:
        return False
    s = _HONORIFICS.sub("", signal.strip())
    # Digits glued to a name are UPI-handle residue ("SHEIKH2"), not part of it.
    tokens = [_GLUED_DIGITS_RE.sub("", t) for t in s.split()]
    if not (2 <= len(tokens) <= _MAX_NAME_TOKENS):
        return False
    if any(not t.isalpha() for t in tokens):
        return False
    return not any(t.lower() in _COMMERCIAL_TOKENS for t in tokens)


def has_commercial_keyword(signal: str) -> bool:
    """Used by the rules layer too — true if any token signals a business."""
    if not signal:
        return False
    return any(t.lower() in _COMMERCIAL_TOKENS for t in signal.split())


__all__ = [
    "detect_opaque",
    "looks_like_person",
    "has_commercial_keyword",
]
