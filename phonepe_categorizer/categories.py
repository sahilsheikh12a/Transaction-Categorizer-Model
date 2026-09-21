"""Canonical category taxonomy — the single place category names are defined.

Everything downstream (rules, merchant directory, fuzzy matcher, evaluation
report, UI) imports from here. Changing the taxonomy means editing THIS file
and the rule table in `rules.py` — nothing else should hardcode a category
string.

Design constraints (from the V1 spec):
  * Small, day-to-day useful set. Not dozens of buckets.
  * Every category must be assignable from a merchant string alone; we do not
    have MCC codes or bank-supplied category hints in PhonePe statements.
  * `OTHER` is a real answer, not a dumping ground — it always pairs with
    `needs_review=True` so the user can teach the system instead of the system
    guessing.

LEGACY BRIDGE
-------------
The existing FinPilot DB/API/UI speak an older, coarser vocabulary
("food", "bills", "contact", ...). The engine emits canonical categories;
`to_legacy()` maps them at the API boundary so stored rows and the frontend
contract stay unchanged. Delete the bridge once the UI is migrated.
"""
from __future__ import annotations

from typing import Final

# --- canonical categories -------------------------------------------------

GROCERIES: Final = "GROCERIES"
FOOD_AND_DINING: Final = "FOOD_AND_DINING"
SHOPPING: Final = "SHOPPING"
BILLS_AND_UTILITIES: Final = "BILLS_AND_UTILITIES"
MOBILE_AND_INTERNET: Final = "MOBILE_AND_INTERNET"
TRANSPORT: Final = "TRANSPORT"
HEALTH: Final = "HEALTH"
ENTERTAINMENT: Final = "ENTERTAINMENT"
TRAVEL: Final = "TRAVEL"
FINANCE: Final = "FINANCE"
PERSONAL_TRANSFER: Final = "PERSONAL_TRANSFER"
INCOME: Final = "INCOME"
OTHER: Final = "OTHER"

ALL: Final[tuple[str, ...]] = (
    GROCERIES,
    FOOD_AND_DINING,
    SHOPPING,
    BILLS_AND_UTILITIES,
    MOBILE_AND_INTERNET,
    TRANSPORT,
    HEALTH,
    ENTERTAINMENT,
    TRAVEL,
    FINANCE,
    PERSONAL_TRANSFER,
    INCOME,
    OTHER,
)

VALID: Final[frozenset[str]] = frozenset(ALL)

# Human-readable scope, kept next to the constant so rule authors and the
# review UI read the same definition.
DEFINITIONS: Final[dict[str, str]] = {
    GROCERIES: "Supermarkets, kirana stores, daily/general stores, grocery delivery, "
               "fruit/vegetable/dairy/meat vendors, flour mills.",
    FOOD_AND_DINING: "Restaurants, cafes, bakeries, sweet shops, street food, tea stalls, "
                     "canteens, food delivery.",
    SHOPPING: "Amazon/Flipkart, clothing, footwear, electronics, stationery, books, "
              "furniture, general non-grocery retail.",
    BILLS_AND_UTILITIES: "Electricity, water, piped gas, LPG, DTH, municipal and other "
                         "recurring utility bills.",
    MOBILE_AND_INTERNET: "Mobile recharge, prepaid/postpaid telecom, broadband and "
                         "internet service providers.",
    TRANSPORT: "Fuel and petrol pumps, ride-hailing, metro, rail, bus, parking, tolls, "
               "vehicle service and spares.",
    HEALTH: "Pharmacies, medical stores, clinics, hospitals, diagnostic labs, opticians.",
    ENTERTAINMENT: "Streaming subscriptions, cinemas, ticketing, games, film/DVD rental, "
                   "sports and recreation venues.",
    TRAVEL: "Airlines, hotels and lodging, tour operators, travel bookings.",
    FINANCE: "Bank and payment charges, insurance premiums, loan repayments, investments.",
    PERSONAL_TRANSFER: "Money moved to or from an individual — friends, family, or a "
                       "person-to-person UPI transfer.",
    INCOME: "Money received from a business, employer, or institution, plus refunds "
            "and cashback.",
    OTHER: "Not confidently classifiable. Always paired with needs_review=True.",
}

# --- classification sources ----------------------------------------------
# Which layer produced a result. Ordered highest-authority first.

SRC_USER_OVERRIDE: Final = "USER_OVERRIDE"
SRC_EXACT_MERCHANT: Final = "EXACT_MERCHANT"
SRC_RULE: Final = "RULE"
SRC_FUZZY_MATCH: Final = "FUZZY_MATCH"
SRC_TXN_TYPE: Final = "TXN_TYPE"
SRC_ML_MODEL: Final = "ML_MODEL"
SRC_FALLBACK: Final = "FALLBACK"

SOURCES: Final[tuple[str, ...]] = (
    SRC_USER_OVERRIDE,
    SRC_EXACT_MERCHANT,
    SRC_RULE,
    SRC_FUZZY_MATCH,
    SRC_TXN_TYPE,
    SRC_ML_MODEL,
    SRC_FALLBACK,
)

# --- legacy bridge --------------------------------------------------------
# Canonical -> the vocabulary the existing DB rows / API / UI already use.
# Lossy by design: GROCERIES and FOOD_AND_DINING both collapse to "food"
# because the old taxonomy had no grocery bucket.

_TO_LEGACY: Final[dict[str, str]] = {
    GROCERIES: "food",
    FOOD_AND_DINING: "food",
    SHOPPING: "shopping",
    BILLS_AND_UTILITIES: "bills",
    MOBILE_AND_INTERNET: "bills",
    TRANSPORT: "transport",
    HEALTH: "health",
    ENTERTAINMENT: "ent",
    TRAVEL: "transport",
    FINANCE: "bills",
    PERSONAL_TRANSFER: "contact",
    INCOME: "income",
    OTHER: "Other",
}


def to_legacy(category: str) -> str:
    """Map a canonical category to the legacy UI/DB vocabulary."""
    return _TO_LEGACY.get(category, "Other")


def is_valid(category: str | None) -> bool:
    return bool(category) and category in VALID


def coerce(category: str | None) -> str:
    """Accept a canonical name (any case) or a legacy name; return canonical.

    Used when reading user overrides written before the taxonomy change, and
    when accepting a category from the recategorize API.
    """
    if not category:
        return OTHER
    up = category.strip().upper().replace(" ", "_").replace("-", "_")
    if up in VALID:
        return up
    return _FROM_LEGACY.get(category.strip().lower(), OTHER)


# Legacy -> canonical. Ambiguous legacy buckets resolve to the most common
# canonical meaning; the user can always override per merchant.
_FROM_LEGACY: Final[dict[str, str]] = {
    "food": FOOD_AND_DINING,
    "groceries": GROCERIES,
    "grocery": GROCERIES,
    "transport": TRANSPORT,
    "travel": TRAVEL,
    "shopping": SHOPPING,
    "bills": BILLS_AND_UTILITIES,
    "mobile": MOBILE_AND_INTERNET,
    "ent": ENTERTAINMENT,
    "entertainment": ENTERTAINMENT,
    "health": HEALTH,
    "finance": FINANCE,
    "contact": PERSONAL_TRANSFER,
    "transfer": PERSONAL_TRANSFER,
    "income": INCOME,
    "other": OTHER,
    # `edu` has no canonical home in V1 — see the evaluation report's
    # recommendation to add an EDUCATION category.
    "edu": OTHER,
}


__all__ = [
    "GROCERIES", "FOOD_AND_DINING", "SHOPPING", "BILLS_AND_UTILITIES",
    "MOBILE_AND_INTERNET", "TRANSPORT", "HEALTH", "ENTERTAINMENT", "TRAVEL",
    "FINANCE", "PERSONAL_TRANSFER", "INCOME", "OTHER",
    "ALL", "VALID", "DEFINITIONS",
    "SRC_USER_OVERRIDE", "SRC_EXACT_MERCHANT", "SRC_RULE", "SRC_FUZZY_MATCH",
    "SRC_TXN_TYPE", "SRC_ML_MODEL", "SRC_FALLBACK", "SOURCES",
    "to_legacy", "coerce", "is_valid",
]
