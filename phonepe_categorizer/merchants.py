"""Stage 3 — exact merchant directory.

A normalized-name -> category map for merchants whose category is a *fact*
rather than a keyword inference. Two populations live here:

  BUILT_IN   national brands and chains. These generalize to every user, so
             they ship with the app.
  learned    merchants the user (or a previous import) already resolved,
             loaded from the DB at runtime and merged on top.

Why this exists separately from `rules.py`: a rule says "anything containing
'bakery' is dining". A directory entry says "'ebenezer bakery' IS dining".
Exact identity beats keyword inference, so this layer runs first, and it is
also the corpus the fuzzy matcher compares against.

Keys are `normalize_merchant()` output. Never write a raw statement string
here — use the normalized form so lookups actually hit.
"""
from __future__ import annotations

from . import categories as C
from .normalize import normalize_merchant

EXACT_CONFIDENCE = 0.97


# Normalized name -> category. Keep alphabetical within each block.
BUILT_IN: dict[str, str] = {
    # ── food delivery & QSR ──────────────────────────────────────────
    "swiggy": C.FOOD_AND_DINING,
    "zomato": C.FOOD_AND_DINING,
    "dominos": C.FOOD_AND_DINING,
    "dominos pizza": C.FOOD_AND_DINING,
    "kfc": C.FOOD_AND_DINING,
    "mcdonalds": C.FOOD_AND_DINING,
    "burger king": C.FOOD_AND_DINING,
    "subway": C.FOOD_AND_DINING,
    "starbucks": C.FOOD_AND_DINING,
    "cafe coffee day": C.FOOD_AND_DINING,
    "haldiram": C.FOOD_AND_DINING,
    "haldirams": C.FOOD_AND_DINING,
    "bikanerwala": C.FOOD_AND_DINING,
    "hardcastle restaurants": C.FOOD_AND_DINING,      # McDonald's West/South India
    "restaurant brands asia": C.FOOD_AND_DINING,      # Burger King India
    "nepenthe coffee and chocolates": C.FOOD_AND_DINING,

    # ── grocery & quick commerce ─────────────────────────────────────
    "bigbasket": C.GROCERIES,
    "blinkit": C.GROCERIES,
    "zepto": C.GROCERIES,
    "jiomart": C.GROCERIES,
    "dmart": C.GROCERIES,
    "avenue supermarts": C.GROCERIES,                 # DMart's legal entity
    "reliance fresh": C.GROCERIES,
    "reliance retail": C.GROCERIES,
    "vishal mega mart": C.GROCERIES,
    "amazon pay groceries": C.GROCERIES,

    # ── marketplaces & retail ────────────────────────────────────────
    "amazon": C.SHOPPING,
    "amazon india": C.SHOPPING,
    "amazon pay": C.SHOPPING,
    "amazon seller services": C.SHOPPING,
    "flipkart": C.SHOPPING,
    "myntra": C.SHOPPING,
    "ajio": C.SHOPPING,
    "meesho": C.SHOPPING,
    "nykaa": C.SHOPPING,
    "croma": C.SHOPPING,
    "reliance digital": C.SHOPPING,
    "decathlon": C.SHOPPING,
    "ikea": C.SHOPPING,
    "the souled store": C.SHOPPING,
    "bad brains streetstyle": C.SHOPPING,
    "premier sales promotions": C.SHOPPING,           # gift-voucher issuer
    "vr vidarbha": C.SHOPPING,                        # VR Mall, Nagpur
    "trends": C.SHOPPING,                             # Reliance Trends
    "bonkerscorner": C.SHOPPING,                      # apparel brand
    "bonkers corner": C.SHOPPING,
    "thedermaco": C.SHOPPING,                         # skincare brand
    "the derma co": C.SHOPPING,

    # ── telecom & internet ───────────────────────────────────────────
    "jio": C.MOBILE_AND_INTERNET,
    "jio prepaid recharges": C.MOBILE_AND_INTERNET,
    "reliance jio infocomm": C.MOBILE_AND_INTERNET,
    "airtel": C.MOBILE_AND_INTERNET,
    "bharti airtel": C.MOBILE_AND_INTERNET,
    "vodafoneidea": C.MOBILE_AND_INTERNET,
    "vodafone idea": C.MOBILE_AND_INTERNET,
    "bsnl": C.MOBILE_AND_INTERNET,
    "mobile recharge": C.MOBILE_AND_INTERNET,
    "net distribution services": C.MOBILE_AND_INTERNET,

    # ── utilities ────────────────────────────────────────────────────
    "electricity": C.BILLS_AND_UTILITIES,
    "maharashtra state electricity distribution company": C.BILLS_AND_UTILITIES,
    "msedcl": C.BILLS_AND_UTILITIES,
    "tata power": C.BILLS_AND_UTILITIES,
    "adani electricity": C.BILLS_AND_UTILITIES,

    # ── transport & fuel ─────────────────────────────────────────────
    "uber": C.TRANSPORT,
    "ola": C.TRANSPORT,
    "rapido": C.TRANSPORT,
    "irctc": C.TRANSPORT,
    "indian railways": C.TRANSPORT,
    "maharashtra metro rail corporation": C.TRANSPORT,
    "indian oil": C.TRANSPORT,
    "bharat petroleum": C.TRANSPORT,
    "hindustan petroleum": C.TRANSPORT,
    "reliance bp mobility": C.TRANSPORT,
    "rbml solutions india": C.TRANSPORT,
    "redbus": C.TRANSPORT,

    # ── entertainment ────────────────────────────────────────────────
    "netflix": C.ENTERTAINMENT,
    "jiohotstar": C.ENTERTAINMENT,
    "disney hotstar": C.ENTERTAINMENT,
    "spotify": C.ENTERTAINMENT,
    "bookmyshow": C.ENTERTAINMENT,
    "bigtree entertainment": C.ENTERTAINMENT,
    "pvr inox": C.ENTERTAINMENT,
    "gge sports and entertainment": C.ENTERTAINMENT,
    # Google bills Play Store content, app purchases and media subscriptions
    # through these two names. A judgment call — see the evaluation report's
    # ambiguous list; the user can override per merchant.
    "google": C.ENTERTAINMENT,
    "google india digital services": C.ENTERTAINMENT,

    # ── health ───────────────────────────────────────────────────────
    "apollo pharmacy": C.HEALTH,
    "pharmeasy": C.HEALTH,
    "netmeds": C.HEALTH,
    "1mg": C.HEALTH,
    "hcg nagpur": C.HEALTH,                           # HCG cancer centre
    "children first mental health institute": C.HEALTH,

    # ── travel ───────────────────────────────────────────────────────
    "makemytrip": C.TRAVEL,
    "goibibo": C.TRAVEL,
    "cleartrip": C.TRAVEL,
    "oyo": C.TRAVEL,
    "indigo": C.TRAVEL,
    "air india": C.TRAVEL,

    # ── finance ──────────────────────────────────────────────────────
    "lic of india": C.FINANCE,
    "groww": C.FINANCE,
    "zerodha": C.FINANCE,
    "bajaj finance": C.FINANCE,
}


def lookup(signal: str | None, learned: dict[str, str] | None = None) -> tuple[str, float] | None:
    """Exact directory hit for a merchant string.

    `learned` (normalized name -> category) is merged on top of BUILT_IN so a
    merchant the user has already resolved wins over the shipped default.
    """
    if not signal:
        return None
    key = normalize_merchant(signal)
    if not key:
        return None
    if learned and key in learned:
        return learned[key], EXACT_CONFIDENCE
    hit = BUILT_IN.get(key)
    if hit is not None:
        return hit, EXACT_CONFIDENCE
    return None


def corpus(learned: dict[str, str] | None = None) -> dict[str, str]:
    """Full known-merchant map used as the fuzzy matcher's comparison set."""
    out = dict(BUILT_IN)
    if learned:
        out.update(learned)
    return out


__all__ = ["BUILT_IN", "lookup", "corpus", "EXACT_CONFIDENCE"]
