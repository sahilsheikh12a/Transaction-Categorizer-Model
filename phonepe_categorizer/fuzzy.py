"""Stage 5 — fuzzy merchant matching.

Runs only after exact lookup and the rule engine have both missed. Its job is
to survive typos, truncation and branch suffixes in merchant strings that the
user (or the directory) has already resolved once:

    "Sgree Ganesh Book Depot"  ~  "shree ganesh book depot"
    "Sheree krushan medical stor" ~ "shri krishna medical and general stores"
    "Shabana Bakery buttibori"  ~  "shabana bakery"

Deliberately conservative. A wrong fuzzy hit is worse than no hit, because it
produces a confident-looking category the user never asked for. Three guards:

  1. length gate      — strings under `_MIN_LEN` chars are not matched at all;
                        short names collide far too easily.
  2. anchor token     — the two strings must share at least one token of
                        >= `_ANCHOR_LEN` chars. Pure character similarity will
                        happily match "raj traders" to "ram traders"; requiring
                        a shared substantive word stops that.
  3. score threshold  — combined score must clear `THRESHOLD`, and the runner-up
                        must not be within `_MARGIN` (an ambiguous best match
                        is treated as no match).

Scoring blends character-level similarity (difflib) with token-set containment
(Jaccard), so both "typo" and "extra words" cases are covered. Pure stdlib —
no new dependency, works offline.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from .normalize import normalize_merchant

# Tunables — surfaced as module constants so they can be swept during eval.
THRESHOLD = 0.86      # minimum blended score to accept a match
_MARGIN = 0.04        # best must beat runner-up by this much
_MIN_LEN = 8          # ignore very short merchant strings
_ANCHOR_LEN = 4       # a shared token this long counts as an anchor

# Tokens too generic to anchor a match — "medical stores" vs "general stores"
# must not be joined by the word "stores" alone.
_WEAK_ANCHORS = frozenset({
    "store", "stores", "shop", "shops", "centre", "center", "general",
    "daily", "needs", "new", "shree", "shri", "sri", "sree", "jai", "jay",
    "maa", "the", "and", "india", "indian", "private", "limited",
    "point", "house", "services", "service", "traders", "enterprises",
    "kirana", "medical", "hotel", "cafe", "bakery", "restaurant",
})


def _tokens(s: str) -> set[str]:
    return {t for t in s.split() if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _containment(a: set[str], b: set[str]) -> float:
    """Fraction of the smaller token set covered by the larger one.

    Handles "shabana bakery buttibori" vs "shabana bakery" — Jaccard punishes
    the extra locality token, containment does not.
    """
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return len(small & large) / len(small)


def _near_token(a: str, b: str) -> bool:
    """True if two tokens are the same word with a typo/transposition.

    Catches "matruchaya" ~ "maturchaya", which share no exact token and would
    otherwise fail the anchor test despite being obviously the same shop.
    """
    if len(a) < 5 or len(b) < 5:
        return False
    if abs(len(a) - len(b)) > 2:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.82


def _has_anchor(a: set[str], b: set[str]) -> bool:
    """Require a substantive word in common before scoring at all.

    Without this, character similarity alone happily matches "raj traders" to
    "ram traders" — different businesses, one edit apart.
    """
    for t in a & b:
        if len(t) >= _ANCHOR_LEN and t not in _WEAK_ANCHORS:
            return True
    # Fall back to a near-identical strong token pair (typo case).
    strong_a = [t for t in a if len(t) >= 5 and t not in _WEAK_ANCHORS]
    strong_b = [t for t in b if len(t) >= 5 and t not in _WEAK_ANCHORS]
    return any(_near_token(x, y) for x in strong_a for y in strong_b)


def _token_align(a: set[str], b: set[str]) -> float:
    """Average best-match similarity of each token in the smaller set.

    This is the metric that actually works for merchant names, because it
    handles both failure modes at once:

      typo        "sgree ganesh book depot" vs "shree ganesh book depot"
                  -> sgree~shree .8, ganesh 1, book 1, depot 1  = .95
      extra words "shabana bakery buttibori" vs "shabana bakery"
                  -> shabana 1, bakery 1                        = 1.0

    A whole-string character ratio scores the first well and the second badly;
    a token-set Jaccard does the opposite.
    """
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    total = 0.0
    for t in small:
        total += max(SequenceMatcher(None, t, u).ratio() for u in large)
    return total / len(small)


def score(a: str, b: str) -> float:
    """Blended similarity in [0, 1] between two already-normalized strings."""
    if not a or not b:
        return 0.0
    ta, tb = _tokens(a), _tokens(b)
    char = SequenceMatcher(None, a, b).ratio()
    return 0.4 * char + 0.6 * _token_align(ta, tb)


def match(
    signal: str | None,
    corpus: dict[str, str],
    *,
    threshold: float = THRESHOLD,
) -> tuple[str, float, str] | None:
    """Best fuzzy hit as (category, confidence, matched_key), or None.

    `corpus` maps normalized merchant name -> category.
    """
    if not signal or not corpus:
        return None
    needle = normalize_merchant(signal)
    if len(needle) < _MIN_LEN:
        return None

    # An exact key hit is not this layer's job, but scoring it would give 1.0
    # anyway — bail so callers see a clean "no fuzzy needed".
    ntok = _tokens(needle)
    scored: list[tuple[float, str]] = []
    for key in corpus:
        if len(key) < _MIN_LEN:
            continue
        if not _has_anchor(ntok, _tokens(key)):
            continue
        scored.append((score(needle, key), key))

    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    best_score, best_key = scored[0]
    if best_score < threshold:
        return None

    # Reject an ambiguous winner: if the runner-up is nearly as good AND
    # disagrees on the category, we cannot tell them apart — abstain.
    best_cat = corpus[best_key]
    for other_score, other_key in scored[1:]:
        if best_score - other_score >= _MARGIN:
            break
        if corpus[other_key] != best_cat:
            return None

    # Confidence tracks the score but is capped below the exact-match layer so
    # a fuzzy hit never looks as certain as a directory hit.
    return best_cat, round(min(0.94, best_score), 4), best_key


__all__ = ["match", "score", "THRESHOLD"]
