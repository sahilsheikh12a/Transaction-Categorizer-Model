# Categorizer Architecture

How a raw PhonePe counterparty string becomes a category, and why the design is
shaped the way it is. Figures come from the shipped statement
(`transactions2.csv`, 2,005 transactions, 607 unique merchants, Jul 2024 – Apr 2026).

---

## 1. The problem

An Indian UPI statement gives you one usable field per transaction: a
counterparty string typed by a merchant, a bank, or nobody at all.

```
Paid to        NEW MATRUCHAYA DAILY AND GENERAL STORE
Paid to        Ritik Pandey
Paid           Mobile Recharge
Bill paid      Electricity
Received from  Sajid Sheikh
Paid to        ******6258
```

There is no MCC code, no bank-supplied category, no merchant ID. Three
properties of this data drive every decision below:

**Most transactions are to people, not shops.** 716 of 2,005 rows here are
person-to-person. A classifier that only knows spending categories will file
them all under something wrong.

**Merchant names and person names are structurally identical.** `Swiggy Order`
and `Ritik Pandey` are both two alphabetic tokens. No shape heuristic can
separate them.

**Small vendors collect on personal accounts.** `CHETMANI DWARKAPRASAD SAHU`
takes 34 payments at a ₹38 median. It is a shop wearing a person's name, and
nothing in the string says so.

---

## 2. Pipeline

```
                    raw counterparty string
                              │
                    ┌─────────▼──────────┐
                    │  normalization     │  normalize.py
                    │  strip noise       │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
        stage 1     │  transaction type  │  txn_type.py
                    │  person? merchant? │
                    │  received? refund? │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
        stage 2     │  user overrides    │  db/  ── wins over everything
                    └─────────┬──────────┘
                              │
                  ┌───────────┴────────────┐
        incoming? │                        │ outgoing
                  ▼                        ▼
          INCOME / PERSONAL       ┌────────────────────┐
          _TRANSFER               │ exact merchant     │ stage 3  merchants.py
                                  └─────────┬──────────┘
                                            │ miss
                                  ┌─────────▼──────────┐
                                  │ keyword + brand    │ stage 4  rules.py
                                  │ rules              │
                                  └─────────┬──────────┘
                                            │ miss
                                  ┌─────────▼──────────┐
                                  │ person short-      │ stage 1 result,
                                  │ circuit            │ applied here
                                  └─────────┬──────────┘
                                            │ not a person
                                  ┌─────────▼──────────┐
                                  │ fuzzy match        │ stage 5  fuzzy.py
                                  └─────────┬──────────┘
                                            │ miss
                                  ┌─────────▼──────────┐
                                  │ OTHER              │ stage 6
                                  │ needs_review=True  │
                                  └────────────────────┘
```

Implemented as one pure function, `engine.classify_core`, ~100 lines. No I/O, no
globals, no database. Learned state arrives as two plain dicts.

---

## 3. The two ordering decisions that matter

Everything else is a table you can edit. These two are the design.

### Type detection runs before categorization

Asking "which spending category is `Paid to Ritik`?" is the wrong question — the
answer is *none*. Deciding what **kind** of transaction it is first makes the
mistake structurally impossible rather than something rules have to defend
against.

This is why `PERSONAL_TRANSFER` and `INCOME` are in the same enum as
`GROCERIES`: they are the answers you need when the question "what did I spend
this on" doesn't apply.

### But merchant evidence outranks name shape

The person heuristic cannot run as soon as it fires, because
`Swiggy Order`, `Sony Center` and `Aditya Food` all pass a person-name test.
So stages 3 and 4 run **first**, and the person short-circuit applies only to
strings no merchant layer recognized.

This is not theoretical. Before the reorder, `khemka digital home` — a ₹55,000
electronics purchase — was filed as a transfer to a friend, along with
₹78,000 of other spending.

**Combined rule:** a curated fact about a merchant beats a guess about a name;
a guess about a name beats nothing. User overrides beat both, because only a
human can tell you that `CHETMANI DWARKAPRASAD SAHU` is a grocery shop.

---

## 4. Modules

| Module | Lines | Role |
|---|---:|---|
| `categories.py` | 185 | 13 categories, definitions, source constants, legacy bridge |
| `normalize.py` | 162 | two normalizers — storage key and matching key |
| `triage.py` | 131 | person-vs-business shape heuristics, 258 commercial tokens |
| `txn_type.py` | 136 | stage 1 |
| `merchants.py` | 182 | stage 3, 101 built-in brands |
| `rules.py` | 514 | stage 4, 363 ordered rules |
| `fuzzy.py` | 184 | stage 5 |
| `engine.py` | 170 | the orchestrator |
| `schema.py` | 70 | `ClassifyResult` |
| `importer.py` | 361 | CSV → transactions, role-based, resilient |
| `db/` | 553 | optional persistence + the correction loop |

The engine depends on nothing but the pure modules. `importer`, `db`, `export`,
`report`, `web` and `cli` all depend on the engine and never the reverse.

---

## 5. Normalization: two keys, not one

The single most load-bearing subtlety in the codebase.

```python
canon("Maturchaya Kirana 2")              # "maturchaya kirana 2"   STORAGE key
normalize_merchant("Maturchaya Kirana 2") # "maturchaya kirana"     MATCHING key
```

`canon` is conservative because it is **persisted** — it keys the override and
merchant tables. If its output ever shifts, every stored override silently
orphans and every cache lookup misses. It is lowercase-and-strip-punctuation,
and nothing more, forever.

`normalize_merchant` is aggressive because it is **recomputed** every time. It
strips statement verbs (`Paid to`), reference IDs, corporate suffixes
(`Pvt Ltd`), masked fragments, and trailing outlet codes (`U178`, the `2` in
`Kirana 2`). It can evolve freely.

They diverge on purpose, and conflating them is a bug: overrides were once
written with `canon` and read with `normalize_merchant`, so a correction made in
the UI never applied to anything.

Normalization never destroys brand information. Every removal is anchored to a
shape that cannot be a brand: `BIG 5` keeps its `5` because the digit is not
trailing; `Hotel Cafe 8` loses its `8` because it is.

---

## 6. Rule ordering is load-bearing

363 rules in one ordered table, first match wins. `phrase` rules substring-match;
`token` rules match whole words, for short strings like `ola` that would
otherwise match inside `chocolate`.

Specific patterns must precede generic ones. Real cases from this statement:

| String | Wins | Beats | Why |
|---|---|---|---|
| `GAJANAN MEDICAL GENERAL STORES` | `medical general` → HEALTH | `general store` → GROCERIES | a pharmacy, not a kirana |
| `ISHANT DAIRY AND ICECREAM PARLOUR` | `icecream` → FOOD | `dairy` → GROCERIES | you eat there |
| `Shree Ganesh Book Depot … General Sto` | `book depot` → SHOPPING | `general` → GROCERIES | head noun wins |
| `Roshan Mobile and Repairing Centre` | `mobile and repairing` → SHOPPING | `recharge` → MOBILE | a repair shop |
| `A U K HOTELS PRIVATE LIMITED` | `hotels private` → TRAVEL | `hotel` → FOOD | corporate = lodging |
| `Aapna hotel` | `hotel` → FOOD | — | a standalone Indian "hotel" is an eatery |

Phrase rules match against **two** haystacks: the normalized string and a
corporate-preserving one. Without the second, `hotels private` can never fire,
because normalization has already removed `private limited`.

Some phrases are listed as **ambiguous** (`internet cafe`, `cyber cafe`) and
force a fall-through to `OTHER` rather than matching `cafe` → FOOD. Abstaining
is a valid answer.

---

## 7. Fuzzy matching is deliberately reluctant

Stage 5 runs only after exact lookup and rules have both missed. A wrong fuzzy
hit is worse than no hit, because it produces a confident-looking category
nobody asked for. Four guards:

- **length gate** — strings under 8 characters are never matched
- **anchor** — the two strings must share a substantive token, or contain a
  near-identical one (`matruchaya` ~ `maturchaya`). Without this, character
  similarity happily matches `raj traders` to `ram traders`
- **threshold** — blended score ≥ 0.86
- **ambiguity margin** — if the runner-up is within 0.04 *and* disagrees on the
  category, abstain

Scoring blends whole-string character similarity (40%) with **token alignment**
(60%) — each token's best match in the other string, averaged. Token alignment
handles both failure modes at once, which neither a character ratio nor a token
Jaccard does alone:

```
"sgree ganesh book depot"  vs "shree ganesh book depot"   typo         → .95
"shabana bakery buttibori" vs "shabana bakery"            extra words  → 1.0
```

It fires 3 times in 2,005 transactions. That is the correct volume for a layer
whose job is catching typos, not doing the work.

---

## 8. Output contract

```python
ClassifyResult(
    category='GROCERIES',              # one of 13
    confidence=0.90,
    source='RULE',                     # which layer decided
    needs_review=False,
    txn_type='PAYMENT_TO_MERCHANT',
    normalized_merchant='new matruchaya daily and general store',
    evidence="rule: 'daily and general'",
)
```

`evidence` is why a wrong answer is debuggable without re-running anything.

`needs_review` separates *"this is miscellaneous"* from *"I could not tell"*.
Only the second should ever reach a user as a question, and `OTHER` always
carries it. This is what "never hallucinate a category" means operationally.

Confidences are **ordered preferences, not calibrated probabilities**:

| Source | Conf | Meaning |
|---|---:|---|
| `USER_OVERRIDE` | 1.00 | a human said so |
| `EXACT_MERCHANT` | 0.97 | this merchant *is* this |
| `RULE` | 0.90 | a curated keyword |
| `FUZZY_MATCH` | ≤0.94 | looks like something known |
| `TXN_TYPE` | 0.60–0.99 | decided by flow, not merchant |
| `FALLBACK` | 0.00–0.70 | a guess or a refusal |

Anything below 0.75 sets `needs_review`.

---

## 9. Learning

The engine is stateless; learning lives entirely in `db/`.

| Table | Written by | Read back at |
|---|---|---|
| `merchant_overrides` | user corrections | stage 2 |
| `merchants` | confident imports | stage 3 + fuzzy corpus |
| `classification_rules` | user-defined rules | between 2 and 4 |
| `transactions` | import | audit, review queue, reclassify |

A correction back-applies to history and forward to future imports, keyed on the
normalized merchant so one answer covers every spelling variant.

**The directory records merchant identity, never transaction direction.** Only
outgoing payments resolved by an identity layer (`EXACT_MERCHANT`, `RULE`,
`FUZZY_MATCH`) are learned. Learning from `TXN_TYPE` once taught the system that
a garage was `INCOME`, because a refund came from it — and since the directory
doubles as the fuzzy corpus, that error spread. The guard cut learned merchants
from 497 to 178, all genuine, and made `reclassify` a no-op immediately after
import instead of changing 260 categories.

`raw_counterparty` is **never** overwritten. Everything else is derived, so
`reclassify` can always rebuild from the original text.

---

## 10. Results and honest limits

| | |
|---|---:|
| Transactions | 2,005 |
| Auto-classified | **1,800 — 89.8%** |
| Flagged for review | 205 — 10.2% |
| `OTHER` | 91 — 4.5% |
| Speed | ~120 µs/txn |

By deciding layer: `TXN_TYPE` 971 · `RULE` 634 · `FALLBACK` 201 ·
`EXACT_MERCHANT` 196 · `FUZZY_MATCH` 3.

**89.8% is coverage, not accuracy.** It means 1,800 transactions got a confident
answer — not that 1,800 are correct. There is no labelled ground truth for this
statement, so precision is unmeasured. The report's suspicious-classification
sections exist precisely because of that gap.

Known limits:

- **No `EDUCATION` category.** ₹144,350 of college fees sits in `OTHER`. The
  largest single gap; a `categories.py` edit plus two rules.
- **Person-named vendors.** Unlearnable from the string. The report flags them
  by behaviour (many payments, small median); one correction each fixes it.
- **Incoming money from a person is `PERSONAL_TRANSFER`, not `INCOME`** — by
  design. Counting a friend's repayment as income corrupts every budget.
- **Masked counterparties** (`******6258`) carry no merchant information;
  treated as personal transfers at 0.70, always flagged.

---

## 11. Why no ML

~90% of these transactions are decided by an exact merchant name or a single
keyword. That is a lookup table's job, and the table already reaches 89.8%
while remaining reproducible and explainable — two properties a model would
cost and neither of which the residual 10% needs.

Training on this system's own output would be circular: the labels are the rule
engine's answers, so a model would learn to imitate `rules.py` including its
mistakes, while trading away determinism and `evidence`.

The labels that would be worth something are the **human** ones in
`merchant_overrides`, and the design already accumulates them: the review queue
is an unlabelled pool ranked by spend, and `previous_category` records what the
classifier got wrong so frequently-corrected *rules* can be fixed directly.

If a model is ever added, it belongs **after** the deterministic layers as a
suggestion for `needs_review` rows — never replacing them.
