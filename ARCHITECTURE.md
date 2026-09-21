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
                                  │ ONNX model         │ stage 6  ml.py
                                  │ taken if p ≥ 0.75  │ (not for masked
                                  └─────────┬──────────┘  or ambiguous)
                                            │ unsure: kept as a hint
                                  ┌─────────▼──────────┐
                                  │ masked / mobile no │ stage 7
                                  │ short name         │ flagged transfers
                                  └─────────┬──────────┘
                                            │ none of those
                                  ┌─────────▼──────────┐
                                  │ OTHER              │ + MiniLM suggestion
                                  │ needs_review=True  │   in the evidence
                                  └────────────────────┘
```

Implemented as one function, `engine.classify_core`, ~120 lines. No I/O and no
database; learned state arrives as two plain dicts. The only process state is
the two models, each loaded once on first use and replaceable with
`set_default_model` (tests switch both off by default).

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
| `categories.py` | 193 | 14 categories, definitions, source constants, legacy bridge |
| `normalize.py` | 162 | two normalizers — storage key and matching key |
| `triage.py` | 131 | person-vs-business shape heuristics, 258 commercial tokens |
| `txn_type.py` | 136 | stage 1 |
| `merchants.py` | 182 | stage 3, 101 built-in brands |
| `rules.py` | 664 | stage 4, 479 ordered rules |
| `fuzzy.py` | 184 | stage 5 |
| `ml.py` | 177 | stage 6 — featurizer + ONNX runtime, optional |
| `semantic.py` | 220 | MiniLM embeddings and neighbour vote (hint only), fetch/index |
| `train.py` | 332 | builds the dataset, trains, exports the ONNX model, rebuilds the MiniLM index |
| `engine.py` | 226 | the orchestrator |
| `schema.py` | 70 | `ClassifyResult` |
| `importer.py` | 361 | CSV → transactions, role-based, resilient |
| `db/` | 553 | optional persistence + the correction loop |

The engine depends on nothing but the pure modules and `ml.py`, which
degrades to "no answer" when onnxruntime or the model file is missing.
`importer`, `db`, `export`, `report`, `web`, `cli` and `train` all depend on
the engine and never the reverse.

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

479 rules in one ordered table, first match wins. `phrase` rules substring-match;
`token` rules match whole words, for short strings like `ola` that would
otherwise match inside `chocolate`.

Specific patterns must precede generic ones. Real cases from this statement:

| String | Wins | Beats | Why |
|---|---|---|---|
| `GAJANAN MEDICAL GENERAL STORES` | `medical general` → HEALTH | `general store` → GROCERIES | a pharmacy, not a kirana |
| `ISHANT DAIRY AND ICECREAM PARLOUR` | `icecream` → FOOD | `dairy` → GROCERIES | you eat there |
| `Shree Ganesh Book Depot … General Sto` | `book depot` → SHOPPING | `general` → GROCERIES | head noun wins |
| `Roshan Mobile and Repairing Centre` | `mobile and repairing` → SHOPPING | `recharge` → MOBILE | a repair shop |
| `Aapna hotel`, `A U K HOTELS PRIVATE LIMITED` | `hotel` → FOOD | — | in India every "hotel" is an eatery; stays go through travel brands |

Phrase rules match against **two** haystacks: the normalized string and a
corporate-preserving one. Without the second, `gas limited` can never fire,
because normalization has already removed `private limited`.

Some phrases are listed as **ambiguous** (`internet cafe`, `cyber cafe`) and
force a fall-through to `OTHER` rather than matching `cafe` → FOOD. Abstaining
is a valid answer.

EDUCATION shows the same principle from the other side. "medical college" has
to sit at the **top** of the table, or `medical` → HEALTH claims it. Every
other education word (`college`, `school`, `academy`, `institute`,
`classes`…) sits at the very **end**, because merchant names carry addresses:
`Sharma Kirana College Road` is a kirana, and the education rule only fires
when no other keyword in the string says what the business is.

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
    category='GROCERIES',              # one of 14
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
| `ML_MODEL` | 0.75–0.89 | the model's probability, capped below `RULE` |
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
import instead of changing 260 categories. `ML_MODEL` is outside the guard for
the same reason: a model's guess is not a fact about a merchant.

`raw_counterparty` is **never** overwritten. Everything else is derived, so
`reclassify` can always rebuild from the original text.

---

## 10. Results and honest limits

| | |
|---|---:|
| Transactions | 2,005 |
| Auto-classified | **1,848 — 92.2%** |
| Flagged for review | 157 — 7.8% |
| `OTHER` | 47 — 2.3% |
| Speed | ~150 µs/txn average, ~3 ms on a row that ends as OTHER |

By deciding layer: `TXN_TYPE` 967 · `RULE` 683 · `EXACT_MERCHANT` 196 ·
`FALLBACK` 153 · `ML_MODEL` 3 · `FUZZY_MATCH` 3.

**92.2% is coverage, not accuracy.** It means 1,848 transactions got a confident
answer, not that 1,848 are correct.

**Accuracy on a held-out statement.** Jun–Sep 2026 (363 rows, 69 of 111
merchants never seen in training), hand-labelled blind, scored with
`evaluate`. The committed system scored 78.0% (81.3% when not flagged, 60
silent errors). The fixes it prompted brought that to 96.1% (96.6%, 10 silent
errors), but because they were found by reading that statement, 96% overstates
real accuracy; the next unseen statement is the honest test. There is no labelled ground truth for this
statement, so precision is unmeasured. The report's suspicious-classification
sections exist precisely because of that gap.

Known limits:

- **Person-named vendors.** Unlearnable from the string. The report flags them
  by behaviour (many payments, small median); one correction each fixes it.
- **Incoming money from a person is `PERSONAL_TRANSFER`, not `INCOME`** — by
  design. Counting a friend's repayment as income corrupts every budget.
- **Masked counterparties** (`******6258`) carry no merchant information;
  treated as personal transfers at 0.70, always flagged.

---

## 11. The model layers, and why they are last

~90% of these transactions are decided by an exact merchant name or a single
keyword. That is a lookup table's job, and the table does it reproducibly and
with `evidence`. So the model does not replace any layer. It runs in the one
place where the table has nothing to say: rows about to fall back under the
review threshold.

**What it is.** Logistic regression over hashed character n-grams (2–4 chars
plus whole words, CRC32 into 16,384 buckets, L2-normalized), exported with
skl2onnx and run with onnxruntime. The featurizer lives in `ml.py`, not in the
graph, so the graph is a single matrix multiply that any ONNX runtime can
execute; its spec string is stored in the model's metadata and checked at
load. The model is ~900 KB and costs ~80 µs on the rows that reach it.

**What it learns from.** There is no hand-labelled ground truth, so it is
distilled from the curated layers: 101 built-in brands, 479 rule patterns, the
statement's confidently-classified outgoing merchants (identity layers, plus
person-to-person payments as `PERSONAL_TRANSFER`), learned merchants, and user
overrides at 3× weight. Statement labels are produced with the model switched
off, so it never trains on its own answers. `INCOME` and `OTHER` are never
targets: one is a property of a payment's direction, the other is the absence
of an answer.

That is circular on purpose. The model cannot know more than the rules; the
bet is that it generalizes their spelling patterns to strings no rule matches.

**Guards.**

- Never asked about masked handles, incoming rows, or `AMBIGUOUS_PHRASES`
  (without this, `akshay internet cafe` → FOOD at 0.81).
- A `PERSONAL_TRANSFER` answer is refused when the string has a commercial
  keyword (`bharat traders` is not a friend).
- Taken only at p ≥ 0.75. Below that the row stays flagged and the guess is
  appended to `evidence` as a hint for the reviewer.
- Confidence capped at 0.89, below `RULE`.
- Never learned into the merchant directory (§9).

**What it buys — measured, not hoped.** 5-fold cross-validation gives 99.2%
agreement with the curated layers on the 57% of held-out merchants it clears
0.75 on. That number flatters it, because held-out merchants usually still
contain a keyword the training set has. On the 201 rows that actually reach
stage 6 it resolves **5** (after the EDUCATION retrain): `raju`, `rupesh`,
`kamlesh`, `shubham` ×2, all as transfers. An earlier training run also let
through `sweety` → FOOD, probably a person, so its output shifts from one
retrain to the next. Most of its guesses below 0.75 on the other rows are
wrong. The residual is not a spelling problem: shops wearing people's names,
and bare `… traders` / `… enterprises`. No text-only model fixes those.

**Where real gains would come from.** The human labels in
`merchant_overrides`. The review queue is an unlabelled pool ranked by spend,
`train` already weights corrections 3×, and `previous_category` records what
the classifier got wrong so frequently-corrected *rules* can be fixed directly.
Retrain after a batch of corrections; measure against those corrections, not
against the rules.

### MiniLM: from last resort to hint

It was added so the system would never give up with `OTHER`: all-MiniLM-L6-v2
(ONNX, pinned to revision `1110a24`, sha256-verified, ~90 MB, fetched by
`setup-minilm` and kept out of git) embeds the merchant, and the 5 nearest of
the same labelled merchants the n-gram model trains on vote, weighted by
cosine similarity.

Evaluated against the hand-labelled Jun–Sep 2026 statement, it got 1 of the
40 rows it decided right and cost about 10 points of accuracy, nearly all by
replacing a correct `OTHER` (e.g. `Google Asia Pacific`) with a wrong guess.
It is now a hint: rows that end as `OTHER` stay `OTHER` and flagged, with
its suggestion in the evidence for the reviewer. Its guard still applies:
never suggest `PERSONAL_TRANSFER` for a commercial name.
