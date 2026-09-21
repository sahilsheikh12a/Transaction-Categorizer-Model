# PhonePe Transaction Categorizer

Deterministic, offline categorization of PhonePe statement transactions into a
small set of day-to-day expense categories. No LLM and no network at run
time: five hand-curated layers, then two ONNX models that only see what those
could not place: a small n-gram classifier, then MiniLM as a reviewer's hint. The
same input always produces the same output.

Built and evaluated against a real 2,009-row statement (`transactions2.csv`,
Jul 2024 – Apr 2026).

## Results on that statement

| | |
|---|---|
| Transactions parsed | **2,005** of 2,009 (2 duplicates collapsed, 2 blank rows skipped) |
| Classified automatically | **1,848 — 92.2%** (3 by the n-gram model) |
| Flagged for review | 157 — 7.8% |
| Fell through to `OTHER` | 47 — 2.3%, each with a MiniLM suggestion in the evidence |
| Unique merchants | 607 |
| Speed | ~150 µs per transaction on average; rows ending as OTHER ~3 ms (MiniLM hint) |
| Tests | 308 passing |

## Quick start

```bash
./start.sh
```

Creates the venv, installs requirements, frees the port if a previous run is
still holding it, starts the UI on <http://127.0.0.1:8000> and opens a browser.

```bash
./start.sh --port 9000     # different port
./start.sh --no-db         # scratch session, corrections not saved
./start.sh --check         # run the tests and the linter instead of serving
```

Or set it up by hand:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Use

### Browser UI

```bash
.venv/bin/python -m phonepe_categorizer serve      # http://127.0.0.1:8000
```

Classify a single counterparty and see the full verdict, or drop a CSV in to get
summary tiles, a category breakdown, and a filterable table of every
transaction with the evidence that decided it. Rows needing review are
highlighted; the `Correct to…` dropdown on each row teaches the system and
re-runs the whole file so you can watch the correction propagate.

Zero dependencies — it is `http.server` and one self-contained HTML file, no
CDN and no build step. It binds to `127.0.0.1` only; this is financial data and
there is no authentication, so do not expose it. It also refuses requests
that another web page in the same browser could forge: POSTs must use the
endpoint's content type (JSON or CSV), and the Host header must be a loopback
name. `--no-db` gives a scratch session whose corrections are not saved.

### CSV in, CSV out

```bash
.venv/bin/python -m phonepe_categorizer categorize transactions2.csv -o categorized.csv
```

```
2009 input rows -> categorized.csv
  classified     2007
  needs review   206
  unparseable    2
  duplicates     0
```

The output keeps every original column untouched and appends eight:

| Column | Example |
|---|---|
| `category` | `GROCERIES` |
| `category_confidence` | `0.90` |
| `category_source` | `RULE` |
| `category_needs_review` | `no` |
| `category_txn_type` | `PAYMENT_TO_MERCHANT` |
| `category_normalized_merchant` | `new matruchaya daily and general store` |
| `category_evidence` | `rule: 'daily and general'` |
| `status` | `ok` |

Three guarantees:

- **Every input row appears in the output, in order.** Rows the importer can't
  parse are written with `status = skipped: <reason>` and empty category cells,
  never dropped — otherwise the output can't be reconciled against the source.
- **Original columns are copied verbatim.** If the input already has a
  `category` or `status` column, the appended one becomes `category_2` /
  `status_2` rather than shadowing it, so the command is safe to re-run on its
  own output.
- **Duplicates are marked, not removed**, unless you pass `--dedupe`.

If a `categorizer.db` exists, its overrides and learned merchants are applied
automatically, so corrections you've made show up in the exported CSV.

### Other commands

```bash
# Classify a whole statement and print the full report
.venv/bin/python -m phonepe_categorizer report transactions2.csv

# Classify one string
.venv/bin/python -m phonepe_categorizer classify "Paid to NEW MATRUCHAYA GENERAL STORE"

# Import into SQLite, then teach it
.venv/bin/python -m phonepe_categorizer import  transactions2.csv
.venv/bin/python -m phonepe_categorizer review  --limit 20
.venv/bin/python -m phonepe_categorizer correct "Smart Point NAGPUR U178" GROCERIES
.venv/bin/python -m phonepe_categorizer reclassify

# Retrain the n-gram model (and the MiniLM hint index) after corrections
.venv/bin/python -m phonepe_categorizer train transactions2.csv
.venv/bin/python -m phonepe_categorizer setup-minilm transactions2.csv    # once, ~90 MB

# Measure accuracy on a statement the system has never seen
.venv/bin/python -m phonepe_categorizer label-sheet test.csv -o reports/labels.csv
#   ...fill in true_category, blind...
.venv/bin/python -m phonepe_categorizer evaluate test.csv reports/labels.csv --train transactions2.csv
```

As a library:

```python
from phonepe_categorizer import classify, importer

stmt = importer.parse_file("transactions2.csv")
for txn in stmt.rows:
    r = classify(txn.counterparty, is_credit=txn.is_credit, direction=txn.direction)
    print(r.category, r.confidence, r.source, r.needs_review)
```

## Architecture

[ARCHITECTURE.md](ARCHITECTURE.md) covers the design in full: why the stage
ordering is what it is, the two-normalizer split, how rule precedence is
resolved, the fuzzy matcher's guards, and what the learning loop is allowed to
learn from.

## The pipeline

```
raw counterparty
  → merchant normalization       normalize.py    strip prefixes, ref IDs, branch codes
  → transaction-type detection   txn_type.py     person / merchant / received / refund
  → user overrides               db/             highest priority, always
  → exact merchant lookup        merchants.py    ~110 brands + learned merchants
  → keyword & brand rules        rules.py        479 ordered patterns
  → fuzzy merchant matching      fuzzy.py        typos, truncation, branch suffixes
  → ONNX n-gram model            ml.py           only below the review threshold
  → OTHER + needs_review         engine.py       with a MiniLM suggestion (semantic.py)
```

Two ordering decisions carry most of the accuracy:

**Type detection runs before categorization.** "Paid to Ritik" is not a spending
category, so asking *which category* first is the wrong question. This is what
structurally prevents a person's name from becoming SHOPPING.

**But merchant evidence outranks name shape.** `"Swiggy Order"` and
`"Sony Center"` are two alphabetic tokens and read as Indian person names to any
shape heuristic. The exact-merchant and rule layers therefore run *before* the
person short-circuit — without this a ₹55,000 electronics purchase gets filed as
a transfer to a friend.

User overrides sit above both. Small Indian vendors routinely collect on
personal UPI accounts, and only the user can resolve that.

## The model layers

### Stage 6: n-gram model

Stage 6 is a logistic-regression classifier over hashed character n-grams,
shipped as `phonepe_categorizer/models/categorizer.onnx` (~900 KB) and run with
onnxruntime. It is asked only about outgoing rows every curated layer missed —
never about masked handles, incoming money, or phrases the rules deliberately
abstain on. At probability ≥ 0.75 its answer is taken (`source=ML_MODEL`,
confidence capped at 0.89, below every curated layer); below that the row stays
flagged and the evidence carries its guess:

```
evidence    no layer matched; model suggests FOOD_AND_DINING (p=0.49)
```

Its answers are never learned into the merchant directory, so a wrong one
cannot spread into exact or fuzzy matching.

It learns from what the curated layers already know — the built-in brands,
every rule pattern, the statement's confidently-classified merchants, and your
corrections (weighted 3×). Retrain after a new statement or a batch of
corrections:

```bash
.venv/bin/pip install -r requirements-train.txt   # scikit-learn, skl2onnx
.venv/bin/python -m phonepe_categorizer train transactions2.csv
```

`train` also rebuilds the MiniLM index below. Set
`PHONEPE_CATEGORIZER_ML=off` to run without the n-gram model. If onnxruntime is missing or
the model file is absent, the layer switches itself off.

**What it does and doesn't buy.** It resolves 5 of the rows that reach it on
this statement, all plain first names (`raju`, `rupesh`, `kamlesh`,
`shubham`) as transfers. Text alone can't place the rest: person-named shops
and bare `… traders` / `… enterprises`. Cross-validation shows 99% agreement with the
curated layers when it clears 0.75, but that measures agreement with the rules,
not correctness — there is still no hand-labelled ground truth.

### MiniLM: a hint for the reviewer

A row that nothing above can place ends as `OTHER`, flagged. Before it does,
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(ONNX, pinned revision, checksum-verified) finds the 5 most similar known
merchants, and their vote goes into the evidence as a suggestion. **It never
sets the category.**

```
OTHER  0.00 ⚠  no layer matched; minilm suggests GROCERIES (nearest 'nanda pradeep labhane' 0.47, 81% of vote)
```

It used to replace `OTHER` with its own answer. Measured against a
hand-labelled statement it got 1 of 40 of those rows right and cost about 10
points of accuracy, almost entirely by overwriting a correct `OTHER`, so it
was demoted to a hint.

Setup is one command, and `start.sh` runs it on first launch:

```bash
.venv/bin/python -m phonepe_categorizer setup-minilm transactions2.csv   # ~90 MB, once
```

Set `PHONEPE_CATEGORIZER_MINILM=off` to skip it. Without the files it switches
itself off.

## Categories

14, defined in one place (`categories.py`) so they can be changed without
touching the rules:

`GROCERIES` · `FOOD_AND_DINING` · `SHOPPING` · `BILLS_AND_UTILITIES` ·
`MOBILE_AND_INTERNET` · `TRANSPORT` · `HEALTH` · `ENTERTAINMENT` · `TRAVEL` ·
`FINANCE` · `EDUCATION` · `PERSONAL_TRANSFER` · `INCOME` · `OTHER`

`OTHER` always pairs with `needs_review=True` — it means "could not tell", never
"miscellaneous".

## Every result explains itself

```python
ClassifyResult(
    category='GROCERIES',
    confidence=0.90,
    source='RULE',                 # USER_OVERRIDE | EXACT_MERCHANT | RULE
                                   # | FUZZY_MATCH | TXN_TYPE | ML_MODEL
                                   # | FALLBACK
    needs_review=False,
    txn_type='PAYMENT_TO_MERCHANT',
    normalized_merchant='new matruchaya daily and general store',
    evidence="rule: 'daily and general'",
)
```

Confidences are ordered preferences, not calibrated probabilities:
override 1.0 > exact 0.97 > rule 0.90 > fuzzy ≤0.94 > model ≤0.89 > fallback ≤0.70.

## Learning from corrections

One correction fixes the past and the future:

```bash
.venv/bin/python -m phonepe_categorizer correct "Smart Point NAGPUR U178" GROCERIES
# → override saved; 33 past transaction(s) updated
```

Overrides are keyed on the *normalized* merchant, so `"XYZ General Store"`,
`"Paid to XYZ General Store"` and `"XYZ General Store 2"` are one merchant and
need one correction.

The original statement text is never overwritten — `raw_counterparty` and the
full source row are kept verbatim, so `reclassify` can always re-derive
everything from the original data.

## Storage

SQLite via SQLAlchemy, and entirely optional — the engine takes learned state as
plain dicts and never touches a database.

| Table | Holds |
|---|---|
| `transactions` | raw + normalized text, amount, date, type, category, confidence, source, evidence, needs_review |
| `merchants` | known merchant → category directory, also the fuzzy corpus |
| `merchant_overrides` | user corrections, plus what they replaced |
| `classification_rules` | user-defined keyword rules |

## Known limits

- **Person-named vendors.** `swapna lava gotivada` takes 85 payments at a ₹30
  median — a shop, not a friend. Deterministic rules cannot resolve this from the
  string alone; the report flags them and one correction each fixes it.
- **Incoming money from a person is `PERSONAL_TRANSFER`, not `INCOME`** — by
  design. Counting a friend's repayment as income would corrupt every budget.
- **Fuzzy matching is intentionally conservative.** It requires a shared
  substantive token and abstains when two candidates tie in different categories.

## Layout

```
phonepe_categorizer/
  categories.py   taxonomy — edit here to change categories
  normalize.py    canon() storage key + normalize_merchant() matching key
  triage.py       person vs business shape heuristics
  txn_type.py     stage 1 — transaction type
  merchants.py    stage 3 — exact directory
  rules.py        stage 4 — 479 ordered rules
  fuzzy.py        stage 5 — token-alignment matcher
  ml.py           stage 6 — ONNX model runtime + featurizer
  semantic.py     MiniLM embeddings and neighbour vote (hint only), setup
  train.py        trains and exports the stage-6 model
  models/         the shipped categorizer.onnx; minilm/ is fetched, not in git
  engine.py       the orchestrator (pure, no I/O)
  importer.py     PhonePe CSV → transactions, resilient to bad rows
  export.py       CSV in -> categorized CSV out
  web.py          local browser UI (stdlib http.server)
  static/         the UI's single HTML page
  report.py       evaluation report
  cli.py          command line
  db/             optional SQLAlchemy persistence + correction loop
tests/            308 tests
```

## Develop

```bash
.venv/bin/python -m pytest        # 308 tests
.venv/bin/python -m ruff check .
```
