"""Unit tests for the V1 deterministic categorization pipeline.

Every merchant string in this file is a real counterparty from the PhonePe
statement the pipeline was built against — typos, truncations and all — so a
regression here means a regression on real data, not on invented examples.

Covers, in order: normalization, person-to-person detection, exact merchant
matching, rule matching, fuzzy matching, user overrides, unknown merchants,
incoming transactions, refunds, and malformed CSV rows.
"""
from __future__ import annotations

import pytest

from phonepe_categorizer import categories as C
from phonepe_categorizer import fuzzy, merchants, rules, triage, txn_type
from phonepe_categorizer.engine import classify_core
from phonepe_categorizer.importer import parse
from phonepe_categorizer.normalize import canon, normalize_merchant
from phonepe_categorizer.txn_type import TxnType


def _paid(merchant: str, **kw):
    return classify_core(merchant, direction="Paid to", **kw)


# ── 1. merchant normalization ────────────────────────────────────────────

class TestNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("Paid to NEW MATRUCHAYA DAILY AND GENERAL STORE",
         "new matruchaya daily and general store"),
        ("MAH NAG Mate Square KFC 6453", "mah nag mate square kfc"),
        ("Maturchaya Kirana 2", "maturchaya kirana"),
        ("Nimrah_Cafe_And_Bakery", "nimrah cafe and bakery"),
        ("Smart Point NAGPUR U178", "smart point nagpur"),
        ("PVR INOX Limited", "pvr inox"),
        ("M S ADRIJA CLINIC", "adrija clinic"),
        ("  YUNUSKHA  PATHAN-KIRANA!! ", "yunuskha pathan kirana"),
    ])
    def test_strips_noise_keeps_merchant(self, raw, expected):
        assert normalize_merchant(raw) == expected

    def test_brand_numbers_survive_when_not_trailing(self):
        # "BIG 5" is part of the brand; only trailing outlet codes are dropped.
        assert normalize_merchant("R.S BIG 5 FASHION STUDIO") == "r s big 5 fashion studio"

    def test_masked_account_yields_no_merchant(self):
        # Must not degrade to the bare digits — "6258" is not a merchant name.
        for raw in ("******6258", "xxxxxx3159", "******0064"):
            assert normalize_merchant(raw) == ""

    def test_doubled_signal_is_collapsed(self):
        # Adapters that set merchant == text produce "X X"; a doubled name
        # breaks every token-count heuristic downstream.
        assert normalize_merchant("Sajid Sheikh Sajid Sheikh") == "sajid sheikh"

    def test_canon_contract_is_unchanged(self):
        # canon() keys the override + cache tables. If this drifts, every
        # stored override silently orphans.
        assert canon("  YUNUSKHA  PATHAN-KIRANA!! ") == "yunuskha pathan kirana"
        assert canon("") == "" and canon(None) == ""

    def test_idempotent(self):
        for raw in ("Maturchaya Kirana 2", "PVR INOX Limited", "EBENEZER BAKERY"):
            once = normalize_merchant(raw)
            assert normalize_merchant(once) == once


# ── 2. person-to-person detection ────────────────────────────────────────

class TestPersonDetection:
    @pytest.mark.parametrize("name", [
        "Ritik Pandey", "Sajid Sheikh", "Joel Jacob Varghese",
        "CHETMANI DWARKAPRASAD SAHU", "ABDUL FAIZAL ABDUL SALIM SHEIKH",
        "ARVINDER KOUR SARANJEET SINGH NAGI", "K G ATKAR", "Mr YASH AJABRAO RAUT",
    ])
    def test_real_people_are_personal_transfers(self, name):
        r = _paid(name)
        assert r.category == C.PERSONAL_TRANSFER, r
        assert r.txn_type == TxnType.PAYMENT_TO_PERSON.value

    @pytest.mark.parametrize("name", [
        "NEW MATRUCHAYA DAILY AND GENERAL STORE", "EBENEZER BAKERY",
        "Taj Chicken And Mutton Centre", "J F C Fruit Shop", "M S ADRIJA CLINIC",
    ])
    def test_businesses_are_not_people(self, name):
        assert triage.looks_like_person(name) is False
        assert _paid(name).category != C.PERSONAL_TRANSFER

    def test_merchant_evidence_beats_name_shape(self):
        # "Swiggy Order" is two alphabetic tokens and reads as a name to any
        # shape heuristic. A curated rule must win.
        r = classify_core("Swiggy Order")
        assert r.category == C.FOOD_AND_DINING
        assert r.source == C.SRC_RULE

    def test_spec_example_paid_to_person_is_not_shopping(self):
        for name in ("Ritik", "tannu", "Pratik"):
            assert _paid(name).category == C.PERSONAL_TRANSFER


# ── 3. exact merchant matching ───────────────────────────────────────────

class TestExactMerchant:
    @pytest.mark.parametrize("name,expected", [
        ("KFC", C.FOOD_AND_DINING),
        ("Mobile Recharge", C.MOBILE_AND_INTERNET),
        ("Electricity", C.BILLS_AND_UTILITIES),
        ("Amazon India", C.SHOPPING),
        ("Netflix", C.ENTERTAINMENT),
        ("AVENUE SUPERMARTS LTD", C.GROCERIES),
    ])
    def test_directory_hits(self, name, expected):
        r = classify_core(name, direction="Paid to")
        assert r.category == expected
        assert r.source == C.SRC_EXACT_MERCHANT
        assert r.confidence == merchants.EXACT_CONFIDENCE

    def test_learned_merchant_beats_builtin(self):
        r = classify_core("KFC", direction="Paid to", learned={"kfc": C.GROCERIES})
        assert r.category == C.GROCERIES

    def test_unknown_name_is_not_a_directory_hit(self):
        assert merchants.lookup("Zzz Nonexistent Merchant") is None


# ── 4. rule matching ─────────────────────────────────────────────────────

class TestRules:
    @pytest.mark.parametrize("name,expected", [
        ("Maturchaya Kirana 2", C.GROCERIES),
        ("NEW MATRUCHAYA DAILY AND GENERAL STORE", C.GROCERIES),
        ("Mahavir daily needs", C.GROCERIES),
        ("EBENEZER BAKERY", C.FOOD_AND_DINING),
        ("Jhoya Chinese Center", C.FOOD_AND_DINING),
        ("Zoya Chat Center", C.FOOD_AND_DINING),
        ("Parmatma Ek Tea Stall", C.FOOD_AND_DINING),
        ("GRACE STATIONARY", C.SHOPPING),
        ("Shree Ganesh Book Depot", C.SHOPPING),
        ("Hp Auto Care Chinchbhvan", C.TRANSPORT),
        ("Indian Oil Petrol Pump - Sujit Automobiles", C.TRANSPORT),
        ("MAHARASHTRA METRO RAIL CORPORATION LIMITED", C.TRANSPORT),
        ("Nilesh medical", C.HEALTH),
        ("KOTHARI HOSPITAL", C.HEALTH),
        ("Ridan Films Rent House", C.ENTERTAINMENT),
        ("PVR INOX Limited", C.ENTERTAINMENT),
        ("Jio Prepaid Recharges", C.MOBILE_AND_INTERNET),
    ])
    def test_real_merchants(self, name, expected):
        hit = rules.lookup(name)
        assert hit is not None, f"no rule matched {name!r}"
        assert hit[0] == expected, f"{name!r} -> {hit[0]}, expected {expected}"

    @pytest.mark.parametrize("name,expected,why", [
        ("GAJANAN MEDICAL GENERAL STORES", C.HEALTH, "medical beats general store"),
        ("Shri Krishna Medical and General Stores", C.HEALTH, "medical beats general store"),
        ("ISHANT DAIRY AND ICECREAM PARLOUR", C.FOOD_AND_DINING, "icecream beats dairy"),
        ("Shree Ganesh Book Depot Bag House and General Sto", C.SHOPPING,
         "book depot beats general"),
        ("Roshan Mobile and Repairing Centre", C.SHOPPING, "repair beats recharge"),
        ("A U K HOTELS PRIVATE LIMITED", C.FOOD_AND_DINING, "every Indian 'hotel' is an eatery"),
        ("Aapna hotel", C.FOOD_AND_DINING, "a standalone Indian 'hotel' is an eatery"),
        ("BHOYAR RESTAURANT DAILY NEEDS", C.FOOD_AND_DINING, "restaurant beats daily needs"),
    ])
    def test_ordering_is_load_bearing(self, name, expected, why):
        assert rules.lookup(name)[0] == expected, why

    def test_ambiguous_phrases_abstain(self):
        # "Internet Cafe" is neither food nor telecom — refuse rather than guess.
        assert rules.lookup("Akshay Internet Cafe") is None

    def test_no_match_returns_none(self):
        assert rules.lookup("Ritik Pandey") is None
        assert rules.lookup("") is None


class TestBrandStores:
    """National brand stores. Unlike the rest of this file these strings are
    not from the shipped statement — they are the shapes a new user's
    statement brings, which is exactly what the statement-curated rules
    missed."""

    @pytest.mark.parametrize("merchant,category", [
        ("Samsung Store", C.SHOPPING),
        ("Samsung Smart Plaza", C.SHOPPING),
        ("Realme Store", C.SHOPPING),
        ("Mi Store", C.SHOPPING),
        ("Xiaomi", C.SHOPPING),
        ("Apple Store", C.SHOPPING),
        ("OnePlus Store", C.SHOPPING),
        ("One Plus Service Center", C.SHOPPING),
        ("Lenovo Exclusive Store", C.SHOPPING),
        ("Nike Store", C.SHOPPING),
        ("Bata Store", C.SHOPPING),
        ("Titan World", C.SHOPPING),        # would otherwise read as a person
        ("Titan Eye Plus", C.HEALTH),       # eyewear, before bare "titan"
        ("Lenskart", C.HEALTH),
    ])
    def test_brand(self, merchant, category):
        r = _paid(merchant)
        assert r.category == category
        assert r.source == C.SRC_RULE

    @pytest.mark.parametrize("merchant", [
        "Apple Fruit Center",   # a fruit stall, not Apple
        "Zara Sheikh",          # a person, not the brand
        "Raymond Dsouza",
    ])
    def test_brand_words_that_are_not_brands(self, merchant):
        assert _paid(merchant).category != C.SHOPPING


class TestEducation:
    @pytest.mark.parametrize("merchant", [
        "St Vincent Pallotti College of Engineering And Tec",   # from the statement
        "Neso Academy Private Limited",                         # from the statement
        "Rustomjee Academy for Global Careers",                 # from the statement
        "Udemy India LLP",                                      # from the statement
        "Government Medical College Nagpur",
        "Ramesh Coaching Classes",
        "Kendriya Vidyalaya",
    ])
    def test_education(self, merchant):
        assert _paid(merchant).category == C.EDUCATION

    @pytest.mark.parametrize("merchant,category", [
        # An address fragment must not beat what the business actually is.
        ("Sharma Kirana College Road", C.GROCERIES),
        ("Shree Juice Centre School Square", C.FOOD_AND_DINING),
        ("College Canteen", C.FOOD_AND_DINING),
        ("Mental Health Institute", C.HEALTH),
        ("Allen Solly", C.SHOPPING),
    ])
    def test_education_words_lose_to_the_business(self, merchant, category):
        assert _paid(merchant).category == category

    def test_legacy_edu_maps_to_education(self):
        assert C.coerce("edu") == C.EDUCATION
        assert C.to_legacy(C.EDUCATION) == "edu"


class TestBusinessesThatLookLikePeople:
    """A business misread as a person is a silent error: PERSONAL_TRANSFER is
    confident and never reaches the review queue."""

    @pytest.mark.parametrize("merchant", [
        "CINEPOLIS INDIA PRIVATE LIMITED",     # corporate form stripped by normalize
        "Google Asia Pacific Pte.Ltd",
        "Delhivery Limited",
    ])
    def test_corporate_form_is_never_a_person(self, merchant):
        assert _paid(merchant).category != C.PERSONAL_TRANSFER

    @pytest.mark.parametrize("merchant,category", [
        ("Prakash Electricals", C.SHOPPING),
        ("SPICY BITE SHAWARMA", C.FOOD_AND_DINING),
        ("Skyhi Dining", C.FOOD_AND_DINING),
        ("SARANG MADICALS", C.HEALTH),          # misspelt shop board
        ("AMRUT CHAHA", C.FOOD_AND_DINING),     # Marathi for tea
        ("JAIDEEP UPHAR GRUHA", C.SHOPPING),    # gift shop
        ("KHADAKA FEE PLAZA", C.TRANSPORT),     # NHAI toll booth
        ("Smart Point NAGPUR U178", C.GROCERIES),
        ("PRATHAM DEPARTMENTAL STORES", C.GROCERIES),
        ("VR MALL NAGPUR", C.SHOPPING),
        ("MAH NAG Eternity Mall KFC 6451", C.FOOD_AND_DINING),   # the brand beats "mall"
    ])
    def test_shop_words(self, merchant, category):
        assert _paid(merchant).category == category

    def test_rule_does_not_fire_inside_a_surname(self):
        # "patho" (pathology) used to match "PATHODE".
        assert _paid("RAHUL SURESH PATHODE").category == C.PERSONAL_TRANSFER
        assert _paid("Innsight Path Labs").category == C.HEALTH

    @pytest.mark.parametrize("hotel", [
        "HOTEL BLUE STAR", "A U K HOTELS PRIVATE LIMITED", "Aapna hotel",
    ])
    def test_every_hotel_is_food(self, hotel):
        assert _paid(hotel).category == C.FOOD_AND_DINING


class TestMoneyFromPeople:
    @pytest.mark.parametrize("sender", [
        "Rohan \U0001F913",            # emoji in the saved contact name
        "Mrs Farida Rafiq Ansari2",     # digit glued to the surname
        "RAHUL SURESH PATHODE",
    ])
    def test_is_a_transfer_not_income(self, sender):
        r = classify_core(sender, is_credit=True, direction="Received from")
        assert r.category == C.PERSONAL_TRANSFER

    def test_single_plain_name_is_a_flagged_transfer(self):
        r = classify_core("Rohan", is_credit=True, direction="Received from")
        assert r.category == C.PERSONAL_TRANSFER
        assert r.needs_review is True

    def test_business_credit_is_still_income(self):
        r = classify_core("DASHREELS TECHNOLOGIES PRIVATE LIMITED", is_credit=True,
                          direction="Received from")
        assert r.category == C.INCOME


# ── 5. fuzzy matching ────────────────────────────────────────────────────

class TestFuzzy:
    CORPUS = {
        "shree ganesh book depot": C.SHOPPING,
        "shabana bakery": C.FOOD_AND_DINING,
        "maturchaya kirana": C.GROCERIES,
        "jamtha point family restaurant": C.FOOD_AND_DINING,
        "raj traders": C.SHOPPING,
    }

    @pytest.mark.parametrize("typo,expected", [
        ("Sgree Ganesh Book Depot", C.SHOPPING),            # real typo in the data
        ("JAMTHA POINT FAMILY RESTURANT", C.FOOD_AND_DINING),  # real typo
        ("Shabana Bakery buttibori", C.FOOD_AND_DINING),    # locality suffix
        ("Matruchaya kirana store", C.GROCERIES),           # transposition
    ])
    def test_matches_typos_and_suffixes(self, typo, expected):
        hit = fuzzy.match(typo, self.CORPUS)
        assert hit is not None, f"no fuzzy match for {typo!r}"
        assert hit[0] == expected

    @pytest.mark.parametrize("unrelated", [
        "Completely Unrelated Merchant Name",
        "Ram Traders",          # one edit from "raj traders" — a different shop
        "Zoya Chat Center",
    ])
    def test_does_not_match_unrelated(self, unrelated):
        assert fuzzy.match(unrelated, self.CORPUS) is None

    def test_confidence_stays_below_exact_match(self):
        hit = fuzzy.match("Sgree Ganesh Book Depot", self.CORPUS)
        assert hit[1] < merchants.EXACT_CONFIDENCE

    def test_short_strings_are_never_fuzzy_matched(self):
        assert fuzzy.match("abc", self.CORPUS) is None

    def test_ambiguous_winner_abstains(self):
        # Two corpus entries equally close but in different categories.
        corpus = {"shri ram medical": C.HEALTH, "shri ram general": C.GROCERIES}
        assert fuzzy.match("shri ram", corpus) is None


# ── 6. user overrides ────────────────────────────────────────────────────

class TestOverrides:
    def test_override_beats_rules(self):
        r = classify_core("Maturchaya Kirana 2", direction="Paid to",
                          overrides={"maturchaya kirana": C.SHOPPING})
        assert r.category == C.SHOPPING
        assert r.source == C.SRC_USER_OVERRIDE
        assert r.confidence == 1.0
        assert r.needs_review is False

    def test_override_beats_person_detection(self):
        # The spec's hard requirement: a person-named vendor the user has
        # labeled must keep the user's label.
        r = classify_core("CHETMANI DWARKAPRASAD SAHU", direction="Paid to",
                          overrides={"chetmani dwarkaprasad sahu": C.GROCERIES})
        assert r.category == C.GROCERIES
        assert r.source == C.SRC_USER_OVERRIDE

    def test_override_key_survives_branch_numbers(self):
        # "XYZ General Store" and "XYZ General Store 2" are the same shop, so
        # one correction must cover both.
        ov = {"xyz general store": C.GROCERIES}
        for variant in ("Paid to XYZ General Store", "XYZ General Store 2"):
            assert classify_core(variant, direction="Paid to", overrides=ov).source \
                == C.SRC_USER_OVERRIDE

    def test_override_teaches_an_unknown_merchant(self):
        # The spec's worked example: an unrecognized shop starts as OTHER +
        # needs_review, the user picks GROCERIES, and it sticks from then on.
        unknown = "Qwertzu Asdfgh Poiuyt Lkjhgf Mnbvcx Qazwsx"
        before = _paid(unknown)
        assert before.category == C.OTHER and before.needs_review is True
        after = classify_core(f"Paid to {unknown}", direction="Paid to",
                              overrides={"qwertzu asdfgh poiuyt lkjhgf mnbvcx qazwsx": C.GROCERIES})
        assert after.category == C.GROCERIES
        assert after.source == C.SRC_USER_OVERRIDE
        assert after.needs_review is False


# ── 7. unknown merchants ─────────────────────────────────────────────────

class TestUnknownMerchants:
    def test_unknown_business_is_other_and_flagged(self):
        r = _paid("Zzq Unrecognizable Enterprises Holdings")
        assert r.category == C.OTHER
        assert r.source == C.SRC_FALLBACK
        assert r.needs_review is True

    def test_never_invents_a_category(self):
        # Too many tokens to be a name, no commercial or brand signal at all.
        assert _paid("Vbnmqwe Asdzxc Poiuyt Lkjhgf Mnbvcx Qazwsx").category == C.OTHER

    def test_three_token_nonsense_reads_as_a_person_not_a_guess(self):
        # This is correct behavior, not a miss: an unknown 3-token alphabetic
        # string is far likelier to be a person than any spending category.
        r = _paid("Qwertyuiop Asdfghjkl Zxcvbnm")
        assert r.category == C.PERSONAL_TRANSFER
        assert r.category not in (C.SHOPPING, C.FOOD_AND_DINING, C.GROCERIES)

    def test_empty_merchant(self):
        r = classify_core("", direction="Paid")
        assert r.category == C.OTHER and r.needs_review is True

    def test_masked_account_is_reviewable_transfer(self):
        r = _paid("******6258")
        assert r.category == C.PERSONAL_TRANSFER
        assert r.needs_review is True

    def test_every_result_carries_full_provenance(self):
        r = _paid("Maturchaya Kirana 2")
        assert r.category in C.VALID
        assert r.source in C.SOURCES
        assert 0.0 <= r.confidence <= 1.0
        assert isinstance(r.needs_review, bool)
        assert r.normalized_merchant and r.evidence


# ── 8. incoming transactions ─────────────────────────────────────────────

class TestIncoming:
    def test_credit_from_person_is_a_transfer_not_income(self):
        # Money from a friend is not earnings; counting it as income would
        # corrupt every budget and insight built on top.
        r = classify_core("Sajid Sheikh", is_credit=True, direction="Received from")
        assert r.category == C.PERSONAL_TRANSFER
        assert r.txn_type == TxnType.MONEY_RECEIVED.value

    def test_credit_from_business_is_income(self):
        r = classify_core("A U K HOTELS PRIVATE LIMITED", is_credit=True,
                          direction="Received from")
        assert r.category == C.INCOME

    def test_salary_keyword_is_income(self):
        # Two alphabetic tokens — reads as a person name without the keyword.
        r = classify_core("Salary Credit", is_credit=True)
        assert r.category == C.INCOME

    def test_credit_from_masked_account_is_a_transfer(self):
        r = classify_core("******6258", is_credit=True, direction="Received from")
        assert r.category == C.PERSONAL_TRANSFER

    def test_direction_overrides_amount_sign(self):
        r = classify_core("Sajid Sheikh", is_credit=False, direction="Received from")
        assert r.txn_type == TxnType.MONEY_RECEIVED.value


# ── 9. refunds ───────────────────────────────────────────────────────────

class TestRefunds:
    def test_refund_direction_is_income(self):
        r = classify_core("Amazon India", is_credit=True, direction="Refund from")
        assert r.category == C.INCOME
        assert r.txn_type == TxnType.REFUND.value

    def test_refund_keyword_on_a_credit(self):
        r = classify_core("Swiggy refund", is_credit=True)
        assert r.txn_type == TxnType.REFUND.value
        assert r.category == C.INCOME

    def test_refund_wins_over_merchant_category(self):
        # An incoming Amazon row is a refund, not SHOPPING spend.
        r = classify_core("Amazon India", is_credit=True, direction="Refund from")
        assert r.category != C.SHOPPING

    def test_outgoing_refund_word_is_not_a_refund(self):
        assert txn_type.detect("Refundable Deposit Shop", is_credit=False,
                               direction="Paid to").type is not TxnType.REFUND


# ── 10. malformed CSV rows ───────────────────────────────────────────────

class TestMalformedCsv:
    HEADER = ("date,time,direction,counterparty,transaction_id,utr,"
              "account_flow,account_last4,type,amount\n")

    def _parse(self, body: str):
        return parse((self.HEADER + body).encode())

    def test_blank_counterparty_is_skipped_not_fatal(self):
        res = self._parse(
            "2026-02-27,17:37,Paid,,OMO26,302063287869,Debited From,4950,DEBIT,1131.01\n"
            "2026-02-28,10:00,Paid to,Maturchaya Kirana 2,T1,1,Debited From,4950,DEBIT,50\n"
        )
        assert len(res.rows) == 1
        assert res.problem_counts() == {"blank counterparty": 1}

    def test_unparseable_date_is_skipped(self):
        res = self._parse(
            "not-a-date,17:37,Paid to,EBENEZER BAKERY,T1,1,Debited From,4950,DEBIT,50\n"
            "2026-02-28,10:00,Paid to,EBENEZER BAKERY,T2,2,Debited From,4950,DEBIT,60\n"
        )
        assert len(res.rows) == 1
        assert res.problem_counts() == {"unparseable date": 1}

    def test_zero_amount_is_skipped(self):
        res = self._parse(
            "2026-02-28,10:00,Paid to,EBENEZER BAKERY,T1,1,Debited From,4950,DEBIT,0\n"
        )
        assert len(res.rows) == 0
        assert res.problem_counts() == {"missing or zero amount": 1}

    def test_short_row_does_not_crash(self):
        res = self._parse("2026-02-28,10:00,Paid to\n")
        assert res.total_data_rows >= 0  # parsed without raising

    def test_missing_transaction_id_still_imports(self):
        res = self._parse(
            "2026-02-28,10:00,Paid to,EBENEZER BAKERY,,455710966211,Debited From,4950,DEBIT,50\n"
        )
        assert len(res.rows) == 1

    def test_direction_column_is_preserved(self):
        res = self._parse(
            "2026-02-28,10:00,Bill paid,Electricity,NB1,1,Debited From,4950,DEBIT,790\n"
        )
        assert res.rows[0].direction == "Bill paid"

    def test_one_bad_row_does_not_stop_the_import(self):
        res = self._parse(
            "garbage,,,,,,,,,\n"
            "2026-02-28,10:00,Paid to,EBENEZER BAKERY,T1,1,Debited From,4950,DEBIT,50\n"
            ",,,,,,,,,\n"
            "2026-03-01,11:00,Paid to,Maturchaya Kirana 2,T2,2,Debited From,4950,DEBIT,60\n"
        )
        assert len(res.rows) == 2

    def test_end_to_end_on_a_realistic_slice(self):
        res = self._parse(
            "2024-07-09,19:58,Paid to,Sajid Sheikh,T1,1,Debited From,3159,DEBIT,1.0\n"
            "2024-07-09,20:00,Received from,Sajid Sheikh,T2,2,Credited To,3159,CREDIT,1500.0\n"
            "2024-07-11,08:17,Paid,Mobile Recharge,NX1,3,Debited From,3159,DEBIT,982.0\n"
            "2025-08-03,09:07,Bill paid,Electricity,NB1,4,Debited From,4950,DEBIT,790.0\n"
        )
        cats = [
            classify_core(t.counterparty, is_credit=t.is_credit,
                          direction=t.direction).category
            for t in res.rows
        ]
        assert cats == [
            C.PERSONAL_TRANSFER, C.PERSONAL_TRANSFER,
            C.MOBILE_AND_INTERNET, C.BILLS_AND_UTILITIES,
        ]


# ── determinism ──────────────────────────────────────────────────────────

def test_classification_is_deterministic():
    names = ["Maturchaya Kirana 2", "Ritik Pandey", "******6258", "Akshay Internet Cafe"]
    first = [_paid(n).as_dict() for n in names]
    for _ in range(3):
        assert [_paid(n).as_dict() for n in names] == first


# ── determinism across the whole real statement ──────────────────────────

def test_engine_is_stable_over_the_real_statement():
    """Classifying the shipped statement twice must give identical answers."""
    import pathlib
    csv_path = pathlib.Path(__file__).resolve().parents[1] / "transactions2.csv"
    if not csv_path.exists():
        pytest.skip("statement not present")
    from phonepe_categorizer.importer import parse_file
    rows = parse_file(csv_path).rows

    def run():
        return [
            classify_core(t.counterparty, is_credit=t.is_credit,
                          direction=t.direction).as_dict()
            for t in rows
        ]

    assert run() == run()


def test_punctuated_and_plain_name_agree():
    # "RAHUL_DANDARE_" and "RAHUL DANDARE" are the same person; the type
    # detector reads the normalized form so both take the same path.
    a = classify_core("RAHUL_DANDARE_", direction="Paid to")
    b = classify_core("RAHUL DANDARE", direction="Paid to")
    assert (a.category, a.source) == (b.category, b.source)
    assert a.category == C.PERSONAL_TRANSFER
