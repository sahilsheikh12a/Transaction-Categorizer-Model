"""Web layer tests.

The UI is a testing tool, so these cover the contract the page depends on and
the failure modes that would take the server down: a bad upload, a bad payload,
an unknown route. A handler that raises must return 4xx/5xx and keep serving.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from phonepe_categorizer import categories as C
from phonepe_categorizer import web

HEADER = ("date,time,direction,counterparty,transaction_id,utr,"
          "account_flow,account_last4,type,amount\n")
ROWS = ("2026-01-01,10:00,Paid to,EBENEZER BAKERY,T1,1,Debited From,4950,DEBIT,50\n"
        "2026-01-02,11:00,Paid to,Ritik Pandey,T2,2,Debited From,4950,DEBIT,500\n"
        "2026-01-03,12:00,Paid,Mobile Recharge,T3,3,Debited From,4950,DEBIT,239\n")


@pytest.fixture
def server(tmp_path):
    web.STATE.db_path = str(tmp_path / "web.db")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _req(url, data=None, ctype="application/json", method=None):
    body = None
    if data is not None:
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
    r = urllib.request.Request(url, data=body, method=method,
                               headers={"Content-Type": ctype} if body else {})
    with urllib.request.urlopen(r) as resp:
        return resp.status, json.loads(resp.read() or b"null")


def _expect_error(url, data=None, ctype="application/json"):
    try:
        _req(url, data, ctype)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    pytest.fail("expected an HTTP error")


class TestStatic:
    def test_serves_the_page(self, server):
        with urllib.request.urlopen(server + "/") as r:
            body = r.read().decode()
        assert r.status == 200
        assert "<title>PhonePe Categorizer</title>" in body

    def test_page_has_no_external_references(self, server):
        """It must work with no network — no CDN scripts, fonts or styles."""
        with urllib.request.urlopen(server + "/") as r:
            body = r.read().decode()
        for marker in ("http://", "https://", "//cdn", "integrity="):
            assert marker not in body, f"page references {marker}"

    def test_unknown_route_is_404(self, server):
        code, _ = _expect_error(server + "/nope")
        assert code == 404


class TestMetadata:
    def test_categories(self, server):
        _, body = _req(server + "/api/categories")
        assert body["categories"] == list(C.ALL)
        assert set(body["sources"]) == set(C.SOURCES)
        assert body["definitions"][C.GROCERIES]

    def test_state_reports_learned_counts(self, server):
        _, body = _req(server + "/api/state")
        assert body["overrides"] == 0 and body["learnedMerchants"] == 0


class TestClassify:
    def test_classifies_a_merchant(self, server):
        _, body = _req(server + "/api/classify",
                       {"text": "Maturchaya Kirana 2", "direction": "Paid to"})
        assert body["category"] == C.GROCERIES
        assert body["source"] == C.SRC_RULE
        assert body["needsReview"] is False
        assert body["normalizedMerchant"] == "maturchaya kirana"

    def test_classifies_a_person(self, server):
        _, body = _req(server + "/api/classify",
                       {"text": "Ritik Pandey", "direction": "Paid to"})
        assert body["category"] == C.PERSONAL_TRANSFER

    def test_credit_flag_changes_the_answer(self, server):
        _, paid = _req(server + "/api/classify",
                       {"text": "Sajid Sheikh", "direction": "Paid to"})
        _, got = _req(server + "/api/classify",
                      {"text": "Sajid Sheikh", "direction": "Received from",
                       "isCredit": True})
        assert paid["txnType"] == "PAYMENT_TO_PERSON"
        assert got["txnType"] == "MONEY_RECEIVED"

    def test_empty_text_is_rejected(self, server):
        code, body = _expect_error(server + "/api/classify", {"text": "  "})
        assert code == 400 and "required" in body["error"]


class TestCategorize:
    def test_returns_rows_and_summary(self, server):
        _, body = _req(server + "/api/categorize", (HEADER + ROWS).encode(), "text/csv")
        assert body["summary"]["classified"] == 3
        assert len(body["rows"]) == 3
        assert [r["category"] for r in body["rows"]] == [
            C.FOOD_AND_DINING, C.PERSONAL_TRANSFER, C.MOBILE_AND_INTERNET,
        ]

    def test_rows_carry_everything_the_table_renders(self, server):
        _, body = _req(server + "/api/categorize", (HEADER + ROWS).encode(), "text/csv")
        row = body["rows"][0]
        for field in ("date", "counterparty", "direction", "amount", "isCredit",
                      "category", "confidence", "source", "needsReview",
                      "txnType", "normalizedMerchant", "evidence"):
            assert field in row, field

    def test_malformed_rows_are_counted_not_fatal(self, server):
        payload = HEADER + "garbage,,,,,,,,,\n" + ROWS
        _, body = _req(server + "/api/categorize", payload.encode(), "text/csv")
        assert body["summary"]["classified"] == 3
        assert body["summary"]["skipped"] == 1

    def test_empty_upload_is_rejected(self, server):
        code, _ = _expect_error(server + "/api/categorize", b"", "text/csv")
        assert code == 400

    def test_unrecognizable_csv_returns_422_not_500(self, server):
        code, body = _expect_error(server + "/api/categorize",
                                   b"colour,size\nred,large\n", "text/csv")
        assert code == 422
        assert "date" in body["missing"]


class TestCorrections:
    def test_correction_changes_later_classification(self, server):
        _, before = _req(server + "/api/classify",
                         {"text": "Smart Point NAGPUR U178", "direction": "Paid to"})
        assert before["category"] == C.OTHER
        _req(server + "/api/correct",
             {"merchant": "Smart Point NAGPUR U178", "category": C.GROCERIES})
        _, after = _req(server + "/api/classify",
                        {"text": "Smart Point NAGPUR U178", "direction": "Paid to"})
        assert after["category"] == C.GROCERIES
        assert after["source"] == C.SRC_USER_OVERRIDE

    def test_correction_shows_up_in_a_reupload(self, server):
        body = HEADER + ("2026-01-01,10:00,Paid to,Smart Point NAGPUR U178,"
                         "T9,1,Debited From,4950,DEBIT,90\n")
        _, first = _req(server + "/api/categorize", body.encode(), "text/csv")
        assert first["rows"][0]["needsReview"] is True
        _req(server + "/api/correct",
             {"merchant": "Smart Point NAGPUR U178", "category": C.GROCERIES})
        _, second = _req(server + "/api/categorize", body.encode(), "text/csv")
        assert second["rows"][0]["category"] == C.GROCERIES
        assert second["rows"][0]["needsReview"] is False

    def test_invalid_category_is_rejected(self, server):
        code, _ = _expect_error(server + "/api/correct",
                                {"merchant": "A Shop", "category": "NOT_A_CATEGORY"})
        assert code == 400

    def test_state_reflects_the_correction(self, server):
        _req(server + "/api/correct", {"merchant": "A Shop", "category": C.GROCERIES})
        _, state = _req(server + "/api/state")
        assert state["overrides"] == 1


class TestResilience:
    def test_bad_json_returns_500_and_server_survives(self, server):
        code, _ = _expect_error(server + "/api/classify", b"{not json", "application/json")
        assert code == 500
        # still serving
        _, body = _req(server + "/api/classify", {"text": "KFC"})
        assert body["category"] == C.FOOD_AND_DINING


class TestMerchantsEndpoint:
    def test_returns_two_fields_per_merchant(self, server):
        _, body = _req(server + "/api/merchants", (HEADER + ROWS).encode(), "text/csv")
        assert len(body["merchants"]) == 3
        for m in body["merchants"]:
            assert set(m) == {"merchant", "category", "needsReview"}
            assert m["category"] in C.VALID

    def test_deduplicates_merchants(self, server):
        body = HEADER + ROWS + ROWS.replace("T1", "T4").replace("T2", "T5").replace("T3", "T6")
        _, out = _req(server + "/api/merchants", body.encode(), "text/csv")
        assert len(out["merchants"]) == 3

    def test_sorted_alphabetically(self, server):
        _, body = _req(server + "/api/merchants", (HEADER + ROWS).encode(), "text/csv")
        names = [m["merchant"] for m in body["merchants"]]
        assert names == sorted(names, key=str.lower)

    def test_reflects_a_correction(self, server):
        _req(server + "/api/correct",
             {"merchant": "EBENEZER BAKERY", "category": C.GROCERIES})
        _, body = _req(server + "/api/merchants", (HEADER + ROWS).encode(), "text/csv")
        got = {m["merchant"]: m["category"] for m in body["merchants"]}
        assert got["EBENEZER BAKERY"] == C.GROCERIES

    def test_empty_upload_is_rejected(self, server):
        code, _ = _expect_error(server + "/api/merchants", b"", "text/csv")
        assert code == 400
