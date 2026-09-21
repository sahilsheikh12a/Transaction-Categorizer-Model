"""Stage 7 — MiniLM neighbours, the layer that replaces OTHER.

Contract tests use a stub so they do not need the 90 MB model; the
`real_minilm` tests run only when `setup-minilm` has been done.
"""
from __future__ import annotations

import pytest

from phonepe_categorizer import categories as C
from phonepe_categorizer import semantic
from phonepe_categorizer.engine import classify_core


class _Stub:
    def __init__(self, ranked, similarity=0.9, nearest="somewhere"):
        self.nb = semantic.Neighbours(ranked[0][0], ranked[0][1], similarity, nearest,
                                      tuple(ranked))
        self.calls = 0

    def neighbours(self, text):
        self.calls += 1
        return self.nb


def _paid(merchant: str, **kw):
    return classify_core(merchant, direction="Paid to", **kw)


UNKNOWN = "Zzq Unrecognizable Enterprises Holdings"   # would be OTHER


class TestReplacesOther:
    def test_clear_vote_is_taken_without_review(self):
        semantic.set_default_model(_Stub([(C.GROCERIES, 0.9), (C.HEALTH, 0.1)]))
        r = _paid(UNKNOWN)
        assert r.category == C.GROCERIES
        assert r.source == C.SRC_SEMANTIC
        assert r.confidence == semantic.CONFIDENT
        assert r.needs_review is False

    def test_split_vote_still_answers_but_stays_flagged(self):
        semantic.set_default_model(_Stub([(C.HEALTH, 0.5), (C.GROCERIES, 0.5)]))
        r = _paid(UNKNOWN)
        assert r.category == C.HEALTH                 # an answer, not OTHER
        assert r.needs_review is True
        assert r.confidence <= semantic.GUESS_CEILING

    def test_distant_neighbour_stays_flagged(self):
        semantic.set_default_model(_Stub([(C.HEALTH, 1.0)], similarity=0.3))
        assert _paid(UNKNOWN).needs_review is True

    def test_evidence_names_the_nearest_merchant(self):
        semantic.set_default_model(_Stub([(C.SHOPPING, 1.0)], nearest="textiles"))
        assert "nearest 'textiles'" in _paid(UNKNOWN).evidence

    def test_commercial_name_is_never_a_person(self):
        semantic.set_default_model(
            _Stub([(C.PERSONAL_TRANSFER, 0.9), (C.GROCERIES, 0.1)]))
        r = _paid("Bharat Traders")
        assert r.category == C.GROCERIES
        assert r.needs_review is True

    def test_ambiguous_phrase_always_flagged(self):
        semantic.set_default_model(_Stub([(C.FOOD_AND_DINING, 1.0)]))
        r = _paid("Akshay Internet Cafe")
        assert r.source == C.SRC_SEMANTIC
        assert r.needs_review is True


class TestNeverAsked:
    @pytest.mark.parametrize("merchant", [
        "Swiggy",                    # exact
        "Samsung Store",             # rule
        "Ritik Pandey",              # person
        "******6258",                # masked
        "Nirmala",                   # short-name fallback already answers
    ])
    def test_rows_that_already_have_an_answer(self, merchant):
        stub = _Stub([(C.TRAVEL, 1.0)])
        semantic.set_default_model(stub)
        assert _paid(merchant).source != C.SRC_SEMANTIC
        assert stub.calls == 0


class TestWithoutIt:
    def test_absent_layer_is_the_old_fallback(self):
        r = _paid(UNKNOWN)
        assert r.category == C.OTHER
        assert r.source == C.SRC_FALLBACK

    def test_mobile_number_is_a_flagged_transfer(self):
        r = _paid("9359189581")
        assert r.category == C.PERSONAL_TRANSFER
        assert r.needs_review is True

    def test_env_switch(self, monkeypatch):
        monkeypatch.setenv(semantic.ENV_SWITCH, "off")
        semantic.reset_default_model()
        assert semantic.default_model() is None


class TestFetch:
    def test_checksum_mismatch_discards_the_download(self, tmp_path, monkeypatch):
        monkeypatch.setattr(semantic, "FILES", {"tokenizer.json": ("tokenizer.json", "0" * 64)})
        monkeypatch.setattr(semantic.urllib.request, "urlretrieve",
                            lambda url, dest: dest.write_bytes(b"not the file"))
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            semantic.fetch(tmp_path, log=lambda *_: None)
        assert list(tmp_path.iterdir()) == []

    def test_good_download_is_kept(self, tmp_path, monkeypatch):
        import hashlib

        body = b"tokenizer bytes"
        monkeypatch.setattr(semantic, "FILES", {
            "tokenizer.json": ("tokenizer.json", hashlib.sha256(body).hexdigest())})
        monkeypatch.setattr(semantic.urllib.request, "urlretrieve",
                            lambda url, dest: dest.write_bytes(body))
        semantic.fetch(tmp_path, log=lambda *_: None)
        assert (tmp_path / "tokenizer.json").read_bytes() == body


@pytest.fixture
def real_minilm():
    pytest.importorskip("tokenizers")
    if not (semantic.is_installed() and (semantic.MODEL_DIR / semantic.INDEX_FILE).exists()):
        pytest.skip("run `python -m phonepe_categorizer setup-minilm` first")
    semantic.reset_default_model()
    model = semantic.default_model()
    semantic.set_default_model(model)
    return model


class TestRealModel:
    def test_embeddings_are_unit_length(self, real_minilm):
        import numpy as np

        v = real_minilm.embedder.embed(["chat center", "textiles"])
        assert v.shape == (2, 384)
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)

    def test_an_unknown_merchant_gets_a_category(self, real_minilm):
        r = _paid("Akhtar Textile")
        assert r.source == C.SRC_SEMANTIC
        assert r.category == C.SHOPPING
