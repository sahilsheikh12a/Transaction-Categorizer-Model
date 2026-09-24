"""Stage 7 — MiniLM nearest shops: decides on a clear vote, suggests otherwise.

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


class TestClearVoteDecides:
    def test_clear_vote_sets_the_category(self):
        semantic.set_default_model(_Stub([(C.GROCERIES, 0.9), (C.HEALTH, 0.1)],
                                         similarity=0.6))
        r = _paid(UNKNOWN)
        assert r.category == C.GROCERIES
        assert r.source == C.SRC_SEMANTIC
        assert r.confidence == semantic.CONFIDENT
        assert r.needs_review is False
        assert "nearest shop" in r.evidence

    def test_it_ranks_below_the_n_gram_model(self):
        assert semantic.CONFIDENT < 0.89

    def test_an_ambiguous_phrase_is_still_refused(self):
        # "internet cafe" is a phrase the rules deliberately abstain on.
        semantic.set_default_model(_Stub([(C.FOOD_AND_DINING, 1.0)], similarity=0.9))
        r = _paid("Akshay Internet Cafe")
        assert r.category == C.OTHER
        assert r.needs_review is True


class TestUnclearVoteOnlySuggests:
    """A wrong category is worse than OTHER, so an unsure vote stays a hint."""

    def test_split_vote(self):
        semantic.set_default_model(_Stub([(C.HEALTH, 0.5), (C.GROCERIES, 0.5)],
                                         similarity=0.9))
        r = _paid(UNKNOWN)
        assert r.category == C.OTHER
        assert r.source == C.SRC_FALLBACK
        assert r.needs_review is True
        assert "minilm suggests HEALTH" in r.evidence

    def test_distant_nearest_shop(self):
        semantic.set_default_model(_Stub([(C.HEALTH, 1.0)], similarity=0.3))
        r = _paid(UNKNOWN)
        assert r.category == C.OTHER
        assert "minilm suggests HEALTH" in r.evidence

    def test_evidence_names_the_nearest_shop(self):
        semantic.set_default_model(_Stub([(C.SHOPPING, 0.5)], nearest="textiles"))
        assert "nearest shop 'textiles'" in _paid(UNKNOWN).evidence


class TestIndexHoldsShopsOnly:
    """People are settled by stage 1; in the index they only mislead."""

    def test_person_and_direction_labels_are_left_out(self, tmp_path, monkeypatch):
        recorded = {}
        monkeypatch.setattr(semantic, "Embedder",
                            lambda model_dir=None: type("E", (), {
                                "embed": lambda self, texts: _fake_vectors(texts, recorded)})())
        n = semantic.build_index(
            ["kirana store", "ritik pandey", "acme ltd", "salary"],
            [C.GROCERIES, C.PERSONAL_TRANSFER, C.SHOPPING, C.INCOME],
            tmp_path)
        assert n == 2
        assert recorded["texts"] == ["kirana store", "acme ltd"]

    def test_an_index_with_no_shops_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(semantic, "Embedder", lambda model_dir=None: None)
        with pytest.raises(ValueError, match="no indexable shops"):
            semantic.build_index(["ritik pandey"], [C.PERSONAL_TRANSFER], tmp_path)


def _fake_vectors(texts, recorded):
    import numpy as np

    recorded["texts"] = list(texts)
    return np.zeros((len(texts), 384), dtype="float32")


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
        _paid(merchant)
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

    def test_an_unknown_merchant_gets_an_answer_or_a_suggestion(self, real_minilm):
        r = _paid("Zzq Unrecognizable Enterprises Holdings")
        assert r.source in (C.SRC_SEMANTIC, C.SRC_FALLBACK)
        assert r.source == C.SRC_SEMANTIC or "minilm suggests" in r.evidence

    def test_the_shipped_index_holds_no_people(self, real_minilm):
        assert not (set(real_minilm.labels) & semantic.EXCLUDE_LABELS)
