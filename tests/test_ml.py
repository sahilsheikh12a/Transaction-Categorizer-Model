"""Stage 6 — the ONNX model layer.

Runs against the shipped `models/categorizer.onnx`. Assertions are about the
layer's contract (when it may speak, how its answers are marked, what it must
never touch), not about specific predictions, so a retrain does not break
them. The one exception, `test_resolves_a_plain_first_name`, pins a behaviour
a retrain should keep.
"""
from __future__ import annotations

import pytest

from phonepe_categorizer import categories as C
from phonepe_categorizer import ml
from phonepe_categorizer.engine import REVIEW_THRESHOLD, classify_core

pytest.importorskip("onnxruntime")


@pytest.fixture
def onnx_model():
    if not ml.MODEL_PATH.exists():
        pytest.skip("no shipped model")
    model = ml.Model.load()
    ml.set_default_model(model)
    return model


class _Fixed:
    """A stand-in model that always says the same thing."""

    def __init__(self, category: str, probability: float):
        self.pred = ml.Prediction(category, probability)
        self.calls = 0

    def predict(self, text):
        self.calls += 1
        return self.pred


def _paid(merchant: str, **kw):
    return classify_core(merchant, direction="Paid to", **kw)


class TestFeatures:
    def test_deterministic_and_normalized(self):
        a, b = ml.features("maturchaya kirana"), ml.features("maturchaya kirana")
        assert a == b
        assert sum(v * v for v in a.values()) == pytest.approx(1.0)

    def test_empty_string_has_no_features(self):
        assert ml.features("") == {}

    def test_indices_fit_the_model_input(self):
        assert all(0 <= i < ml.FEATURE_DIM for i in ml.features("st vincent pallotti"))


class TestShippedModel:
    def test_loads_with_matching_feature_spec(self, onnx_model):
        assert C.PERSONAL_TRANSFER in onnx_model.classes
        # Direction-derived answers are never model targets.
        assert C.INCOME not in onnx_model.classes
        assert C.OTHER not in onnx_model.classes

    def test_probability_is_a_probability(self, onnx_model):
        pred = onnx_model.predict("new matruchaya daily and general store")
        assert pred.category in C.VALID
        assert 0.0 <= pred.probability <= 1.0

    def test_resolves_a_plain_first_name(self, onnx_model):
        r = _paid("Raju")
        assert r.source == C.SRC_ML_MODEL
        assert r.category == C.PERSONAL_TRANSFER
        assert r.needs_review is False

    def test_rejects_a_model_with_other_features(self, tmp_path, onnx_model):
        import onnx

        m = onnx.load(str(ml.MODEL_PATH))
        for prop in m.metadata_props:
            if prop.key == "feature_spec":
                prop.value = "something else"
        bad = tmp_path / "bad.onnx"
        onnx.save(m, str(bad))
        with pytest.raises(ValueError, match="retrain"):
            ml.Model.load(bad)


class TestGate:
    def test_confident_answer_is_taken_and_capped(self):
        ml.set_default_model(_Fixed(C.GROCERIES, 0.99))
        r = _paid("Zzq Unrecognizable Enterprises Holdings")
        assert r.category == C.GROCERIES
        assert r.source == C.SRC_ML_MODEL
        assert r.confidence == ml.MAX_CONFIDENCE
        assert r.needs_review is False

    def test_unsure_answer_is_only_a_hint(self):
        ml.set_default_model(_Fixed(C.GROCERIES, REVIEW_THRESHOLD - 0.01))
        r = _paid("Zzq Unrecognizable Enterprises Holdings")
        assert r.category == C.OTHER
        assert r.source == C.SRC_FALLBACK
        assert r.needs_review is True
        assert "model suggests GROCERIES" in r.evidence

    def test_threshold_is_inclusive(self):
        ml.set_default_model(_Fixed(C.HEALTH, REVIEW_THRESHOLD))
        assert _paid("Zzq Unrecognizable Enterprises Holdings").source == C.SRC_ML_MODEL

    def test_commercial_keyword_blocks_a_person_answer(self):
        ml.set_default_model(_Fixed(C.PERSONAL_TRANSFER, 0.99))
        r = _paid("Bharat Traders")
        assert r.source == C.SRC_FALLBACK
        assert r.needs_review is True


class TestNeverAsked:
    """Cases where stage 6 must not even run."""

    @pytest.mark.parametrize("merchant,kw", [
        ("Swiggy", {}),                                   # exact merchant
        ("NEW MATRUCHAYA DAILY AND GENERAL STORE", {}),   # rule
        ("Ritik Pandey", {}),                             # person short-circuit
        ("Unknownium Store", {"overrides": {"unknownium store": C.HEALTH}}),
        ("******6258", {}),                               # masked handle
        ("Akshay Internet Cafe", {}),                     # deliberate abstention
    ])
    def test_outgoing(self, merchant, kw):
        fixed = _Fixed(C.TRAVEL, 0.99)
        ml.set_default_model(fixed)
        r = _paid(merchant, **kw)
        assert r.source != C.SRC_ML_MODEL
        assert fixed.calls == 0

    def test_incoming_rows_are_left_to_txn_type(self):
        fixed = _Fixed(C.TRAVEL, 0.99)
        ml.set_default_model(fixed)
        r = classify_core("Zzq Unrecognizable Holdings", is_credit=True,
                          direction="Received from")
        assert r.source == C.SRC_TXN_TYPE
        assert fixed.calls == 0


class TestAbsent:
    def test_disabled_layer_is_the_old_pipeline(self):
        ml.set_default_model(None)
        r = _paid("Zzq Unrecognizable Enterprises Holdings")
        assert r.category == C.OTHER
        assert r.evidence == "no layer matched"

    def test_env_switch_turns_it_off(self, monkeypatch):
        monkeypatch.setenv(ml.ENV_SWITCH, "off")
        ml.reset_default_model()
        assert ml.default_model() is None


class TestNotLearned:
    def test_model_answers_never_enter_the_directory(self):
        from phonepe_categorizer.db import CategorizerRepository

        ml.set_default_model(_Fixed(C.GROCERIES, 0.99))
        r = _paid("Zzq Unrecognizable Enterprises Holdings")
        assert r.source == C.SRC_ML_MODEL
        assert CategorizerRepository._is_learnable(r) is False
