"""
Unit tests for the end-to-end inference engine and explanation generation.
"""

import pytest
from src.models.inference import SafetyInferenceEngine
from src.utils.helpers import explain_prediction


def test_explain_prediction():
    explanation = explain_prediction(
        intent="Educational",
        context="Academic",
        actionability="Conceptual",
        risk_level="Low",
        ambiguity="Low",
    )
    assert "Low risk" in explanation
    assert "Educational" in explanation
    assert "Academic" in explanation


def test_inference_engine_lazy_or_keyword():
    # Test normalization and empty prompt handling
    engine = SafetyInferenceEngine(model_name_or_path="google/muril-base-cased")
    res_empty = engine.analyze_prompt("")
    assert "error" in res_empty

    # Test valid Odia text analysis
    res_odia = engine.analyze_prompt("ସବୁଜ ଉଦ୍ଭିଦରେ ଆଲୋକଶ୍ଳେଷଣ ପ୍ରକ୍ରିୟା")
    assert "predictions" in res_odia
    assert "confidences" in res_odia
    assert "explanation" in res_odia
    assert res_odia["latency_ms"] > 0
