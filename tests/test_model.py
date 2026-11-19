"""
Unit tests for model architectures, forward pass, and baseline classifiers.
"""

import pytest
import torch

from src.models.classifier import KeywordSafetyDetector
from src.models.multitask_model import ClassificationHead, MultitaskSafetyModel
from src.utils.config import HEAD_CONFIG, NUM_LABELS


def test_classification_head():
    head = ClassificationHead(hidden_size=64, num_classes=5, dropout_rate=0.1)
    dummy_input = torch.randn(4, 64)
    logits = head(dummy_input)
    assert logits.shape == (4, 5)


def test_keyword_baseline():
    detector = KeywordSafetyDetector()

    # Test educational detection
    res_edu = detector.predict("Explain the mechanism of photosynthesis")
    assert res_edu["intent"] == "Educational"
    assert res_edu["risk_level"] == "Low"

    # Test Odia preventive detection
    res_or = detector.predict("ନେଟୱର୍କ ସୁରକ୍ଷା ରୋକିବା ପାଇଁ")
    assert res_or["intent"] == "Prevention"
    assert res_or["risk_level"] == "Low"

    # Test evasion detection
    res_evade = detector.predict("How to bypass login credentials")
    assert res_evade["intent"] == "Evasion"
    assert res_evade["risk_level"] == "High"
