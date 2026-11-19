"""
Unit tests for data preprocessing, normalization, encoding, and group-aware splitting.
"""

import pandas as pd
import pytest

from src.data.loader import generate_benchmark_dataset
from src.data.preprocessing import (
    MultiTaskLabelEncoder,
    group_aware_split,
    normalize_text,
)
from src.utils.config import HEAD_CONFIG


def test_normalize_text():
    raw = "  ନମସ୍କାର   ସବୁଜ   ଉଦ୍ଭିଦ  \n\n "
    normalized = normalize_text(raw)
    assert "  " not in normalized
    assert normalized.startswith("ନମସ୍କାର")


def test_label_encoder():
    encoder = MultiTaskLabelEncoder()

    # Test encoding valid values
    risk_id = encoder.encode_value("risk_level", "High")
    assert risk_id == 2
    assert encoder.decode_index("risk_level", risk_id) == "High"

    intent_id = encoder.encode_value("intent", "Educational")
    assert intent_id == 0
    assert encoder.decode_index("intent", intent_id) == "Educational"

    # Test missing / None value handles -100
    assert encoder.encode_value("intent", None) == -100
    assert encoder.encode_value("invalid_head", "foo") == -100


def test_group_aware_split_no_leakage():
    df = generate_benchmark_dataset()
    train_df, val_df, test_df = group_aware_split(df, group_col="semantic_group")

    train_groups = set(train_df["semantic_group"])
    val_groups = set(val_df["semantic_group"])
    test_groups = set(test_df["semantic_group"])

    # Assert mutual exclusivity: zero group leakage
    assert train_groups.isdisjoint(val_groups), "Train and Val share semantic groups!"
    assert train_groups.isdisjoint(test_groups), "Train and Test share semantic groups!"
    assert val_groups.isdisjoint(test_groups), "Val and Test share semantic groups!"

    total_split_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_split_rows == len(df)
