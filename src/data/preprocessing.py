"""
Data preprocessing, label encoding, group-aware splitting,
and PyTorch dataset implementation for Multi-Task Multilingual AI Safety.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.utils.config import HEAD_CONFIG, NUM_LABELS


def normalize_text(text: str) -> str:
    """Normalize whitespace and Unicode text across Indian scripts and English."""
    if not isinstance(text, str):
        return ""
    # NFKC Unicode normalization for unified representation of diacritics
    text = unicodedata.normalize("NFKC", text)
    # Collapse multiple whitespaces and trim
    text = re.sub(r"\s+", " ", text).strip()
    return text


class MultiTaskLabelEncoder:
    """
    Encodes categorical labels into integer indices for multi-task heads.
    Uses -100 as the ignore_index for missing/unlabeled targets in PyTorch.
    """

    def __init__(self, head_config: Dict[str, List[str]] = HEAD_CONFIG):
        self.head_config = head_config
        self.label_to_id: Dict[str, Dict[str, int]] = {}
        self.id_to_label: Dict[str, Dict[int, str]] = {}

        for head, classes in self.head_config.items():
            self.label_to_id[head] = {cls_name: i for i, cls_name in enumerate(classes)}
            self.id_to_label[head] = {i: cls_name for i, cls_name in enumerate(classes)}

    def encode_value(self, head: str, value: Optional[str]) -> int:
        """Encode a single label string to integer index or -100 if missing/invalid."""
        if value is None or pd.isna(value):
            return -100
        val_str = str(value).strip()
        return self.label_to_id.get(head, {}).get(val_str, -100)

    def decode_index(self, head: str, idx: int) -> str:
        """Decode an integer prediction index back to human-readable label."""
        return self.id_to_label.get(head, {}).get(idx, "Unknown")

    def encode_dataframe(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Encode all head columns from a pandas DataFrame into integer arrays."""
        encoded: Dict[str, np.ndarray] = {}
        for head in self.head_config.keys():
            if head in df.columns:
                encoded[head] = np.array(
                    [self.encode_value(head, val) for val in df[head]],
                    dtype=np.int64,
                )
            else:
                encoded[head] = np.full(len(df), -100, dtype=np.int64)
        return encoded


def group_aware_split(
    df: pd.DataFrame,
    group_col: str = "semantic_group",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset strictly at the semantic_group level to avoid data leakage
    across cross-lingual paraphrases and linguistic variations.
    Default distribution: 70% Train, 15% Validation, 15% Test.
    """
    if group_col not in df.columns or df[group_col].nunique() < 3:
        # Fallback to random splitting if semantic_group is unavailable or too few
        shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            shuffled.iloc[:n_train],
            shuffled.iloc[n_train : n_train + n_val],
            shuffled.iloc[n_train + n_val :],
        )

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df, groups=df[group_col]))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)

    # Second split: Val vs Test
    test_sub_ratio = test_ratio / temp_ratio
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_sub_ratio, random_state=random_state)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_col]))

    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)

    return train_df, val_df, test_df


class MultitaskSafetyDataset(Dataset):
    """
    PyTorch Dataset for multi-task multilingual AI safety prompt analysis.
    Yields input_ids, attention_mask, and multi-head target labels.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[Dict[str, Union[np.ndarray, List[int]]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: int = 128,
    ):
        self.texts = [normalize_text(t) for t in texts]
        self.labels = labels or {}
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
            if "token_type_ids" in encoding:
                item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        else:
            item = {"text": text}

        # Attach multi-head targets
        target_dict = {}
        for head in HEAD_CONFIG.keys():
            if head in self.labels and len(self.labels[head]) > idx:
                target_dict[head] = torch.tensor(self.labels[head][idx], dtype=torch.long)
            else:
                target_dict[head] = torch.tensor(-100, dtype=torch.long)

        item["labels"] = target_dict
        return item
