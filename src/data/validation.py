"""
Dataset validation module for AI Safety Analysis System.
Validates schemas, category values, missing values, duplicates, and language coverage.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
from src.utils.config import DATASET_COLUMNS, HEAD_CONFIG, SUPPORTED_LANGUAGES


class DatasetValidator:
    """Validates AI Safety prompt datasets against research specification."""

    def __init__(self, expected_columns: List[str] = DATASET_COLUMNS):
        self.expected_columns = expected_columns
        self.head_config = HEAD_CONFIG

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check if all required columns are present in the dataframe."""
        missing = [col for col in self.expected_columns if col not in df.columns]
        is_valid = len(missing) == 0
        return is_valid, missing

    def validate_labels(self, df: pd.DataFrame) -> Dict[str, List[Any]]:
        """Identify invalid label values across all multi-task heads."""
        invalid_labels: Dict[str, List[Any]] = {}
        for head, valid_values in self.head_config.items():
            if head in df.columns:
                unique_vals = df[head].dropna().unique()
                invalid = [val for val in unique_vals if val not in valid_values]
                if invalid:
                    invalid_labels[head] = invalid
        return invalid_labels

    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive quality metrics on the dataset."""
        schema_valid, missing_cols = self.validate_schema(df)
        invalid_labels = self.validate_labels(df)

        total_rows = len(df)
        duplicate_texts = int(df["text"].duplicated().sum()) if "text" in df.columns else 0
        missing_values_per_col = {col: int(df[col].isna().sum()) for col in df.columns}
        
        language_dist = df["language"].value_counts().to_dict() if "language" in df.columns else {}
        semantic_groups_count = int(df["semantic_group"].nunique()) if "semantic_group" in df.columns else 0

        # Assess balance across supported languages
        language_balance = {}
        if "language" in df.columns:
            for lang_code in SUPPORTED_LANGUAGES.keys():
                count = int((df["language"] == lang_code).sum())
                pct = round((count / total_rows * 100), 2) if total_rows > 0 else 0
                language_balance[lang_code] = {"count": count, "percentage": pct}

        return {
            "total_rows": total_rows,
            "schema_valid": schema_valid,
            "missing_columns": missing_cols,
            "invalid_labels": invalid_labels,
            "duplicate_texts": duplicate_texts,
            "missing_values": missing_values_per_col,
            "semantic_groups_count": semantic_groups_count,
            "language_distribution": language_dist,
            "language_balance": language_balance,
            "is_ready_for_training": (
                schema_valid and len(invalid_labels) == 0 and total_rows > 0
            ),
        }
