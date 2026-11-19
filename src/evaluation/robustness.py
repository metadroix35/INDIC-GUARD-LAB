"""
Cross-lingual robustness and semantic consistency evaluation.
Measures whether identical semantic intents receive consistent classifications
across English, Odia, Odia-English code-mixed, Romanized Odia, and Hindi.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np


class CrossLingualRobustnessEvaluator:
    """Evaluates prediction invariance across language translations and code-mixing."""

    def __init__(self, target_heads: List[str] = ["risk_level", "intent"]):
        self.target_heads = target_heads

    def evaluate_semantic_consistency(
        self,
        df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        group_col: str = "semantic_group",
        lang_col: str = "language",
    ) -> Dict[str, Any]:
        """
        Evaluate pair-wise prediction consistency across languages for identical semantic groups.
        """
        merged = df.copy()
        for head in self.target_heads:
            pred_col = f"pred_{head}"
            if pred_col in predictions_df.columns:
                merged[pred_col] = predictions_df[pred_col]
            elif head in predictions_df.columns:
                merged[pred_col] = predictions_df[head]

        groups = merged.groupby(group_col)
        consistency_results: Dict[str, Any] = {
            "total_groups_evaluated": 0,
            "head_consistency_rates": {},
            "language_pair_agreements": {},
            "divergent_cases": [],
        }

        group_count = 0
        head_matches = {head: [] for head in self.target_heads}
        divergent_cases = []

        for group_id, group_data in groups:
            if len(group_data) < 2:
                continue
            group_count += 1

            # Check if all variants in this group agreed
            for head in self.target_heads:
                pred_col = f"pred_{head}"
                if pred_col in group_data.columns:
                    preds = group_data[pred_col].tolist()
                    is_consistent = len(set(preds)) == 1
                    head_matches[head].append(1 if is_consistent else 0)

                    if not is_consistent:
                        # Log divergent case
                        divergent_cases.append(
                            {
                                "semantic_group": group_id,
                                "head": head,
                                "variants": group_data[
                                    ["id", "text", lang_col, pred_col]
                                ].to_dict(orient="records"),
                            }
                        )

        consistency_results["total_groups_evaluated"] = group_count

        for head, matches in head_matches.items():
            rate = float(np.mean(matches)) if matches else 1.0
            consistency_results["head_consistency_rates"][head] = round(rate, 4)

        consistency_results["divergent_cases"] = divergent_cases[:10]  # Cap top 10 examples
        return consistency_results
