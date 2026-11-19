"""
Evaluation metrics computation for Multi-Task AI Safety Analysis.
Calculates Accuracy, Precision, Recall, Macro-F1, and Per-Class F1 scores.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from src.utils.config import HEAD_CONFIG


class MultitaskEvaluator:
    """Evaluates multi-head classification predictions against ground truth labels."""

    def __init__(self, head_config: Dict[str, List[str]] = HEAD_CONFIG):
        self.head_config = head_config

    def evaluate_head(
        self,
        y_true: List[int],
        y_pred: List[int],
        head_name: str,
    ) -> Dict[str, Any]:
        """Compute metrics for a single prediction head ignoring masked (-100) labels."""
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        valid_mask = (y_true_arr != -100)
        filtered_true = y_true_arr[valid_mask]
        filtered_pred = y_pred_arr[valid_mask]

        if len(filtered_true) == 0:
            return {"error": "No valid ground truth labels found"}

        classes = self.head_config.get(head_name, [])
        labels_indices = list(range(len(classes)))

        acc = float(accuracy_score(filtered_true, filtered_pred))
        prec_macro = float(precision_score(filtered_true, filtered_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(filtered_true, filtered_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(filtered_true, filtered_pred, average="macro", zero_division=0))

        # Per-class F1
        per_class_f1 = f1_score(
            filtered_true, filtered_pred, labels=labels_indices, average=None, zero_division=0
        )
        per_class_dict = {
            classes[i]: round(float(per_class_f1[i]), 4)
            for i in range(min(len(classes), len(per_class_f1)))
        }

        return {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "per_class_f1": per_class_dict,
            "total_samples": int(len(filtered_true)),
        }

    def evaluate_all(
        self,
        true_dict: Dict[str, List[int]],
        pred_dict: Dict[str, List[int]],
    ) -> Dict[str, Any]:
        """Compute full multi-task evaluation report across all five heads."""
        results = {}
        macro_f1_sum = 0.0
        valid_heads = 0

        for head in self.head_config.keys():
            if head in true_dict and head in pred_dict:
                head_res = self.evaluate_head(true_dict[head], pred_dict[head], head)
                results[head] = head_res
                if "f1_macro" in head_res:
                    macro_f1_sum += head_res["f1_macro"]
                    valid_heads += 1

        overall_macro_f1 = round(macro_f1_sum / valid_heads, 4) if valid_heads > 0 else 0.0
        results["overall_macro_f1"] = overall_macro_f1
        return results
