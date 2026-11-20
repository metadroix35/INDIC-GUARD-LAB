"""
Evaluate the trained Multi-Task Model on the test set and save final evaluation results.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModel, AutoConfig

from src.data.preprocessing import normalize_text
from src.utils.config import TRAINED_MODEL_DIR, RESULTS_DIR, HEAD_CONFIG, NUM_LABELS
from src.utils.helpers import get_device, set_seed, save_json
from scripts.train_on_dataset import (
    EnhancedMultitaskModel,
    encode_labels,
    extract_features_batched,
    LABEL_MAPS,
)


def main():
    set_seed(42)
    device = get_device()
    model_name = "google/muril-base-cased"

    print("Loading test dataset...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    test_labels = encode_labels(test_df)
    test_texts = [normalize_text(str(t)) for t in test_df["text"]]

    print(f"Loading base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EnhancedMultitaskModel(model_name)

    heads_path = TRAINED_MODEL_DIR / "multitask_heads.pt"
    print(f"Loading trained heads from: {heads_path}")
    heads_state = torch.load(heads_path, map_location=device, weights_only=False)
    model.heads.load_state_dict(heads_state)
    model.to(device)
    model.eval()

    print("Extracting test features...")
    features = extract_features_batched(model, tokenizer, test_texts, device, batch_size=32)
    features_d = features.to(device)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION REPORT ON TEST SET (1,914 samples)")
    print("=" * 60)

    final_metrics = {}
    reports = {}
    cm_dir = RESULTS_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for h in HEAD_CONFIG:
        with torch.no_grad():
            logits = model.heads[h](features_d)
        preds = logits.argmax(dim=-1).cpu().numpy()
        true = test_labels[h]
        mask = true != -100
        p, t = preds[mask], true[mask]

        acc = accuracy_score(t, p)
        f1w = f1_score(t, p, average="weighted", zero_division=0)
        f1m = f1_score(t, p, average="macro", zero_division=0)
        final_metrics[h] = {"accuracy": float(acc), "f1_weighted": float(f1w), "f1_macro": float(f1m)}

        rep = classification_report(t, p, target_names=HEAD_CONFIG[h], output_dict=True, zero_division=0)
        reports[h] = rep

        rep_str = classification_report(t, p, target_names=HEAD_CONFIG[h], zero_division=0)
        print(f"\n┌─── {h.upper()} ───")
        print(f"│ Accuracy: {acc:.4f} | F1-weighted: {f1w:.4f} | F1-macro: {f1m:.4f}")
        print(f"├───────────────────────────────")
        print(rep_str)

        # Save confusion matrix as CSV
        cm = confusion_matrix(t, p, labels=list(range(len(HEAD_CONFIG[h]))))
        cm_df = pd.DataFrame(cm, index=HEAD_CONFIG[h], columns=HEAD_CONFIG[h])
        cm_df.to_csv(cm_dir / f"{h}_confusion_matrix.csv")

    avg_acc = float(np.mean([m["accuracy"] for m in final_metrics.values()]))
    avg_f1w = float(np.mean([m["f1_weighted"] for m in final_metrics.values()]))
    avg_f1m = float(np.mean([m["f1_macro"] for m in final_metrics.values()]))

    print(f"\n{'='*60}")
    print(f"OVERALL AVERAGE: Accuracy={avg_acc:.4f} | F1-weighted={avg_f1w:.4f} | F1-macro={avg_f1m:.4f}")
    print(f"{'='*60}")

    results_data = {
        "final_metrics": final_metrics,
        "avg_accuracy": avg_acc,
        "avg_f1_weighted": avg_f1w,
        "avg_f1_macro": avg_f1m,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "classification_reports": reports,
    }

    save_json(results_data, RESULTS_DIR / "evaluation_results.json")
    print(f"Saved evaluation metrics to: {RESULTS_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
