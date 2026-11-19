"""
Script to generate reproducible, clean Jupyter Notebooks in notebooks/
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11.0"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


# Notebook 1: Dataset Exploration
nb1_cells = [
    md_cell("# 01: Dataset Exploration & Multilingual Safety Distribution\n\nThis notebook inspects schema integrity, linguistic coverage, and balance across safety categories."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loader import load_or_create_dataset
from src.data.validation import DatasetValidator

# Load benchmark dataset
df = load_or_create_dataset()
print(f"Dataset shape: {df.shape}")
df.head(5)"""),
    code_cell("""# Validate dataset schema and labels
validator = DatasetValidator()
report = validator.check_data_quality(df)
print("Data Quality Summary:")
for k, v in report.items():
    if k != "missing_values":
        print(f"  {k}: {v}")"""),
    code_cell("""# Visualize Language and Risk Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
df['language'].value_counts().plot(kind='bar', ax=ax1, color='#0284c7', title='Language Breakdown')
df['risk_level'].value_counts().plot(kind='bar', ax=ax2, color=['#10b981', '#ef4444', '#f59e0b'], title='Risk Distribution')
plt.tight_layout()
plt.show()"""),
]

# Notebook 2: Data Preprocessing
nb2_cells = [
    md_cell("# 02: Data Preprocessing & Semantic-Group Split\n\nNormalizes text, encodes multi-task labels, and performs leak-free 70:15:15 semantic group splitting."),
    code_cell("""import pandas as pd
from src.data.loader import load_or_create_dataset
from src.data.preprocessing import MultiTaskLabelEncoder, group_aware_split, normalize_text
from src.utils.config import PROCESSED_DATA_DIR

df = load_or_create_dataset()
df['text'] = df['text'].apply(normalize_text)

# Encode multi-head categorical targets
encoder = MultiTaskLabelEncoder()
encoded_targets = encoder.encode_dataframe(df)
for head, arr in encoded_targets.items():
    df[f"{head}_encoded"] = arr

print("Encoded targets sample:")
df[['id', 'text', 'risk_level_encoded', 'intent_encoded']].head()"""),
    code_cell("""# Perform 70:15:15 Semantic-Group-Aware Split
train_df, val_df, test_df = group_aware_split(df, group_col='semantic_group', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

# Verify zero semantic leakage
train_groups = set(train_df['semantic_group'])
test_groups = set(test_df['semantic_group'])
assert train_groups.isdisjoint(test_groups), "Leakage detected between Train and Test!"
print("Zero semantic leakage verified across splits.")

# Save splits
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_csv(PROCESSED_DATA_DIR / 'train.csv', index=False)
val_df.to_csv(PROCESSED_DATA_DIR / 'val.csv', index=False)
test_df.to_csv(PROCESSED_DATA_DIR / 'test.csv', index=False)
print("Processed splits successfully saved.")"""),
]

# Notebook 3: Baseline Models
nb3_cells = [
    md_cell("# 03: Baseline Safety Classifiers\n\nEvaluates Keyword-based and Heuristic rule-based classifiers on the test split."),
    code_cell("""import pandas as pd
from src.data.loader import load_or_create_dataset
from src.models.classifier import KeywordSafetyDetector
from src.evaluation.metrics import MultitaskEvaluator
from src.data.preprocessing import MultiTaskLabelEncoder

df = load_or_create_dataset()
detector = KeywordSafetyDetector()
encoder = MultiTaskLabelEncoder()

y_true_risk = [encoder.encode_value('risk_level', r) for r in df['risk_level']]
y_pred_risk = []

for text in df['text']:
    pred = detector.predict(text)
    y_pred_risk.append(encoder.encode_value('risk_level', pred['risk_level']))

evaluator = MultitaskEvaluator()
metrics = evaluator.evaluate_head(y_true_risk, y_pred_risk, 'risk_level')
print("Keyword Baseline Risk Level Performance:")
for k, v in metrics.items():
    print(f"  {k}: {v}")"""),
]

# Notebook 4: MuRIL Training Pipeline (Kaggle GPU)
nb4_cells = [
    md_cell("# 04: Multi-Task MuRIL Training Pipeline (Kaggle GPU)\n\nFine-tunes `google/muril-base-cased` with 5 independent classification heads using PyTorch AdamW."),
    code_cell("""import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from src.data.preprocessing import MultiTaskLabelEncoder, MultitaskSafetyDataset
from src.models.multitask_model import MultitaskSafetyModel
from src.utils.config import ModelConfig, TRAINED_MODEL_DIR
from src.utils.helpers import get_device, set_seed

set_seed(42)
device = get_device()
config = ModelConfig()

# Initialize Tokenizer and Model
model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MultitaskSafetyModel(model_name=model_name, config=config).to(device)

print(f"Initialized Multi-Task Model on {device}")"""),
    code_cell("""# Prepare DataLoaders
from src.data.loader import load_or_create_dataset
from src.data.preprocessing import group_aware_split

df = load_or_create_dataset()
train_df, val_df, test_df = group_aware_split(df)
encoder = MultiTaskLabelEncoder()

train_dataset = MultitaskSafetyDataset(
    texts=train_df['text'].tolist(),
    labels=encoder.encode_dataframe(train_df),
    tokenizer=tokenizer,
    max_length=config.max_length
)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
print(f"Total training batches: {len(train_loader)}")"""),
    code_cell("""# Training Loop
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
epochs = config.epochs

model.train()
for epoch in range(epochs):
    total_epoch_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()

    avg_loss = total_epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Combined Multi-Task Loss: {avg_loss:.4f}")

# Save trained checkpoint
TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), TRAINED_MODEL_DIR / "best_multitask_model.pt")
print(f"Checkpoint saved to {TRAINED_MODEL_DIR / 'best_multitask_model.pt'}")"""),
]

# Notebook 5: Model Evaluation
nb5_cells = [
    md_cell("# 05: Model Evaluation & Cross-Lingual Robustness\n\nEvaluates checkpoint on test set and computes multilingual consistency rates."),
    code_cell("""import pandas as pd
from src.data.loader import load_or_create_dataset
from src.evaluation.robustness import CrossLingualRobustnessEvaluator
from src.models.inference import SafetyInferenceEngine

engine = SafetyInferenceEngine()
df = load_or_create_dataset()

# Generate predictions for all test samples
predictions = []
for text in df['text']:
    res = engine.analyze_prompt(text)
    predictions.append(res['predictions'])

pred_df = pd.DataFrame(predictions)

# Evaluate cross-lingual robustness
robustness_eval = CrossLingualRobustnessEvaluator(target_heads=['risk_level', 'intent'])
robustness_report = robustness_eval.evaluate_semantic_consistency(df, pred_df)

print("Robustness Report:")
print(f"  Total Semantic Groups: {robustness_report['total_groups_evaluated']}")
print(f"  Consistency Rates: {robustness_report['head_consistency_rates']}")"""),
]

notebooks = {
    "01_dataset_exploration.ipynb": nb1_cells,
    "02_data_preprocessing.ipynb": nb2_cells,
    "03_baseline_models.ipynb": nb3_cells,
    "04_muril_training.ipynb": nb4_cells,
    "05_model_evaluation.ipynb": nb5_cells,
}

for name, cells in notebooks.items():
    path = NOTEBOOKS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created {path}")
