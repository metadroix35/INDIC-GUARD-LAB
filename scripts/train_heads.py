"""
Script to calibrate and save multi-task classification heads for MuRIL / XLM-R.
Produces models/trained/multitask_heads.pt so that trained weights are committed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.data.loader import load_or_create_dataset
from src.data.preprocessing import MultiTaskLabelEncoder, normalize_text
from src.models.multitask_model import MultitaskSafetyModel
from src.utils.config import HEAD_CONFIG, TRAINED_MODEL_DIR, ModelConfig
from src.utils.helpers import get_device, set_seed

def main():
    set_seed(42)
    device = get_device()
    config = ModelConfig()

    print("Loading benchmark dataset...")
    df = load_or_create_dataset()
    texts = [normalize_text(t) for t in df["text"]]

    encoder = MultiTaskLabelEncoder()
    encoded_labels = encoder.encode_dataframe(df)

    print("Initializing Multi-Task Model...")
    model = MultitaskSafetyModel(model_name="google/muril-base-cased", config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

    # Extract pooled embeddings for dataset
    print(f"Tokenizing {len(texts)} samples...")
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    token_type_ids = encodings.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    print("Extracting contextual embeddings from MuRIL backbone...")
    model.encoder.eval()
    with torch.no_grad():
        features = model.extract_pooled_features(input_ids, attention_mask, token_type_ids)

    # Train only the 5 classification heads
    print("Training 5 classification heads with AdamW...")
    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    target_tensors = {
        k: torch.tensor(v, dtype=torch.long, device=device) for k, v in encoded_labels.items()
    }

    model.heads.train()
    for step in range(60):
        optimizer.zero_grad()
        step_loss = torch.tensor(0.0, device=device)
        for head_name, head_mod in model.heads.items():
            logits = head_mod(features)
            loss = loss_fn(logits, target_tensors[head_name])
            weight = config.head_weights.get(head_name, 1.0)
            step_loss += weight * loss

        step_loss.backward()
        optimizer.step()

        if (step + 1) % 15 == 0:
            print(f"Step {step + 1}/60 - Multi-Task Loss: {step_loss.item():.4f}")

    # Save trained heads checkpoint
    TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    heads_path = TRAINED_MODEL_DIR / "multitask_heads.pt"
    torch.save(model.heads.state_dict(), heads_path)
    print(f"Successfully saved calibrated heads to {heads_path} ({heads_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
