"""
Enhanced training script for the Multi-Task MuRIL Safety Model.
Improvements over v1:
  - 2-layer MLP classification heads (768 → 256 → N) with ReLU + BatchNorm
  - Class-weighted CrossEntropyLoss to handle severe imbalance
  - Phase 1: Train enhanced heads on frozen encoder features (fast, ~2 min)
  - Phase 2: End-to-end fine-tuning with last 2 encoder layers unfrozen (~30 min on CPU)
  - Proper learning rate scheduling, gradient accumulation, early stopping
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import functools
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel, AutoConfig

from src.data.preprocessing import normalize_text
from src.utils.config import TRAINED_MODEL_DIR, RESULTS_DIR, ModelConfig
from src.utils.helpers import get_device, set_seed, save_json

print = functools.partial(print, flush=True)

# ─── Label mappings ───────────────────────────────────────

INTENT_MAP = {
    "EDUCATIONAL": "Educational", "HISTORICAL": "Educational", "CREATIVE_WRITING": "Educational",
    "SAFETY_INFO": "Analysis", "SUPPORT_SEEKING": "Analysis", "CONSENT_AND_PRIVACY": "Analysis",
    "REPORTING": "Analysis", "OPTIMIZATION": "Analysis", "ACCOUNT_RECOVERY": "Analysis",
    "DEFENSIVE_CYBER": "Prevention", "PRIVACY_PROTECTION": "Prevention", "FRAUD_PREVENTION": "Prevention",
    "INSTRUCTIONAL": "Execution", "HARMFUL_INSTRUCTIONAL": "Execution", "HARMFUL_USE": "Execution",
    "PROCUREMENT": "Execution", "FORGERY": "Execution", "SABOTAGE": "Execution",
    "SCAM": "Execution", "ENCOURAGEMENT": "Execution", "THREATS_AND_EVASION": "Execution",
    "EVASION": "Evasion", "PRIVACY_INVASION": "Evasion", "CREDENTIAL_THEFT": "Evasion",
    "UNAUTHORIZED_ACCESS": "Evasion", "SOCIAL_ENGINEERING": "Evasion", "SURVEILLANCE": "Evasion",
    "IMPERSONATION": "Evasion",
}
RISK_MAP = {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High", "Low": "Low", "Medium": "Medium", "High": "High"}
AMBIGUITY_MAP = {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High", "Low": "Low", "Medium": "Medium", "High": "High"}
CONTEXT_MAP = {"Academic": "Academic", "Historical": "Historical", "Fictional": "Fictional",
               "Preventive": "Preventive", "Real-world": "Real-world"}
ACTIONABILITY_MAP = {"Conceptual": "Conceptual", "General guidance": "General guidance",
                     "Procedural": "Procedural", "Highly actionable": "Highly actionable"}

HEAD_CONFIG = {
    "intent": ["Educational", "Analysis", "Prevention", "Execution", "Evasion"],
    "context": ["Academic", "Historical", "Fictional", "Preventive", "Real-world"],
    "actionability": ["Conceptual", "General guidance", "Procedural", "Highly actionable"],
    "risk_level": ["Low", "Medium", "High"],
    "ambiguity": ["Low", "Medium", "High"],
}
NUM_LABELS = {k: len(v) for k, v in HEAD_CONFIG.items()}
LABEL_MAPS = {
    "intent": INTENT_MAP, "context": CONTEXT_MAP, "actionability": ACTIONABILITY_MAP,
    "risk_level": RISK_MAP, "ambiguity": AMBIGUITY_MAP,
}


# ─── Enhanced Classification Head ─────────────────────────

class EnhancedHead(nn.Module):
    """2-layer MLP head: hidden_size → 256 → num_classes with BN + ReLU + Dropout."""
    def __init__(self, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class EnhancedMultitaskModel(nn.Module):
    """MuRIL encoder + 5 enhanced classification heads."""
    def __init__(self, model_name="google/muril-base-cased"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hs = self.config.hidden_size  # 768

        self.heads = nn.ModuleDict({
            head: EnhancedHead(hs, n_cls) for head, n_cls in NUM_LABELS.items()
        })

    def get_pooled(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.encoder(**kw)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        pooled = self.get_pooled(input_ids, attention_mask, token_type_ids)
        logits = {h: head(pooled) for h, head in self.heads.items()}
        return logits, pooled


# ─── Helpers ──────────────────────────────────────────────

def encode_labels(df):
    """Encode all head columns to integer IDs."""
    result = {}
    for head, mapping in LABEL_MAPS.items():
        l2id = {cls: i for i, cls in enumerate(HEAD_CONFIG[head])}
        ids = []
        for val in df[head]:
            mapped = mapping.get(str(val).strip())
            ids.append(l2id[mapped] if mapped and mapped in l2id else -100)
        result[head] = np.array(ids, dtype=np.int64)
    return result


def compute_class_weights(label_array, num_classes):
    """Inverse-frequency class weights, capped at 10x the minimum weight."""
    counts = Counter(int(v) for v in label_array if v != -100)
    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(total / (num_classes * c))
    # Cap at 10x to avoid extreme weights
    min_w = min(weights)
    weights = [min(w, min_w * 10) for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


def extract_features_batched(model, tokenizer, texts, device, batch_size=32):
    """Extract pooled features from frozen encoder."""
    model.eval()
    all_feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        tids = enc.get("token_type_ids")
        if tids is not None:
            tids = tids.to(device)
        with torch.no_grad():
            pooled = model.get_pooled(ids, mask, tids)
        all_feats.append(pooled.cpu())
        done = min(i + batch_size, len(texts))
        if (done // batch_size) % 20 == 0 or done == len(texts):
            print(f"    {done}/{len(texts)} extracted...")
    return torch.cat(all_feats, dim=0)


def evaluate(model_heads, features, labels, device):
    """Evaluate all heads and return metrics dict."""
    results = {}
    for h in HEAD_CONFIG:
        with torch.no_grad():
            logits = model_heads[h](features.to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()
        true = labels[h]
        mask = true != -100
        p, t = preds[mask], true[mask]
        if len(t) == 0:
            continue
        acc = accuracy_score(t, p)
        f1w = f1_score(t, p, average="weighted", zero_division=0)
        f1m = f1_score(t, p, average="macro", zero_division=0)
        results[h] = {"accuracy": acc, "f1_weighted": f1w, "f1_macro": f1m}
    return results


def print_final_report(model_heads, features, labels, device):
    """Print per-class classification reports for all heads."""
    for h in HEAD_CONFIG:
        with torch.no_grad():
            logits = model_heads[h](features.to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()
        true = labels[h]
        mask = true != -100
        p, t = preds[mask], true[mask]
        if len(t) == 0:
            continue
        acc = accuracy_score(t, p)
        f1w = f1_score(t, p, average="weighted", zero_division=0)
        f1m = f1_score(t, p, average="macro", zero_division=0)
        report = classification_report(t, p, target_names=HEAD_CONFIG[h], zero_division=0)
        print(f"\n┌─── {h.upper()} ───")
        print(f"│ Accuracy: {acc:.4f} | F1-weighted: {f1w:.4f} | F1-macro: {f1m:.4f}")
        print(f"├───────────────────────────────")
        print(report)


# ─── Main ─────────────────────────────────────────────────

def main():
    set_seed(42)
    device = get_device()
    model_name = "google/muril-base-cased"

    # Load data
    print(f"\n{'='*60}")
    print("Loading saved train/test splits...")
    print(f"{'='*60}")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    train_labels = encode_labels(train_df)
    test_labels = encode_labels(test_df)

    train_texts = [normalize_text(str(t)) for t in train_df["text"]]
    test_texts = [normalize_text(str(t)) for t in test_df["text"]]

    # Class weights
    print("\nComputing class weights...")
    class_weights = {}
    for h in HEAD_CONFIG:
        w = compute_class_weights(train_labels[h], NUM_LABELS[h])
        class_weights[h] = w.to(device)
        wstr = ", ".join(f"{HEAD_CONFIG[h][i]}={w[i]:.2f}" for i in range(len(w)))
        print(f"  {h}: {wstr}")

    # Model
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EnhancedMultitaskModel(model_name).to(device)
    print(f"Encoder: {sum(p.numel() for p in model.encoder.parameters())/1e6:.1f}M params")
    print(f"Heads:   {sum(p.numel() for p in model.heads.parameters())/1e3:.1f}K params")

    # ══════════════════════════════════════════════════════
    # PHASE 1: Train enhanced heads on frozen features
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 1: Training enhanced MLP heads (frozen encoder)")
    print(f"{'='*60}")

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    # Extract features
    print("\nExtracting train features...")
    train_feats = extract_features_batched(model, tokenizer, train_texts, device, batch_size=32)
    print(f"  Shape: {train_feats.shape}")
    print("Extracting test features...")
    test_feats = extract_features_batched(model, tokenizer, test_texts, device, batch_size=32)
    print(f"  Shape: {test_feats.shape}")

    train_feats_d = train_feats.to(device)
    test_feats_d = test_feats.to(device)
    train_label_tensors = {h: torch.tensor(v, dtype=torch.long, device=device) for h, v in train_labels.items()}

    # Weighted loss functions
    loss_fns = {
        h: nn.CrossEntropyLoss(weight=class_weights[h], ignore_index=-100)
        for h in HEAD_CONFIG
    }
    head_weights = {"risk_level": 1.5, "intent": 1.2, "context": 1.0, "actionability": 1.0, "ambiguity": 0.8}

    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=3e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)

    n_train = len(train_feats_d)
    batch_size = 256
    best_f1 = 0.0
    patience_counter = 0
    patience = 50

    for epoch in range(500):
        model.heads.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start+batch_size]
            feats = train_feats_d[idx]
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for h, head in model.heads.items():
                logits = head(feats)
                loss = loss_fns[h](logits, train_label_tensors[h][idx])
                total_loss = total_loss + head_weights[h] * loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.heads.eval()
            metrics = evaluate(model.heads, test_feats_d, test_labels, device)
            f1s = [m["f1_weighted"] for m in metrics.values()]
            avg_f1 = np.mean(f1s) if f1s else 0
            mstr = " | ".join(f"{h}:{m['f1_weighted']:.3f}" for h, m in metrics.items())
            print(f"  Ep {epoch+1:3d} | Loss:{avg_loss:.4f} | AvgF1:{avg_f1:.4f} | {mstr}")

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                patience_counter = 0
                TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.heads.state_dict(), TRAINED_MODEL_DIR / "enhanced_heads_phase1.pt")
            else:
                patience_counter += 20
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Reload best phase 1 heads
    model.heads.load_state_dict(
        torch.load(TRAINED_MODEL_DIR / "enhanced_heads_phase1.pt", map_location=device, weights_only=True)
    )
    print(f"\nPhase 1 best Avg F1-weighted: {best_f1:.4f}")
    model.heads.eval()
    print("\n── Phase 1 Test Results ──")
    print_final_report(model.heads, test_feats_d, test_labels, device)

    # ══════════════════════════════════════════════════════
    # PHASE 2: End-to-end fine-tuning (last 2 encoder layers)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: End-to-end fine-tuning (last 2 encoder layers + heads)")
    print(f"{'='*60}")

    # Unfreeze last 2 transformer layers + pooler
    encoder_layers = model.encoder.encoder.layer
    num_layers = len(encoder_layers)
    print(f"Encoder has {num_layers} layers. Unfreezing layers {num_layers-2} to {num_layers-1} + pooler.")

    for i in range(num_layers - 2, num_layers):
        for p in encoder_layers[i].parameters():
            p.requires_grad = True
    if hasattr(model.encoder, "pooler") and model.encoder.pooler is not None:
        for p in model.encoder.pooler.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M total ({trainable/total*100:.1f}%)")

    # Optimizer with differential LR
    encoder_params = []
    for i in range(num_layers - 2, num_layers):
        encoder_params.extend(encoder_layers[i].parameters())
    if hasattr(model.encoder, "pooler") and model.encoder.pooler is not None:
        encoder_params.extend(model.encoder.pooler.parameters())

    optimizer2 = torch.optim.AdamW([
        {"params": encoder_params, "lr": 2e-5, "weight_decay": 0.01},
        {"params": model.heads.parameters(), "lr": 5e-4, "weight_decay": 0.01},
    ])

    # DataLoader-style iteration with tokenizer
    num_epochs = 3
    batch_size2 = 8
    grad_accum = 4  # effective batch = 32
    total_steps = (len(train_texts) // (batch_size2 * grad_accum) + 1) * num_epochs
    warmup = int(0.1 * total_steps)

    from transformers import get_linear_schedule_with_warmup
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, warmup, total_steps)

    best_f1_p2 = best_f1  # start from phase 1 best
    training_log = []

    for epoch in range(num_epochs):
        model.train()
        # Shuffle indices
        perm = np.random.permutation(len(train_texts))
        epoch_loss = 0.0
        n_steps = 0
        optimizer2.zero_grad()
        t0 = time.time()

        for batch_start in range(0, len(train_texts), batch_size2):
            idx = perm[batch_start:batch_start + batch_size2]
            batch_texts = [train_texts[i] for i in idx]

            enc = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            tids = enc.get("token_type_ids")
            if tids is not None:
                tids = tids.to(device)

            logits_dict, _ = model(input_ids, attn_mask, tids)

            loss = torch.tensor(0.0, device=device)
            for h in HEAD_CONFIG:
                targets = torch.tensor(train_labels[h][idx], dtype=torch.long, device=device)
                loss = loss + head_weights[h] * loss_fns[h](logits_dict[h], targets)
            loss = loss / grad_accum
            loss.backward()

            epoch_loss += loss.item() * grad_accum
            n_steps += 1

            if n_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer2.step()
                scheduler2.step()
                optimizer2.zero_grad()

            if n_steps % 100 == 0:
                elapsed = time.time() - t0
                avg_l = epoch_loss / n_steps
                rate = n_steps / elapsed
                eta = (len(train_texts) // batch_size2 - n_steps) / max(rate, 0.01) / 60
                print(f"    Epoch {epoch+1} | Step {n_steps}/{len(train_texts)//batch_size2} | Loss:{avg_l:.4f} | {rate:.1f} it/s | ETA:{eta:.0f}min")

        # Flush remaining gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer2.step()
        optimizer2.zero_grad()

        elapsed = time.time() - t0
        avg_epoch_loss = epoch_loss / max(n_steps, 1)
        print(f"\n  Epoch {epoch+1}/{num_epochs} done in {elapsed/60:.1f}min | Avg Loss: {avg_epoch_loss:.4f}")

        # Evaluate
        model.eval()
        # Re-extract test features with updated encoder
        print("  Re-extracting test features with fine-tuned encoder...")
        test_feats_ft = extract_features_batched(model, tokenizer, test_texts, device, batch_size=32)
        test_feats_ft_d = test_feats_ft.to(device)

        metrics = evaluate(model.heads, test_feats_ft_d, test_labels, device)
        f1s = [m["f1_weighted"] for m in metrics.values()]
        avg_f1 = np.mean(f1s) if f1s else 0
        mstr = " | ".join(f"{h}:{m['f1_weighted']:.3f}" for h, m in metrics.items())
        print(f"  TEST | AvgF1:{avg_f1:.4f} | {mstr}")

        training_log.append({"epoch": epoch+1, "loss": avg_epoch_loss, "avg_f1": avg_f1, "heads": {
            h: {k: float(v) for k, v in m.items()} for h, m in metrics.items()
        }})

        if avg_f1 > best_f1_p2:
            best_f1_p2 = avg_f1
            TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_f1": best_f1_p2,
                "head_config": HEAD_CONFIG,
                "num_labels": NUM_LABELS,
                "model_name": model_name,
            }, TRAINED_MODEL_DIR / "multitask_full_model.pt")
            torch.save(model.heads.state_dict(), TRAINED_MODEL_DIR / "multitask_heads.pt")
            print(f"  ✓ New best model! F1={best_f1_p2:.4f}")

    # ── Final evaluation ──
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (Best Model)")
    print(f"{'='*60}")

    if best_f1_p2 > best_f1 and (TRAINED_MODEL_DIR / "multitask_full_model.pt").exists():
        best_ckpt = torch.load(TRAINED_MODEL_DIR / "multitask_full_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"Loaded Phase 2 end-to-end best model (F1={best_f1_p2:.4f})")
    else:
        heads_state = torch.load(TRAINED_MODEL_DIR / "enhanced_heads_phase1.pt", map_location=device, weights_only=False)
        model.heads.load_state_dict(heads_state)
        print(f"Loaded Phase 1 enhanced heads best model (F1={best_f1:.4f})")
    model.eval()

    final_test_feats = extract_features_batched(model, tokenizer, test_texts, device, batch_size=32)
    final_test_feats_d = final_test_feats.to(device)

    print_final_report(model.heads, final_test_feats_d, test_labels, device)

    final_metrics = evaluate(model.heads, final_test_feats_d, test_labels, device)
    avg_acc = np.mean([m["accuracy"] for m in final_metrics.values()])
    avg_f1w = np.mean([m["f1_weighted"] for m in final_metrics.values()])
    avg_f1m = np.mean([m["f1_macro"] for m in final_metrics.values()])

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json({
        "phase1_best_f1": float(best_f1),
        "phase2_best_f1": float(best_f1_p2),
        "final_metrics": {h: {k: float(v) for k, v in m.items()} for h, m in final_metrics.items()},
        "avg_accuracy": float(avg_acc),
        "avg_f1_weighted": float(avg_f1w),
        "avg_f1_macro": float(avg_f1m),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "training_log": training_log,
    }, RESULTS_DIR / "evaluation_results.json")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Phase 1 (frozen encoder + MLP heads): F1={best_f1:.4f}")
    print(f"Phase 2 (fine-tuned last 2 layers):   F1={best_f1_p2:.4f}")
    print(f"\n{'Head':<20s} {'Accuracy':>10s} {'F1-weighted':>12s} {'F1-macro':>10s}")
    print(f"{'─'*52}")
    for h, m in final_metrics.items():
        print(f"{h:<20s} {m['accuracy']:>10.4f} {m['f1_weighted']:>12.4f} {m['f1_macro']:>10.4f}")
    print(f"{'─'*52}")
    print(f"{'AVERAGE':<20s} {avg_acc:>10.4f} {avg_f1w:>12.4f} {avg_f1m:>10.4f}")
    print(f"\nModel: {TRAINED_MODEL_DIR / 'multitask_full_model.pt'}")
    print(f"Results: {RESULTS_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
