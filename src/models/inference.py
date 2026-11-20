"""
Inference engine for Multilingual AI Safety Analysis.
Performs tokenization, forward pass, semantic safety validation,
confidence extraction, and comprehensive safety explanation.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer

from src.data.preprocessing import MultiTaskLabelEncoder, normalize_text
from src.models.classifier import SemanticSafetyClassifier
from src.models.multitask_model import MultitaskSafetyModel
from src.models.risk_scorer import DynamicHarmScorer
from src.utils.config import (
    CHECKPOINTS_DIR,
    HEAD_CONFIG,
    TRAINED_MODEL_DIR,
    ModelConfig,
)
from src.utils.helpers import explain_prediction, get_device, logger


class SafetyInferenceEngine:
    """
    Production-grade inference pipeline supporting MuRIL / XLM-R multi-task models.
    Combines transformer contextual representations with semantic safety alignment
    to ensure high-risk queries (e.g. violence, evasion, attacks) and low-risk
    educational queries are classified with high fidelity and calibrated confidence.
    """

    def __init__(
        self,
        model_name_or_path: str = "google/muril-base-cased",
        weights_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.model_name = model_name_or_path
        self.label_encoder = MultiTaskLabelEncoder()
        self.semantic_classifier = SemanticSafetyClassifier()
        self.harm_scorer = DynamicHarmScorer()

        logger.info(f"Loading tokenizer for: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = MultitaskSafetyModel(model_name=model_name_or_path)
        self.is_fine_tuned = False

        # Attempt to load custom fine-tuned weights in priority order
        checkpoint_candidates = [
            TRAINED_MODEL_DIR / "multitask_full_model.pt",
            TRAINED_MODEL_DIR / "best_multitask_model.pt",
            TRAINED_MODEL_DIR / "enhanced_heads_phase1.pt",
            TRAINED_MODEL_DIR / "multitask_heads.pt",
        ]
        if weights_path:
            checkpoint_candidates.insert(0, Path(weights_path))

        for ckpt_path in checkpoint_candidates:
            if ckpt_path.exists():
                try:
                    self.model.load_weights_flexible(ckpt_path)
                    self.is_fine_tuned = True
                    logger.info(f"Loaded model weights from {ckpt_path}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load checkpoint from {ckpt_path}: {e}")

        self.model.to(self.device)
        self.model.eval()

    def analyze_prompt(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze user-submitted text and predict:
        - Intent
        - Context
        - Actionability
        - Risk Level
        - Ambiguity
        With confidence distributions and safety explanation.
        """
        clean_text = normalize_text(text)
        if not clean_text:
            return {
                "error": "Input text is empty.",
                "is_valid": False,
            }

        start_time = time.time()

        # Tokenize input and run transformer forward pass
        encoding = self.tokenizer(
            clean_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        token_type_ids = None
        if "token_type_ids" in encoding:
            token_type_ids = encoding["token_type_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Run semantic safety classifier and dynamic harm scorer
        semantic_res = self.semantic_classifier.predict(clean_text)
        risk_metrics = self.harm_scorer.calculate_risk(clean_text)

        predictions: Dict[str, str] = {}
        confidences: Dict[str, float] = {}
        all_probabilities: Dict[str, Dict[str, float]] = {}

        probs_dict = outputs["probabilities"]

        # If model has fine-tuned weights, blend with semantic engine;
        # otherwise use semantic engine distributions so predictions are always calibrated.
        if self.is_fine_tuned:
            import numpy as np
            for head, classes in HEAD_CONFIG.items():
                nn_probs = probs_dict[head][0].cpu().numpy()
                sem_probs = np.array([semantic_res["probabilities"][head].get(c, 0.0) for c in classes], dtype=float)

                # Blended probability
                blended = 0.6 * nn_probs + 0.4 * sem_probs

                # Dynamically elevate risk if negative words or jailbreak pretext are present
                if head == "risk_level" and risk_metrics["risk_score"] >= 70.0:
                    blended = np.array([0.01, 0.04, 0.95], dtype=float)
                elif head == "risk_level" and risk_metrics["risk_score"] >= 40.0 and blended[2] < 0.5:
                    blended = np.array([0.15, 0.70, 0.15], dtype=float)
                elif head == "context" and risk_metrics.get("is_jailbreak_concealment"):
                    # Fictional is index 2 in CONTEXT_CLASSES: ["Academic", "Historical", "Fictional", "Preventive", "Real-world"]
                    blended = np.array([0.02, 0.02, 0.92, 0.02, 0.02], dtype=float)
                elif head == "intent" and risk_metrics.get("is_jailbreak_concealment"):
                    # Evasion is index 4 in INTENT_CLASSES: ["Educational", "Analysis", "Prevention", "Execution", "Evasion"]
                    blended = np.array([0.02, 0.02, 0.02, 0.04, 0.90], dtype=float)
                elif head == "ambiguity" and risk_metrics.get("is_jailbreak_concealment"):
                    # High is index 2 in AMBIGUITY_CLASSES: ["Low", "Medium", "High"]
                    blended = np.array([0.03, 0.05, 0.92], dtype=float)
                elif head == "actionability" and risk_metrics.get("is_jailbreak_concealment"):
                    # Highly actionable is index 3 in ACTIONABILITY_CLASSES
                    blended = np.array([0.03, 0.05, 0.12, 0.80], dtype=float)

                pred_idx = int(blended.argmax())
                pred_label = self.label_encoder.decode_index(head, pred_idx)
                confidence = float(blended[pred_idx])

                predictions[head] = pred_label
                confidences[head] = round(confidence, 4)
                all_probabilities[head] = {
                    classes[i]: round(float(blended[i]), 4) for i in range(len(classes))
                }
        else:
            # Calibrated semantic safety predictions
            for head, classes in HEAD_CONFIG.items():
                pred_label = semantic_res.get(head, classes[0])
                conf = semantic_res["confidences"].get(head, 0.90)

                # Dynamically elevate risk if negative words are present
                if head == "risk_level" and risk_metrics["risk_score"] >= 70.0:
                    pred_label = "High"
                    conf = max(conf, 0.95)

                predictions[head] = pred_label
                confidences[head] = round(conf, 4)
                all_probabilities[head] = semantic_res["probabilities"].get(head, {})

        # Generate explanatory breakdown
        explanation = explain_prediction(
            intent=predictions.get("intent", "Analysis"),
            context=predictions.get("context", "Academic"),
            actionability=predictions.get("actionability", "Conceptual"),
            risk_level=predictions.get("risk_level", "Low"),
            ambiguity=predictions.get("ambiguity", "Low"),
        )

        return {
            "input_text": clean_text,
            "language": language or "Auto-detected",
            "is_fine_tuned": self.is_fine_tuned,
            "latency_ms": latency_ms,
            "predictions": predictions,
            "confidences": confidences,
            "probabilities": all_probabilities,
            "explanation": explanation,
            "risk_score": risk_metrics["risk_score"],
            "detected_negative_words": risk_metrics["detected_negative_words"],
            "detected_mitigators": risk_metrics["detected_mitigators"],
            "device": str(self.device),
            "model_name": self.model_name,
        }
