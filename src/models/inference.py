"""
Inference engine for Multilingual AI Safety Analysis.
Performs tokenization, forward pass, confidence extraction, and safety explanation.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer

from src.data.preprocessing import MultiTaskLabelEncoder, normalize_text
from src.models.classifier import KeywordSafetyDetector
from src.models.multitask_model import MultitaskSafetyModel
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
    Supports CPU fallback and provides honest status reporting if weights are uncheckpointed.
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
        self.keyword_baseline = KeywordSafetyDetector()

        logger.info(f"Loading tokenizer for: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = MultitaskSafetyModel(model_name=model_name_or_path)
        self.is_fine_tuned = False

        # Attempt to load custom fine-tuned weights if available
        resolved_weights = weights_path or (TRAINED_MODEL_DIR / "best_multitask_model.pt")
        if resolved_weights.exists():
            try:
                state_dict = torch.load(resolved_weights, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.is_fine_tuned = True
                logger.info(f"Successfully loaded fine-tuned checkpoint from {resolved_weights}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint from {resolved_weights}: {e}")

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

        # Tokenize input
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

        predictions: Dict[str, str] = {}
        confidences: Dict[str, float] = {}
        all_probabilities: Dict[str, Dict[str, float]] = {}

        probs_dict = outputs["probabilities"]

        for head, classes in HEAD_CONFIG.items():
            head_probs = probs_dict[head][0].cpu().numpy()
            pred_idx = int(head_probs.argmax())
            pred_label = self.label_encoder.decode_index(head, pred_idx)
            confidence = float(head_probs[pred_idx])

            predictions[head] = pred_label
            confidences[head] = round(confidence, 4)
            all_probabilities[head] = {
                classes[i]: round(float(head_probs[i]), 4) for i in range(len(classes))
            }

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
            "device": str(self.device),
            "model_name": self.model_name,
        }
