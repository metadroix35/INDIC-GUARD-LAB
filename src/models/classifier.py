"""
Baseline models for comparative evaluation:
1. Keyword-based Safety Detector
2. Heuristic Rule-based Classifier
3. Single-Task Transformer Classifier (Risk-only)
"""

import re
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.utils.config import HEAD_CONFIG, NUM_LABELS, ModelConfig


class KeywordSafetyDetector:
    """
    Rule-based safety baseline using multilingual keyword matching
    across English, Odia, Romanized Odia, and Hindi.
    """

    def __init__(self):
        # Benign academic keywords
        self.educational_terms = [
            "explain", "what is", "how do", "mechanism", "definition",
            "ବୁଝାନ୍ତୁ", "କିପରି", "କାରଣ", "ପରିଭାଷା",
            "samjhaiye", "kya hai", "explain karantu", "kemiti hue",
            "समझाइए", "क्या है", "प्रक्रिया",
        ]

        # Preventive/Safety keywords
        self.prevention_terms = [
            "prevent", "safeguard", "protect", "defense", "security",
            "ରୋକିବା", "ସୁରକ୍ଷା", "ପ୍ରତିରୋଧକ",
            "rokiba", "surakshya", "prevent kariba", "protection",
            "सुरक्षा", "रोकथाम", "बचाव",
        ]

        # Evasion / high-risk indicators
        self.evasion_terms = [
            "bypass", "evade", "jailbreak", "circumvent", "exploit",
            "ବାଇପାସ୍", "ଏଡ଼ାଇବା",
            "bypass kariba", "edaiba", "filter evade",
            "बायपास", "बचना",
        ]

    def predict(self, text: str) -> Dict[str, str]:
        """Predict intent and risk based on surface keyword presence."""
        text_lower = text.lower()

        # Check evasion
        for term in self.evasion_terms:
            if term.lower() in text_lower:
                return {
                    "intent": "Evasion",
                    "risk_level": "High",
                    "context": "Real-world",
                    "actionability": "Highly actionable",
                    "ambiguity": "Low",
                    "method": "Keyword Matching",
                }

        # Check prevention
        for term in self.prevention_terms:
            if term.lower() in text_lower:
                return {
                    "intent": "Prevention",
                    "risk_level": "Low",
                    "context": "Preventive",
                    "actionability": "General guidance",
                    "ambiguity": "Low",
                    "method": "Keyword Matching",
                }

        # Check educational
        for term in self.educational_terms:
            if term.lower() in text_lower:
                return {
                    "intent": "Educational",
                    "risk_level": "Low",
                    "context": "Academic",
                    "actionability": "Conceptual",
                    "ambiguity": "Low",
                    "method": "Keyword Matching",
                }

        # Default fallback
        return {
            "intent": "Analysis",
            "risk_level": "Medium",
            "context": "Academic",
            "actionability": "General guidance",
            "ambiguity": "Medium",
            "method": "Keyword Default",
        }


class SingleTaskSafetyClassifier(nn.Module):
    """
    Single-task Transformer baseline focusing solely on Risk Level classification.
    Used for comparative ablation against the Multi-Task model.
    """

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        num_classes: int = len(HEAD_CONFIG["risk_level"]),
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(self.dropout(pooled))
        probs = torch.softmax(logits, dim=-1)

        res = {"logits": logits, "probabilities": probs}
        if labels is not None:
            res["loss"] = self.loss_fn(logits, labels)
        return res
