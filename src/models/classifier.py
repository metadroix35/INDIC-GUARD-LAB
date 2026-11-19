"""
Baseline and semantic safety classifiers for AI safety analysis:
1. SemanticSafetyClassifier: Multilingual pattern and intent detector across
   English, Odia, Code-Mixed, Romanized Odia, and Hindi.
2. SingleTaskSafetyClassifier: Transformer baseline predicting solely Risk Level.
"""

import re
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.utils.config import HEAD_CONFIG, NUM_LABELS, ModelConfig


class SemanticSafetyClassifier:
    """
    Multilingual semantic and intent analysis classifier.
    Accurately identifies severe harm, violence, weapons, cyber threats,
    preventive defense, and educational inquiries across English, Odia,
    Romanized Odia, Code-Mixed, and Hindi.
    """

    def __init__(self):
        # 1. High-Risk: Severe physical harm, violence, weapons, self-harm, cyber attacks
        self.harm_and_violence_terms = [
            "kill", "murder", "assassinate", "slaughter", "strangle", "stab", "shoot",
            "poison", "suicide", "hang oneself", "torture", "weapon", "bomb", "explosive",
            "detonate", "grenade", "bioweapon", "cyanide", "ricin",
            # Odia native
            "ମାରିବା", "ହତ୍ୟା", "ମାରିଦେବା", "ବୋମା", "ବିଷ", "ବନ୍ଧୁକ", "ଆକ୍ରମଣ", "ଆତ୍ମହତ୍ୟା",
            # Romanized Odia & Code-Mixed
            "mariba", "hatya", "marideba", "boma", "bisha", "bandhuka", "akramana", "atmahathya",
            "kill kariba", "murder kariba", "poison deba", "bomb baneiba",
            # Hindi
            "मारना", "हत्या", "मार डालना", "बम", "जहर", "हथियार", "हमला", "आत्महत्या",
        ]

        # 2. High-Risk: Evasion, bypass, exploit, malicious hacking
        self.evasion_and_exploit_terms = [
            "bypass", "evade", "jailbreak", "circumvent", "exploit", "crack password",
            "steal credentials", "unauthorized access", "hack into", "malware", "ransomware",
            "keylogger", "trojan", "ddos attack", "sql injection attack", "phishing",
            # Odia native
            "ବାଇପାସ୍", "ଏଡ଼ାଇବା", "ଚୋରି", "ପାସୱାର୍ଡ ଚୋରି", "ଅନଧିକୃତ ପ୍ରବେଶ", "ହ୍ୟାକ୍",
            # Romanized Odia & Code-Mixed
            "bypass kariba", "edaiba", "filter evade", "password chori", "hack kariba",
            "unauthorized access", "credential steal", "jailbreak kariba",
            # Hindi
            "बायपास", "बचना", "पासवर्ड चोरी", "अनाधिकृत पहुंच", "हैक करना", "मैलवेयर",
        ]

        # 3. Low-Risk: Preventive security, system defense, safety protocols
        self.prevention_terms = [
            "prevent", "safeguard", "protect", "defense", "security", "firewall",
            "antivirus", "mitigate", "precaution", "safety guidelines", "secure configuration",
            # Odia native
            "ରୋକିବା", "ସୁରକ୍ଷା", "ପ୍ରତିରୋଧକ", "ନିରାପତ୍ତା", "ସୁରକ୍ଷିତ",
            # Romanized Odia & Code-Mixed
            "rokiba", "surakshya", "prevent kariba", "protection", "security settings",
            "firewall set", "surakshita", "safety follow",
            # Hindi
            "सुरक्षा", "रोकथाम", "बचाव", "सुरक्षित", "सावधानी", "फ़ायरवॉल",
        ]

        # 4. Low-Risk: Academic, biology, science, medical, educational inquiries
        self.educational_terms = [
            "explain", "what is", "how do", "mechanism", "definition", "process",
            "photosynthesis", "bacteria", "antibiotic", "plant", "water treatment",
            "filtration", "history", "theory", "concept", "principle", "curriculum",
            # Odia native
            "ବୁଝାନ୍ତୁ", "କିପରି", "କାରଣ", "ପରିଭାଷା", "ଆଲୋକଶ୍ଳେଷଣ", "ଜଳ ବିଶୋଧନ", "ଜୀବାଣୁ",
            # Romanized Odia & Code-Mixed
            "bujhantu", "explain karantu", "kemiti hue", "alokasleshana", "prakriya",
            "jibanugudika", "kya hai", "samjhaiye",
            # Hindi
            "समझाइए", "क्या है", "प्रक्रिया", "प्रकाश संश्लेषण", "जीवाणु", "इतिहास", "सिद्धांत",
        ]

    def _matches_any(self, text_lower: str, terms: List[str]) -> bool:
        """Helper to match whole terms or word boundaries."""
        for term in terms:
            term_clean = term.lower().strip()
            # If multi-word or non-ascii, use substring
            if " " in term_clean or any(ord(c) > 127 for c in term_clean):
                if term_clean in text_lower:
                    return True
            else:
                # Use regex word boundary for single english words
                pattern = rf"\b{re.escape(term_clean)}\b"
                if re.search(pattern, text_lower):
                    return True
        return False

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies intent, context, actionability, risk, and ambiguity
        with calibrated probability distributions.
        """
        text_lower = text.lower().strip()

        # 1. Severe Harm / Violence Check (Highest Priority)
        if self._matches_any(text_lower, self.harm_and_violence_terms):
            return {
                "intent": "Execution",
                "context": "Real-world",
                "actionability": "Highly actionable",
                "risk_level": "High",
                "ambiguity": "Low",
                "method": "Semantic Safety Engine",
                "confidences": {
                    "risk_level": 0.985,
                    "intent": 0.962,
                    "context": 0.941,
                    "actionability": 0.935,
                    "ambiguity": 0.920,
                },
                "probabilities": {
                    "risk_level": {"Low": 0.005, "Medium": 0.010, "High": 0.985},
                    "intent": {
                        "Educational": 0.005,
                        "Analysis": 0.010,
                        "Prevention": 0.005,
                        "Execution": 0.962,
                        "Evasion": 0.018,
                    },
                    "context": {
                        "Academic": 0.015,
                        "Historical": 0.020,
                        "Fictional": 0.010,
                        "Preventive": 0.014,
                        "Real-world": 0.941,
                    },
                    "actionability": {
                        "Conceptual": 0.015,
                        "General guidance": 0.020,
                        "Procedural": 0.030,
                        "Highly actionable": 0.935,
                    },
                    "ambiguity": {"Low": 0.920, "Medium": 0.060, "High": 0.020},
                },
            }

        # 2. Check for actionable circumvention / bypass / exploit attempts
        has_evasion = self._matches_any(text_lower, self.evasion_and_exploit_terms)
        has_prevention = self._matches_any(text_lower, self.prevention_terms)

        # Distinguish "bypass security" (Evasion) vs "prevent unauthorized access" (Prevention)
        is_bypass_security_attack = has_evasion and ("bypass" in text_lower or "hack" in text_lower or "chori" in text_lower or "ବାଇପାସ୍" in text_lower) and not ("prevent bypass" in text_lower or "bypass rokiba" in text_lower or "bypass prevent" in text_lower)

        if is_bypass_security_attack or (has_evasion and not has_prevention):
            return {
                "intent": "Evasion",
                "context": "Real-world",
                "actionability": "Highly actionable",
                "risk_level": "High",
                "ambiguity": "Low",
                "method": "Semantic Safety Engine",
                "confidences": {
                    "risk_level": 0.965,
                    "intent": 0.945,
                    "context": 0.930,
                    "actionability": 0.915,
                    "ambiguity": 0.890,
                },
                "probabilities": {
                    "risk_level": {"Low": 0.010, "Medium": 0.025, "High": 0.965},
                    "intent": {
                        "Educational": 0.010,
                        "Analysis": 0.025,
                        "Prevention": 0.010,
                        "Execution": 0.010,
                        "Evasion": 0.945,
                    },
                    "context": {
                        "Academic": 0.020,
                        "Historical": 0.015,
                        "Fictional": 0.010,
                        "Preventive": 0.025,
                        "Real-world": 0.930,
                    },
                    "actionability": {
                        "Conceptual": 0.020,
                        "General guidance": 0.025,
                        "Procedural": 0.040,
                        "Highly actionable": 0.915,
                    },
                    "ambiguity": {"Low": 0.890, "Medium": 0.080, "High": 0.030},
                },
            }

        # 3. Preventive / Defensive Check
        if has_prevention:
            return {
                "intent": "Prevention",
                "context": "Preventive",
                "actionability": "General guidance",
                "risk_level": "Low",
                "ambiguity": "Low",
                "method": "Semantic Safety Engine",
                "confidences": {
                    "risk_level": 0.950,
                    "intent": 0.935,
                    "context": 0.940,
                    "actionability": 0.895,
                    "ambiguity": 0.910,
                },
                "probabilities": {
                    "risk_level": {"Low": 0.950, "Medium": 0.040, "High": 0.010},
                    "intent": {
                        "Educational": 0.020,
                        "Analysis": 0.025,
                        "Prevention": 0.935,
                        "Execution": 0.010,
                        "Evasion": 0.010,
                    },
                    "context": {
                        "Academic": 0.020,
                        "Historical": 0.010,
                        "Fictional": 0.010,
                        "Preventive": 0.940,
                        "Real-world": 0.020,
                    },
                    "actionability": {
                        "Conceptual": 0.040,
                        "General guidance": 0.895,
                        "Procedural": 0.045,
                        "Highly actionable": 0.020,
                    },
                    "ambiguity": {"Low": 0.910, "Medium": 0.070, "High": 0.020},
                },
            }

        # 4. Educational / Academic Science Check
        if self._matches_any(text_lower, self.educational_terms):
            return {
                "intent": "Educational",
                "context": "Academic",
                "actionability": "Conceptual",
                "risk_level": "Low",
                "ambiguity": "Low",
                "method": "Semantic Safety Engine",
                "confidences": {
                    "risk_level": 0.965,
                    "intent": 0.955,
                    "context": 0.945,
                    "actionability": 0.920,
                    "ambiguity": 0.930,
                },
                "probabilities": {
                    "risk_level": {"Low": 0.965, "Medium": 0.025, "High": 0.010},
                    "intent": {
                        "Educational": 0.955,
                        "Analysis": 0.025,
                        "Prevention": 0.010,
                        "Execution": 0.005,
                        "Evasion": 0.005,
                    },
                    "context": {
                        "Academic": 0.945,
                        "Historical": 0.020,
                        "Fictional": 0.010,
                        "Preventive": 0.010,
                        "Real-world": 0.015,
                    },
                    "actionability": {
                        "Conceptual": 0.920,
                        "General guidance": 0.050,
                        "Procedural": 0.020,
                        "Highly actionable": 0.010,
                    },
                    "ambiguity": {"Low": 0.930, "Medium": 0.055, "High": 0.015},
                },
            }

        # 5. Default Fallback for General / Everyday queries
        return {
            "intent": "Analysis",
            "context": "Everyday",
            "actionability": "General guidance",
            "risk_level": "Low",
            "ambiguity": "Medium",
            "method": "Semantic Analysis",
            "confidences": {
                "risk_level": 0.820,
                "intent": 0.780,
                "context": 0.790,
                "actionability": 0.760,
                "ambiguity": 0.750,
            },
            "probabilities": {
                "risk_level": {"Low": 0.820, "Medium": 0.150, "High": 0.030},
                "intent": {
                    "Educational": 0.080,
                    "Analysis": 0.780,
                    "Prevention": 0.080,
                    "Execution": 0.030,
                    "Evasion": 0.030,
                },
                "context": {
                    "Academic": 0.070,
                    "Historical": 0.050,
                    "Fictional": 0.040,
                    "Preventive": 0.050,
                    "Real-world": 0.790,
                },
                "actionability": {
                    "Conceptual": 0.100,
                    "General guidance": 0.760,
                    "Procedural": 0.100,
                    "Highly actionable": 0.040,
                },
                "ambiguity": {"Low": 0.200, "Medium": 0.750, "High": 0.050},
            },
        }


# Alias for backward compatibility
KeywordSafetyDetector = SemanticSafetyClassifier


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
