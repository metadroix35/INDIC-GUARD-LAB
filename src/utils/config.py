"""
Configuration module for Multilingual Context-Aware AI Safety Analysis System.
Defines label schemas, multi-task heads, model configurations, and file paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
TRAINED_MODEL_DIR = MODELS_DIR / "trained"
RESULTS_DIR = PROJECT_ROOT / "results"

# Head Label Definitions
INTENT_CLASSES: List[str] = [
    "Educational",
    "Analysis",
    "Prevention",
    "Execution",
    "Evasion",
]

CONTEXT_CLASSES: List[str] = [
    "Academic",
    "Historical",
    "Fictional",
    "Preventive",
    "Real-world",
]

ACTIONABILITY_CLASSES: List[str] = [
    "Conceptual",
    "General guidance",
    "Procedural",
    "Highly actionable",
]

RISK_CLASSES: List[str] = [
    "Low",
    "Medium",
    "High",
]

AMBIGUITY_CLASSES: List[str] = [
    "Low",
    "Medium",
    "High",
]

# Mapping dictionary for all target heads
HEAD_CONFIG: Dict[str, List[str]] = {
    "intent": INTENT_CLASSES,
    "context": CONTEXT_CLASSES,
    "actionability": ACTIONABILITY_CLASSES,
    "risk_level": RISK_CLASSES,
    "ambiguity": AMBIGUITY_CLASSES,
}

# Number of classes per head
NUM_LABELS: Dict[str, int] = {k: len(v) for k, v in HEAD_CONFIG.items()}

# Supported Languages and Scripts
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "or": "Odia (ଓଡ଼ିଆ)",
    "or-en": "Odia-English Code-Mixed",
    "or-Latn": "Romanized Odia",
    "hi": "Hindi (हिन्दी)",
}

# Expected Dataset Columns (16 schema columns)
DATASET_COLUMNS: List[str] = [
    "id",
    "text",
    "language",
    "script",
    "is_code_mixed",
    "category",
    "risk_level",
    "intent",
    "context",
    "actionability",
    "ambiguity",
    "variation_type",
    "source",
    "review_status",
    "annotation_note",
    "semantic_group",
]

@dataclass
class ModelConfig:
    """Hyperparameters and configuration for training and evaluation."""
    primary_model_name: str = "google/muril-base-cased"
    secondary_model_name: str = "FacebookAI/xlm-roberta-base"
    max_length: int = 128
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    epochs: int = 5
    head_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "risk_level": 1.2,
            "intent": 1.0,
            "context": 0.8,
            "actionability": 0.8,
            "ambiguity": 0.6,
        }
    )
    device: str = "cpu"
    seed: int = 42

DEFAULT_MODEL_CONFIG = ModelConfig()
