"""
Utility functions for logging, reproducibility, device selection,
explanations, and serialization.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ai_safety")


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Detect available hardware accelerator or fallback to CPU gracefully."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device for computation")
    return device


def save_json(data: Any, path: Path) -> None:
    """Save serializable data to JSON file with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON data to {path}")


def load_json(path: Path) -> Optional[Any]:
    """Safely load JSON file if it exists, otherwise return None."""
    if not path.exists():
        logger.warning(f"File {path} does not exist.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def explain_prediction(
    intent: str,
    context: str,
    actionability: str,
    risk_level: str,
    ambiguity: str,
) -> str:
    """
    Generate a clear, human-understandable explanation for why the combination
    of intent, context, and actionability produces the assessed risk level.
    """
    explanations = []

    # Risk level contextualization
    if risk_level == "High":
        explanations.append(
            "High safety concern: The query presents indicators of potentially unsafe execution "
            "or security circumvention, or requests actionable guidance in sensitive operational domains."
        )
    elif risk_level == "Medium":
        explanations.append(
            "Moderate risk: The prompt contains sensitive domain terminology or dual-use context "
            "that warrants caution, though explicit malicious intent is not confirmed."
        )
    else:
        explanations.append(
            "Low risk: The query is safe, educational, analytical, or preventive in nature."
        )

    # Intent and Context interplay
    if intent in ["Educational", "Analysis", "Prevention"]:
        explanations.append(
            f"The primary intent is '{intent}' within a '{context}' framing, which strongly "
            f"supports legitimate academic and defensive inquiry."
        )
    elif intent in ["Execution", "Evasion"]:
        explanations.append(
            f"The intent is '{intent}' under a '{context}' setting, signaling a desire for direct "
            f"operational assistance rather than theoretical comprehension."
        )

    # Actionability aspect
    if actionability in ["Procedural", "Highly actionable"]:
        explanations.append(
            f"Actionability is '{actionability}', meaning step-by-step or executable details are "
            f"involved. This elevates scrutiny when combined with high-risk topics."
        )
    else:
        explanations.append(
            f"Actionability is '{actionability}', indicating high-level conceptual discussion "
            f"with low risk of real-world misuse."
        )

    # Ambiguity note
    if ambiguity == "High":
        explanations.append(
            "Ambiguity is 'High', suggesting the phrasing is vague or obfuscated (e.g. through "
            "cross-lingual transliteration or indirect metaphors). Further context is advised."
        )

    return " ".join(explanations)
