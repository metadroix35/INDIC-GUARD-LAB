"""
Multi-Task Multilingual Transformer Architecture for AI Safety Analysis.
Uses a shared encoder (MuRIL / XLM-R) with five independent classification heads:
1. Intent Head
2. Context Head
3. Actionability Head
4. Risk Head
5. Ambiguity Head
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.utils.config import HEAD_CONFIG, NUM_LABELS, ModelConfig


class ClassificationHead(nn.Module):
    """Feed-forward classification head with dropout and linear projection."""

    def __init__(self, hidden_size: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        logits = self.classifier(x)
        return logits


class EnhancedClassificationHead(nn.Module):
    """2-layer MLP head: hidden_size -> 256 -> num_classes with BatchNorm + ReLU + Dropout."""

    def __init__(self, hidden_size: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MultitaskSafetyModel(nn.Module):
    """
    Multi-Task Neural Architecture for multilingual prompt safety analysis.
    Shares a pretrained multilingual encoder across five specialized prediction heads.
    """

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        config: Optional[ModelConfig] = None,
        custom_num_labels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.config = config or ModelConfig()
        self.model_name = model_name
        self.num_labels = custom_num_labels or NUM_LABELS

        # Load transformer encoder backbone
        self.transformer_config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.transformer_config)
        hidden_size = self.transformer_config.hidden_size

        # Five independent classification heads
        self.heads = nn.ModuleDict(
            {
                head_name: ClassificationHead(
                    hidden_size=hidden_size,
                    num_classes=num_classes,
                    dropout_rate=self.config.dropout_rate,
                )
                for head_name, num_classes in self.num_labels.items()
            }
        )

        # Cross entropy loss with ignore_index=-100 to support partial/masked labels
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def configure_enhanced_heads(self, hidden_size: Optional[int] = None, dropout_rate: float = 0.3):
        """Switch classification heads to 2-layer MLP (hidden_size -> 256 -> num_classes)."""
        hs = hidden_size or self.transformer_config.hidden_size
        self.heads = nn.ModuleDict(
            {
                head_name: EnhancedClassificationHead(
                    hidden_size=hs,
                    num_classes=num_classes,
                    dropout_rate=dropout_rate,
                )
                for head_name, num_classes in self.num_labels.items()
            }
        )

    def load_weights_flexible(self, checkpoint_path_or_dict: Union[str, Path, Dict[str, Any]]) -> bool:
        """
        Dynamically inspects checkpoint structure and loads:
        - Full model state dict (encoder + heads)
        - Checkpoint dict containing 'model_state_dict'
        - Heads-only state dict (either linear or enhanced MLP)
        """
        if isinstance(checkpoint_path_or_dict, (str, Path)):
            state = torch.load(checkpoint_path_or_dict, map_location="cpu", weights_only=False)
        else:
            state = checkpoint_path_or_dict

        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        # Check if heads use 'net.' (EnhancedClassificationHead)
        has_enhanced_heads = any(".net." in k for k in state.keys())
        if has_enhanced_heads:
            self.configure_enhanced_heads()

        # Check if this is full model or heads only
        has_encoder = any(k.startswith("encoder.") for k in state.keys())
        if has_encoder:
            self.load_state_dict(state, strict=False)
            return True
        else:
            # Heads-only: strip 'heads.' prefix if present
            cleaned_state = {}
            for k, v in state.items():
                if k.startswith("heads."):
                    cleaned_state[k[len("heads."):]] = v
                else:
                    cleaned_state[k] = v
            self.heads.load_state_dict(cleaned_state, strict=False)
            return True

    def extract_pooled_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract [CLS] token representation or mean pooling across tokens."""
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)

        # Use pooler_output if present, otherwise take representation of the first token [CLS]
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass computing logits and probabilities for each head,
        and calculating multi-task weighted loss when labels are provided.
        """
        pooled_features = self.extract_pooled_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        logits_dict: Dict[str, torch.Tensor] = {}
        probs_dict: Dict[str, torch.Tensor] = {}

        for head_name, head_module in self.heads.items():
            head_logits = head_module(pooled_features)
            logits_dict[head_name] = head_logits
            probs_dict[head_name] = torch.softmax(head_logits, dim=-1)

        result: Dict[str, Any] = {
            "logits": logits_dict,
            "probabilities": probs_dict,
            "pooled_features": pooled_features,
        }

        # Multi-task loss computation
        if labels is not None:
            total_loss = torch.tensor(0.0, device=input_ids.device)
            head_losses = {}

            for head_name, target in labels.items():
                if head_name in logits_dict:
                    loss = self.loss_fn(logits_dict[head_name], target)
                    weight = self.config.head_weights.get(head_name, 1.0)
                    total_loss += weight * loss
                    head_losses[head_name] = loss

            result["loss"] = total_loss
            result["head_losses"] = head_losses

        return result
