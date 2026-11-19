"""
Multi-Task Multilingual Transformer Architecture for AI Safety Analysis.
Uses a shared encoder (MuRIL / XLM-R) with five independent classification heads:
1. Intent Head
2. Context Head
3. Actionability Head
4. Risk Head
5. Ambiguity Head
"""

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
