"""
DistilBERT-based Sentiment Classifier.

A production-ready sentiment analysis model built on DistilBERT,
fine-tuned for binary/multi-class sentiment classification.
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class SentimentClassifier(nn.Module):
    """DistilBERT-based sentiment classifier with dropout and linear head."""

    def __init__(
        self,
        pretrained_model: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            pretrained_model: HuggingFace model identifier.
            num_labels: Number of sentiment classes (2 = binary, 3 = positive/neutral/negative).
            dropout: Dropout rate for the classification head.
            freeze_backbone: If True, freeze DistilBERT weights (linear probing).
        """
        super().__init__()
        self.num_labels = num_labels
        self.backbone = DistilBertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.

        Returns:
            dict with 'loss' (optional), 'logits', 'probabilities', 'predictions'
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        result = {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
        }

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result["loss"] = loss

        return result

    def predict_proba(self, input_ids, attention_mask=None):
        """Return class probabilities for given inputs."""
        self.eval()
        with torch.no_grad():
            outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        return outputs["probabilities"]


def load_model(checkpoint_path: str, num_labels: int = 2, device: str = "cpu"):
    """Load a trained model from a checkpoint directory."""
    model = SentimentClassifier(
        pretrained_model=checkpoint_path,
        num_labels=num_labels,
    )
    model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
