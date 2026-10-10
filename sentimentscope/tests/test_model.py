"""
Unit tests for SentimentScope model.
"""
import pytest
import torch

from src.model.architecture import SentimentClassifier, count_parameters


@pytest.fixture
def model():
    return SentimentClassifier(
        pretrained_model="distilbert-base-uncased",
        num_labels=2,
        dropout=0.1,
    )


@pytest.fixture
def sample_batch():
    batch_size = 4
    seq_len = 128
    return {
        "input_ids": torch.randint(0, 30000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.tensor([0, 1, 0, 1]),
    }


def test_model_forward(model, sample_batch):
    """Test that forward pass returns expected keys."""
    outputs = model(
        input_ids=sample_batch["input_ids"],
        attention_mask=sample_batch["attention_mask"],
        labels=sample_batch["labels"],
    )
    assert "loss" in outputs
    assert "logits" in outputs
    assert "probabilities" in outputs
    assert "predictions" in outputs
    assert outputs["loss"].dim() == 0  # scalar


def test_model_predict_shape(model, sample_batch):
    """Test prediction output shapes."""
    probs = model.predict_proba(
        input_ids=sample_batch["input_ids"],
        attention_mask=sample_batch["attention_mask"],
    )
    assert probs.shape == (4, 2)
    # Probabilities should sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_model_parameters():
    """Test parameter counting."""
    model = SentimentClassifier(num_labels=3, freeze_backbone=True)
    total, trainable = count_parameters(model)
    assert total > 0
    assert trainable < total  # Frozen backbone means fewer trainable params
    assert trainable > 0


def test_model_multiclass():
    """Test 3-class model."""
    model = SentimentClassifier(num_labels=3)
    input_ids = torch.randint(0, 30000, (2, 64))
    attention_mask = torch.ones(2, 64)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs["predictions"].max() < 3
    assert outputs["probabilities"].shape == (2, 3)
