"""
Model predictor for SentimentScope API.
Handles loading and inference with the trained model.
"""
import time
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import DistilBertTokenizerFast

from src.model.architecture import SentimentClassifier
from src.model.config import model_config


class SentimentPredictor:
    """Production-ready sentiment predictor."""

    def __init__(
        self,
        model_path: str = None,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        device: str = None,
    ):
        """
        Args:
            model_path: Path to saved model. If None, uses pretrained.
            tokenizer_name: Tokenizer to use.
            max_length: Max sequence length.
            device: 'cpu', 'cuda', or None (auto-detect).
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
        self.model = None
        self.num_labels = model_config.num_labels
        self.label_names = ["negative", "positive"] if self.num_labels == 2 else [
            "negative", "neutral", "positive"
        ]

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load a trained model from checkpoint or HuggingFace."""
        print(f"🧠 Loading model from {model_path}...")

        try:
            # Try loading as checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.num_labels = checkpoint.get("num_labels", 2)
                self.model = SentimentClassifier(
                    pretrained_model=model_config.pretrained_model,
                    num_labels=self.num_labels,
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Try loading as HuggingFace model
                self.model = SentimentClassifier(
                    pretrained_model=model_path,
                    num_labels=self.num_labels,
                )
        except Exception:
            # Load from HuggingFace Hub
            self.model = SentimentClassifier(
                pretrained_model=model_path,
                num_labels=self.num_labels,
            )

        self.model.to(self.device)
        self.model.eval()
        print(f"   ✅ Model loaded on {self.device}")

    def predict(self, text: str, return_probabilities: bool = False) -> dict:
        """
        Predict sentiment for a single text.

        Returns:
            dict with 'sentiment', 'confidence', and optionally 'probabilities'
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        inference_time = (time.time() - start_time) * 1000  # ms

        probs = outputs["probabilities"].squeeze().cpu().numpy()
        pred = int(outputs["predictions"].item())
        confidence = float(probs[pred])

        result = {
            "sentiment": self.label_names[pred],
            "confidence": confidence,
            "inference_time_ms": inference_time,
        }

        if return_probabilities:
            result["probabilities"] = {
                self.label_names[i]: float(probs[i])
                for i in range(self.num_labels)
            }

        return result

    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Predict sentiment for a batch of texts."""
        return [self.predict(text) for text in texts]

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
