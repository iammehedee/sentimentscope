"""
Evaluation script for SentimentScope.

Loads a trained model and runs comprehensive evaluation with metrics,
confusion matrix, and detailed error analysis.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.model.architecture import SentimentClassifier
from src.model.config import model_config, training_config
from src.data.preprocess import create_dataloaders


def evaluate_model(model_path: str, dataset_name: str = "imdb", device: str = "cpu"):
    """
    Load a trained model and evaluate it on the test set.

    Args:
        model_path: Path to the saved model checkpoint or HuggingFace model ID.
        dataset_name: Dataset to evaluate on.
        device: 'cpu' or 'cuda'.

    Returns:
        dict with all evaluation metrics
    """
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # Load data
    print("📋 Loading data...")
    dataloaders = create_dataloaders(
        dataset_name=dataset_name,
        max_length=model_config.max_length,
        batch_size=training_config.batch_size,
    )

    # Load model
    print(f"🧠 Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    model = SentimentClassifier(
        pretrained_model=model_config.pretrained_model,
        num_labels=checkpoint.get("num_labels", 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Evaluate
    print("🧪 Running evaluation...")
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloaders["test"]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs["probabilities"]
            preds = outputs["predictions"]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    # Per-class metrics
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("\n  Per-class metrics:")
    for i, (p, r, f, s) in enumerate(zip(precision_per, recall_per, f1_per, support_per)):
        print(f"    Class {i}: P={p:.4f} R={r:.4f} F1={f:.4f} (n={s})")

    print("\n  Confusion Matrix:")
    print(f"  {cm}")

    print("\n" + "=" * 60)
    print("  Detailed Classification Report:")
    print("=" * 60)
    print(classification_report(all_labels, all_preds))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SentimentScope model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        dataset_name=args.dataset,
        device=args.device,
    )
