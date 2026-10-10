"""
Training script for SentimentScope with full MLflow experiment tracking.

Usage:
    python src/train.py --dataset imdb --epochs 3 --batch-size 16
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.model.architecture import SentimentClassifier, count_parameters
from src.model.config import model_config, training_config, ops_config
from src.data.preprocess import create_dataloaders


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch: int):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    from tqdm import tqdm
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model on a dataloader."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = len(dataloader)

    from tqdm import tqdm
    pbar = tqdm(dataloader, desc="[Eval]", leave=False)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]
            preds = outputs["predictions"]

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches

    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SentimentScope model")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze DistilBERT")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    args = parser.parse_args()

    set_seed(training_config.seed)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Override config with CLI args
    training_config.num_epochs = args.epochs
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.lr
    model_config.max_length = args.max_length
    model_config.freeze_backbone = args.freeze_backbone

    # Load data
    print("\n📋 Loading and preprocessing data...")
    dataloaders = create_dataloaders(
        dataset_name=args.dataset,
        max_length=model_config.max_length,
        batch_size=training_config.batch_size,
    )

    # Initialize model
    print(f"\n🧠 Initializing {model_config.pretrained_model}...")
    model = SentimentClassifier(
        pretrained_model=model_config.pretrained_model,
        num_labels=dataloaders["num_labels"],
        dropout=model_config.dropout,
        freeze_backbone=model_config.freeze_backbone,
    )
    model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    num_training_steps = len(dataloaders["train"]) * training_config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # MLflow setup
    mlflow.set_tracking_uri(ops_config.mlflow_tracking_uri)
    mlflow.set_experiment(ops_config.mlflow_experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print(f"\n📊 MLflow run: {run_id}")

        # Log parameters
        params = {
            "model": model_config.pretrained_model,
            "dataset": args.dataset,
            "num_labels": dataloaders["num_labels"],
            "num_epochs": training_config.num_epochs,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "max_length": model_config.max_length,
            "weight_decay": training_config.weight_decay,
            "warmup_steps": training_config.warmup_steps,
            "dropout": model_config.dropout,
            "freeze_backbone": model_config.freeze_backbone,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
        mlflow.log_params(params)

        # Training loop
        best_val_f1 = 0
        no_improve = 0

        print(f"\n🚀 Starting training for {training_config.num_epochs} epochs...")
        for epoch in range(1, training_config.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = train_epoch(
                model, dataloaders["train"], optimizer, scheduler, device, epoch
            )

            # Evaluate on validation
            val_metrics = evaluate(model, dataloaders["validation"], device)

            epoch_time = time.time() - epoch_start

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "epoch_time": epoch_time,
                },
                step=epoch,
            )

            print(f"\n📈 Epoch {epoch}/{training_config.num_epochs} ({epoch_time:.1f}s)")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_metrics['loss']:.4f}")
            print(f"   Val Acc:    {val_metrics['accuracy']:.4f}")
            print(f"   Val F1:     {val_metrics['f1']:.4f}")

            # Early stopping & model checkpointing
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                no_improve = 0

                # Save best model
                checkpoint_path = training_config.output_dir / f"best_model_{run_id[:8]}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": model_config,
                        "num_labels": dataloaders["num_labels"],
                    },
                    checkpoint_path,
                )
                mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
                print(f"   ✅ New best F1: {best_val_f1:.4f} — checkpoint saved")
            else:
                no_improve += 1
                if no_improve >= training_config.early_stopping_patience:
                    print(f"\n⏹️  Early stopping after {epoch} epochs")
                    break

        # Final test evaluation
        print("\n🧪 Evaluating on test set...")
        test_metrics = evaluate(model, dataloaders["test"], device)

        mlflow.log_metrics(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
            }
        )

        print(f"\n🎯 Test Results:")
        print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall:    {test_metrics['recall']:.4f}")
        print(f"   F1 Score:  {test_metrics['f1']:.4f}")

        # Log classification report
        from sklearn.metrics import classification_report
        report = classification_report(test_metrics["labels"], test_metrics["predictions"])
        report_path = Path("classification_report.txt")
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        # Log model
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=ops_config.model_registry_name,
        )

        print(f"\n✅ Model logged to MLflow as '{ops_config.model_registry_name}'")
        print(f"🔗 MLflow UI: {ops_config.mlflow_tracking_uri}")
        print(f"🏁 Run ID: {run_id}")


if __name__ == "__main__":
    main()
