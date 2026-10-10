"""
Data preprocessing pipeline for SentimentScope.

Handles tokenization, train/validation/test splitting, and DataLoader creation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast
from datasets import load_from_disk

from src.model.config import data_config, model_config, PROCESSED_DATA_DIR, RAW_DATA_DIR


class SentimentDataset(Dataset):
    """PyTorch Dataset wrapping tokenized sentiment data."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


def tokenize_dataset(dataset, tokenizer, max_length: int = 128):
    """Tokenize a HuggingFace dataset using the provided tokenizer."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    # Remove columns that tokenizer doesn't need
    columns_to_remove = [col for col in dataset.column_names if col not in ["label"]]
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=columns_to_remove)
    return tokenized


def get_label_column(dataset) -> str:
    """Determine the label column name."""
    if "label" in dataset.column_names:
        return "label"
    elif "labels" in dataset.column_names:
        return "labels"
    else:
        raise ValueError(f"No label column found. Available: {dataset.column_names}")


def create_dataloaders(
    dataset_name: str = "imdb",
    max_length: int = 128,
    batch_size: int = 16,
    num_workers: int = 4,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
):
    """
    Full preprocessing pipeline: load → tokenize → split → DataLoader.

    Returns:
        dict with 'train', 'val', 'test' DataLoaders and 'tokenizer'
    """
    # Load dataset
    dataset_path = raw_dir / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run `python src/data/download.py` first."
        )

    dataset = load_from_disk(str(dataset_path))
    print(f"📂 Loaded dataset from {dataset_path}")

    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_config.pretrained_model)

    # Get split names
    available_splits = list(dataset.keys())
    print(f"   Available splits: {available_splits}")

    # Tokenize all splits
    tokenized_splits = {}
    for split_name in available_splits:
        tokenized_splits[split_name] = tokenize_dataset(
            dataset[split_name], tokenizer, max_length
        )

    # Determine label column
    label_col = get_label_column(tokenized_splits[available_splits[0]])
    num_labels = len(set(tokenized_splits[available_splits[0]][label_col]))

    # Update model config with detected number of labels
    model_config.num_labels = num_labels
    print(f"   Detected {num_labels} classes")

    # Handle splits — create val/test if not present
    if "train" in tokenized_splits and "validation" not in tokenized_splits:
        # Split train into train/val
        train_val = tokenized_splits["train"].train_test_split(
            test_size=0.1, seed=42
        )
        tokenized_splits["train"] = train_val["train"]
        tokenized_splits["validation"] = train_val["test"]

    if "test" not in tokenized_splits and "test" in available_splits:
        pass  # test exists
    elif "test" not in tokenized_splits:
        # Split validation further if no test set
        if "validation" in tokenized_splits:
            val_test = tokenized_splits["validation"].train_test_split(
                test_size=0.5, seed=42
            )
            tokenized_splits["validation"] = val_test["train"]
            tokenized_splits["test"] = val_test["test"]

    # Create DataLoaders
    dataloaders = {}
    for split_name, split_data in tokenized_splits.items():
        if split_name not in ["train", "validation", "test"]:
            continue
        labels = split_data[label_col]
        dataset_obj = SentimentDataset(
            encodings={
                "input_ids": split_data["input_ids"],
                "attention_mask": split_data["attention_mask"],
            },
            labels=labels,
        )

        shuffle = split_name == "train"
        dataloaders[split_name] = DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"   {split_name}: {len(dataset_obj)} samples → DataLoader (batch_size={batch_size})")

    dataloaders["tokenizer"] = tokenizer
    dataloaders["num_labels"] = num_labels

    return dataloaders


if __name__ == "__main__":
    dataloaders = create_dataloaders(
        dataset_name=data_config.dataset_name,
        max_length=model_config.max_length,
        batch_size=training_config.batch_size,
        num_workers=data_config.num_workers,
    )
    print("\n✅ Data preprocessing complete!")
    print(f"   Train batches: {len(dataloaders['train'])}")
    print(f"   Val batches: {len(dataloaders['validation'])}")
    print(f"   Test batches: {len(dataloaders['test'])}")
    print(f"   Num labels: {dataloaders['num_labels']}")
