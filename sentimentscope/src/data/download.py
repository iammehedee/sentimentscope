"""
Dataset downloader for SentimentScope.

Supports multiple sentiment analysis datasets:
- IMDb (movie reviews, binary)
- TweetEval (Twitter sentiment, 3-class)
- SST-2 (Stanford Sentiment Treebank, binary)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from src.model.config import data_config, RAW_DATA_DIR


def download_dataset(
    dataset_name: str = "imdb",
    num_samples: int = -1,
    output_dir: Path = RAW_DATA_DIR,
):
    """
    Download and save a sentiment analysis dataset.

    Args:
        dataset_name: One of 'imdb', 'tweet_eval', 'sst2'
        num_samples: Max samples to keep (-1 = all)
        output_dir: Where to save the dataset

    Returns:
        Path to saved dataset directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading '{dataset_name}' dataset...")

    if dataset_name == "imdb":
        dataset = load_dataset("imdb")
        label_map = {0: "negative", 1: "positive"}
    elif dataset_name == "tweet_eval":
        dataset = load_dataset("tweet_eval", "sentiment")
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
    elif dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")
        label_map = {0: "negative", 1: "positive"}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Subsample if requested
    if num_samples > 0:
        for split in dataset:
            split_size = min(num_samples, len(dataset[split]))
            dataset[split] = dataset[split].shuffle(seed=42).select(range(split_size))

    # Save to disk
    save_path = output_dir / dataset_name
    dataset.save_to_disk(str(save_path))

    # Print summary
    print(f"\n✅ Dataset '{dataset_name}' downloaded and saved to {save_path}")
    print(f"\n📊 Dataset Summary:")
    print(f"   Label mapping: {label_map}")
    for split in dataset:
        print(f"   {split}: {len(dataset[split])} examples")

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download sentiment analysis dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default=data_config.dataset_name,
        choices=["imdb", "tweet_eval", "sst2"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=data_config.num_samples,
        help="Number of samples (-1 for full dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Output directory",
    )
    args = parser.parse_args()

    download_dataset(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )
