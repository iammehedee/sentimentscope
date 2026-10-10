"""
Centralized configuration for SentimentScope.
All hyperparameters and settings live here for reproducibility.
"""
from dataclasses import dataclass, field
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / ".cache"


@dataclass
class ModelConfig:
    """Model architecture settings."""
    pretrained_model: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 128
    dropout: float = 0.1
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    output_dir: Path = field(default_factory=lambda: MODEL_DIR)
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    gradient_clip: float = 1.0
    seed: int = 42
    fp16: bool = True
    early_stopping_patience: int = 3


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "imdb"  # Options: imdb, tweet_eval, glue (sst2)
    num_samples: int = 5000  # Set to -1 for full dataset
    test_split: float = 0.2
    val_split: float = 0.1
    num_workers: int = 4


@dataclass
class OpsConfig:
    """LLM Ops settings."""
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "sentimentscope"
    model_registry_name: str = "sentiment_classifier"
    prometheus_port: int = 9090
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Singleton instances
model_config = ModelConfig()
training_config = TrainingConfig()
data_config = DataConfig()
ops_config = OpsConfig()
