"""
Pytest configuration and shared fixtures.
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment variables
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///test_mlflow.db"
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
