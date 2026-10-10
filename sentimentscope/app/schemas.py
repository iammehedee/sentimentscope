"""
Pydantic schemas for SentimentScope API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class PredictionRequest(BaseModel):
    """Request model for single or batch prediction."""
    text: str = Field(..., description="Text to classify", examples=["I loved this movie!"])
    return_probabilities: bool = Field(default=False, description="Return class probabilities")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    texts: List[str] = Field(..., description="List of texts to classify", min_length=1, max_length=128)


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    text: str
    sentiment: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    num_labels: int
    max_length: int
    version: str
    device: str
