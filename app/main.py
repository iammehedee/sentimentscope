"""
FastAPI application for SentimentScope.

Provides REST endpoints for sentiment prediction with health checks,
metrics, and model information.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.predictor import SentimentPredictor
from src.model.config import model_config, ops_config

# Prometheus metrics
REQUEST_COUNT = Counter(
    "sentimentscope_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "sentimentscope_request_duration_seconds",
    "Request latency",
    ["endpoint"],
)
PREDICTION_COUNT = Counter(
    "sentimentscope_predictions_total",
    "Total predictions made",
    ["sentiment"],
)

# Global predictor instance
predictor: SentimentPredictor = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global predictor
    model_path = os.getenv("MODEL_PATH", model_config.pretrained_model)
    predictor = SentimentPredictor(
        model_path=model_path if model_path != "distilbert-base-uncased" else None,
        tokenizer_name=model_config.pretrained_model,
        max_length=model_config.max_length,
    )
    # If no custom model, load pretrained for demo
    if predictor.model is None:
        predictor.load_model(model_config.pretrained_model)
    print("🚀 SentimentScope API ready!")
    yield
    print("👋 Shutting down SentimentScope API...")


app = FastAPI(
    title="SentimentScope API",
    description="Production-Grade Sentiment Analysis API with LLM Ops",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SentimentScope",
        "version": "1.0.0",
        "description": "Production-Grade Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor and predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded if predictor else False,
        uptime_seconds=time.time() - start_time,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    return ModelInfoResponse(
        model_name=model_config.pretrained_model,
        num_labels=predictor.num_labels if predictor else model_config.num_labels,
        max_length=model_config.max_length,
        version="1.0.0",
        device=str(predictor.device) if predictor else "unknown",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict sentiment for a single text."""
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with REQUEST_LATENCY.labels(endpoint="/predict").time():
        try:
            result = predictor.predict(
                request.text,
                return_probabilities=request.return_probabilities,
            )
            REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
            PREDICTION_COUNT.labels(sentiment=result["sentiment"]).inc()

            response = PredictionResponse(
                text=request.text,
                sentiment=result["sentiment"],
                confidence=result["confidence"],
            )
            if request.return_probabilities:
                response.probabilities = result.get("probabilities")
            return response

        except Exception as e:
            REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict sentiment for a batch of texts."""
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with REQUEST_LATENCY.labels(endpoint="/predict/batch").time():
        try:
            results = predictor.predict_batch(request.texts)
            REQUEST_COUNT.labels(endpoint="/predict/batch", status="success").inc()

            predictions = [
                PredictionResponse(
                    text=text,
                    sentiment=result["sentiment"],
                    confidence=result["confidence"],
                )
                for text, result in zip(request.texts, results)
            ]
            return BatchPredictionResponse(predictions=predictions, count=len(predictions))

        except Exception as e:
            REQUEST_COUNT.labels(endpoint="/predict/batch", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def run():
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=ops_config.api_host,
        port=ops_config.api_port,
        reload=True,
    )


if __name__ == "__main__":
    run()
