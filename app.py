"""
SentimentScope - FastAPI REST API
LLM Ops: Model Serving with Health Checks, Logging & Metrics
"""
import os
import time
import pickle
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

MODEL_PATH = "models/sentiment_model.pkl"
LOG_PATH = "evaluations/api_requests.csv"

# Global model instance
model = None
request_count = 0
error_count = 0

# Pydantic models for request/response
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    prediction_id: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_requests: int
    total_errors: int

def load_model():
    """Load the trained model."""
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Train first!")
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")

def log_request(text, prediction, confidence, latency_ms):
    """Log every API request for observability."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)
    
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "text", "prediction", "confidence", "latency_ms"])
        writer.writerow([
            datetime.now().isoformat(),
            text[:100],
            prediction,
            f"{confidence:.2f}",
            f"{latency_ms:.2f}"
        ])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: Load model when API starts."""
    print("🚀 Starting SentimentScope API...")
    load_model()
    yield
    print("👋 Shutting down SentimentScope API...")

# Create FastAPI app
app = FastAPI(
    title="SentimentScope API",
    description="Production-ready Sentiment Analysis API with LLM Ops monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    global request_count, error_count
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        total_requests=request_count,
        total_errors=error_count
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    global request_count, error_count
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        total_requests=request_count,
        total_errors=error_count
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict sentiment of input text.
    
    - **text**: The text to analyze (max 1000 chars recommended)
    - Returns sentiment label and confidence score
    """
    global request_count, error_count
    
    if not request.text or not request.text.strip():
        error_count += 1
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if model is None:
        error_count += 1
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        pred = model.predict([request.text])[0]
        proba = model.predict_proba([request.text])[0]
        confidence = max(proba) * 100
        sentiment = "Positive" if pred == 1 else "Negative"
        latency_ms = (time.time() - start_time) * 1000
        
        request_count += 1
        
        # Log for observability
        log_request(request.text, sentiment, confidence, latency_ms)
        
        return PredictResponse(
            text=request.text[:100],
            sentiment=sentiment,
            confidence=round(confidence, 2),
            prediction_id=request_count
        )
    except Exception as e:
        error_count += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(texts: list[str]):
    """Predict sentiment for multiple texts at once."""
    global request_count, error_count
    
    if not texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for text in texts:
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        confidence = max(proba) * 100
        sentiment = "Positive" if pred == 1 else "Negative"
        
        results.append({
            "text": text[:100],
            "sentiment": sentiment,
            "confidence": round(confidence, 2)
        })
    
    request_count += len(texts)
    return {"predictions": results, "count": len(results)}

@app.get("/metrics")
async def metrics():
    """Simple metrics endpoint for monitoring."""
    global request_count, error_count
    return {
        "total_requests": request_count,
        "total_errors": error_count,
        "error_rate": f"{(error_count / max(request_count, 1)) * 100:.2f}%",
        "model_status": "loaded" if model is not None else "not_loaded"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)