"""
Integration tests for SentimentScope API.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["name"] == "SentimentScope"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict(client):
    response = client.post(
        "/predict",
        json={"text": "I absolutely loved this movie, it was fantastic!", "return_probabilities": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert data["sentiment"] in ["positive", "negative", "neutral"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_batch(client):
    texts = [
        "This is amazing!",
        "I hate this so much.",
        "It's okay, nothing special.",
    ]
    response = client.post(
        "/predict/batch",
        json={"texts": texts},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    for pred in data["predictions"]:
        assert "sentiment" in pred
        assert "confidence" in pred


def test_predict_empty_text(client):
    response = client.post(
        "/predict",
        json={"text": ""},
    )
    # Should still process (empty string is valid input)
    assert response.status_code == 200


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "num_labels" in data
    assert "version" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "sentimentscope" in response.text
