# 🎯 SentimentScope: Production-Grade Sentiment Analysis with Full LLM Ops

> A complete MLOps pipeline for sentiment analysis — from data download to model deployment, with experiment tracking, CI/CD, monitoring, and a beautiful dashboard.

!\[Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
!\[PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c.svg)
!\[FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)
!\[MLflow](https://img.shields.io/badge/MLflow-2.15-0194e0.svg)
!\[Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)
!\[License](https://img.shields.io/badge/License-MIT-green.svg)

\---

## 📋 Table of Contents

* [Overview](#-overview)
* [Architecture](#-architecture)
* [Features](#-features)
* [Quick Start](#-quick-start)
* [Usage Guide](#-usage-guide)
* [API Documentation](#-api-documentation)
* [MLOps Pipeline](#-mlops-pipeline)
* [Testing](#-testing)
* [Docker Deployment](#-docker-deployment)
* [CI/CD Pipeline](#-cicd-pipeline)
* [Monitoring \& Observability](#-monitoring--observability)
* [Project Structure](#-project-structure)
* [Tech Stack](#-tech-stack)
* [Contributing](#-contributing)
* [License](#-license)

\---

## 🎯 Overview

**SentimentScope** is a production-ready sentiment analysis system that demonstrates industry best practices in MLOps. It fine-tunes a DistilBERT model on sentiment datasets (IMDb, SST-2, TweetEval) and serves predictions via a FastAPI endpoint with full observability.

### Key Highlights

* **🤖 State-of-the-art Model**: DistilBERT fine-tuned for sentiment classification
* **📊 Experiment Tracking**: MLflow logs every training run with params, metrics, and artifacts
* **🚀 Model Registry**: Versioned model promotion (Staging → Production)
* **🔄 CI/CD Pipeline**: Automated testing, training validation, Docker build, and deployment
* **🐳 Containerized**: Docker + Docker Compose for reproducible deployments
* **📈 Monitoring**: Prometheus metrics + Grafana dashboards
* **🎨 Interactive Dashboard**: Streamlit UI for testing and visualization
* **🧪 Comprehensive Testing**: Unit tests, integration tests, model validation

\---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit Dashboard                        │
│                   (Interactive UI for users)                      │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Prediction Service                     │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│  │ /predict   │  │ /predict/    │  │ /health + /metrics      │   │
│  │ (single)   │  │ batch        │  │ (Prometheus)            │   │
│  └────────────┘  └──────────────┘  └────────────────────────┘   │
└────────┬─────────────────────────────────┬───────────────────────┘
         │                                 │
         ▼                                 ▼
┌────────────────────┐         ┌────────────────────────┐
│  SentimentPredictor │         │   Prometheus Metrics   │
│  (DistilBERT +     │         │   (request count,      │
│   classification)   │         │    latency, preds)     │
│                    │         │                        │
└────────┬───────────┘         └───────────┬────────────┘
         │                                 │
         ▼                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                      MLflow Tracking Server                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ Experiments  │  │  Runs        │  │  Model Registry     │     │
│  │ (params,     │  │  (versions,  │  │  (Staging → Prod)   │     │
│  │  metrics)    │  │   artifacts) │  │                     │     │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Training Pipeline                            │
│  Download → Preprocess → Train → Evaluate → Register             │
│  (PyTorch + Transformers + MLflow autologging)                   │
└──────────────────────────────────────────────────────────────────┘
```

\---

## ✨ Features

|Feature|Description|
|-|-|
|**Multi-Dataset Support**|IMDb, SST-2, TweetEval (binary \& 3-class)|
|**DistilBERT Fine-tuning**|Transfer learning with HuggingFace Transformers|
|**MLflow Tracking**|Every run logged with hyperparameters, metrics, and artifacts|
|**Model Registry**|Versioned models with staging/production promotion|
|**FastAPI Service**|Async REST API with OpenAPI docs|
|**Batch Predictions**|Process multiple texts in one request|
|**Health Checks**|Kubernetes-ready liveness/readiness probes|
|**Prometheus Metrics**|Request count, latency histograms, prediction distribution|
|**Grafana Dashboards**|Visualize model performance and system health|
|**Streamlit UI**|Beautiful dashboard for interactive testing|
|**Docker Support**|Multi-container setup with docker-compose|
|**CI/CD**|GitHub Actions: lint → test → train → build → deploy|
|**Comprehensive Tests**|Unit tests (model), integration tests (API), coverage reporting|

\---

## 🚀 Quick Start

### Prerequisites

* Python 3.10+
* Docker \& Docker Compose (optional, for containerized deployment)
* Git

### 1\. Clone \& Setup

```bash
git clone https://github.com/yourusername/sentimentscope.git
cd sentimentscope
python -m venv .venv \&\& source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2\. Download Dataset

```bash
# Download IMDb dataset (5000 samples for quick training)
python src/data/download.py --dataset imdb --num-samples 5000

# Or use SST-2 for faster training
python src/data/download.py --dataset sst2 --num-samples 2000
```

### 3\. Train Model

```bash
# Start MLflow server (in another terminal)
mlflow server --host 0.0.0.0 --port 5000

# Train the model
python src/train.py --dataset imdb --epochs 3 --batch-size 16
```

### 4\. Run API

```bash
# Set model path in .env
echo "MODEL\_PATH=$(pwd)/models/best\_model\_\*.pt" >> .env

# Start API
uvicorn app.main:app --reload
```

### 5\. Launch Dashboard

```bash
streamlit run dashboard/streamlit\_app.py
```

\---

## 📖 Usage Guide

### Training a Model

```bash
# Full training with custom parameters
python src/train.py \\
  --dataset imdb \\
  --epochs 3 \\
  --batch-size 16 \\
  --lr 2e-5 \\
  --max-length 128 \\
  --run-name "distilbert-imdb-v1"
```

Training progress is logged to MLflow:

|Metric|Description|
|-|-|
|`train\_loss`|Training cross-entropy loss|
|`val\_loss`|Validation loss|
|`val\_accuracy`|Validation accuracy|
|`val\_f1`|Weighted F1 score|
|`test\_accuracy`|Final test set accuracy|

### Evaluating a Model

```bash
python src/evaluate.py \\
  --model-path models/best\_model\_abc123.pt \\
  --dataset imdb
```

### Making Predictions

```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "text": "I absolutely loved this movie!",
    "return\_probabilities": True
})
print(response.json())
# {'text': 'I absolutely loved this movie!', 'sentiment': 'positive', 'confidence': 0.97, ...}

# Batch prediction
response = requests.post("http://localhost:8000/predict/batch", json={
    "texts": \["Great!", "Terrible!", "Meh."]
})
print(response.json())
```

\---

## 🔌 API Documentation

Once the API is running, visit:

* **Swagger UI**: http://localhost:8000/docs
* **ReDoc**: http://localhost:8000/redoc

### Endpoints

|Method|Endpoint|Description|
|-|-|-|
|`GET`|`/`|API info|
|`GET`|`/health`|Health check|
|`GET`|`/model/info`|Model metadata|
|`POST`|`/predict`|Single text prediction|
|`POST`|`/predict/batch`|Batch prediction (up to 128 texts)|
|`GET`|`/metrics`|Prometheus metrics|

\---

## 🔄 MLOps Pipeline

### Experiment Tracking with MLflow

Every training run automatically logs:

1. **Parameters**: model architecture, hyperparameters, dataset info
2. **Metrics**: loss, accuracy, precision, recall, F1 (train/val/test)
3. **Artifacts**: model checkpoints, classification reports
4. **Model**: Registered in MLflow Model Registry

```bash
# View experiments
mlflow ui --port 5000
# Open http://localhost:5000
```

### Model Registry Workflow

```
┌──────────┐    ┌──────────┐    ┌─────────────┐
│ Training │───▶│ Staging  │───▶│  Production │
│  Run     │    │ (testing)│    │  (serving)  │
└──────────┘    └──────────┘    └─────────────┘
```

1. Train a new model → registered as `version 1` in Staging
2. Run validation tests → promote to Production
3. API automatically loads the latest Production version

\---

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov=app --cov-report=html

# Run specific test files
pytest tests/test\_model.py -v
pytest tests/test\_api.py -v
```

Coverage report is generated in `htmlcov/index.html`.

\---

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Start all services (API + MLflow + Prometheus + Grafana + Dashboard)
docker compose -f docker/docker-compose.yml up -d

# View logs
docker compose -f docker/docker-compose.yml logs -f api

# Stop all services
docker compose -f docker/docker-compose.yml down
```

### Services

|Service|Port|URL|
|-|-|-|
|API|8000|http://localhost:8000|
|API Docs|8000|http://localhost:8000/docs|
|MLflow|5000|http://localhost:5000|
|Prometheus|9090|http://localhost:9090|
|Grafana|3000|http://localhost:3000 (admin/admin)|
|Dashboard|8501|http://localhost:8501|

### Build \& Run Standalone

```bash
docker build -f docker/Dockerfile -t sentimentscope .
docker run -p 8000:8000 -e MODEL\_PATH=/app/models/model.pt sentimentscope
```

\---

## 🔧 CI/CD Pipeline

The GitHub Actions pipeline runs on every push/PR:

```
┌─────────┐  ┌─────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────┐
│  Lint   │─▶│  Test   │─▶│ Train Val.   │─▶│ Docker  │─▶│  Deploy  │
│ (black, │  │ (pytest, │  │ (quick fit)  │  │ (build  │  │ (main    │
│ flake8) │  │ coverage)│  │              │  │  test)  │  │  branch) │
└─────────┘  └─────────┘  └──────────────┘  └─────────┘  └──────────┘
```

* **Lint**: Code style checks (Black, isort, flake8)
* **Test**: Unit + integration tests across Python 3.10/3.11/3.12
* **Train Validation**: Quick training run to validate pipeline
* **Docker**: Build and test container image
* **Deploy**: Deploy to cloud (configure your provider)

\---

## 📊 Monitoring \& Observability

### Prometheus Metrics

|Metric|Type|Description|
|-|-|-|
|`sentimentscope\_requests\_total`|Counter|Total API requests by endpoint/status|
|`sentimentscope\_request\_duration\_seconds`|Histogram|Request latency|
|`sentimentscope\_predictions\_total`|Counter|Predictions by sentiment class|

### Grafana Dashboards

Pre-configured dashboard available at `monitoring/grafana-dashboard.json`.

\---

## 📁 Project Structure

```
sentimentscope/
├── .github/workflows/ci-cd.yml   # GitHub Actions CI/CD
├── app/                            # FastAPI application
│   ├── main.py                     # API endpoints
│   ├── schemas.py                  # Pydantic models
│   └── predictor.py                # Model inference
├── src/                            # Core source code
│   ├── train.py                    # Training pipeline
│   ├── evaluate.py                 # Evaluation script
│   ├── data/
│   │   ├── download.py             # Dataset downloader
│   │   └── preprocess.py           # Tokenization \& DataLoaders
│   └── model/
│       ├── architecture.py         # DistilBERT classifier
│       └── config.py               # Centralized config
├── tests/                          # Test suite
│   ├── test\_model.py               # Model unit tests
│   ├── test\_api.py                 # API integration tests
│   └── conftest.py                 # Pytest fixtures
├── monitoring/                     # Observability
│   ├── prometheus.yml              # Prometheus config
│   └── grafana-dashboard.json      # Grafana dashboard
├── dashboard/                      # Streamlit UI
│   └── streamlit\_app.py
├── docker/                         # Container config
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/                           # Datasets (gitignored)
├── models/                         # Saved models (gitignored)
├── mlruns/                         # MLflow logs (gitignored)
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

\---

## 🛠️ Tech Stack

|Layer|Technology|
|-|-|
|**Model**|PyTorch, HuggingFace Transformers (DistilBERT)|
|**Experiment Tracking**|MLflow|
|**API Framework**|FastAPI, Uvicorn|
|**Data**|HuggingFace Datasets|
|**Metrics**|scikit-learn, Prometheus|
|**Dashboard**|Streamlit, Plotly|
|**Testing**|pytest, httpx|
|**Containerization**|Docker, Docker Compose|
|**CI/CD**|GitHub Actions|
|**Monitoring**|Prometheus, Grafana|

\---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

* **Deep Learning**: Transformer fine-tuning, PyTorch, transfer learning
* **MLOps**: Experiment tracking, model registry, reproducible pipelines
* **Software Engineering**: Clean code, testing, CI/CD, containerization
* **API Development**: RESTful services, async programming, OpenAPI
* **DevOps**: Docker, GitHub Actions, monitoring, observability
* **Data Engineering**: Dataset management, preprocessing pipelines

\---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

\---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

\---

<div align="center">
  <strong>Built with ❤️ for the AI/ML community</strong>
</div>
# SentimentScope 🔍



> A production-ready sentiment analysis API with ML training, evaluation, and observability.



\[!\[Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

\[!\[FastAPI](https://img.shields.io/badge/FastAPI-0.143.0-green.svg)](https://fastapi.tiangolo.com/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



\## 📌 Overview



SentimentScope is an end-to-end sentiment analysis system demonstrating core ML Ops practices:



\- \*\*Training Pipeline\*\* — TF-IDF + Logistic Regression

\- \*\*Evaluation Suite\*\* — Automated testing with logging

\- \*\*REST API\*\* — FastAPI-based model serving

\- \*\*Observability\*\* — Request logging, health checks, metrics

\- \*\*Containerization\*\* — Docker-ready deployment

\- \*\*CI/CD\*\* — GitHub Actions pipeline



\## 🚀 Quick Start



\### 1. Clone \& Setup

