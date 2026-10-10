"""
SentimentScope - LLM Ops Evaluation Pipeline
Logs all predictions, tracks metrics, and saves evaluation results.
"""
import os
import csv
import json
import time
import pickle
from datetime import datetime

MODEL_PATH = "models/sentiment_model.pkl"
EVAL_DIR = "evaluations"
EVAL_LOG = os.path.join(EVAL_DIR, "eval_log.csv")

class EvalLogger:
    """LLM Ops: Logs every prediction with metadata for monitoring."""
    
    def __init__(self):
        os.makedirs(EVAL_DIR, exist_ok=True)
        if not os.path.exists(EVAL_LOG):
            with open(EVAL_LOG, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "text", "prediction", "confidence", "latency_ms"])
    
    def log(self, text, prediction, confidence, latency_ms):
        with open(EVAL_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                text[:100],  # Truncate long texts
                prediction,
                f"{confidence:.2f}",
                f"{latency_ms:.2f}"
            ])

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def evaluate_model(model):
    """Run evaluation on a test set and generate metrics."""
    print("=" * 60)
    print("📊 SentimentScope - LLM Ops Evaluation")
    print("=" * 60)

    # Test set
    test_cases = [
        ("This movie is absolutely fantastic! I loved every minute!", 1),
        ("Terrible film, complete waste of time.", 0),
        ("Amazing cinematography and brilliant acting!", 1),
        ("Worst movie ever made. Don't watch it.", 0),
        ("It was okay, nothing special but not bad either.", 0),
        ("A true masterpiece of modern cinema!", 1),
        ("The plot was confusing and poorly executed.", 0),
        ("I was deeply moved by the story. Five stars!", 1),
        ("Boring from start to finish. I fell asleep.", 0),
        ("Outstanding performance by the entire cast.", 1),
    ]

    logger = EvalLogger()
    correct = 0
    latencies = []

    print(f"\n🧪 Running {len(test_cases)} test cases...\n")
    print(f"{'Text':<45} {'Expected':<12} {'Predicted':<12} {'Confidence':<12} {'Result'}")
    print("-" * 90)

    for text, expected in test_cases:
        start = time.time()
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        confidence = max(proba) * 100
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        label_pred = "Positive" if pred == 1 else "Negative"
        label_exp = "Positive" if expected == 1 else "Negative"
        is_correct = "✅" if pred == expected else "❌"
        result = "PASS" if pred == expected else "FAIL"

        print(f"{text[:43]:<45} {label_exp:<12} {label_pred:<12} {confidence:<12.1f} {result} {is_correct}")

        logger.log(text, label_pred, confidence, latency_ms)
        if pred == expected:
            correct += 1

    accuracy = correct / len(test_cases)
    avg_latency = sum(latencies) / len(latencies)

    print("-" * 90)
    print(f"\n🎯 Accuracy: {accuracy:.2%} ({correct}/{len(test_cases)})")
    print(f"⚡ Avg Latency: {avg_latency:.2f} ms")
    print(f"📁 Logged to: {EVAL_LOG}")

    # Save evaluation report as JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_PATH,
        "total_tests": len(test_cases),
        "correct": correct,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "test_cases": [
            {"text": t, "expected": e, "predicted": int(model.predict([t])[0])}
            for t, e in test_cases
        ]
    }

    report_path = os.path.join(EVAL_DIR, f"eval_report_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📄 Report saved: {report_path}")
    print("=" * 60)

    return accuracy

if __name__ == "__main__":
    model = load_model()
    evaluate_model(model)