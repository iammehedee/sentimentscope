"""
SentimentScope - Fully Offline Sentiment Analysis Training
Uses TF-IDF + Logistic Regression (no HuggingFace needed)
"""
import os
import csv
import random
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Set seed
random.seed(42)
np.random.seed(42)

# Config
DATA_PATH = "data/raw/sentiment_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")

def load_data(filepath):
    texts = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels

def main():
    print("=" * 60)
    print("🚀 SentimentScope - Offline Training Pipeline")
    print("=" * 60)

    # Load data
    print(f"\n📥 Loading data from {DATA_PATH}...")
    texts, labels = load_data(DATA_PATH)
    print(f"✅ Loaded {len(texts)} samples")
    
    pos = sum(labels)
    neg = len(labels) - pos
    print(f"   Positive: {pos} | Negative: {neg}")

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"   Train: {len(train_texts)} | Validation: {len(val_texts)}")

    # Build pipeline: TF-IDF + Logistic Regression
    print("\n🏋️  Training model (TF-IDF + Logistic Regression)...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words="english"
        )),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(train_texts, train_labels)

    # Evaluate
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)

    train_pred = pipeline.predict(train_texts)
    train_acc = accuracy_score(train_labels, train_pred)
    print(f"\n🎯 Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")

    val_pred = pipeline.predict(val_texts)
    val_acc = accuracy_score(val_labels, val_pred)
    print(f"🎯 Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

    print(f"\n📋 Classification Report:")
    print(classification_report(val_labels, val_pred, target_names=["Negative", "Positive"]))

    # Test predictions
    print("\n" + "=" * 60)
    print("🧪 TEST PREDICTIONS")
    print("=" * 60)
    
    test_samples = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible film, complete waste of time. I regret watching it.",
        "It was okay, nothing special but not bad either.",
        "Amazing cinematography and brilliant acting. A true masterpiece!",
        "Worst movie ever made. Don't waste your money on this.",
    ]
    
    for text in test_samples:
        pred = pipeline.predict([text])[0]
        proba = pipeline.predict_proba([text])[0]
        label = "😊 Positive" if pred == 1 else "😞 Negative"
        confidence = max(proba) * 100
        print(f"\n  Text: \"{text}\"")
        print(f"  → {label} (Confidence: {confidence:.1f}%)")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    
    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print("=" * 60)
    print("✅ Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()