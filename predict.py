"""
SentimentScope - Make Predictions with Trained Model
"""
import os
import pickle
import sys

MODEL_PATH = "models/sentiment_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Run 'python train_offline.py' first!")
        sys.exit(1)
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def predict_text(model, text):
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = max(proba) * 100
    label = "Positive" if pred == 1 else "Negative"
    return label, confidence

def main():
    model = load_model()
    print("=" * 60)
    print("🚀 SentimentScope - Prediction Mode")
    print("=" * 60)
    print("\nType a sentence and press Enter to analyze.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            text = input("💬 Enter text: ").strip()
            if text.lower() == "quit":
                print("👋 Goodbye!")
                break
            if not text:
                continue
            
            label, confidence = predict_text(model, text)
            emoji = "😊" if label == "Positive" else "😞"
            print(f"   {emoji} {label} (Confidence: {confidence:.1f}%)\n")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()