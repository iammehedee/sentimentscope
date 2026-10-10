"""
Streamlit Dashboard for SentimentScope.

Interactive UI for testing the sentiment analysis model,
viewing metrics, and exploring predictions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="SentimentScope Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<p class="main-header">🎯 SentimentScope</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Production-Grade Sentiment Analysis Dashboard</p>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict", "📊 Batch Analysis", "📈 Metrics", "ℹ️ Model Info"],
)

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    api_status = health.get("status", "unknown")
except Exception:
    api_status = "offline"

st.sidebar.markdown(f"**API Status:** {'🟢 Online' if api_status == 'healthy' else '🔴 Offline'}")
st.sidebar.markdown(f"**Model Loaded:** {'✅ Yes' if health.get('model_loaded') else '❌ No' if api_status != 'offline' else 'N/A'}")

# ============================================================
# Page: Single Prediction
# ============================================================
if page == "🔮 Predict":
    st.header("🔮 Single Text Prediction")

    col1, col2 = st.columns([2, 1])

    with col1:
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste text here...",
            height=150,
        )
        show_probs = st.checkbox("Show class probabilities", value=True)

        if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={"text": text_input, "return_probabilities": show_probs},
                        )
                        result = response.json()

                        # Result card
                        sentiment = result["sentiment"]
                        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(
                            sentiment, "🤔"
                        )
                        color = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#FF9800"}.get(
                            sentiment, "#9E9E9E"
                        )

                        st.markdown(
                            f"""
                            <div style="
                                background: {color}15;
                                border-left: 4px solid {color};
                                padding: 1.5rem;
                                border-radius: 8px;
                                margin: 1rem 0;
                            ">
                                <h2 style="margin:0; color: {color};">
                                    {emoji} {sentiment.upper()}
                                </h2>
                                <p style="margin:0.5rem 0 0; font-size: 1.1rem;">
                                    Confidence: <strong>{result['confidence']:.1%}</strong>
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Probability bars
                        if show_probs and result.get("probabilities"):
                            st.subheader("📊 Class Probabilities")
                            probs_df = pd.DataFrame({
                                "Sentiment": list(result["probabilities"].keys()),
                                "Probability": list(result["probabilities"].values()),
                            })
                            fig = px.bar(
                                probs_df,
                                x="Sentiment",
                                y="Probability",
                                color="Sentiment",
                                text=probs_df["Probability"].apply(lambda x: f"{x:.1%}"),
                                range_y=[0, 1],
                            )
                            fig.update_traces(textposition="outside")
                            st.plotly_chart(fig, use_container_width=True)

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to API. Make sure it's running on localhost:8000")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Please enter some text to analyze.")

    with col2:
        st.subheader("💡 Tips")
        st.info(
            "Try entering:\n"
            "- Movie reviews\n"
            "- Product feedback\n"
            "- Social media posts\n"
            "- Customer support tickets"
        )

# ============================================================
# Page: Batch Analysis
# ============================================================
elif page == "📊 Batch Analysis":
    st.header("📊 Batch Text Analysis")

    batch_input = st.text_area(
        "Enter multiple texts (one per line):",
        placeholder="I loved this product!\nTerrible service, would not recommend.\nIt was okay, nothing special.",
        height=200,
    )

    if st.button("📈 Analyze Batch", type="primary"):
        if batch_input.strip():
            texts = [t.strip() for t in batch_input.split("\n") if t.strip()]
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        json={"texts": texts},
                    )
                    results = response.json()["predictions"]

                    # Summary metrics
                    df = pd.DataFrame([
                        {
                            "Text": r["text"][:80] + ("..." if len(r["text"]) > 80 else ""),
                            "Sentiment": r["sentiment"],
                            "Confidence": r["confidence"],
                        }
                        for r in results
                    ])

                    col1, col2, col3 = st.columns(3)
                    sentiments = df["Sentiment"].value_counts()
                    col1.metric("😊 Positive", sentiments.get("positive", 0))
                    col2.metric("😐 Neutral", sentiments.get("neutral", 0))
                    col3.metric("😞 Negative", sentiments.get("negative", 0))

                    # Pie chart
                    fig_pie = px.pie(
                        values=sentiments.values,
                        names=sentiments.index,
                        title="Sentiment Distribution",
                        color=sentiments.index,
                        color_discrete_map={
                            "positive": "#4CAF50",
                            "neutral": "#FF9800",
                            "negative": "#F44336",
                        },
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                    )

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API.")
        else:
            st.warning("⚠️ Please enter some texts to analyze.")

# ============================================================
# Page: Metrics
# ============================================================
elif page == "📈 Metrics":
    st.header("📈 System Metrics")

    try:
        metrics_text = requests.get(f"{API_URL}/metrics", timeout=2).text
        st.code(metrics_text, language="text")
    except Exception:
        st.warning("⚠️ Metrics endpoint not available. Start Prometheus + Grafana for full monitoring.")

    st.info(
        "📊 For full observability, run:\n"
        "```bash\ndocker compose -f docker/docker-compose.yml up prometheus grafana\n```\n"
        "Then visit http://localhost:3000 (admin/admin)"
    )

# ============================================================
# Page: Model Info
# ============================================================
elif page == "ℹ️ Model Info":
    st.header("ℹ️ Model Information")

    try:
        info = requests.get(f"{API_URL}/model/info", timeout=2).json()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", info["model_name"])
            st.metric("Version", info["version"])
        with col2:
            st.metric("Number of Labels", info["num_labels"])
            st.metric("Device", info["device"])

        st.subheader("🏗️ Architecture")
        st.code(
            """
SentimentClassifier
├── DistilBertModel (backbone)
├── Dropout (rate=0.1)
└── Linear Classifier (hidden_size → num_labels)
            """,
            language="text",
        )

        st.subheader("🔧 Tech Stack")
        tech_stack = pd.DataFrame([
            {"Component": "Model", "Technology": "DistilBERT (HuggingFace)"},
            {"Component": "Framework", "Technology": "PyTorch"},
            {"Component": "API", "Technology": "FastAPI"},
            {"Component": "Experiment Tracking", "Technology": "MLflow"},
            {"Component": "Containerization", "Technology": "Docker"},
            {"Component": "CI/CD", "Technology": "GitHub Actions"},
            {"Component": "Monitoring", "Technology": "Prometheus + Grafana"},
        ])
        st.dataframe(tech_stack, use_container_width=True, hide_index=True)

    except Exception:
        st.error("❌ Cannot fetch model info. Is the API running?")
