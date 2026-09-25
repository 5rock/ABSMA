"""
Step 5: Train and Evaluate Sentiment Classifier
================================================
Trains a TF-IDF + Logistic Regression sentiment classifier on the full IMDb training set.
Evaluates on the IMDb test set and saves:
  - Model:       models/sentiment_model/tfidf_logreg_model.joblib
  - Metrics:     results/metrics/sentiment_metrics.json
  - Conf matrix: results/metrics/confusion_matrix.png
  - Predictions: results/predictions/sentiment_test_predictions.csv

Run from project root:
    python scripts/train_sentiment.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on path
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import load_imdb_raw, get_preprocessed_dataframe
from src.sentiment_classifier import SentimentClassifier

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
METRICS_DIR = os.path.join(project_root, "results", "metrics")
PREDICTIONS_DIR = os.path.join(project_root, "results", "predictions")
MODEL_DIR = os.path.join(project_root, "models", "sentiment_model")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("[1/5] Loading IMDb dataset...")
dataset = load_imdb_raw()

# Full training set (25,000 reviews)
train_df = get_preprocessed_dataframe(dataset, split="train", sample_size=None)
# Full test set (25,000 reviews)
test_df = get_preprocessed_dataframe(dataset, split="test", sample_size=None)

train_texts = train_df["clean_text"].tolist()
train_labels = train_df["label"].tolist()
test_texts = test_df["clean_text"].tolist()
test_labels = test_df["label"].tolist()

print(f"    Training samples : {len(train_texts):,}")
print(f"    Test samples     : {len(test_texts):,}")

# ─────────────────────────────────────────────────────────────
# 2. TRAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Training TF-IDF + Logistic Regression classifier...")
classifier = SentimentClassifier()
classifier.train(train_texts, train_labels)

# ─────────────────────────────────────────────────────────────
# 3. EVALUATE
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Evaluating on IMDb test set (25,000 reviews)...")
eval_results = classifier.evaluate(test_texts, test_labels)

# ─────────────────────────────────────────────────────────────
# 4. SAVE MODEL
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Saving model...")
classifier.save(model_dir=MODEL_DIR)

# ─────────────────────────────────────────────────────────────
# 5. SAVE METRICS + FIGURES + PREDICTIONS
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Saving metrics, confusion matrix, and predictions...")

# --- 5a. Metrics JSON ---
metrics_to_save = {
    "model": "TF-IDF + Logistic Regression",
    "dataset": "stanfordnlp/imdb",
    "train_samples": len(train_texts),
    "test_samples": len(test_texts),
    "accuracy": eval_results["accuracy"],
    "precision": eval_results["precision"],
    "recall": eval_results["recall"],
    "f1_score": eval_results["f1_score"],
    "confusion_matrix": eval_results["confusion_matrix"],
    "classification_report": eval_results["classification_report"],
    "note": (
        "These metrics reflect REVIEW-LEVEL sentiment classification. "
        "IMDb does not provide aspect-level sentiment labels."
    )
}
metrics_path = os.path.join(METRICS_DIR, "sentiment_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_to_save, f, indent=2)
print(f"  [SAVED] {metrics_path}")

# --- 5b. Confusion Matrix Plot ---
cm = np.array(eval_results["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
    linewidths=0.5, linecolor="gray",
    ax=ax
)
ax.set_title("Sentiment Classifier — Confusion Matrix\n(IMDb Test Set, 25,000 reviews)",
             fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(METRICS_DIR, "confusion_matrix.png")
fig.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  [SAVED] {cm_path}")

# --- 5c. Test Predictions CSV ---
pred_labels = eval_results["predictions"]
confidences = eval_results["confidences"]
label_map = {0: "negative", 1: "positive"}

pred_df = pd.DataFrame({
    "review_id": [f"imdb_test_{i+1}" for i in range(len(test_texts))],
    "text": [t[:300] for t in test_texts],
    "actual_sentiment": [label_map[l] for l in test_labels],
    "predicted_sentiment": [label_map[p] for p in pred_labels],
    "confidence": confidences
})
pred_csv_path = os.path.join(PREDICTIONS_DIR, "sentiment_test_predictions.csv")
pred_df.to_csv(pred_csv_path, index=False)
print(f"  [SAVED] {pred_csv_path} ({len(pred_df):,} rows)")

print(f"\n{'='*60}")
print(f"SENTIMENT CLASSIFIER TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Accuracy  : {eval_results['accuracy']*100:.2f}%")
print(f"  F1 Score  : {eval_results['f1_score']:.4f}")
print(f"  Model     : {os.path.join(MODEL_DIR, 'tfidf_logreg_model.joblib')}")
print(f"{'='*60}")
