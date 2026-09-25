"""
End-to-End ABSA Pipeline Runner
=================================
Runs the complete ABSA project pipeline in correct order:
  1. Load IMDb dataset
  2. Preprocess data
  3. Train sentiment classifier (if not already trained)
  4. Extract aspects from IMDb sample
  5. Run full ABSA on sample reviews
  6. Save predictions + summary metrics

Run from project root:
    python scripts/run_pipeline.py
"""

import os
import sys
import json
import pandas as pd

# Ensure project root is on path
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import load_imdb_raw, get_preprocessed_dataframe
from src.sentiment_classifier import SentimentClassifier
from src.absa_pipeline import ABSAPipeline

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SAMPLE_SIZE = 50            # Number of IMDb reviews for ABSA
SEED = 42
MODEL_DIR = os.path.join(project_root, "models", "sentiment_model")
PREDICTIONS_DIR = os.path.join(project_root, "results", "predictions")
METRICS_DIR = os.path.join(project_root, "results", "metrics")

os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

print("=" * 60)
print("  COMPLETE ABSA PIPELINE")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# [1/6] LOAD DATASET
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading IMDb dataset...")
dataset = load_imdb_raw()
print(f"      Dataset loaded. Train: 25,000 | Test: 25,000 reviews")

# ─────────────────────────────────────────────────────────────
# [2/6] PREPROCESS DATA
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Preprocessing data...")
# Sample for ABSA (train split, mixed pos/neg)
sample_df = get_preprocessed_dataframe(dataset, split="train", sample_size=SAMPLE_SIZE, seed=SEED)
texts = sample_df["clean_text"].tolist()
labels = sample_df["sentiment_label"].tolist()
print(f"      Prepared {len(texts)} review samples.")

# ─────────────────────────────────────────────────────────────
# [3/6] TRAIN SENTIMENT CLASSIFIER (only if not already saved)
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Setting up sentiment classifier...")
model_path = os.path.join(MODEL_DIR, "tfidf_logreg_model.joblib")

if os.path.exists(model_path):
    print(f"      Found existing model at: {model_path}")
    print("      Loading pre-trained model (skipping training).")
    classifier = SentimentClassifier.load(MODEL_DIR)
else:
    print("      No saved model found. Training now on full IMDb train set...")
    print("      (This takes ~30 seconds on CPU)")
    train_df = get_preprocessed_dataframe(dataset, split="train", sample_size=None)
    classifier = SentimentClassifier()
    classifier.train(train_df["clean_text"].tolist(), train_df["label"].tolist())
    classifier.save(MODEL_DIR)
    print("      Sentiment classifier trained and saved.")

# ─────────────────────────────────────────────────────────────
# [4/6] EXTRACT ASPECTS + RUN ABSA
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Initializing ABSA pipeline and extracting aspects...")
absa_pipeline = ABSAPipeline(
    sentiment_classifier=classifier,
    use_transformer_ner=True
)

print(f"\n[5/6] Running ABSA on {SAMPLE_SIZE} IMDb reviews...")
review_ids = [f"imdb_train_{i+1}" for i in range(SAMPLE_SIZE)]
absa_results = absa_pipeline.analyze_batch(texts, review_ids=review_ids, verbose=True)

# Print a sample result for the first review
if absa_results:
    print("\nSample output for Review #1:")
    print(absa_pipeline.format_result_table(absa_results[0]))

# ─────────────────────────────────────────────────────────────
# [6/6] SAVE PREDICTIONS + METRICS
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Saving results...")

# --- 05_absa_predictions.csv ---
rows = []
for result, overall_label in zip(absa_results, labels):
    for asp in result["aspects"]:
        rows.append({
            "review_id": result["review_id"],
            "review_snippet": result["original_text"][:200],
            "overall_imdb_sentiment": overall_label,
            "aspect": asp["aspect"],
            "category": asp["category"],
            "entity_type": asp["entity_type"],
            "sentiment": asp["sentiment"],
            "confidence": asp["confidence"],
            "context_sentence": asp["context_sentence"][:200],
            "source": "absa_pipeline"
        })

absa_df = pd.DataFrame(rows)
absa_csv = os.path.join(PREDICTIONS_DIR, "05_absa_predictions.csv")
absa_df.to_csv(absa_csv, index=False)
print(f"  [SAVED] {absa_csv} ({len(absa_df)} aspect rows)")

# --- 05_absa_summary.json ---
total_aspects = len(rows)
sentiment_dist = absa_df["sentiment"].value_counts().to_dict() if not absa_df.empty else {}
category_dist = absa_df["category"].value_counts().to_dict() if not absa_df.empty else {}
reviews_with_aspects = sum(1 for r in absa_results if r["aspect_count"] > 0)
avg_aspects = total_aspects / len(absa_results) if absa_results else 0

# Count overall sentiment distribution
overall_pos = sum(1 for r in absa_results if r["overall_sentiment"]["sentiment"] == "positive")
overall_neg = sum(1 for r in absa_results if r["overall_sentiment"]["sentiment"] == "negative")

summary = {
    "pipeline": "TF-IDF Logistic Regression + BERT NER + SpaCy + Domain Taxonomy",
    "total_reviews_processed": len(absa_results),
    "total_aspects_extracted": total_aspects,
    "reviews_with_aspects": reviews_with_aspects,
    "avg_aspects_per_review": round(avg_aspects, 2),
    "unique_aspect_categories": len(category_dist),
    "overall_review_sentiment_distribution": {
        "positive": overall_pos,
        "negative": overall_neg
    },
    "aspect_sentiment_distribution": sentiment_dist,
    "category_distribution": category_dist,
    "annotation_note": (
        "IMDb provides overall review-level sentiment labels only. "
        "Aspect-level sentiments are AUTOMATICALLY GENERATED predictions — NOT gold truth."
    )
}

summary_path = os.path.join(METRICS_DIR, "05_absa_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  [SAVED] {summary_path}")

print(f"\n{'='*60}")
print(f"PIPELINE COMPLETE")
print(f"{'='*60}")
print(f"  Reviews processed     : {len(absa_results)}")
print(f"  Total aspects found   : {total_aspects}")
print(f"  Avg aspects/review    : {avg_aspects:.2f}")
print(f"  Sentiment distribution: {sentiment_dist}")
print(f"\n  Output files:")
print(f"    {absa_csv}")
print(f"    {summary_path}")
print(f"{'='*60}")
