"""
Sentiment Classifier Module
============================
Implements a TF-IDF + Logistic Regression sentiment classifier trained on IMDb reviews.
Chosen for CPU efficiency (trains in ~30 seconds) and strong accuracy (~88-90%) on IMDb.

NOTE: This classifier produces REVIEW-LEVEL sentiment predictions.
When used inside the ABSA pipeline, it is applied to individual aspect CONTEXT SENTENCES,
producing ASPECT-LEVEL sentiment predictions that are automatically generated and
NOT gold-standard annotations (IMDb does not provide aspect-level labels).
"""

import os
import json
import joblib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def get_project_root() -> str:
    """Returns absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))


class SentimentClassifier:
    """
    TF-IDF + Logistic Regression sentiment classifier.

    Supports binary sentiment classification: positive / negative.
    Can be saved and reloaded without retraining.
    """

    # Label mapping: IMDb label 0 = negative, 1 = positive
    LABEL_MAP = {0: "negative", 1: "positive"}
    LABEL_REVERSE = {"negative": 0, "positive": 1}

    def __init__(self):
        """Initialize the sklearn pipeline (TF-IDF + Logistic Regression)."""
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50000,     # Top 50k vocabulary terms
                ngram_range=(1, 2),     # Unigrams and bigrams
                sublinear_tf=True,      # Apply sublinear TF scaling (log)
                min_df=3,               # Ignore very rare terms
                max_df=0.95,            # Ignore overly common terms
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b"
            )),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=500,
                solver="lbfgs",
                random_state=42
            ))
        ])
        self.is_trained = False

    def train(self, texts: List[str], labels: List[int]) -> None:
        """
        Train the sentiment classifier.

        Args:
            texts:  List of preprocessed review texts.
            labels: Integer labels (0 = negative, 1 = positive).
        """
        print(f"[INFO] Training sentiment classifier on {len(texts)} samples...")
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        print("[INFO] Training complete.")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.

        Args:
            text: Input review or context sentence.

        Returns:
            Dict with keys: sentiment (str), confidence (float), label (int)
        """
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained. Call train() or load() first.")
        if not text or not text.strip():
            return {"sentiment": "negative", "confidence": 0.5, "label": 0}

        proba = self.pipeline.predict_proba([text])[0]
        pred_label = int(np.argmax(proba))
        confidence = round(float(proba[pred_label]), 4)
        sentiment = self.LABEL_MAP[pred_label]

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "label": pred_label
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of prediction dicts.
        """
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained. Call train() or load() first.")
        if not texts:
            return []

        probas = self.pipeline.predict_proba(texts)
        results = []
        for proba in probas:
            pred_label = int(np.argmax(proba))
            confidence = round(float(proba[pred_label]), 4)
            results.append({
                "sentiment": self.LABEL_MAP[pred_label],
                "confidence": confidence,
                "label": pred_label
            })
        return results

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """
        Evaluate classifier on a labeled test set.

        Args:
            texts:  List of review texts.
            labels: True integer labels.

        Returns:
            Dict with accuracy, precision, recall, f1, confusion_matrix,
            classification_report, and per-sample predictions.
        """
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained. Call train() or load() first.")

        predictions = self.pipeline.predict(texts)
        probas = self.pipeline.predict_proba(texts)
        confidences = [round(float(p[pred]), 4) for p, pred in zip(probas, predictions)]

        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions, average="binary", pos_label=1)
        rec = recall_score(labels, predictions, average="binary", pos_label=1)
        f1 = f1_score(labels, predictions, average="binary", pos_label=1)
        cm = confusion_matrix(labels, predictions).tolist()
        report = classification_report(
            labels, predictions,
            target_names=["negative", "positive"],
            output_dict=True
        )

        print(f"\n[EVAL] Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
        print(f"[EVAL] Precision : {prec:.4f}")
        print(f"[EVAL] Recall    : {rec:.4f}")
        print(f"[EVAL] F1 Score  : {f1:.4f}")
        print(f"\n[EVAL] Classification Report:")
        print(classification_report(labels, predictions, target_names=["negative", "positive"]))

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "classification_report": report,
            "predictions": [int(p) for p in predictions],
            "confidences": confidences
        }

    def save(self, model_dir: Optional[str] = None) -> str:
        """
        Save the trained classifier pipeline to disk using joblib.

        Args:
            model_dir: Directory to save the model. Defaults to models/sentiment_model/.

        Returns:
            Path to saved model file.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained classifier.")
        if model_dir is None:
            model_dir = os.path.join(get_project_root(), "models", "sentiment_model")

        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "tfidf_logreg_model.joblib")
        joblib.dump(self.pipeline, model_path)
        print(f"[INFO] Model saved to: {model_path}")
        return model_path

    @classmethod
    def load(cls, model_dir: Optional[str] = None) -> "SentimentClassifier":
        """
        Load a previously saved classifier from disk.

        Args:
            model_dir: Directory containing the saved model file.

        Returns:
            Loaded SentimentClassifier instance ready for prediction.
        """
        if model_dir is None:
            model_dir = os.path.join(get_project_root(), "models", "sentiment_model")

        model_path = os.path.join(model_dir, "tfidf_logreg_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No saved model found at: {model_path}\n"
                "Run scripts/train_sentiment.py first to train and save the model."
            )
        instance = cls()
        instance.pipeline = joblib.load(model_path)
        instance.is_trained = True
        print(f"[INFO] Model loaded from: {model_path}")
        return instance
