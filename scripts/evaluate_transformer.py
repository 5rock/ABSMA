import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.transformer_model import TransformerSentimentClassifier

def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'], 
                yticklabels=['Negative', 'Neutral', 'Positive'], ax=ax)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main():
    print("Loading datasets...")
    train_path = os.path.join(project_root, "data", "processed", "absa_training.csv")
    test_path = os.path.join(project_root, "data", "processed", "absa_test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    if train_df['sentiment'].dtype == object:
        train_df['sentiment'] = train_df['sentiment'].map(label_map)
    if test_df['sentiment'].dtype == object:
        test_df['sentiment'] = test_df['sentiment'].map(label_map)
        
    print("\nTraining TF-IDF + Logistic Regression on aspect-level training data...")
    # For fair comparison, TF-IDF should train on the exact same context+aspect inputs
    train_texts = train_df['review'] + " " + train_df['aspect']
    test_texts = test_df['review'] + " " + test_df['aspect']
    
    tfidf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    tfidf_pipeline.fit(train_texts, train_df['sentiment'])
    
    print("\nLoading Fine-tuned BERT...")
    transformer_model = TransformerSentimentClassifier()
    
    print("\nRunning inference...")
    baseline_preds = tfidf_pipeline.predict(test_texts)
    transformer_preds = []
    
    str_to_int = {"negative": 0, "neutral": 1, "positive": 2, "unknown": 1}
    for _, row in test_df.iterrows():
        t_pred = transformer_model.predict(row['review'], aspect=row['aspect'])
        transformer_preds.append(str_to_int.get(t_pred['sentiment'], 1))
        
    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        return acc, p, r, f1, cm
        
    true_labels = test_df['sentiment'].tolist()
    b_acc, b_p, b_r, b_f1, b_cm = get_metrics(true_labels, baseline_preds)
    t_acc, t_p, t_r, t_f1, t_cm = get_metrics(true_labels, transformer_preds)
    
    print("\n" + "="*80)
    print("Academic Evaluation Results:")
    print("="*80)
    print(f"{'Model':<35} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-"*80)
    print(f"{'TF-IDF + Logistic Regression':<35} {b_acc:.4f}     {b_p:.4f}      {b_r:.4f}    {b_f1:.4f}")
    print(f"{'Fine-tuned BERT':<35} {t_acc:.4f}     {t_p:.4f}      {t_r:.4f}    {t_f1:.4f}")
    print("="*80)
    
    metrics_dir = os.path.join(project_root, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    plot_cm(b_cm, "TF-IDF + LogReg Baseline", os.path.join(metrics_dir, "tfidf_confusion_matrix.png"))
    plot_cm(t_cm, "Fine-Tuned BERT", os.path.join(metrics_dir, "bert_confusion_matrix.png"))
    
    results_json = {
        "dataset_used": "Development Data (Synthetic Aspect-Level)",
        "number_of_training_samples": len(train_df),
        "number_of_test_samples": len(test_df),
        "class_distribution_train": train_df['sentiment'].value_counts().to_dict(),
        "class_distribution_test": test_df['sentiment'].value_counts().to_dict(),
        "bert_training_configuration": {
            "model_name": "bert-base-uncased",
            "epochs": 40,
            "max_length": 64,
            "batch_size": 16,
            "device": "CPU"
        },
        "bert_metrics": {
            "accuracy": t_acc,
            "precision": t_p,
            "recall": t_r,
            "macro_f1": t_f1
        },
        "tfidf_metrics": {
            "accuracy": b_acc,
            "precision": b_p,
            "recall": b_r,
            "macro_f1": b_f1
        },
        "confusion_matrices_saved": True,
        "note": "Evaluation performed on a proper held-out aspect-level dataset."
    }
    
    with open(os.path.join(metrics_dir, "academic_evaluation.json"), "w") as f:
        json.dump(results_json, f, indent=4)
        
    print("\nSaved academic_evaluation.json and confusion matrices to results/metrics/")

if __name__ == "__main__":
    main()
