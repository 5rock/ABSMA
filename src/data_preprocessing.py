"""
Data Preprocessing and Exploratory Data Analysis (EDA) Module for IMDb Reviews
Provides dataset loading, caching, cleaning, statistical summary, and publication-ready EDA plots.
"""

import os
import re
import html
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, DatasetDict, load_from_disk


def get_project_root() -> str:
    """Returns absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))


def clean_text(text: str) -> str:
    """
    Cleans unstructured movie review text by:
    - Removing HTML tags (e.g. <br />, <p>)
    - Unescaping HTML entities (e.g. &quot;, &#39;)
    - Normalizing irregular whitespace
    - Preserving punctuation and casing relevant to sentiment and NER
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_imdb_raw(
    save_to_disk: bool = True,
    raw_dir: Optional[str] = None
) -> DatasetDict:
    """
    Loads the official stanfordnlp/imdb dataset from Hugging Face Hub or local cache.
    """
    if raw_dir is None:
        raw_dir = os.path.join(get_project_root(), "data", "raw", "imdb")
    
    os.makedirs(raw_dir, exist_ok=True)
    saved_dataset_path = os.path.join(raw_dir, "hf_dataset")
    
    if os.path.exists(saved_dataset_path):
        try:
            print(f"[INFO] Loading cached IMDb dataset from: {saved_dataset_path}")
            return load_from_disk(saved_dataset_path)
        except Exception as e:
            print(f"[WARN] Failed to load from disk ({e}), re-downloading from Hugging Face Hub...")
    
    print("[INFO] Fetching 'stanfordnlp/imdb' from Hugging Face Hub...")
    dataset = load_dataset("stanfordnlp/imdb")
    
    if save_to_disk:
        try:
            print(f"[INFO] Caching dataset locally to: {saved_dataset_path}")
            dataset.save_to_disk(saved_dataset_path)
            
            train_csv = os.path.join(raw_dir, "imdb_train.csv")
            test_csv = os.path.join(raw_dir, "imdb_test.csv")
            
            if not os.path.exists(train_csv):
                dataset["train"].to_pandas().to_csv(train_csv, index=False)
                print(f"[INFO] Saved train CSV: {train_csv}")
            if not os.path.exists(test_csv):
                dataset["test"].to_pandas().to_csv(test_csv, index=False)
                print(f"[INFO] Saved test CSV: {test_csv}")
        except Exception as e:
            print(f"[WARN] Could not cache to disk: {e}")
            
    return dataset


def get_dataset_summary(dataset: DatasetDict) -> Dict[str, Any]:
    """Computes top-level counts, features, and split distributions."""
    summary = {}
    for split_name in dataset.keys():
        ds_split = dataset[split_name]
        df = ds_split.to_pandas()
        
        df["char_length"] = df["text"].astype(str).str.len()
        df["word_count"] = df["text"].astype(str).str.split().str.len()
        
        label_dist = {}
        if "label" in df.columns:
            label_dist = df["label"].value_counts().to_dict()
            
        summary[split_name] = {
            "num_samples": len(df),
            "label_distribution": label_dist,
            "avg_word_count": round(float(df["word_count"].mean()), 2),
            "median_word_count": int(df["word_count"].median()),
            "min_word_count": int(df["word_count"].min()),
            "max_word_count": int(df["word_count"].max()),
            "avg_char_length": round(float(df["char_length"].mean()), 2),
            "missing_texts": int(df["text"].isnull().sum()),
            "duplicate_texts": int(df["text"].duplicated().sum()),
        }
    return summary


def get_preprocessed_dataframe(
    dataset: DatasetDict,
    split: str = "train",
    sample_size: Optional[int] = None,
    seed: int = 42
) -> pd.DataFrame:
    """Converts a dataset split into a pandas DataFrame and applies text cleaning."""
    df = dataset[split].to_pandas()
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        
    df["clean_text"] = df["text"].apply(clean_text)
    df["char_length"] = df["clean_text"].str.len()
    df["word_count"] = df["clean_text"].str.split().str.len()
    
    label_map = {0: "negative", 1: "positive", -1: "unsupervised"}
    df["sentiment_label"] = df["label"].map(label_map)
    return df


def get_ngram_frequencies(
    texts: List[str],
    n: int = 1,
    top_k: int = 20,
    stop_words: Optional[set] = None
) -> List[Tuple[str, int]]:
    """Calculates n-gram frequencies from a list of texts excluding common stop words."""
    if stop_words is None:
        # Standard English stopwords plus common movie filler words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did", "i", "you",
            "he", "she", "it", "we", "they", "this", "that", "these", "those", "my", "your",
            "his", "her", "its", "our", "their", "not", "no", "as", "if", "so", "than", "too",
            "very", "s", "t", "can", "will", "just", "all", "out", "movie", "film"
        }
    
    ngram_counter = Counter()
    for text in texts:
        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        filtered = [t for t in tokens if t not in stop_words]
        if n == 1:
            ngram_counter.update(filtered)
        else:
            ngrams = [" ".join(filtered[i:i+n]) for i in range(len(filtered) - n + 1)]
            ngram_counter.update(ngrams)
            
    return ngram_counter.most_common(top_k)


def generate_eda_visualizations(
    dataset: DatasetDict,
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Generates and saves a complete suite of EDA visualizations:
    1. Class distribution (Train vs Test)
    2. Review word count and character length histograms
    3. Word count distribution by sentiment (Box/Violin)
    4. Top movie-domain keywords & n-grams for positive vs negative reviews
    
    Returns:
        Dict mapping figure name to saved file path.
    """
    if output_dir is None:
        output_dir = os.path.join(get_project_root(), "results", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seaborn style & aesthetics
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "Arial", "figure.autolayout": True})
    
    saved_figures = {}
    
    # Prepare Train and Test DataFrames
    train_df = get_preprocessed_dataframe(dataset, split="train")
    test_df = get_preprocessed_dataframe(dataset, split="test")
    train_df["split"] = "Train"
    test_df["split"] = "Test"
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # -------------------------------------------------------------------------
    # Figure 1: Class Distribution (Train & Test)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"positive": "#2ecc71", "negative": "#e74c3c"}
    sns.countplot(
        data=combined_df,
        x="split",
        hue="sentiment_label",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=1
    )
    ax.set_title("IMDb Dataset Sentiment Class Distribution (Balanced 50/50)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Dataset Split", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Reviews", fontsize=11, fontweight="bold")
    ax.legend(title="Sentiment", loc="upper right")
    
    # Add count labels on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height):,}", (p.get_x() + p.get_width() / 2., height / 2),
                        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    
    fig_path_1 = os.path.join(output_dir, "01_label_distribution.png")
    fig.savefig(fig_path_1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_figures["label_distribution"] = fig_path_1
    print(f"[EDA] Saved: {fig_path_1}")
    
    # -------------------------------------------------------------------------
    # Figure 2: Review Word Length Distribution (Histogram & KDE)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cap at 99th percentile for clean display
    clip_words = np.percentile(train_df["word_count"], 99)
    sns.histplot(
        train_df[train_df["word_count"] <= clip_words]["word_count"],
        kde=True,
        color="#3498db",
        ax=ax1,
        bins=35,
        edgecolor="black"
    )
    ax1.axvline(train_df["word_count"].median(), color="red", linestyle="--", linewidth=1.5, label=f"Median ({int(train_df['word_count'].median())} words)")
    ax1.axvline(train_df["word_count"].mean(), color="orange", linestyle="-.", linewidth=1.5, label=f"Mean ({train_df['word_count'].mean():.1f} words)")
    ax1.set_title("Distribution of Review Word Counts (Train)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Word Count", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=10, fontweight="bold")
    ax1.legend()
    
    # Character length
    clip_chars = np.percentile(train_df["char_length"], 99)
    sns.histplot(
        train_df[train_df["char_length"] <= clip_chars]["char_length"],
        kde=True,
        color="#9b59b6",
        ax=ax2,
        bins=35,
        edgecolor="black"
    )
    ax2.axvline(train_df["char_length"].median(), color="red", linestyle="--", linewidth=1.5, label=f"Median ({int(train_df['char_length'].median())} chars)")
    ax2.axvline(train_df["char_length"].mean(), color="orange", linestyle="-.", linewidth=1.5, label=f"Mean ({train_df['char_length'].mean():.1f} chars)")
    ax2.set_title("Distribution of Review Character Lengths (Train)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Character Length", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=10, fontweight="bold")
    ax2.legend()
    
    fig_path_2 = os.path.join(output_dir, "02_review_length_distribution.png")
    fig.savefig(fig_path_2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_figures["length_distribution"] = fig_path_2
    print(f"[EDA] Saved: {fig_path_2}")
    
    # -------------------------------------------------------------------------
    # Figure 3: Word Count Comparison by Sentiment Class
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    filtered_train = train_df[train_df["word_count"] <= clip_words]
    sns.boxplot(
        data=filtered_train,
        x="sentiment_label",
        y="word_count",
        palette=palette,
        ax=ax1,
        boxprops=dict(alpha=0.8)
    )
    ax1.set_title("Word Count Boxplot by Sentiment", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Sentiment Label", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Word Count", fontsize=10, fontweight="bold")
    
    sns.violinplot(
        data=filtered_train,
        x="sentiment_label",
        y="word_count",
        palette=palette,
        ax=ax2,
        cut=0
    )
    ax2.set_title("Word Count Density (Violin Plot) by Sentiment", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Sentiment Label", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Word Count", fontsize=10, fontweight="bold")
    
    fig_path_3 = os.path.join(output_dir, "03_word_count_by_sentiment.png")
    fig.savefig(fig_path_3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_figures["word_count_by_sentiment"] = fig_path_3
    print(f"[EDA] Saved: {fig_path_3}")
    
    # -------------------------------------------------------------------------
    # Figure 4: Top Characteristic Keywords (Positive vs Negative Reviews)
    # -------------------------------------------------------------------------
    pos_texts = train_df[train_df["label"] == 1]["clean_text"].tolist()
    neg_texts = train_df[train_df["label"] == 0]["clean_text"].tolist()
    
    pos_top = get_ngram_frequencies(pos_texts, n=1, top_k=15)
    neg_top = get_ngram_frequencies(neg_texts, n=1, top_k=15)
    
    pos_words, pos_counts = zip(*pos_top)
    neg_words, neg_counts = zip(*neg_top)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.barplot(x=list(pos_counts), y=list(pos_words), color="#2ecc71", ax=ax1, edgecolor="black")
    ax1.set_title("Top Distinctive Keywords in Positive Reviews", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Frequency Count", fontsize=10, fontweight="bold")
    
    sns.barplot(x=list(neg_counts), y=list(neg_words), color="#e74c3c", ax=ax2, edgecolor="black")
    ax2.set_title("Top Distinctive Keywords in Negative Reviews", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frequency Count", fontsize=10, fontweight="bold")
    
    fig_path_4 = os.path.join(output_dir, "04_top_keywords_comparison.png")
    fig.savefig(fig_path_4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_figures["top_keywords"] = fig_path_4
    print(f"[EDA] Saved: {fig_path_4}")
    
    return saved_figures
