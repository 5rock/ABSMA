"""
ABSA Interactive + Batch Analysis Script
=========================================
Allows a user to:
  - Enter a custom movie review interactively for ABSA analysis
  - Run batch analysis on N IMDb reviews

Usage:
    python scripts/run_absa.py                  # Interactive mode
    python scripts/run_absa.py --batch 10       # Batch mode (10 IMDb reviews)
    python scripts/run_absa.py --batch 50       # Batch mode (50 reviews)
"""

import os
import sys
import argparse

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Ensure project root is on path
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.absa_pipeline import ABSAPipeline
from src.data_preprocessing import load_imdb_raw, get_preprocessed_dataframe


def run_interactive(pipeline: ABSAPipeline):
    """Run interactive review input mode."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE — Enter your own movie review")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        print("\nEnter a movie review:")
        review = input("> ").strip()

        if review.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not review:
            print("[INFO] Please enter a review.")
            continue

        result = pipeline.analyze(review, review_id="user_input")
        print(pipeline.format_result_table(result))


def run_batch(pipeline: ABSAPipeline, n: int):
    """Run ABSA on a batch of N IMDb reviews."""
    print(f"\n" + "=" * 60)
    print(f"  BATCH MODE — Analyzing {n} IMDb reviews")
    print("=" * 60)
    print("NOTE: These are real IMDb reviews. Overall sentiment labels come")
    print("from IMDb. Aspect-level sentiments are machine predictions.\n")

    print("Loading IMDb dataset...")
    dataset = load_imdb_raw()
    df = get_preprocessed_dataframe(dataset, split="test", sample_size=n, seed=99)
    texts = df["clean_text"].tolist()
    labels = df["sentiment_label"].tolist()
    ids = [f"imdb_test_{i+1}" for i in range(len(texts))]

    print(f"Loaded {len(texts)} reviews. Running ABSA...\n")

    for i, (text, label, rid) in enumerate(zip(texts, labels, ids), 1):
        print(f"\n{'─'*60}")
        print(f"Review {i}/{n} | IMDb Overall: {label.upper()}")
        print(f"{'─'*60}")
        result = pipeline.analyze(text, review_id=rid)
        print(pipeline.format_result_table(result))

        if i < n:
            cont = input("\nPress Enter for next review (or 'q' to stop): ").strip()
            if cont.lower() == "q":
                break


def main():
    parser = argparse.ArgumentParser(
        description="Run ABSA on a movie review (interactive or batch)."
    )
    parser.add_argument(
        "--batch", type=int, default=0,
        help="Number of IMDb reviews to analyze in batch mode (0 = interactive)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("   ASPECT-BASED SENTIMENT ANALYSIS (ABSA)")
    print("=" * 60)

    print("\nLoading ABSA pipeline (this may take a moment)...")
    try:
        pipeline = ABSAPipeline(use_transformer_ner=True)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please run: python scripts/train_sentiment.py  first.")
        sys.exit(1)

    if args.batch > 0:
        run_batch(pipeline, n=args.batch)
    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
