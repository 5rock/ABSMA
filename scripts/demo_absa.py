"""
ABSA Demo Script
=================
Demonstrates the complete ABSA pipeline on several predefined movie review examples.
Run from the project root:
    python scripts/demo_absa.py
"""

import os
import sys

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on path
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.absa_pipeline import ABSAPipeline

# ─────────────────────────────────────────────────────────────
# PREDEFINED DEMO REVIEWS
# ─────────────────────────────────────────────────────────────
DEMO_REVIEWS = [
    "The acting was fantastic but the plot was predictable.",
    "The cinematography was beautiful and the soundtrack was amazing.",
    "The movie had terrible pacing and weak characters.",
    "Christopher Nolan directed an absolute masterpiece. "
    "Cillian Murphy's performance was breathtaking, "
    "and the background score was hauntingly beautiful.",
    "The script was terrible and the dialogue felt forced. "
    "The special effects were decent, but the story made no sense.",
    "Tom Hanks delivers a career-best performance. "
    "The screenplay is razor-sharp and the direction is flawless.",
]

def main():
    print("=" * 60)
    print("   ABSA DEMO — Aspect-Based Sentiment Analysis")
    print("=" * 60)
    print("\nNOTE: Aspect-level sentiments are machine predictions.")
    print("IMDb does NOT provide gold-standard aspect-level annotations.\n")

    # Load the pipeline (requires trained model)
    print("Loading ABSA pipeline...")
    try:
        pipeline = ABSAPipeline(use_transformer_ner=True)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please run: python scripts/train_sentiment.py  first.")
        sys.exit(1)

    # Process each demo review
    for i, review in enumerate(DEMO_REVIEWS, 1):
        print(f"\n" + "-" * 60)
        print(f"Demo Review #{i}")
        print("-" * 60)
        result = pipeline.analyze(review, review_id=f"demo_{i}")
        print(pipeline.format_result_table(result))

    print("\n[DEMO COMPLETE]\n")


if __name__ == "__main__":
    main()
