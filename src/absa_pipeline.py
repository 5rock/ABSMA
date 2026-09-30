"""
ABSA Pipeline Module
=====================
Combines AspectExtractor and SentimentClassifier to produce a complete
Aspect-Based Sentiment Analysis (ABSA) result for any movie review.

IMPORTANT DISCLAIMER:
---------------------
The IMDb dataset provides only OVERALL REVIEW-LEVEL sentiment labels (0 = negative, 1 = positive).
It does NOT provide gold-standard aspect-level sentiment annotations.

The aspect-level sentiments produced by this pipeline are AUTOMATICALLY GENERATED
by applying the sentiment classifier to each aspect's context sentence.
They are machine predictions, NOT ground truth labels.

This distinction is critical for honest academic evaluation.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback

# Ensure project root is on path when running from scripts/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.aspect_extraction import AspectExtractor
from src.sentiment_classifier import SentimentClassifier
from src.transformer_model import TransformerSentimentClassifier
from src.data_preprocessing import clean_text


class ABSAPipeline:
    """
    End-to-end Aspect-Based Sentiment Analysis pipeline.

    Pipeline:
        Raw Review Text
            ↓ clean_text()
        Preprocessed Text
            ↓ AspectExtractor.extract_all_aspects()
        Extracted Aspects (with category + context_sentence)
            ↓ SentimentClassifier.predict(context_sentence)
        Aspect-Level Sentiment Predictions
            ↓
        Final ABSA Result (JSON)
    """

    def __init__(
        self,
        aspect_extractor: Optional[AspectExtractor] = None,
        sentiment_classifier: Optional[Any] = None,
        use_transformer_ner: bool = True,
        use_transformer_sentiment: bool = True
    ):
        """
        Initialize the ABSA pipeline.

        Args:
            aspect_extractor:      Pre-initialized AspectExtractor (or None to auto-init).
            sentiment_classifier:  Pre-initialized SentimentClassifier (or None to auto-load).
            use_transformer_ner:   Whether to use BERT NER for actor extraction.
        """
        # Initialize or reuse AspectExtractor
        if aspect_extractor is not None:
            self.aspect_extractor = aspect_extractor
        else:
            print("[ABSA] Initializing AspectExtractor...")
            self.aspect_extractor = AspectExtractor(use_transformer_ner=use_transformer_ner)

        # Initialize or reuse SentimentClassifier
        if sentiment_classifier is not None:
            self.sentiment_classifier = sentiment_classifier
        else:
            if use_transformer_sentiment:
                print("[ABSA] Loading TransformerSentimentClassifier...")
                self.sentiment_classifier = TransformerSentimentClassifier()
                self.using_transformer = True
            else:
                print("[ABSA] Loading baseline SentimentClassifier (TF-IDF)...")
                self.sentiment_classifier = SentimentClassifier.load()
                self.using_transformer = False

        print("[ABSA] Pipeline ready.")

    def analyze(self, review_text: str, review_id: str = "unknown") -> Dict[str, Any]:
        """
        Run the complete ABSA pipeline on a single review.

        Args:
            review_text: Raw movie review text.
            review_id:   Optional identifier for this review.

        Returns:
            Dict with:
                - review_id       : str
                - original_text   : str
                - cleaned_text    : str
                - overall_sentiment: dict (sentiment + confidence for the FULL review)
                - aspects         : list of aspect-level results
                - aspect_count    : int
                - note            : disclaimer about aspect-level labels

        Each aspect dict contains:
            aspect, category, entity_type, sentiment, confidence, context_sentence
        """
        # Step 1: Clean the text
        cleaned = clean_text(review_text)
        if not cleaned.strip():
            return {
                "review_id": review_id,
                "original_text": review_text,
                "cleaned_text": cleaned,
                "overall_sentiment": {"sentiment": "unknown", "confidence": 0.0},
                "aspects": [],
                "aspect_count": 0,
                "note": "Empty review after cleaning."
            }

        # Step 2: Predict overall review sentiment
        overall = self.sentiment_classifier.predict(cleaned)

        # Step 3: Extract aspects (actors + movie aspects)
        raw_aspects = self.aspect_extractor.extract_all_aspects(cleaned)

        # Step 4: For each aspect, predict sentiment on its context sentence
        aspect_results = []
        for asp in raw_aspects:
            context = asp.get("context_sentence", cleaned)
            
            # Use transformer for aspect-level sentiment if enabled
            if getattr(self, 'using_transformer', False):
                asp_sentiment = self.sentiment_classifier.predict(context, aspect=asp["aspect"])
            else:
                asp_sentiment = self.sentiment_classifier.predict(context)

            aspect_results.append({
                "aspect": asp["aspect"],
                "category": asp["category"],
                "entity_type": asp["entity_type"],
                "sentiment": asp_sentiment["sentiment"],
                "confidence": asp_sentiment["confidence"],
                "aspect_confidence": asp.get("confidence", 0.9),
                "context_sentence": context
            })

        return {
            "review_id": review_id,
            "original_text": review_text,
            "cleaned_text": cleaned,
            "overall_sentiment": overall,
            "aspects": aspect_results,
            "aspect_count": len(aspect_results),
            "note": (
                "IMPORTANT: IMDb provides only overall review-level labels. "
                "Aspect-level sentiments are AUTOMATICALLY GENERATED predictions — "
                "NOT gold-standard annotations."
            )
        }

    def analyze_batch(
        self,
        reviews: List[str],
        review_ids: Optional[List[str]] = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run ABSA on a list of reviews.

        Args:
            reviews:    List of raw review texts.
            review_ids: Optional list of IDs (auto-generated if None).
            verbose:    Print progress messages.

        Returns:
            List of ABSA result dicts.
        """
        if review_ids is None:
            review_ids = [f"review_{i+1}" for i in range(len(reviews))]

        results = []
        total = len(reviews)
        for i, (text, rid) in enumerate(zip(reviews, review_ids), 1):
            if verbose and (i % 10 == 0 or i == 1):
                print(f"  Processing review {i}/{total}...")
            result = self.analyze(text, review_id=rid)
            results.append(result)

        return results

    def format_result_table(self, result: Dict[str, Any]) -> str:
        """
        Format a single ABSA result as a human-readable table string.

        Args:
            result: Output from analyze().

        Returns:
            Formatted string for CLI display.
        """
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("ASPECT-BASED SENTIMENT ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"\nReview:\n{result['original_text'][:300]}")
        lines.append(f"\nOverall Sentiment: {result['overall_sentiment']['sentiment'].upper()} "
                     f"(confidence: {result['overall_sentiment']['confidence']:.1%})")

        aspects = result["aspects"]
        if not aspects:
            lines.append("\n[No aspects detected in this review]")
        else:
            lines.append(f"\nAspect-Level Predictions ({len(aspects)} aspects found):")
            lines.append(f"  NOTE: These are machine predictions, not ground truth labels.")
            lines.append("")
            header = f"  {'Aspect':<25} {'Category':<28} {'Sentiment':<12} {'Conf':<6}"
            lines.append(header)
            lines.append("  " + "-" * 73)
            for asp in aspects:
                sent_str = asp["sentiment"].upper()
                line = (f"  {asp['aspect'][:24]:<25} "
                        f"{asp['category'][:27]:<28} "
                        f"{sent_str:<12} "
                        f"{asp['confidence']:.0%}")
                lines.append(line)

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
