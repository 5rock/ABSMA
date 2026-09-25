"""
Step 4: Aspect Extraction Test & Evaluation Script
Runs extraction on 50 IMDb reviews and saves results to results/predictions/
Must be run from the project root: IBM Projects/
"""

import os
import sys
import json
import pandas as pd

# Ensure project root is on path (script is run from project root)
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import load_imdb_raw, get_preprocessed_dataframe, clean_text
from src.aspect_extraction import AspectExtractor

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SAMPLE_SIZE = 50
SEED = 42
OUTPUT_DIR = os.path.join(project_root, "results", "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. FIXED SHOWCASE REVIEWS
# ─────────────────────────────────────────────────────────────
showcase_reviews = [
    "The movie started slowly, but Tom Holland gave an amazing performance. The supporting cast was also good, although the ending was disappointing.",
    "Christopher Nolan directed an absolute masterpiece. Cillian Murphy and Emily Blunt were both sensational, and the soundtrack by Ludwig Goransson was spine-tingling.",
    "Tom's performance was brilliant, but the plot became boring toward the end.",
    "Terrible script and predictable plot completely ruined this film. The pacing was sluggish throughout.",
    "Robert De Niro brings gravitas to every scene. The cinematography is stunning and the screenplay is razor-sharp.",
    "The special effects are breathtaking, but the characters lack depth and the dialogue feels forced.",
    "Meryl Streep is phenomenal as always, delivering another Oscar-worthy performance.",
    "The film has great music but a weak storyline and disappointing ending.",
    "The direction is masterful, but the acting from some supporting cast members was wooden.",
    "An absolutely brilliant screenplay complimented by exceptional cinematography and a hauntingly beautiful score.",
]

# ─────────────────────────────────────────────────────────────
# 2. LOAD IMDb SAMPLE
# ─────────────────────────────────────────────────────────────
print("Loading IMDb dataset (cached)...")
dataset = load_imdb_raw()
train_df = get_preprocessed_dataframe(dataset, split="train", sample_size=SAMPLE_SIZE, seed=SEED)
imdb_texts = train_df["clean_text"].tolist()
imdb_labels = train_df["sentiment_label"].tolist()
print(f"Loaded {len(imdb_texts)} IMDb review samples.")

# ─────────────────────────────────────────────────────────────
# 3. INITIALIZE EXTRACTOR
# ─────────────────────────────────────────────────────────────
print("\nInitializing AspectExtractor (Transformer NER + SpaCy)...")
extractor = AspectExtractor(use_transformer_ner=True)

# ─────────────────────────────────────────────────────────────
# 4. EVALUATE SHOWCASE REVIEWS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SHOWCASE REVIEW EXTRACTION RESULTS")
print("NOTE: IMDb does not contain gold-standard aspect annotations.")
print("These extractions are automatically generated and NOT ground truth.")
print("="*80)

showcase_records = []
for idx, review in enumerate(showcase_reviews, 1):
    print(f"\nReview #{idx}: \"{review[:150]}\"")
    aspects = extractor.extract_all_aspects(review)
    if not aspects:
        print("  [No aspects detected]")
    else:
        persons = [a for a in aspects if a["entity_type"] == "PERSON"]
        movie_aspects = [a for a in aspects if a["entity_type"] == "ASPECT"]
        if persons:
            print(f"  PERSONS  : {[p['aspect'] for p in persons]}")
        if movie_aspects:
            print(f"  ASPECTS  : {[(a['aspect'], a['category']) for a in movie_aspects[:6]]}")
    for a in aspects:
        showcase_records.append({
            "review_id": f"showcase_{idx}",
            "review_snippet": review[:200],
            "aspect": a["aspect"],
            "category": a["category"],
            "entity_type": a["entity_type"],
            "confidence": a["confidence"],
            "context_sentence": a["context_sentence"],
            "source": "showcase"
        })

# ─────────────────────────────────────────────────────────────
# 5. EVALUATE IMDb SAMPLE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("IMDb SAMPLE EXTRACTION (50 reviews)")
print("NOTE: Automatically generated — NOT gold-standard ABSA annotations.")
print("="*80)

imdb_records = []
category_counts = {}
review_aspect_counts = []

for idx, (review, label) in enumerate(zip(imdb_texts, imdb_labels), 1):
    aspects = extractor.extract_all_aspects(review)
    review_aspect_counts.append(len(aspects))
    persons = [a for a in aspects if a["entity_type"] == "PERSON"]
    movie_asp = [a for a in aspects if a["entity_type"] == "ASPECT"]

    if idx <= 20:
        print(f"\nIMDb Review #{idx} | Sentiment: {label.upper()}")
        print(f"  Preview : \"{review[:120]}...\"")
        print(f"  Persons : {[p['aspect'] for p in persons]}")
        print(f"  Aspects : {[(a['aspect'], a['category']) for a in movie_asp[:4]]}")

    for a in aspects:
        imdb_records.append({
            "review_id": f"imdb_train_{idx}",
            "review_snippet": review[:200],
            "overall_sentiment": label,
            "aspect": a["aspect"],
            "category": a["category"],
            "entity_type": a["entity_type"],
            "confidence": a["confidence"],
            "context_sentence": a["context_sentence"],
            "source": "imdb"
        })
        category_counts[a["category"]] = category_counts.get(a["category"], 0) + 1

# ─────────────────────────────────────────────────────────────
# 6. STATISTICS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("EXTRACTION STATISTICS")
print("="*80)
total_aspects = len(imdb_records)
reviews_with_aspects = sum(1 for c in review_aspect_counts if c > 0)
reviews_no_aspects = sum(1 for c in review_aspect_counts if c == 0)
avg_aspects = sum(review_aspect_counts) / len(review_aspect_counts) if review_aspect_counts else 0

print(f"  Total aspects extracted     : {total_aspects}")
print(f"  Reviews with >= 1 aspect    : {reviews_with_aspects} / {SAMPLE_SIZE}")
print(f"  Reviews with 0 aspects      : {reviews_no_aspects} / {SAMPLE_SIZE}")
print(f"  Avg aspects per review      : {avg_aspects:.2f}")
print(f"\n  Category Distribution:")
for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
    pct = cnt / total_aspects * 100 if total_aspects > 0 else 0
    print(f"    {cat:<28}: {cnt:>4}  ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 7. SAVE RESULTS
# ─────────────────────────────────────────────────────────────
all_records = showcase_records + imdb_records
df_out = pd.DataFrame(all_records)
output_csv = os.path.join(OUTPUT_DIR, "04_aspect_extraction_sample.csv")
df_out.to_csv(output_csv, index=False)
print(f"\n[SAVED] {output_csv} ({len(df_out)} records)")

stats = {
    "total_reviews_sampled": SAMPLE_SIZE,
    "total_aspects_extracted": total_aspects,
    "reviews_with_aspects": reviews_with_aspects,
    "reviews_without_aspects": reviews_no_aspects,
    "avg_aspects_per_review": round(avg_aspects, 2),
    "category_distribution": category_counts,
    "annotation_note": "Automatically generated via Transformer NER + SpaCy + domain taxonomy. NOT gold-truth ABSA annotations."
}
stats_json = os.path.join(OUTPUT_DIR, "04_extraction_stats.json")
with open(stats_json, "w") as f:
    json.dump(stats, f, indent=2)
print(f"[SAVED] {stats_json}")
print("\nStep 4 complete. Ready for Step 5.")
