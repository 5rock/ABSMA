"""
Aspect Extraction Module for Movie Reviews
Combines Named Entity Recognition (NER) for actors/persons and syntactic/semantic
aspect extraction for general movie facets (plot, direction, pacing, music, etc.).
"""

import os
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


def _is_subword_fragment(text: str) -> bool:
    """
    Detects BERT subword fragments that should not appear as standalone aspects.
    These occur when aggregation_strategy fails to merge tokens properly.
    Examples: '##ill', '##ian Murphy', '##ryl Streep', '##th'
    """
    stripped = text.strip()
    # Starts with ## — classic BERT subword prefix
    if stripped.startswith("##"):
        return True
    # Very short fragments (1-2 chars) are noise
    if len(stripped) <= 2:
        return True
    # Regex: starts with 2+ lowercase letters that look like a word suffix
    # but only if the full text is suspiciously short or lacks a capital
    if re.match(r'^[a-z]{1,3}\b', stripped) and len(stripped) < 5:
        return True
    return False


def _normalize_person_name(name: str) -> str:
    """Lowercase, strip, collapse whitespace for deduplication comparison."""
    return re.sub(r'\s+', ' ', name.strip().lower())


class AspectExtractor:
    """
    Extracts actors, persons, and domain-specific movie aspects from unstructured reviews.
    Utilizes:
    1. Pretrained Transformer NER (BERT-NER) for Actor/Person extraction with SpaCy fallback.
    2. SpaCy Dependency Parsing and Noun Chunk Extraction for syntactic aspect phrases.
    3. Domain-specific movie aspect taxonomy and semantic categorization.
    """

    # Comprehensive Movie Aspect Taxonomy
    ASPECT_TAXONOMY = {
        "Actor/Performance": {
            "keywords": [
                "actor", "actress", "cast", "acting", "performance", "performances",
                "role", "roles", "lead", "supporting cast", "chemistry", "portrayal",
                "portrayed", "characterization", "cameo", "ensemble", "starring",
                "co-star", "voice acting"
            ],
            "synonyms": ["acting", "actor", "actress", "performance", "cast", "portrayal"]
        },
        "Plot/Story": {
            "keywords": [
                "plot", "story", "storyline", "twist", "narrative", "premise",
                "script", "screenplay", "writing", "writers", "climax", "sub-plot",
                "plothole", "plot hole", "story arc", "concept", "structure", "tale",
                "drama", "suspense", "mystery", "lore"
            ],
            "synonyms": ["plot", "story", "narrative", "script", "storyline"]
        },
        "Ending": {
            "keywords": [
                "ending", "climax", "finale", "conclusion", "resolution",
                "final act", "final scene", "ending twist", "last 20 minutes",
                "last scene", "credits"
            ],
            "synonyms": ["ending", "finale", "conclusion", "climax"]
        },
        "Direction": {
            "keywords": [
                "direction", "director", "directed", "filmmaking", "filmmaker",
                "directing", "vision", "helmed", "staging", "execution"
            ],
            "synonyms": ["direction", "director", "directing", "filmmaking"]
        },
        "Cinematography/Visuals": {
            "keywords": [
                "cinematography", "visuals", "visual", "camerawork", "camera work",
                "shots", "lighting", "photography", "color palette", "framing",
                "aesthetic", "scenery", "locations", "visual style"
            ],
            "synonyms": ["cinematography", "visuals", "photography", "camerawork"]
        },
        "Music/Soundtrack": {
            "keywords": [
                "music", "soundtrack", "score", "songs", "sound", "bgm",
                "background score", "theme song", "sound design", "audio",
                "composer", "sound effects"
            ],
            "synonyms": ["music", "soundtrack", "score", "audio", "sound"]
        },
        "Dialogues/Screenplay": {
            "keywords": [
                "dialogue", "dialogues", "screenplay", "lines", "monologue",
                "conversations", "one-liners", "scriptwriting", "banter"
            ],
            "synonyms": ["dialogue", "screenplay", "lines", "monologue"]
        },
        "Pacing/Editing": {
            "keywords": [
                "pacing", "pace", "tempo", "runtime", "dragged", "dragging",
                "slow start", "rushed", "sluggish", "editing", "cuts", "flow",
                "length", "transitions", "slow burn"
            ],
            "synonyms": ["pacing", "pace", "editing", "flow", "runtime"]
        },
        "Characters": {
            "keywords": [
                "character", "characters", "protagonist", "antagonist", "villain",
                "hero", "heroine", "side character", "side characters", "main character",
                "character development", "character arc", "depth"
            ],
            "synonyms": ["character", "characters", "protagonist", "villain"]
        },
        "Special Effects/CGI": {
            "keywords": [
                "special effects", "effects", "cgi", "vfx", "action scenes",
                "action sequences", "stunts", "explosions", "practical effects",
                "choreography", "fight scenes", "animation"
            ],
            "synonyms": ["special effects", "cgi", "vfx", "effects", "stunts"]
        },
        "Atmosphere/Tone": {
            "keywords": [
                "atmosphere", "tone", "vibe", "mood", "ambiance", "humor",
                "comedy", "dark tone", "emotional depth", "tension", "feel"
            ],
            "synonyms": ["atmosphere", "mood", "tone", "ambiance"]
        }
    }

    # Non-person words often capitalized that NER might mistakenly tag
    FALSE_POSITIVE_PERSONS = {
        "Hollywood", "Oscar", "Oscars", "Academy", "IMDb", "CGI", "VFX",
        "DVD", "Blu-ray", "Netflix", "Disney", "Marvel", "DC", "Warner",
        "Sony", "Universal", "Paramount", "Cinema", "Film", "Movie", "Broadway",
        "American", "British", "English", "French", "German", "God", "Jesus", "Satan"
    }

    def __init__(self, use_transformer_ner: bool = True):
        """
        Initializes SpaCy pipeline and optional pretrained Transformer NER model.
        """
        # Load SpaCy model for syntactic parsing and noun chunks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.use_transformer_ner = use_transformer_ner
        self.ner_pipeline = None
        
        if use_transformer_ner:
            try:
                print("[INFO] Initializing pretrained BERT NER pipeline (dslim/bert-base-NER)...")
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                    device="cpu"
                )
                print("[INFO] Pretrained Transformer NER initialized successfully.")
            except Exception as e:
                print(f"[WARN] Could not load Transformer NER ({e}). Falling back to SpaCy NER.")
                self.ner_pipeline = None

    def extract_actors_ner(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts actor/person mentions with exact character spans using Transformer NER or SpaCy NER.
        Filters BERT subword fragments (e.g. '##ill', '##ian Murphy') and deduplicates
        repeated person names (case-insensitive).
        """
        actors = []
        seen_spans = set()
        # Track normalized names to avoid duplicate person entries
        seen_names: Set[str] = set()

        if self.ner_pipeline is not None:
            try:
                ner_results = self.ner_pipeline(text)
                for ent in ner_results:
                    if ent.get("entity_group") == "PER":
                        name = ent["word"].strip()
                        start = ent["start"]
                        end = ent["end"]

                        # Skip BERT subword fragments (e.g. '##ill', '##ryl Streep')
                        if _is_subword_fragment(name):
                            continue

                        # Skip obvious false positives and very short names
                        if len(name) < 3 or name in self.FALSE_POSITIVE_PERSONS:
                            continue

                        # Deduplicate: skip if same person name already seen
                        norm_name = _normalize_person_name(name)
                        if norm_name in seen_names:
                            continue

                        if (start, end) not in seen_spans:
                            actors.append({
                                "aspect": name,
                                "category": "Actor/Performance",
                                "entity_type": "PERSON",
                                "start": start,
                                "end": end,
                                "confidence": round(float(ent.get("score", 0.9)), 3)
                            })
                            seen_spans.add((start, end))
                            seen_names.add(norm_name)
            except Exception as e:
                print(f"[WARN] Transformer NER error ({e}), using SpaCy NER fallback.")

        # Fallback / augmentation with SpaCy NER
        if not actors:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    # Skip subword fragments and false positives
                    if _is_subword_fragment(name):
                        continue
                    if len(name) < 3 or name in self.FALSE_POSITIVE_PERSONS:
                        continue
                    norm_name = _normalize_person_name(name)
                    if norm_name in seen_names:
                        continue
                    start = ent.start_char
                    end = ent.end_char
                    if (start, end) not in seen_spans:
                        actors.append({
                            "aspect": name,
                            "category": "Actor/Performance",
                            "entity_type": "PERSON",
                            "start": start,
                            "end": end,
                            "confidence": 0.85
                        })
                        seen_spans.add((start, end))
                        seen_names.add(norm_name)

        return actors

    def categorize_noun_chunk(self, chunk_text: str, chunk_doc) -> Optional[str]:
        """
        Maps a noun chunk or phrase to a predefined movie aspect category.
        """
        chunk_lower = chunk_text.lower().strip()
        
        # 1. Exact or multi-word match in taxonomy keywords
        for category, cat_info in self.ASPECT_TAXONOMY.items():
            for kw in cat_info["keywords"]:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, chunk_lower):
                    return category

        # 2. Lemma-based token matching
        for token in chunk_doc:
            lemma = token.lemma_.lower()
            for category, cat_info in self.ASPECT_TAXONOMY.items():
                if lemma in cat_info["synonyms"]:
                    return category

        return None

    def extract_movie_aspects(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts syntactic noun phrases and domain aspects from review text.
        """
        doc = self.nlp(text)
        aspects = []
        seen_spans = set()

        # Step A: Syntactic noun chunks
        for chunk in doc.noun_chunks:
            # Clean chunk text
            clean_chunk = re.sub(r"^(the|a|an|this|that|these|those|its|their|his|her)\s+", "", chunk.text, flags=re.IGNORECASE).strip()
            if len(clean_chunk) < 3:
                continue

            category = self.categorize_noun_chunk(clean_chunk, chunk)
            if category is not None:
                start = chunk.start_char
                end = chunk.end_char
                if (start, end) not in seen_spans:
                    aspects.append({
                        "aspect": clean_chunk,
                        "category": category,
                        "entity_type": "ASPECT",
                        "start": start,
                        "end": end,
                        "confidence": 0.90
                    })
                    seen_spans.add((start, end))

        # Step B: Direct taxonomy search for missed aspect occurrences
        for category, cat_info in self.ASPECT_TAXONOMY.items():
            for kw in cat_info["keywords"]:
                for match in re.finditer(r"\b" + re.escape(kw) + r"\b", text, flags=re.IGNORECASE):
                    start = match.start()
                    end = match.end()
                    # Check if already covered by an existing chunk span
                    overlapping = any(s <= start and end <= e for (s, e) in seen_spans)
                    if not overlapping:
                        matched_text = match.group(0)
                        aspects.append({
                            "aspect": matched_text,
                            "category": category,
                            "entity_type": "ASPECT",
                            "start": start,
                            "end": end,
                            "confidence": 0.88
                        })
                        seen_spans.add((start, end))

        return aspects

    def extract_all_aspects(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts both actors/persons (via NER) and general movie aspects (via syntactic & semantic extraction).
        Sorts results by their appearance in the review text and attaches local sentence context.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return []

        # 1. Extract actors via NER
        actors = self.extract_actors_ner(text)
        
        # 2. Extract movie aspects via taxonomy & dependency parsing
        movie_aspects = self.extract_movie_aspects(text)

        # Merge results and deduplicate overlapping spans
        all_aspects = []
        covered_spans = []

        # Give priority to NER actor spans
        for actor in actors:
            all_aspects.append(actor)
            covered_spans.append((actor["start"], actor["end"]))

        for aspect in movie_aspects:
            a_start, a_end = aspect["start"], aspect["end"]
            # Avoid duplicate if overlaps closely with an existing actor span
            overlaps = any(
                max(a_start, s) < min(a_end, e)
                for (s, e) in covered_spans
            )
            if not overlaps:
                all_aspects.append(aspect)
                covered_spans.append((a_start, a_end))

        # Sort by character start position
        all_aspects.sort(key=lambda x: x["start"])

        # Attach local contextual sentence for each aspect
        doc = self.nlp(text)
        sentences = list(doc.sents)

        for item in all_aspects:
            item_start = item["start"]
            item_end = item["end"]
            
            # Find the sentence containing this aspect
            containing_sent = text
            for sent in sentences:
                if sent.start_char <= item_start and item_end <= sent.end_char:
                    containing_sent = sent.text.strip()
                    break
                    
            item["context_sentence"] = containing_sent

        return all_aspects
