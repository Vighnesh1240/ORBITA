# src/nlp_analyzer.py
"""
ORBITA Manual NLP Analyzer

Research Justification:
    This module makes ORBITA a HYBRID system — combining:
    1. Rule-based NLP (VADER, spaCy, TF-IDF) ← THIS FILE
    2. Generative AI (Gemini) ← existing agents

    The hybrid approach provides:
    - Independent validation of AI-generated bias scores
    - Reproducible, deterministic analysis (no API needed)
    - Grounded factual context for AI agents
    - Stronger research claim: "our system uses both
      manual NLP and LLM, with each validating the other"

What This Module Computes:
    Per Article:
        - VADER sentiment scores (pos, neg, neu, compound)
        - spaCy Named Entities (people, places, organizations)
        - Sentence count, avg sentence length
        - Subjectivity indicators

    Across All Articles (Corpus Level):
        - TF-IDF top keywords (what topics dominate)
        - Entity frequency (who/what is mentioned most)
        - Aggregate sentiment distribution
        - Manual bias score (independent from Gemini)

    For Agent Context:
        - Formatted text summary for injection into prompts
        - Gives agents grounded facts to reason about

Libraries Used:
    vaderSentiment: Rule-based sentiment (Hutto & Gilbert 2014)
    spaCy:          Industrial NLP (Honnibal & Montani 2017)
    scikit-learn:   TF-IDF implementation
    wordcloud:      Keyword visualization

Author: [Your Name]
Project: ORBITA — B.Tech 6th Sem, AIML 2026
"""

import re
import json
import time
import warnings
from collections import Counter, defaultdict
from typing      import Optional

import numpy as np

# Suppress warnings from libraries
warnings.filterwarnings("ignore", category=UserWarning)

# ── VADER Sentiment ───────────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_available = True
except ImportError:
    _vader_available = False
    print(
        "[nlp_analyzer] WARNING: vaderSentiment not installed.\n"
        "  Run: pip install vaderSentiment"
    )

# ── spaCy NER ─────────────────────────────────────────────────────────────────
try:
    import spacy
    try:
        _nlp_model = spacy.load("en_core_web_sm")
        _spacy_available = True
    except OSError:
        _spacy_available = False
        _nlp_model       = None
        print(
            "[nlp_analyzer] WARNING: spaCy model not found.\n"
            "  Run: python -m spacy download en_core_web_sm"
        )
except ImportError:
    _spacy_available = False
    _nlp_model       = None
    print(
        "[nlp_analyzer] WARNING: spaCy not installed.\n"
        "  Run: pip install spacy"
    )

# ── TF-IDF ────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

# ── WordCloud ─────────────────────────────────────────────────────────────────
try:
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend for Streamlit
    import matplotlib.pyplot as plt
    _wordcloud_available = True
except ImportError:
    _wordcloud_available = False
    print(
        "[nlp_analyzer] WARNING: wordcloud not installed.\n"
        "  Run: pip install wordcloud matplotlib"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# spaCy entity types we care about for news analysis
# These are the most informative for media bias research
RELEVANT_ENTITY_TYPES = {
    "PERSON":  "People",          # politicians, activists, experts
    "ORG":     "Organizations",   # parties, companies, NGOs
    "GPE":     "Places",          # countries, cities, states
    "LOC":     "Locations",       # geographical features
    "LAW":     "Laws/Policies",   # legislation, court cases
    "EVENT":   "Events",          # protests, elections, conflicts
    "NORP":    "Groups",          # nationalities, political groups
    "MONEY":   "Money",           # financial amounts
    "PERCENT": "Percentages",     # statistics
    "DATE":    "Dates",           # temporal references
}

# Stopwords for TF-IDF keyword extraction
# Added domain-specific news stopwords beyond standard English
NEWS_STOPWORDS = [
    # Standard English stopwords
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "he", "she", "we", "you", "i", "me", "my", "our", "your", "his",
    "her", "their", "which", "who", "what", "when", "where", "how",
    "if", "then", "than", "so", "as", "up", "out", "about", "into",
    # News-specific non-informative words
    "said", "says", "according", "told", "added", "noted", "stated",
    "also", "however", "although", "while", "since", "after", "before",
    "new", "also", "just", "more", "one", "two", "three", "year",
    "years", "time", "people", "day", "week", "month", "like", "make",
    "made", "take", "took", "come", "came", "get", "got", "go", "went",
    "know", "think", "see", "use", "used", "first", "last", "many",
    "some", "most", "other", "other", "such", "very", "can", "not",
    "no", "all", "any", "each", "every", "both", "few", "own", "same",
    "than", "too", "very", "just", "because", "though", "through",
]

# Minimum word length for TF-IDF keywords
MIN_WORD_LENGTH = 3

# Maximum number of top keywords to extract
TOP_KEYWORDS_COUNT = 20

# Maximum text length for spaCy processing (avoid memory issues)
SPACY_MAX_CHARS = 100_000

# VADER compound score thresholds
VADER_POSITIVE_THRESHOLD  =  0.05
VADER_NEGATIVE_THRESHOLD  = -0.05


# ─────────────────────────────────────────────────────────────────────────────
# VADER SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# Initialize VADER once at module level (expensive to create repeatedly)
_vader_analyzer = None

def _get_vader():
    """Lazy initialization of VADER analyzer."""
    global _vader_analyzer
    if _vader_analyzer is None and _vader_available:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def analyze_sentiment_vader(text: str) -> dict:
    """
    Compute VADER sentiment scores for a text.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is
    specifically tuned for news and social media text. It handles:
    - Negations ("not good" → negative)
    - Intensifiers ("VERY good" → more positive)
    - Punctuation ("great!!!" → more positive)
    - Emoticons (not relevant for news but shows VADER's design)

    Reference:
        Hutto & Gilbert (2014) "VADER: A Parsimonious Rule-based Model
        for Sentiment Analysis of Social Media Text" — ICWSM 2014

    Args:
        text: article text or any string to analyze

    Returns:
        dict with scores:
            neg:      negative sentiment [0, 1]
            neu:      neutral sentiment  [0, 1]
            pos:      positive sentiment [0, 1]
            compound: overall score [-1, +1]
            label:    "positive" | "negative" | "neutral"

        All four scores sum to 1.0 (neg + neu + pos = 1.0)
        compound is the most useful single score.

    Example:
        "The government's new policy is a complete disaster"
        → neg=0.35, neu=0.65, pos=0.0, compound=-0.54 (Negative)

        "Farmers peacefully protested, demanding fair prices"
        → neg=0.0, neu=0.72, pos=0.28, compound=0.34 (Positive)
    """
    vader = _get_vader()

    if vader is None:
        # Fallback if VADER not available
        return {
            "neg":      0.0,
            "neu":      1.0,
            "pos":      0.0,
            "compound": 0.0,
            "label":    "neutral",
            "error":    "VADER not available",
        }

    if not text or len(text.strip()) < 10:
        return {
            "neg":      0.0,
            "neu":      1.0,
            "pos":      0.0,
            "compound": 0.0,
            "label":    "neutral",
            "error":    "Text too short",
        }

    # VADER works best on sentences, not huge blocks
    # Process in chunks of 5000 chars and average
    chunk_size    = 5000
    all_scores    = []
    text_stripped = text.strip()

    for i in range(0, min(len(text_stripped), 50000), chunk_size):
        chunk = text_stripped[i:i + chunk_size]
        if chunk.strip():
            scores = vader.polarity_scores(chunk)
            all_scores.append(scores)

    if not all_scores:
        return {
            "neg": 0.0, "neu": 1.0, "pos": 0.0,
            "compound": 0.0, "label": "neutral",
        }

    # Average scores across chunks
    avg = {
        "neg":      round(float(np.mean([s["neg"]      for s in all_scores])), 4),
        "neu":      round(float(np.mean([s["neu"]      for s in all_scores])), 4),
        "pos":      round(float(np.mean([s["pos"]      for s in all_scores])), 4),
        "compound": round(float(np.mean([s["compound"] for s in all_scores])), 4),
    }

    # Assign label based on compound score
    if avg["compound"] >= VADER_POSITIVE_THRESHOLD:
        avg["label"] = "positive"
    elif avg["compound"] <= VADER_NEGATIVE_THRESHOLD:
        avg["label"] = "negative"
    else:
        avg["label"] = "neutral"

    return avg


def analyze_sentiment_by_sentence(text: str) -> dict:
    """
    Compute sentence-level VADER sentiment.

    This gives more granular analysis than whole-text sentiment.
    Useful for identifying which PARTS of an article are most
    emotionally charged.

    Returns:
        dict with:
            sentence_scores:  list of per-sentence compound scores
            most_positive:    the most positive sentence
            most_negative:    the most negative sentence
            sentiment_shift:  how much sentiment varies (std deviation)
    """
    vader = _get_vader()
    if vader is None:
        return {}

    if not text:
        return {}

    # Split into sentences
    sentences = [
        s.strip()
        for s in re.split(r'[.!?]+', text)
        if len(s.strip().split()) >= 4
    ][:100]    # Cap at 100 sentences

    if not sentences:
        return {}

    scores = []
    for sent in sentences:
        s = vader.polarity_scores(sent)
        scores.append({
            "text":     sent[:100],
            "compound": round(s["compound"], 4),
        })

    compound_vals = [s["compound"] for s in scores]

    return {
        "sentence_scores":    scores[:10],   # Top 10 for display
        "most_positive":      max(scores, key=lambda x: x["compound"]),
        "most_negative":      min(scores, key=lambda x: x["compound"]),
        "sentiment_shift":    round(float(np.std(compound_vals)), 4),
        "sentiment_range":    round(
            max(compound_vals) - min(compound_vals), 4
        ),
        "n_sentences":        len(sentences),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NAMED ENTITY RECOGNITION
# ─────────────────────────────────────────────────────────────────────────────

def extract_entities_spacy(text: str) -> dict:
    """
    Extract named entities from article text using spaCy.

    This runs on ARTICLE TEXT (not user query like intent_decoder.py).
    It tells us WHO and WHAT each article is about.

    Why this matters for bias detection:
        If 8/10 articles focus on "Government" and only 2/10
        focus on "Farmers", that is COVERAGE BIAS.

        If articles consistently use "activist" vs "terrorist"
        for the same group, that is FRAMING BIAS.

    Args:
        text: full article text

    Returns:
        dict with:
            entities:    list of {text, label, count} dicts
            by_type:     entities grouped by entity type
            top_persons: most mentioned people
            top_orgs:    most mentioned organizations
            top_places:  most mentioned places
    """
    if not _spacy_available or _nlp_model is None:
        return {
            "entities":    [],
            "by_type":     {},
            "top_persons": [],
            "top_orgs":    [],
            "top_places":  [],
            "error":       "spaCy not available",
        }

    if not text:
        return {
            "entities": [], "by_type": {},
            "top_persons": [], "top_orgs": [], "top_places": [],
        }

    # Truncate to avoid memory issues with very long articles
    text_truncated = text[:SPACY_MAX_CHARS]

    try:
        doc = _nlp_model(text_truncated)
    except Exception as e:
        return {
            "entities": [], "by_type": {},
            "top_persons": [], "top_orgs": [], "top_places": [],
            "error": str(e),
        }

    # Count entity occurrences
    entity_counts = defaultdict(lambda: defaultdict(int))

    for ent in doc.ents:
        # Only keep relevant entity types
        if ent.label_ not in RELEVANT_ENTITY_TYPES:
            continue

        # Clean entity text
        ent_text = ent.text.strip()

        # Skip very short or very long entities
        if len(ent_text) < 2 or len(ent_text) > 50:
            continue

        # Skip entities that are just numbers
        if ent_text.isdigit():
            continue

        entity_counts[ent.label_][ent_text] += 1

    # Build structured output
    by_type = {}
    all_entities = []

    for label, entities in entity_counts.items():
        # Sort by frequency
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]    # Top 10 per type

        by_type[label] = [
            {
                "text":  ent_text,
                "count": count,
                "label": label,
                "label_name": RELEVANT_ENTITY_TYPES.get(label, label),
            }
            for ent_text, count in sorted_entities
        ]
        all_entities.extend(by_type[label])

    # Sort all entities by count
    all_entities.sort(key=lambda x: x["count"], reverse=True)

    return {
        "entities":    all_entities[:30],    # Top 30 overall
        "by_type":     by_type,
        "top_persons": by_type.get("PERSON", [])[:5],
        "top_orgs":    by_type.get("ORG",    [])[:5],
        "top_places":  by_type.get("GPE",    [])[:5],
    }


def extract_corpus_entities(articles: list) -> dict:
    """
    Extract and aggregate entities across ALL articles.

    This gives corpus-level entity analysis:
    - Which people are mentioned most across all articles?
    - Which organizations appear in both supportive and critical coverage?
    - Do different-stance articles focus on different entities?

    Args:
        articles: list of article dicts with 'full_text' and 'stance'

    Returns:
        dict with corpus-level entity statistics
    """
    all_entity_counts = defaultdict(int)
    entity_by_stance  = defaultdict(lambda: defaultdict(int))
    per_article       = []

    for article in articles:
        text   = article.get("full_text", "") or ""
        stance = article.get("stance", "Neutral")
        source = article.get("source", "Unknown")

        if len(text.split()) < 20:
            continue

        ner_result = extract_entities_spacy(text)

        # Aggregate counts
        for entity in ner_result.get("entities", []):
            ent_key = f"{entity['text']}|{entity['label']}"
            all_entity_counts[ent_key] += entity["count"]
            entity_by_stance[stance][ent_key] += entity["count"]

        per_article.append({
            "source":   source,
            "stance":   stance,
            "entities": ner_result.get("entities", [])[:10],
        })

    # Build top entities across corpus
    top_entities = []
    for ent_key, count in sorted(
        all_entity_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:25]:
        text, label = ent_key.rsplit("|", 1)
        top_entities.append({
            "text":  text,
            "label": label,
            "count": count,
            "label_name": RELEVANT_ENTITY_TYPES.get(label, label),
        })

    # Build by_type structure expected by UI charts
    by_type_map = defaultdict(list)
    for item in top_entities:
        by_type_map[item["label"]].append(item)

    return {
        "top_entities":    top_entities,
        "by_type":         dict(by_type_map),
        "per_article":     per_article,
        "entity_by_stance": {
            stance: sorted(entities.items(), key=lambda x: x[1], reverse=True)[:10]
            for stance, entities in entity_by_stance.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF KEYWORD EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_tfidf_keywords(
    articles:     list,
    top_n:        int = TOP_KEYWORDS_COUNT,
) -> dict:
    """
    Extract the most important keywords from all articles using TF-IDF.

    TF-IDF (Term Frequency — Inverse Document Frequency):
        TF  = how often a word appears in THIS article
        IDF = how rare a word is ACROSS all articles
        TF-IDF = TF * IDF (high = word is frequent here but rare elsewhere)

    Why this is better than simple word counts:
        Simple count: "India" appears 500 times → seems important
        TF-IDF:       "India" appears in ALL articles → IDF is low
                      "demonetization" appears in 3 articles → IDF is high
        TF-IDF correctly identifies "demonetization" as more distinctive.

    Args:
        articles: list of article dicts with 'full_text'
        top_n:    number of top keywords to return

    Returns:
        dict with:
            top_keywords:     list of {word, score, rank} dicts
            per_stance:       keywords broken down by article stance
            word_frequencies: raw word counts for word cloud
    """
    # Collect texts by stance for per-stance analysis
    texts_by_stance = defaultdict(list)
    all_texts       = []
    valid_articles  = []

    for article in articles:
        text   = article.get("full_text", "") or ""
        stance = article.get("stance", "Neutral")

        if len(text.split()) < 15:  # Changed from 30 to 15
            continue

        # Basic preprocessing
        cleaned = _preprocess_text_for_tfidf(text)
        all_texts.append(cleaned)
        texts_by_stance[stance].append(cleaned)
        valid_articles.append(article)

    if not all_texts:
        return {
            "top_keywords":     [],
            "per_stance":       {},
            "word_frequencies": {},
            "error":            "No valid article texts",
        }

    # ── Global TF-IDF across all articles ────────────────────────
    try:
        vectorizer = TfidfVectorizer(
            stop_words   = NEWS_STOPWORDS,
            min_df       = 1,
            max_df       = 0.95,       # Ignore words in >95% of docs
            max_features = 500,
            ngram_range  = (1, 2),     # Include bigrams like "farm laws"
            token_pattern= r'\b[a-zA-Z][a-zA-Z0-9]{' +
                           str(MIN_WORD_LENGTH - 1) + r',}\b',
        )

        tfidf_matrix = vectorizer.fit_transform(all_texts)
        feature_names = vectorizer.get_feature_names_out()

        # Sum TF-IDF scores across all documents
        # This gives importance across the corpus
        tfidf_sum = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

        # Sort by score
        top_indices = tfidf_sum.argsort()[::-1][:top_n]

        top_keywords = [
            {
                "word":  feature_names[idx],
                "score": round(float(tfidf_sum[idx]), 4),
                "rank":  i + 1,
            }
            for i, idx in enumerate(top_indices)
        ]

    except Exception as e:
        return {
            "top_keywords":     [],
            "per_stance":       {},
            "word_frequencies": {},
            "error":            str(e),
        }

    # ── Per-stance TF-IDF ─────────────────────────────────────────
    per_stance = {}
    for stance, stance_texts in texts_by_stance.items():
        if len(stance_texts) < 1:
            continue
        try:
            sv = TfidfVectorizer(
                stop_words   = NEWS_STOPWORDS,
                min_df       = 1,
                max_features = 100,
                ngram_range  = (1, 2),
                token_pattern= r'\b[a-zA-Z][a-zA-Z0-9]{2,}\b',
            )
            sm = sv.fit_transform(stance_texts)
            sf = sv.get_feature_names_out()
            ss = np.asarray(sm.sum(axis=0)).flatten()
            si = ss.argsort()[::-1][:10]

            per_stance[stance] = [
                {
                    "word":  sf[idx],
                    "score": round(float(ss[idx]), 4),
                }
                for idx in si
            ]
        except Exception:
            per_stance[stance] = []

    # ── Word frequencies for word cloud ──────────────────────────
    all_words_text = " ".join(all_texts)
    word_freq      = _compute_word_frequencies(all_words_text)

    return {
        "top_keywords":     top_keywords,
        "per_stance":       per_stance,
        "word_frequencies": word_freq,
        "n_articles":       len(all_texts),
    }


def _preprocess_text_for_tfidf(text: str) -> str:
    """
    Clean text for TF-IDF processing.

    Removes:
    - URLs
    - Email addresses
    - Numbers (standalone)
    - Special characters
    - Extra whitespace
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', ' ', text)

    # Remove special characters, keep letters and spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Lowercase
    text = text.lower()

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def _compute_word_frequencies(text: str) -> dict:
    """
    Compute raw word frequencies for word cloud generation.

    Returns dict of {word: frequency} after removing stopwords.
    """
    words = text.lower().split()
    stopwords_set = set(NEWS_STOPWORDS)

    filtered = [
        w for w in words
        if (w not in stopwords_set and
            len(w) >= MIN_WORD_LENGTH and
            w.isalpha())
    ]

    return dict(Counter(filtered).most_common(200))


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL BIAS SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_manual_bias_score(
    articles: list,
    sentiment_results: list,
) -> dict:
    """
    Compute a manual bias score independent of Gemini.

    This is the KEY RESEARCH CONTRIBUTION for addressing
    the mentor's concern. It gives an INDEPENDENT VALIDATION
    of the Gemini-generated bias score.

    Method:
        1. Take VADER compound scores for all articles
        2. Take stance labels (Supportive/Critical/Neutral)
        3. Combine into a manual bias estimate
        4. Compare with Gemini's bias_score

    If manual_bias ≈ gemini_bias → System is validated
    If they diverge significantly → Worth investigating why

    Args:
        articles:          list of article dicts with 'stance'
        sentiment_results: list of VADER results per article

    Returns:
        dict with manual_bias_score and validation metrics
    """
    if not articles or not sentiment_results:
        return {
            "manual_bias_score": 0.0,
            "method":            "insufficient data",
            "validation_note":   "Cannot compute",
        }

    # ── Component 1: Sentiment-based score ───────────────────────
    # Negative average sentiment → articles are critical in tone
    # Positive average sentiment → articles are supportive in tone
    compound_scores = [
        r.get("compound", 0.0)
        for r in sentiment_results
        if r.get("label") is not None
    ]
    avg_compound = float(np.mean(compound_scores)) if compound_scores else 0.0

    # ── Component 2: Stance distribution ─────────────────────────
    stance_map   = {"Supportive": -1.0, "Critical": 1.0, "Neutral": 0.0}
    stance_scores = [
        stance_map.get(a.get("stance", "Neutral"), 0.0)
        for a in articles
    ]
    avg_stance = float(np.mean(stance_scores)) if stance_scores else 0.0

    # ── Component 3: Sentiment extremity ─────────────────────────
    # Articles with very high or very low compound scores
    # indicate more biased coverage
    if compound_scores:
        extremity = float(np.std(compound_scores))    # High std = polarized
    else:
        extremity = 0.0

    # ── Combine into manual bias score ───────────────────────────
    # Sentiment sign tells us DIRECTION of bias
    # Stance confirms direction
    # We average them (both are [-1, +1])
    #
    # Note: VADER compound is OPPOSITE to bias direction:
    # Positive compound (happy tone) → Supportive coverage → negative bias
    # Negative compound (negative tone) → Critical coverage → positive bias
    # So we NEGATE the compound component
    manual_bias = (
        0.40 * (-avg_compound) +     # Negate: negative tone = critical bias
        0.50 * avg_stance      +     # Stance direction
        0.10 * extremity             # Extremity adds magnitude
    )

    manual_bias = round(float(np.clip(manual_bias, -1.0, 1.0)), 4)

    # ── Sentiment label distribution ─────────────────────────────
    label_counts = Counter(
        r.get("label", "neutral") for r in sentiment_results
    )

    # ── Validation note ───────────────────────────────────────────
    if abs(avg_compound) < 0.05:
        validation_note = "Sentiment near-neutral — topic may be balanced"
    elif avg_compound < -0.2:
        validation_note = "Strong negative sentiment — critical coverage dominant"
    elif avg_compound > 0.2:
        validation_note = "Strong positive sentiment — supportive coverage dominant"
    else:
        validation_note = "Mild sentiment — mixed coverage"

    return {
        "manual_bias_score":    manual_bias,
        "avg_vader_compound":   round(avg_compound, 4),
        "avg_stance_score":     round(avg_stance,   4),
        "sentiment_extremity":  round(extremity,    4),
        "sentiment_distribution": dict(label_counts),
        "validation_note":      validation_note,
        "n_articles_scored":    len(compound_scores),
    }


def validate_against_gemini(
    manual_bias_score: float,
    gemini_bias_score: float,
) -> dict:
    """
    Compare manual NLP bias score with Gemini's bias score.

    This is the VALIDATION step that your paper needs.
    It answers: "Do our two independent methods agree?"

    Args:
        manual_bias_score: from compute_manual_bias_score()
        gemini_bias_score: from bias_model.compute_bias_vector()

    Returns:
        dict with agreement metrics
    """
    difference  = abs(manual_bias_score - gemini_bias_score)
    correlation = 1.0 - (difference / 2.0)   # Simple normalized agreement

    # Direction agreement (do they agree on which side?)
    manual_sign = np.sign(manual_bias_score)
    gemini_sign = np.sign(gemini_bias_score)
    direction_agrees = bool(manual_sign == gemini_sign)

    if difference < 0.15:
        agreement_level = "Strong Agreement"
        note = (
            "Manual NLP and Gemini AI produce consistent bias scores. "
            "This validates the system's reliability."
        )
    elif difference < 0.35:
        agreement_level = "Moderate Agreement"
        note = (
            "Some divergence between manual NLP and Gemini AI. "
            "This may reflect different aspects of bias being measured."
        )
    else:
        agreement_level = "Low Agreement"
        note = (
            "Significant divergence between manual NLP and Gemini AI. "
            "Investigate which articles are causing the discrepancy."
        )

    return {
        "manual_score":      manual_bias_score,
        "gemini_score":      gemini_bias_score,
        "absolute_diff":     round(difference,   4),
        "agreement_score":   round(correlation,  4),
        "agreement_level":   agreement_level,
        "direction_agrees":  direction_agrees,
        "validation_note":   note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WORD CLOUD GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_word_cloud_image(
    word_frequencies: dict,
    width:            int = 800,
    height:           int = 400,
    background_color: str = "#0a0f1e",   # ORBITA navy background
    colormap:         str = "YlOrBr",    # Gold/amber colors matching ORBITA
) -> Optional[object]:
    """
    Generate a word cloud image from word frequencies.

    The word cloud is styled to match ORBITA's dark navy + gold theme.
    Larger words = higher TF-IDF importance.

    Args:
        word_frequencies: dict of {word: frequency} from extract_tfidf_keywords
        width:            image width in pixels
        height:           image height in pixels
        background_color: background color (ORBITA navy = #0a0f1e)
        colormap:         matplotlib colormap for word colors

    Returns:
        matplotlib Figure object for Streamlit display,
        or None if wordcloud not available
    """
    if not _wordcloud_available:
        return None

    if not word_frequencies:
        return None

    try:
        wc = WordCloud(
            width            = width,
            height           = height,
            background_color = background_color,
            colormap         = colormap,
            max_words        = 100,
            min_font_size    = 10,
            max_font_size    = 80,
            prefer_horizontal= 0.7,
            collocations     = False,   # Avoid duplicate phrases
        ).generate_from_frequencies(word_frequencies)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_facecolor(background_color)
        plt.tight_layout(pad=0)

        return fig

    except Exception as e:
        print(f"[nlp_analyzer] Word cloud error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AGENT CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_nlp_context_for_agents(nlp_results: dict) -> str:
    """
    Format NLP analysis results as text for injection into agent prompts.

    This is the bridge between the NLP analyzer and the AI agents.
    Instead of agents reasoning blind, they now receive:
    - Sentiment scores per article
    - Key entities mentioned
    - Top keywords
    - Manual bias estimate

    Format is designed to be:
    1. Concise (agents have token limits)
    2. Structured (easy for agents to parse)
    3. Factual (no interpretation, just numbers)

    Args:
        nlp_results: output from run_nlp_analysis()

    Returns:
        formatted string for injection into agent prompts
    """
    if not nlp_results:
        return ""

    lines = ["NLP ANALYSIS RESULTS (Manual — Independent of AI):"]

    # ── Sentiment Summary ─────────────────────────────────────────
    sentiment_summary = nlp_results.get("sentiment_summary", {})
    if sentiment_summary:
        dist = sentiment_summary.get("distribution", {})
        avg  = sentiment_summary.get("avg_compound", 0)
        lines.append(
            f"\nSentiment Distribution:"
            f" Positive={dist.get('positive', 0)},"
            f" Negative={dist.get('negative', 0)},"
            f" Neutral={dist.get('neutral', 0)}"
            f" | Average VADER compound: {avg:+.3f}"
        )

    # ── Per-article sentiment ─────────────────────────────────────
    per_article = nlp_results.get("per_article_sentiment", [])
    if per_article:
        lines.append("\nPer-Article Sentiment (VADER compound):")
        for art in per_article[:6]:    # Cap at 6 for token budget
            source   = art.get("source",   "?")[:20]
            stance   = art.get("stance",   "?")
            compound = art.get("compound",  0)
            label    = art.get("label",    "?")
            lines.append(
                f"  [{stance}] {source}: "
                f"compound={compound:+.3f} ({label})"
            )

    # ── Top Entities ──────────────────────────────────────────────
    entity_data = nlp_results.get("entity_analysis", {})
    top_entities = entity_data.get("top_entities", [])
    if top_entities:
        top5 = top_entities[:5]
        entity_str = ", ".join(
            f"{e['text']} ({e['label_name']}, {e['count']}x)"
            for e in top5
        )
        lines.append(f"\nMost Mentioned Entities: {entity_str}")

    # ── Top Keywords ──────────────────────────────────────────────
    keyword_data = nlp_results.get("keyword_analysis", {})
    top_keywords = keyword_data.get("top_keywords", [])
    if top_keywords:
        kw_str = ", ".join(
            kw["word"] for kw in top_keywords[:10]
        )
        lines.append(f"\nTop TF-IDF Keywords: {kw_str}")

    # ── Manual Bias Estimate ──────────────────────────────────────
    manual_bias = nlp_results.get("manual_bias", {})
    if manual_bias:
        score = manual_bias.get("manual_bias_score", 0)
        note  = manual_bias.get("validation_note", "")
        lines.append(
            f"\nManual NLP Bias Estimate: {score:+.3f}"
            f" | {note}"
        )

    # ── Validation Against Gemini ─────────────────────────────────
    validation = nlp_results.get("gemini_validation", {})
    if validation:
        agreement = validation.get("agreement_level", "")
        diff      = validation.get("absolute_diff",   0)
        dir_ok    = validation.get("direction_agrees", True)
        lines.append(
            f"\nGemini vs Manual Validation: {agreement}"
            f" (diff={diff:.3f}, direction={'agrees' if dir_ok else 'disagrees'})"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_nlp_analysis(
    articles:          list,
    gemini_bias_score: float = 0.0,
) -> dict:
    """
    Run complete manual NLP analysis on all articles.

    This is the MAIN FUNCTION called from pipeline.py.
    It runs after scraping + deduplication, before agents.

    Pipeline position:
        scraper → deduplicator → [THIS FUNCTION] → chunker → agents

    Args:
        articles:          list of scraped article dicts
        gemini_bias_score: Gemini bias score for validation
                           (pass 0.0 if agents haven't run yet —
                           validation runs after agents complete)

    Returns:
        Complete NLP analysis dict with all metrics.
        This dict is stored in the pipeline result and
        passed to agents as nlp_context.
    """
    start = time.time()

    print(f"\n[nlp_analyzer] Starting manual NLP analysis...")
    print(f"  Articles: {len(articles)}")
    print(f"  VADER available:     {_vader_available}")
    print(f"  spaCy available:     {_spacy_available}")
    print(f"  WordCloud available: {_wordcloud_available}")

    # ── Step 1: VADER Sentiment per article ───────────────────────
    print("\n  [1/4] VADER sentiment analysis...")

    per_article_sentiment = []
    for article in articles:
        text   = article.get("full_text", "") or ""
        source = article.get("source",    "Unknown")
        stance = article.get("stance",    "Neutral")
        title  = (article.get("title",    "")  or "")[:50]

        sentiment = analyze_sentiment_vader(text)

        per_article_sentiment.append({
            "source":   source,
            "stance":   stance,
            "title":    title,
            "compound": sentiment.get("compound", 0.0),
            "pos":      sentiment.get("pos",      0.0),
            "neg":      sentiment.get("neg",      0.0),
            "neu":      sentiment.get("neu",      0.0),
            "label":    sentiment.get("label",    "neutral"),
        })

        print(
            f"    {source[:20]:<20} "
            f"[{stance}] "
            f"compound={sentiment.get('compound', 0):+.3f} "
            f"({sentiment.get('label', 'neutral')})"
        )

    # Aggregate sentiment summary
    compounds = [a["compound"] for a in per_article_sentiment]
    labels    = [a["label"]    for a in per_article_sentiment]

    sentiment_summary = {
        "avg_compound": round(float(np.mean(compounds)), 4) if compounds else 0.0,
        "std_compound": round(float(np.std(compounds)),  4) if compounds else 0.0,
        "distribution": dict(Counter(labels)),
        "most_positive_source": (
            max(per_article_sentiment, key=lambda x: x["compound"])
            if per_article_sentiment else {}
        ),
        "most_negative_source": (
            min(per_article_sentiment, key=lambda x: x["compound"])
            if per_article_sentiment else {}
        ),
    }

    print(f"\n  Sentiment summary: avg={sentiment_summary['avg_compound']:+.4f}")

    # ── Step 2: spaCy NER on article texts ───────────────────────
    print("\n  [2/4] spaCy Named Entity Recognition...")

    entity_analysis = extract_corpus_entities(articles)

    top_ents = entity_analysis.get("top_entities", [])[:5]
    if top_ents:
        print(f"  Top entities: " +
              ", ".join(f"{e['text']}({e['count']})" for e in top_ents))
    else:
        print("  No entities extracted")

    # ── Step 3: TF-IDF Keywords ───────────────────────────────────
    print("\n  [3/4] TF-IDF keyword extraction...")

    keyword_analysis = extract_tfidf_keywords(articles, top_n=TOP_KEYWORDS_COUNT)

    top_kws = keyword_analysis.get("top_keywords", [])[:5]
    if top_kws:
        print(f"  Top keywords: " +
              ", ".join(kw["word"] for kw in top_kws))
    else:
        print("  No keywords extracted")

    # ── Step 4: Manual Bias Score ─────────────────────────────────
    print("\n  [4/4] Computing manual bias score...")

    manual_bias = compute_manual_bias_score(
        articles          = articles,
        sentiment_results = per_article_sentiment,
    )

    print(f"  Manual bias score: {manual_bias['manual_bias_score']:+.4f}")
    print(f"  Note: {manual_bias['validation_note']}")

    # ── Gemini Validation ─────────────────────────────────────────
    # Only meaningful if gemini_bias_score was provided
    gemini_validation = {}
    if gemini_bias_score != 0.0:
        gemini_validation = validate_against_gemini(
            manual_bias_score = manual_bias["manual_bias_score"],
            gemini_bias_score = gemini_bias_score,
        )
        print(
            f"\n  Gemini validation: "
            f"{gemini_validation.get('agreement_level', 'N/A')} "
            f"(diff={gemini_validation.get('absolute_diff', 0):.4f})"
        )

    elapsed = round(time.time() - start, 2)

    print(f"\n[nlp_analyzer] Complete in {elapsed}s")

    # ── Compile final result ──────────────────────────────────────
    result = {
        "per_article_sentiment":  per_article_sentiment,
        "sentiment_summary":      sentiment_summary,
        "entity_analysis":        entity_analysis,
        "keyword_analysis":       keyword_analysis,
        "manual_bias":            manual_bias,
        "gemini_validation":      gemini_validation,
        "elapsed_seconds":        elapsed,
        "n_articles":             len(articles),
        "libraries_used": {
            "vader":     _vader_available,
            "spacy":     _spacy_available,
            "wordcloud": _wordcloud_available,
        },
    }

    # Add formatted context string for agents
    result["agent_context"] = build_nlp_context_for_agents(result)

    return result