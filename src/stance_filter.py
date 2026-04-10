# src/stance_filter.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from .config import STANCE_LABELS, ARTICLES_PER_STANCE
except ImportError:
    from config import STANCE_LABELS, ARTICLES_PER_STANCE


# Keyword signals for each stance
# These act as "reference documents" for zero-shot classification
STANCE_SIGNALS = {
    "Supportive": [
        "benefit growth positive support success boost opportunity"
        " progress advance approval praise welcome favour advantage"
        " development improvement achievement milestone promising"
    ],
    "Critical": [
        "problem failure risk danger oppose concern criticism"
        " protest controversy negative harm damage threat oppose"
        " reject against warning flaw corruption scandal backlash"
        " violation controversial problematic ineffective poor"
    ],
    "Neutral": [
        "report analysis study according data government says stated"
        " announced official policy statement explained described"
        " information details background context overview facts"
        " reviewed examined survey research update published"
    ],
}


def _get_article_text(article: dict) -> str:
    """
    Combine all available text fields for classification.
    Uses title + description since full_text isn't scraped yet.
    """
    parts = [
        article.get("title", ""),
        article.get("description", ""),
        article.get("raw_content", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def classify_stance(article: dict) -> str:
    """
    Classify a single article as Supportive, Critical, or Neutral
    using TF-IDF cosine similarity against stance signal documents.

    Returns one of: "Supportive", "Critical", "Neutral"
    """
    article_text = _get_article_text(article)

    if not article_text:
        return "Neutral"

    # Build corpus: [article_text, supportive_signals, critical_signals, neutral_signals]
    corpus = [article_text] + [STANCE_SIGNALS[label][0] for label in STANCE_LABELS]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return "Neutral"

    # Compare article (index 0) against each stance signal (indices 1,2,3)
    article_vec  = tfidf_matrix[0]
    stance_vecs  = tfidf_matrix[1:]

    similarities = cosine_similarity(article_vec, stance_vecs)[0]

    # Pick the stance with highest similarity
    best_idx = int(np.argmax(similarities))
    return STANCE_LABELS[best_idx]


def label_all_articles(articles: list[dict]) -> list[dict]:
    """
    Run stance classification on every article.
    Adds a 'stance' field to each article dict.
    """
    print(f"\n[stance_filter] Classifying {len(articles)} articles...")

    for article in articles:
        article["stance"] = classify_stance(article)

    # Print distribution
    distribution = {label: 0 for label in STANCE_LABELS}
    for article in articles:
        distribution[article["stance"]] += 1

    print(f"  Stance distribution: {distribution}")
    return articles


def rebalance_articles(articles: list[dict]) -> list[dict]:
    """
    Ensure we have a mix of stances.
    If all articles are the same stance, keep the best from each available stance
    and flag that rebalancing occurred.

    Strategy:
    - Group articles by stance
    - If any stance has 0 articles, that's fine (topic may genuinely be one-sided)
    - If ONE stance dominates with 90%+ of articles, warn the user
    - Always return at least MIN_ARTICLES articles total
    """
    if not articles:
        return articles

    grouped = {label: [] for label in STANCE_LABELS}
    for article in articles:
        stance = article.get("stance", "Neutral")
        grouped[stance].append(article)

    counts = {k: len(v) for k, v in grouped.items()}
    total  = sum(counts.values())

    print(f"\n[stance_filter] Rebalancing check:")
    for label, count in counts.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {label}: {count} articles ({pct:.0f}%)")

    # Check if one stance dominates (>= 90%)
    dominant = max(counts, key=counts.get)
    if counts[dominant] / total >= 0.9:
        print(f"  WARNING: '{dominant}' dominates ({counts[dominant]}/{total} articles).")
        print(f"  This may mean the topic genuinely lacks diverse coverage,")
        print(f"  or NewsAPI's results are biased. Proceeding with available data.")

    # Build balanced output:
    # Take up to ARTICLES_PER_STANCE from each stance, then fill remaining slots
    balanced = []
    for label in STANCE_LABELS:
        balanced.extend(grouped[label][:ARTICLES_PER_STANCE])

    # Fill remaining slots with leftover articles (any stance)
    max_total = 10
    remaining_slots = max_total - len(balanced)
    if remaining_slots > 0:
        already_in = set(id(a) for a in balanced)
        extras = [a for a in articles if id(a) not in already_in]
        balanced.extend(extras[:remaining_slots])

    print(f"  Final balanced set: {len(balanced)} articles")
    return balanced