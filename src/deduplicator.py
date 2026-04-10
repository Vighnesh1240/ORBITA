# src/deduplicator.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from .config import SIMILARITY_THRESHOLD
except ImportError:
    from config import SIMILARITY_THRESHOLD


def _deduplicate_by_url(articles: list[dict]) -> list[dict]:
    """
    Remove exact URL duplicates. Keeps the first occurrence.
    """
    seen_urls = set()
    unique    = []

    for article in articles:
        url = article.get("url", "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(article)

    removed = len(articles) - len(unique)
    if removed > 0:
        print(f"  URL dedup: removed {removed} exact duplicate(s)")

    return unique


def _deduplicate_by_content(articles: list[dict]) -> list[dict]:
    """
    Remove near-duplicate articles using TF-IDF cosine similarity
    on the full article text.

    If two articles have similarity >= SIMILARITY_THRESHOLD,
    keep only the first one.
    """
    if len(articles) < 2:
        return articles

    texts = [
        (article.get("full_text") or article.get("description") or "")
        for article in articles
    ]

    # Need at least some text to compare
    non_empty = [t for t in texts if t.strip()]
    if len(non_empty) < 2:
        return articles

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return articles  # Not enough vocabulary to compare

    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Greedy deduplication: mark articles as duplicate if too similar
    # to any already-kept article
    keep_flags = [True] * len(articles)

    for i in range(len(articles)):
        if not keep_flags[i]:
            continue
        for j in range(i + 1, len(articles)):
            if not keep_flags[j]:
                continue
            if similarity_matrix[i][j] >= SIMILARITY_THRESHOLD:
                keep_flags[j] = False  # mark j as duplicate of i

    unique   = [a for a, keep in zip(articles, keep_flags) if keep]
    removed  = len(articles) - len(unique)

    if removed > 0:
        print(f"  Content dedup: removed {removed} near-duplicate(s) "
              f"(similarity >= {SIMILARITY_THRESHOLD})")

    return unique


def deduplicate(articles: list[dict]) -> list[dict]:
    """
    Main function. Runs both URL and content deduplication.

    Args:
        articles: list of article dicts

    Returns:
        deduplicated list
    """
    print(f"\n[deduplicator] Running deduplication on {len(articles)} articles...")

    articles = _deduplicate_by_url(articles)
    articles = _deduplicate_by_content(articles)

    print(f"  Final count after deduplication: {len(articles)} articles")
    return articles