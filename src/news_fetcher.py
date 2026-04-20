import time
import requests
from datetime import datetime, timedelta

try:
    from .config import NEWS_API_KEY, MAX_ARTICLES, MIN_ARTICLES
except ImportError:
    from config import NEWS_API_KEY, MAX_ARTICLES, MIN_ARTICLES


NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"


def _fetch_for_query(query: str, page_size: int = 10) -> list[dict]:
    """
    Call NewsAPI for a single query.
    Returns a list of raw article dicts from the API.
    """
    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY not found in .env file.")

    # Use a wider lookback window so sparse topics still return enough results.
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    params = {
        "q":          query,
        "pageSize":   page_size,
        "apiKey":     NEWS_API_KEY,
        "language":   "en",
        "sortBy":     "relevancy",
        "from":       from_date,
    }

    try:
        response = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=10)
        data     = response.json()

        if response.status_code == 200 and data.get("status") == "ok":
            return data.get("articles", [])

        elif response.status_code == 401:
            raise RuntimeError("Invalid NewsAPI key. Check your .env file.")

        elif response.status_code == 429:
            print("    [news_fetcher] Rate limit hit — waiting 5 seconds...")
            time.sleep(5)
            return []

        else:
            print(f"    [news_fetcher] API warning: {data.get('message', 'Unknown')}")
            return []

    except requests.exceptions.Timeout:
        print(f"    [news_fetcher] Timeout for query: '{query}'")
        return []

    except requests.exceptions.ConnectionError:
        raise RuntimeError("No internet connection. Cannot reach NewsAPI.")


def _normalize_article(raw: dict) -> dict | None:
    """
    Convert a raw NewsAPI article dict into ORBITA's standard format.
    Returns None if the article is missing essential fields.
    """
    url     = (raw.get("url") or "").strip()
    title   = (raw.get("title") or "").strip()
    source  = (raw.get("source") or {}).get("name", "Unknown")
    desc    = (raw.get("description") or "").strip()
    content = (raw.get("content") or "").strip()

    # Discard articles without URL or title
    if not url or not title:
        return None

    # Discard removed/deleted articles
    if "[Removed]" in title or "[Removed]" in (desc or ""):
        return None

    return {
        "url":         url,
        "title":       title,
        "source":      source,
        "description": desc,
        "raw_content": content,   # NewsAPI gives only ~200 chars; scraper gets full text
        "stance":      None,      # filled in by stance_filter.py
        "full_text":   None,      # filled in by scraper.py
    }


def _is_relevant(article: dict, topic: str, relaxed: bool = False) -> bool:
    """
    Quick relevance filter — discard articles with zero 
    topic overlap to avoid polluting the vector store.
    """
    topic_words = set(topic.lower().split())
    # Remove very common words
    stop = {"vs", "the", "a", "an", "and", "or", "in", "of"}
    topic_words -= stop

    if not topic_words:
        return True  # Can't filter if no meaningful words

    text = (
        (article.get("title", "") or "") + " " +
        (article.get("description", "") or "")
    ).lower()

    # In fallback mode we relax relevance to recover more candidates.
    matches = sum(1 for word in topic_words if word in text)
    if relaxed:
        return matches >= 1
    return matches >= max(1, len(topic_words) // 3)


def _build_fallback_queries(topic: str) -> list[str]:
    """
    Build broader backup queries when initial recall is low.
    """
    topic = (topic or "").strip()
    lower = topic.lower()

    candidates = [
        f"{topic} policy",
        f"{topic} law",
        f"{topic} RBI",
        f"{topic} government",
    ]

    if "crypto" in lower or "cryptocurrency" in lower or "bitcoin" in lower:
        candidates.extend([
            "cryptocurrency regulation india",
            "crypto policy india RBI",
            "bitcoin regulation india",
        ])

    seen = set()
    unique = []
    for q in candidates:
        q = " ".join(q.split())
        if not q:
            continue
        key = q.lower()
        if key in seen or key == lower:
            continue
        seen.add(key)
        unique.append(q)

    return unique[:6]


def fetch_articles(search_queries: list[str]) -> list[dict]:
    """Main function — now with relevance filtering."""
    print(f"\n[news_fetcher] Fetching for {len(search_queries)} queries...")

    all_articles    = []
    seen_urls       = set()
    topic           = search_queries[0] if search_queries else ""

    for i, query in enumerate(search_queries):
        per_query = 10
        print(f"  Query {i+1}: '{query}'")
        raw_articles = _fetch_for_query(query, page_size=per_query)

        added = 0
        for raw in raw_articles:
            normalized = _normalize_article(raw)
            if normalized is None:
                continue
            if not _is_relevant(normalized, topic):
                print(f"    SKIPPED (off-topic): {normalized['title'][:50]}")
                continue
            url = normalized["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                all_articles.append(normalized)
                added += 1

        print(f"    → {added} relevant articles added")
        if i < len(search_queries) - 1:
            time.sleep(0.5)

    # Fallback pass: broaden query strategy and relax relevance when recall is low.
    if len(all_articles) < MIN_ARTICLES and topic:
        print(
            f"\n[news_fetcher] Low recall ({len(all_articles)}/{MIN_ARTICLES}). "
            "Running fallback queries..."
        )
        fallback_queries = _build_fallback_queries(topic)

        for i, query in enumerate(fallback_queries):
            if len(all_articles) >= MAX_ARTICLES:
                break

            print(f"  Fallback {i+1}: '{query}'")
            raw_articles = _fetch_for_query(query, page_size=12)

            added = 0
            for raw in raw_articles:
                normalized = _normalize_article(raw)
                if normalized is None:
                    continue
                if not _is_relevant(normalized, topic, relaxed=True):
                    continue

                url = normalized["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_articles.append(normalized)
                    added += 1

                    if len(all_articles) >= MAX_ARTICLES:
                        break

            print(f"    → {added} fallback articles added")
            if i < len(fallback_queries) - 1:
                time.sleep(0.5)

    print(f"\n[news_fetcher] Total: {len(all_articles)} articles")
    return all_articles[:MAX_ARTICLES]