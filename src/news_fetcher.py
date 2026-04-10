import time
import requests
from datetime import datetime, timedelta

try:
    from .config import NEWS_API_KEY, MAX_ARTICLES
except ImportError:
    from config import NEWS_API_KEY, MAX_ARTICLES


NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"


def _fetch_for_query(query: str, page_size: int = 5) -> list[dict]:
    """
    Call NewsAPI for a single query.
    Returns a list of raw article dicts from the API.
    """
    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY not found in .env file.")

    # Only fetch articles from the last 7 days for relevance (free plan limit)
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

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


def fetch_articles(search_queries: list[str]) -> list[dict]:
    """
    Main function. Takes a list of search queries.
    Fetches articles for each, deduplicates by URL, and returns
    a combined list capped at MAX_ARTICLES.

    Args:
        search_queries: list of query strings from decode_intent()

    Returns:
        list of normalized article dicts
    """
    print(f"\n[news_fetcher] Fetching articles for {len(search_queries)} queries...")

    all_articles    = []
    seen_urls       = set()
    articles_needed = MAX_ARTICLES

    for i, query in enumerate(search_queries):
        # How many to request from this query
        per_query = max(3, articles_needed // len(search_queries) + 2)

        print(f"  Query {i+1}/{len(search_queries)}: '{query}' (requesting {per_query})")
        raw_articles = _fetch_for_query(query, page_size=per_query)

        added = 0
        for raw in raw_articles:
            normalized = _normalize_article(raw)
            if normalized is None:
                continue
            url = normalized["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                all_articles.append(normalized)
                added += 1

        print(f"    → {added} new articles added (total: {len(all_articles)})")

        # Small delay between queries to be polite to the API
        if i < len(search_queries) - 1:
            time.sleep(0.5)

    print(f"\n[news_fetcher] Done. Total unique articles fetched: {len(all_articles)}")
    return all_articles[:MAX_ARTICLES]