# src/scraper.py

import time
from newspaper import Article

try:
    from .config import SCRAPE_TIMEOUT, MIN_ARTICLE_CHARS
except ImportError:
    from config import SCRAPE_TIMEOUT, MIN_ARTICLE_CHARS


def _scrape_single(url: str) -> str | None:
    """
    Download and parse a single article URL using newspaper4k.
    Returns full text string, or None if scraping fails.
    """
    try:
        article = Article(url, request_timeout=SCRAPE_TIMEOUT)
        article.download()
        article.parse()

        text = article.text.strip()

        if len(text) < MIN_ARTICLE_CHARS:
            return None  # Too short — likely a paywall or JS-rendered page

        return text

    except Exception:
        # newspaper4k raises many different exceptions for failed pages
        # (paywalls, 403s, JS-only pages, timeouts) — catch all silently
        return None


def scrape_articles(articles: list[dict]) -> list[dict]:
    """
    Main function. Attempts to scrape full text for each article.
    Articles that fail scraping are removed from the list.

    Args:
        articles: list of article dicts (must have 'url' field)

    Returns:
        list of articles that were successfully scraped,
        each with 'full_text' field populated
    """
    print(f"\n[scraper] Scraping full text for {len(articles)} articles...")

    successful = []
    failed     = 0

    for i, article in enumerate(articles):
        url   = article.get("url", "")
        title = article.get("title", "")[:50]

        print(f"  [{i+1}/{len(articles)}] {title}...")

        full_text = _scrape_single(url)

        if full_text:
            article["full_text"] = full_text
            successful.append(article)
            word_count = len(full_text.split())
            print(f"    OK — {word_count} words scraped")
        else:
            failed += 1
            print(f"    SKIPPED — paywall, JS page, or too short")

        # Small delay between requests — be polite to servers
        if i < len(articles) - 1:
            time.sleep(0.3)

    print(f"\n[scraper] Done. {len(successful)} scraped, {failed} failed/skipped.")
    return successful