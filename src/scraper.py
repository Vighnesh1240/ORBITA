# src/scraper.py
# MODIFY the existing _scrape_single function to also extract images

import time
from newspaper import Article

try:
    from .config import SCRAPE_TIMEOUT, MIN_ARTICLE_CHARS
except ImportError:
    from config import SCRAPE_TIMEOUT, MIN_ARTICLE_CHARS


def _scrape_single(url: str) -> dict:
    """
    Download and parse a single article URL using newspaper4k.

    MODIFIED: Now returns a dict with both text AND image URLs,
    instead of just returning the text string.

    Returns:
        dict with 'text', 'images', 'top_image', or None if failed
    """
    try:
        article = Article(url, request_timeout=SCRAPE_TIMEOUT)
        article.download()
        article.parse()

        text = article.text.strip()

        if len(text) < MIN_ARTICLE_CHARS:
            return None

        # Extract image URLs from newspaper4k
        images = []
        try:
            if hasattr(article, "images") and article.images:
                images = list(article.images)
        except Exception:
            images = []

        top_image = ""
        try:
            if hasattr(article, "top_image") and article.top_image:
                top_image = article.top_image
        except Exception:
            top_image = ""

        return {
            "text":      text,
            "images":    images,
            "top_image": top_image,
        }

    except Exception:
        return None


def scrape_articles(articles: list[dict]) -> list[dict]:
    """
    Main function. Attempts to scrape full text AND images.

    MODIFIED: Now also populates 'image_urls' and 'top_image'
    fields on each article dict.
    """
    print(f"\n[scraper] Scraping full text for {len(articles)} articles...")

    successful = []
    failed     = 0

    for i, article in enumerate(articles):
        url   = article.get("url", "")
        title = article.get("title", "")[:50]

        print(f"  [{i+1}/{len(articles)}] {title}...")

        scraped = _scrape_single(url)

        if scraped:
            article["full_text"]  = scraped["text"]
            article["image_urls"] = scraped["images"]    # NEW
            article["top_image"]  = scraped["top_image"] # NEW

            successful.append(article)
            word_count  = len(scraped["text"].split())
            image_count = len(scraped["images"])
            print(f"    OK — {word_count} words, "
                  f"{image_count} images found")
        else:
            failed += 1
            print(f"    SKIPPED — paywall, JS page, or too short")

        if i < len(articles) - 1:
            time.sleep(0.3)

    print(f"\n[scraper] Done. {len(successful)} scraped, "
          f"{failed} failed/skipped.")
    return successful