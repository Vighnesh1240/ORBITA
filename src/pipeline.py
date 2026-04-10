# src/pipeline.py

import os
import json
from datetime import datetime

from src.bias_timeline import save_bias_entry
from src.fact_density import compute_fact_density

try:
    from .intent_decoder import decode_intent
    from .news_fetcher   import fetch_articles
    from .stance_filter  import label_all_articles, rebalance_articles
    from .scraper        import scrape_articles
    from .deduplicator   import deduplicate
    from .chunker        import chunk_all_articles
    from .embedder       import embed_chunks
    from .vector_store   import store_chunks, get_collection_stats
    from .agents         import run_all_agents
    from .config         import DATA_DIR

except ImportError:

    from intent_decoder import decode_intent
    from news_fetcher   import fetch_articles
    from stance_filter  import label_all_articles, rebalance_articles
    from scraper        import scrape_articles
    from deduplicator   import deduplicate
    from chunker        import chunk_all_articles
    from embedder       import embed_chunks
    from vector_store   import store_chunks, get_collection_stats
    from agents         import run_all_agents
    from config         import DATA_DIR


def save_articles(articles, topic):

    os.makedirs(DATA_DIR, exist_ok=True)

    safe = "".join(
        c if c.isalnum() else "_"
        for c in topic
    )[:40]

    ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = os.path.join(
        DATA_DIR,
        f"{safe}_{ts}.json"
    )

    data = []

    for a in articles:

        data.append({
            "url": a.get("url"),
            "title": a.get("title"),
            "source": a.get("source"),
            "stance": a.get("stance"),
            "word_count":
                len((a.get("full_text") or "").split()),
            "fact_density":
                a.get("fact_density", 0)
        })

    with open(path, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"[pipeline] Metadata saved → {path}")


def run_pipeline(user_input: str) -> dict:

    print("\n" + "=" * 65)
    print("ORBITA — Full Pipeline")
    print("=" * 65)

    # ── STEP 2 ─────────────────────────

    print("\n>>> STEP 2: DATA ENGINEERING")

    intent = decode_intent(user_input)

    print(f"Topic: {intent['topic']}")

    articles = fetch_articles(
        intent["search_queries"]
    )

    if not articles:
        raise RuntimeError(
            "No articles found."
        )

    articles = label_all_articles(articles)

    articles = rebalance_articles(articles)

    articles = scrape_articles(articles)

    articles = deduplicate(articles)

    # ── FACT DENSITY ─────────────────

    for article in articles:

        text = article.get(
            "full_text",
            ""
        )

        density = compute_fact_density(text)

        article["fact_density"] = density

    save_articles(
        articles,
        intent["topic"]
    )

    print(
        f"Step 2 done: {len(articles)} articles"
    )

    # ── STEP 3 ─────────────────────────

    print("\n>>> STEP 3: EMBEDDINGS")

    chunks = chunk_all_articles(
        articles
    )

    embedded_chunks = embed_chunks(
        chunks
    )

    store_chunks(
        embedded_chunks,
        reset=True
    )

    stats = get_collection_stats()

    print(
        f"Step 3 done: "
        f"{stats['total_chunks']} chunks"
    )

    # ── STEP 4 ─────────────────────────

    print("\n>>> STEP 4: AGENTS")

    report = run_all_agents(
        intent["topic"]
    )

    # ── SAVE TIMELINE ─────────────────

    try:

        bias_score = report.get(
            "bias_score",
            0.0
        )

        save_bias_entry(
            topic=intent["topic"],
            bias_score=bias_score
        )

    except Exception as e:

        print(
            f"[timeline error] {e}"
        )

    return {

        "articles": articles,
        "stats": stats,
        "report": report,
        "topic": intent["topic"],

    }