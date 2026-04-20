# src/cache_manager.py

import os
import json
import hashlib
from datetime import datetime, timedelta
from src.config import CACHE_DIR

CACHE_TTL_HOURS = 6   # cache is valid for 6 hours


def _topic_key(topic: str) -> str:
    """Create a stable cache key from a topic string."""
    normalised = topic.lower().strip()
    return hashlib.md5(normalised.encode()).hexdigest()


def _cache_path(topic: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{_topic_key(topic)}.json")


def get_cached_result(topic: str) -> dict | None:
    """
    Return cached pipeline result for this topic if it exists
    and is less than CACHE_TTL_HOURS old. Returns None otherwise.
    """
    path = _cache_path(topic)
    if not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
        age       = datetime.now() - cached_at

        if age > timedelta(hours=CACHE_TTL_HOURS):
            os.remove(path)
            return None

        print(f"[cache] Cache hit for '{topic}' "
              f"(cached {int(age.total_seconds() / 60)} min ago)")
        return data.get("result")

    except Exception:
        return None


def save_to_cache(topic: str, result: dict) -> None:
    """
    Save a pipeline result to the cache.

    MODIFIED: Now also caches nlp_analysis so NLP charts
    work correctly when loading from cache.
    """
    path = _cache_path(topic)

    # Strip large fields not needed for display
    cacheable = {
        "report":   result.get("report", {}),
        "articles": [
            {k: v for k, v in a.items() if k != "full_text"}
            for a in result.get("articles", [])
        ],
        "stats":        result.get("stats",        {}),
        "topic":        result.get("topic",        ""),
        "intent":       result.get("intent",       {}),
        "phase_timings": result.get("phase_timings", {}),
        "elapsed_seconds": result.get("elapsed_seconds", 0),

        # NEW: cache NLP analysis for chart display
        # Remove large intermediate objects, keep display data
        "nlp_analysis": _make_nlp_cacheable(
            result.get("nlp_analysis", {})
        ),

        # Keep image summary but not raw image bytes
        "image_analysis": {
            k: v for k, v in
            result.get("image_analysis", {}).items()
            if k != "article_analyses"
        },
    }

    # Remove retrieved_chunks from agent outputs (too large)
    report = cacheable.get("report", {})
    for agent_key in ("agent_a", "agent_b", "agent_c"):
        if agent_key in report:
            report[agent_key] = {
                k: v for k, v in report[agent_key].items()
                if k != "retrieved_chunks"
            }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cached_at": datetime.now().isoformat(),
                    "result":    cacheable,
                },
                f,
                indent        = 2,
                ensure_ascii  = False,
            )
        print(f"[cache] Saved result for '{topic}'")
    except Exception as e:
        print(f"[cache] Save failed: {e}")


def _make_nlp_cacheable(nlp_results: dict) -> dict:
    """
    Make nlp_results safe to cache — remove huge objects.

    Keeps all display data, removes raw intermediate data.
    """
    if not nlp_results:
        return {}

    return {
        # Keep all display data
        "per_article_sentiment":  nlp_results.get(
            "per_article_sentiment", []
        ),
        "sentiment_summary":      nlp_results.get(
            "sentiment_summary",     {}
        ),
        "manual_bias":            nlp_results.get(
            "manual_bias",           {}
        ),
        "gemini_validation":      nlp_results.get(
            "gemini_validation",     {}
        ),
        "keyword_analysis": {
            # Keep top keywords and word frequencies
            "top_keywords":     nlp_results.get(
                "keyword_analysis", {}
            ).get("top_keywords",     []),
            "word_frequencies": nlp_results.get(
                "keyword_analysis", {}
            ).get("word_frequencies", {}),
            "per_stance":       nlp_results.get(
                "keyword_analysis", {}
            ).get("per_stance",       {}),
        },
        "entity_analysis": {
            # Keep top entities and by_type
            "top_entities": nlp_results.get(
                "entity_analysis", {}
            ).get("top_entities", []),
            "by_type":      nlp_results.get(
                "entity_analysis", {}
            ).get("by_type",      {}),
        },
        # Keep metadata
        "elapsed_seconds": nlp_results.get("elapsed_seconds", 0),
        "n_articles":      nlp_results.get("n_articles",      0),
        "libraries_used":  nlp_results.get("libraries_used",  {}),
    }


def list_cached_topics() -> list[dict]:
    """
    Return a list of all cached topics with their age.
    Used by the sidebar history panel.
    """
    if not os.path.exists(CACHE_DIR):
        return []

    topics = []
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(CACHE_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            cached_at = datetime.fromisoformat(
                data.get("cached_at", "2000-01-01")
            )
            age = datetime.now() - cached_at
            if age > timedelta(hours=CACHE_TTL_HOURS):
                os.remove(path)
                continue
            result = data.get("result", {})
            topics.append({
                "topic":     result.get("topic", "Unknown"),
                "bias_score": result.get("report", {}).get("bias_score", 0.0),
                "n_articles": len(result.get("articles", [])),
                "age_mins":  int(age.total_seconds() / 60),
                "cache_key": fname.replace(".json", ""),
            })
        except Exception:
            continue

    return sorted(topics, key=lambda x: x["age_mins"])


def clear_cache() -> int:
    """Delete all cache files. Returns number deleted."""
    if not os.path.exists(CACHE_DIR):
        return 0
    count = 0
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith(".json"):
            os.remove(os.path.join(CACHE_DIR, fname))
            count += 1
    return count