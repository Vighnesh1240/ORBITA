# src/demo_manager.py
"""
ORBITA Demo Manager
===================
Handles pre-cached analysis results for viva/demo presentations.

Why This Exists:
    During live demos, API calls take 3-5 minutes and can fail.
    This module lets ORBITA load pre-computed results instantly
    from local JSON files — zero internet dependency.

How It Works:
    1. Before your viva, run create_demo_cache.py once
    2. Results are saved to demo_cache/ folder
    3. During demo, Demo Mode loads from those files
    4. Examiner sees instant, polished results

Usage:
    from src.demo_manager import DemoManager
    dm = DemoManager()
    result = dm.load("India Elections 2024")
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Constants ─────────────────────────────────────────────────
DEMO_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "demo_cache"
)

# Pre-defined demo topics with display names
# Add more as you pre-run them
DEMO_TOPICS = {
    "India Elections 2024": {
        "filename":    "india_elections_2024.json",
        "description": "BJP vs Congress — Media bias across 10+ sources",
        "icon":        "🗳️",
        "tags":        ["Politics", "India", "Democracy"],
    },
    "Russia Ukraine War": {
        "filename":    "russia_ukraine_war.json",
        "description": "Western vs Eastern media framing analysis",
        "icon":        "⚔️",
        "tags":        ["Geopolitics", "War", "International"],
    },
    "Elon Musk Twitter": {
        "filename":    "elon_musk_twitter.json",
        "description": "Coverage bias of X/Twitter acquisition",
        "icon":        "🐦",
        "tags":        ["Technology", "Business", "Media"],
    },
    "AI Regulation India": {
        "filename":    "ai_regulation_india.json",
        "description": "How Indian media covers AI policy debates",
        "icon":        "🤖",
        "tags":        ["Technology", "Policy", "India"],
    },
    "Cryptocurrency Regulation": {
        "filename":    "cryptocurrency_regulation.json",
        "description": "Pro-crypto vs regulatory framing analysis",
        "icon":        "₿",
        "tags":        ["Finance", "Technology", "Policy"],
    },
    "Electric Vehicles India": {
        "filename":    "electric_vehicles_india.json",
        "description": "EV adoption coverage — optimism vs skepticism",
        "icon":        "⚡",
        "tags":        ["Technology", "Environment", "India"],
    },
}


class DemoManager:
    """
    Manages pre-cached ORBITA results for demo/viva mode.

    Features:
        - Load pre-computed results instantly
        - List available demo topics
        - Validate cache integrity
        - Show cache metadata (when created, topic stats)
    """

    def __init__(self, cache_dir: str = DEMO_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Core Methods ──────────────────────────────────────────

    def is_available(self, topic_display_name: str) -> bool:
        """Check if a demo cache exists for this topic."""
        config = DEMO_TOPICS.get(topic_display_name)
        if not config:
            return False
        path = self.cache_dir / config["filename"]
        return path.exists()

    def load(self, topic_display_name: str) -> Optional[dict]:
        """
        Load pre-cached result for a demo topic.

        Args:
            topic_display_name: key from DEMO_TOPICS dict

        Returns:
            Pipeline result dict (same format as live pipeline)
            or None if not found
        """
        config = DEMO_TOPICS.get(topic_display_name)
        if not config:
            print(f"[demo] Unknown topic: {topic_display_name}")
            return None

        path = self.cache_dir / config["filename"]
        if not path.exists():
            print(f"[demo] Cache not found: {path}")
            return None

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            # Inject demo metadata into result
            result = data.get("result", data)
            result["_demo_mode"]      = True
            result["_demo_topic"]     = topic_display_name
            result["_demo_cached_at"] = data.get("cached_at", "Unknown")
            result["_demo_config"]    = config

            print(
                f"[demo] Loaded: {topic_display_name} "
                f"(cached {data.get('cached_at', 'unknown date')})"
            )
            return result

        except json.JSONDecodeError as e:
            print(f"[demo] JSON error in {path}: {e}")
            return None
        except Exception as e:
            print(f"[demo] Load error: {e}")
            return None

    def save(self, topic_display_name: str, result: dict) -> bool:
        """
        Save a pipeline result as a demo cache file.

        Args:
            topic_display_name: key from DEMO_TOPICS dict
            result: complete pipeline result dict

        Returns:
            True if saved successfully
        """
        config = DEMO_TOPICS.get(topic_display_name)
        if not config:
            # Auto-create config for unknown topic
            safe_name = "".join(
                c if c.isalnum() else "_"
                for c in topic_display_name.lower()
            )
            filename = f"{safe_name}.json"
        else:
            filename = config["filename"]

        path = self.cache_dir / filename

        # Clean result before saving (remove huge full_text)
        clean = self._make_saveable(result)

        try:
            payload = {
                "cached_at":    datetime.now().isoformat(),
                "topic":        topic_display_name,
                "orbita_version": "1.0",
                "result":       clean,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            size_kb = path.stat().st_size / 1024
            print(
                f"[demo] Saved: {topic_display_name} → "
                f"{filename} ({size_kb:.1f} KB)"
            )
            return True

        except Exception as e:
            print(f"[demo] Save error: {e}")
            return False

    def get_available_topics(self) -> list:
        """
        Return list of topics that have cached results.

        Returns:
            List of dicts with topic info and availability
        """
        available = []
        for name, config in DEMO_TOPICS.items():
            path = self.cache_dir / config["filename"]
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    cached_at = data.get("cached_at", "Unknown")
                    # Get bias score from cached result
                    result    = data.get("result", {})
                    report    = result.get("report", {})
                    bias      = report.get("bias_score", 0.0)
                    n_art     = len(result.get("articles", []))
                    size_kb   = path.stat().st_size / 1024

                    available.append({
                        "name":        name,
                        "icon":        config.get("icon", "◉"),
                        "description": config.get("description", ""),
                        "tags":        config.get("tags", []),
                        "cached_at":   cached_at,
                        "bias_score":  bias,
                        "n_articles":  n_art,
                        "size_kb":     size_kb,
                        "available":   True,
                    })
                except Exception:
                    pass

        # Sort: most recent first
        available.sort(key=lambda x: x.get("cached_at", ""), reverse=True)
        return available

    def get_all_topics_with_status(self) -> list:
        """Return ALL topics with available/unavailable status."""
        result = []
        for name, config in DEMO_TOPICS.items():
            path      = self.cache_dir / config["filename"]
            is_avail  = path.exists()
            result.append({
                "name":        name,
                "icon":        config.get("icon", "◉"),
                "description": config.get("description", ""),
                "tags":        config.get("tags", []),
                "available":   is_avail,
                "filename":    config["filename"],
            })
        return result

    def delete(self, topic_display_name: str) -> bool:
        """Delete cached result for a topic."""
        config = DEMO_TOPICS.get(topic_display_name)
        if not config:
            return False
        path = self.cache_dir / config["filename"]
        if path.exists():
            path.unlink()
            print(f"[demo] Deleted cache: {topic_display_name}")
            return True
        return False

    def get_stats(self) -> dict:
        """Return summary stats about the demo cache."""
        topics  = self.get_available_topics()
        total_kb = sum(t.get("size_kb", 0) for t in topics)
        return {
            "total_cached":  len(topics),
            "total_possible": len(DEMO_TOPICS),
            "total_size_kb":  round(total_kb, 1),
            "topics":         topics,
        }

    # ── Private Methods ───────────────────────────────────────

    def _make_saveable(self, result: dict) -> dict:
        """
        Strip large fields to keep cache files small.
        Preserves all display data.
        """
        import copy
        clean = copy.deepcopy(result)

        # Strip full_text from articles (too large)
        for article in clean.get("articles", []):
            article.pop("full_text",   None)
            article.pop("raw_content", None)

        # Strip retrieved_chunks from agents
        report = clean.get("report", {})
        for agent_key in ("agent_a", "agent_b", "agent_c"):
            agent = report.get(agent_key, {})
            agent.pop("retrieved_chunks", None)

        # Strip heavy NLP fields
        nlp = clean.get("nlp_analysis", {})
        if nlp:
            nlp.pop("tfidf_matrix", None)
            nlp.pop("raw_vectors",  None)

        return clean