# src/heatmap_manager.py
"""
ORBITA Bias Heatmap Manager
============================
Manages the Sources × Topics bias heatmap data.

The heatmap is the single most powerful visualization
in ORBITA — it shows at a glance which sources
are biased and on which topics.

Data is accumulated from past pipeline runs
and stored in a local JSON database.
"""

import os
import json
from datetime import datetime
from typing import Optional
import numpy as np


HEATMAP_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "heatmap_data.json"
)


class HeatmapManager:
    """Manages bias heatmap data accumulation and retrieval."""

    def __init__(self, db_path: str = HEATMAP_DB_PATH):
        self.db_path = db_path
        self._data   = self._load()

    # ── Public API ────────────────────────────────────────────

    def record_run(
        self,
        topic:    str,
        articles: list,
        report:   dict,
    ) -> None:
        """
        Record bias scores per source for a completed run.
        Call this after every pipeline run.

        Args:
            topic:    topic string
            articles: list of article dicts with source + stance
            report:   final report dict with bias_score
        """
        topic_key = self._normalize_topic(topic)
        timestamp = datetime.now().isoformat()

        if topic_key not in self._data["topics"]:
            self._data["topics"][topic_key] = {
                "display": topic,
                "runs":    [],
            }

        # Collect per-source scores
        from src.source_credibility import get_source_info
        per_source = {}

        for article in articles:
            source = article.get("source", "Unknown")
            if not source or source == "Unknown":
                continue

            stance = article.get("stance", "Neutral")
            stance_num = (
                -1.0 if stance == "Supportive"
                else  1.0 if stance == "Critical"
                else  0.0
            )

            if source not in per_source:
                per_source[source] = {
                    "scores":      [],
                    "credibility": get_source_info(source).get("credibility", 0.60),
                }
            per_source[source]["scores"].append(stance_num)

        # Average per source
        source_averages = {}
        for src, data in per_source.items():
            if data["scores"]:
                source_averages[src] = {
                    "bias":        round(
                        float(np.mean(data["scores"])), 4
                    ),
                    "n_articles":  len(data["scores"]),
                    "credibility": data["credibility"],
                }

        self._data["topics"][topic_key]["runs"].append({
            "timestamp":    timestamp,
            "overall_bias": round(
                float(report.get("bias_score", 0.0)), 4
            ),
            "per_source":   source_averages,
        })

        # Track all seen sources
        for src in source_averages:
            if src not in self._data["sources"]:
                self._data["sources"][src] = True

        self._save()

    def get_matrix(
        self,
        min_topics:  int = 1,
        min_sources: int = 1,
        max_topics:  int = 10,
        max_sources: int = 12,
    ) -> dict:
        """
        Build the Sources × Topics matrix for the heatmap.

        Returns:
            dict with topics, sources, matrix, and metadata
        """
        if not self._data["topics"]:
            return self._empty_matrix()

        # Collect all topics and sources
        all_topics  = list(self._data["topics"].keys())[:max_topics]
        all_sources = set()

        for tk in all_topics:
            runs = self._data["topics"][tk].get("runs", [])
            if runs:
                last_run = runs[-1]  # use most recent run
                all_sources.update(last_run.get("per_source", {}).keys())

        all_sources = sorted(list(all_sources))[:max_sources]

        if not all_topics or not all_sources:
            return self._empty_matrix()

        # Build matrix: rows=sources, cols=topics
        matrix = []
        for src in all_sources:
            row = []
            for tk in all_topics:
                runs = self._data["topics"][tk].get("runs", [])
                if not runs:
                    row.append(None)
                    continue
                last_run = runs[-1]
                src_data = last_run.get("per_source", {}).get(src)
                if src_data:
                    row.append(src_data.get("bias", None))
                else:
                    row.append(None)
            matrix.append(row)

        topic_displays = [
            self._data["topics"][tk].get("display", tk)[:25]
            for tk in all_topics
        ]

        return {
            "topics":          topic_displays,
            "topic_keys":      all_topics,
            "sources":         all_sources,
            "matrix":          matrix,
            "n_topics":        len(all_topics),
            "n_sources":       len(all_sources),
            "has_data":        True,
        }

    def get_source_bias_profile(self, source: str) -> dict:
        """Get bias scores for one source across all topics."""
        profile = {}
        for tk, tdata in self._data["topics"].items():
            display = tdata.get("display", tk)
            runs    = tdata.get("runs", [])
            if not runs:
                continue
            last   = runs[-1]
            srcdat = last.get("per_source", {}).get(source)
            if srcdat:
                profile[display] = srcdat.get("bias", 0.0)
        return profile

    def get_topic_list(self) -> list:
        """Return list of all topics with run counts."""
        result = []
        for tk, tdata in self._data["topics"].items():
            result.append({
                "key":       tk,
                "display":   tdata.get("display", tk),
                "n_runs":    len(tdata.get("runs", [])),
                "last_bias": (
                    tdata["runs"][-1].get("overall_bias", 0.0)
                    if tdata.get("runs") else 0.0
                ),
            })
        return sorted(result, key=lambda x: x["n_runs"], reverse=True)

    def get_stats(self) -> dict:
        return {
            "n_topics":  len(self._data["topics"]),
            "n_sources": len(self._data["sources"]),
            "db_path":   self.db_path,
        }

    def clear(self) -> None:
        """Reset the heatmap database."""
        self._data = {"topics": {}, "sources": {}}
        self._save()

    # ── Private ───────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"topics": {}, "sources": {}}

    def _save(self) -> None:
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[heatmap] Save error: {e}")

    def _normalize_topic(self, topic: str) -> str:
        return "_".join(topic.lower().split())[:40]

    def _empty_matrix(self) -> dict:
        return {
            "topics":    [],
            "sources":   [],
            "matrix":    [],
            "n_topics":  0,
            "n_sources": 0,
            "has_data":  False,
        }