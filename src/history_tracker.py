# src/history_tracker.py
"""
ORBITA Historical Bias Tracker
================================
SQLite-based database that records every analysis run.

Why SQLite:
    - Built into Python — zero extra dependencies
    - Persistent across sessions (unlike session_state)
    - Queryable — can filter by topic, date, bias range
    - Fast — sub-millisecond reads for dashboard
    - File-based — easy to backup, share, demo

Schema:
    runs table:
        id, topic, timestamp, bias_score, weighted_bias,
        n_articles, n_chunks, vader_score, synthesis_words,
        mean_credibility, high_cred_count, elapsed_seconds

    articles table:
        id, run_id, source, stance, credibility, word_count

    agents table:
        id, run_id, agent, confidence, n_arguments

Research Value:
    After collecting 30+ runs across 10+ topics,
    you can show:
    - Which topics are consistently biased
    - How bias changes over time for same topic
    - Which sources appear most across topics
    - Credibility distribution of news coverage
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional
import numpy as np


# ── Database path ─────────────────────────────────────────────
DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "orbita_history.db"
)


# ─────────────────────────────────────────────────────────────
# DATABASE INITIALIZATION
# ─────────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Get SQLite connection with row_factory for dict access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    """
    Create all tables if they don't exist.
    Safe to call multiple times — uses IF NOT EXISTS.
    """
    conn = _get_connection()
    cur  = conn.cursor()

    # Main runs table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            topic             TEXT    NOT NULL,
            topic_normalized  TEXT    NOT NULL,
            timestamp         TEXT    NOT NULL,
            date_only         TEXT    NOT NULL,
            bias_score        REAL    DEFAULT 0.0,
            weighted_bias     REAL    DEFAULT 0.0,
            bias_direction    TEXT    DEFAULT 'balanced',
            n_articles        INTEGER DEFAULT 0,
            n_chunks          INTEGER DEFAULT 0,
            vader_score       REAL    DEFAULT 0.0,
            synthesis_words   INTEGER DEFAULT 0,
            mean_credibility  REAL    DEFAULT 0.60,
            high_cred_count   INTEGER DEFAULT 0,
            low_cred_count    INTEGER DEFAULT 0,
            elapsed_seconds   REAL    DEFAULT 0.0,
            agent_a_conf      REAL    DEFAULT 0.0,
            agent_b_conf      REAL    DEFAULT 0.0,
            n_arguments       INTEGER DEFAULT 0,
            n_counter_args    INTEGER DEFAULT 0,
            n_hallucinations  INTEGER DEFAULT 0,
            n_loaded_phrases  INTEGER DEFAULT 0,
            is_demo           INTEGER DEFAULT 0
        )
    """)

    # Per-article records
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER NOT NULL,
            source      TEXT,
            stance      TEXT,
            credibility REAL    DEFAULT 0.60,
            word_count  INTEGER DEFAULT 0,
            vader_score REAL    DEFAULT 0.0,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)

    # Source aggregation (for heatmap queries)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS source_bias (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER NOT NULL,
            source      TEXT    NOT NULL,
            topic       TEXT    NOT NULL,
            bias_score  REAL    DEFAULT 0.0,
            credibility REAL    DEFAULT 0.60,
            n_articles  INTEGER DEFAULT 1,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)

    # Indexes for fast queries
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_runs_topic
        ON runs(topic_normalized)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_runs_date
        ON runs(date_only)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_source_bias_source
        ON source_bias(source)
    """)

    conn.commit()
    conn.close()
    print(f"[history] Database ready: {DB_PATH}")


# ─────────────────────────────────────────────────────────────
# SAVE RUN
# ─────────────────────────────────────────────────────────────

def save_run(
    pipeline_result: dict,
    elapsed_seconds: float = 0.0,
    is_demo:         bool  = False,
) -> Optional[int]:
    """
    Save a complete pipeline result to SQLite.

    Args:
        pipeline_result: full result dict from run_pipeline()
        elapsed_seconds: total pipeline time
        is_demo:         True if loaded from demo cache

    Returns:
        run_id (int) or None if save failed
    """
    initialize_database()

    try:
        report   = pipeline_result.get("report",   {})
        articles = pipeline_result.get("articles", [])
        topic    = pipeline_result.get("topic",    "Unknown")
        nlp      = pipeline_result.get("nlp_analysis", {})

        now      = datetime.now()
        ts       = now.isoformat()
        date_str = now.strftime("%Y-%m-%d")

        # Extract metrics
        bias_score  = float(report.get("bias_score", 0.0))
        direction   = (
            "supportive" if bias_score < -0.2
            else "critical" if bias_score > 0.2
            else "balanced"
        )
        agent_a     = report.get("agent_a", {})
        agent_b     = report.get("agent_b", {})
        agent_c     = report.get("agent_c", {})
        synthesis   = report.get("synthesis_report", "")
        vader_score = float(
            nlp.get("sentiment_summary", {}).get("avg_compound",
            nlp.get("corpus_sentiment", {}).get("mean_compound", 0.0))
        )

        # Credibility metrics from articles
        cred_scores = []
        for a in articles:
            c = a.get("credibility_score",
                a.get("credibility_info", {}).get("credibility", 0.60))
            cred_scores.append(float(c))

        mean_cred  = float(np.mean(cred_scores)) if cred_scores else 0.60
        high_cred  = sum(1 for c in cred_scores if c >= 0.80)
        low_cred   = sum(1 for c in cred_scores if c  < 0.60)

        # Weighted bias
        from src.source_credibility import compute_credibility_weighted_bias
        cred_result  = compute_credibility_weighted_bias(
            articles, bias_score
        )
        weighted_bias = float(
            cred_result.get("weighted_bias_score", bias_score)
        )

        conn = _get_connection()
        cur  = conn.cursor()

        # Insert run
        cur.execute("""
            INSERT INTO runs (
                topic, topic_normalized, timestamp, date_only,
                bias_score, weighted_bias, bias_direction,
                n_articles, n_chunks, vader_score, synthesis_words,
                mean_credibility, high_cred_count, low_cred_count,
                elapsed_seconds, agent_a_conf, agent_b_conf,
                n_arguments, n_counter_args,
                n_hallucinations, n_loaded_phrases, is_demo
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            topic,
            _normalize(topic),
            ts,
            date_str,
            round(bias_score,   4),
            round(weighted_bias,4),
            direction,
            len(articles),
            pipeline_result.get("stats", {}).get("total_chunks", 0),
            round(vader_score, 4),
            len(synthesis.split()),
            round(mean_cred,   4),
            high_cred,
            low_cred,
            round(elapsed_seconds, 2),
            float(agent_a.get("confidence_score", 0.0)),
            float(agent_b.get("confidence_score", 0.0)),
            len(agent_a.get("arguments",         [])),
            len(agent_b.get("counter_arguments", [])),
            len(report.get("hallucination_flags", [])),
            len(report.get("loaded_language_removed", [])),
            int(is_demo),
        ))
        run_id = cur.lastrowid

        # Insert per-article records
        for a in articles:
            cred = float(a.get(
                "credibility_score",
                a.get("credibility_info", {}).get("credibility", 0.60)
            ))
            cur.execute("""
                INSERT INTO articles
                (run_id, source, stance, credibility, word_count)
                VALUES (?,?,?,?,?)
            """, (
                run_id,
                a.get("source", "Unknown"),
                a.get("stance",  "Neutral"),
                round(cred, 3),
                len((a.get("full_text") or "").split()),
            ))

        # Insert source-bias records
        source_groups: dict = {}
        for a in articles:
            src    = a.get("source", "Unknown")
            stance = a.get("stance", "Neutral")
            s_num  = (
                -1.0 if stance == "Supportive"
                else 1.0 if stance == "Critical"
                else 0.0
            )
            cred = float(a.get(
                "credibility_score",
                a.get("credibility_info", {}).get("credibility", 0.60)
            ))
            if src not in source_groups:
                source_groups[src] = {
                    "scores":      [],
                    "credibility": cred,
                }
            source_groups[src]["scores"].append(s_num)

        for src, data in source_groups.items():
            avg_bias = float(np.mean(data["scores"]))
            cur.execute("""
                INSERT INTO source_bias
                (run_id, source, topic, bias_score, credibility, n_articles)
                VALUES (?,?,?,?,?,?)
            """, (
                run_id,
                src,
                topic,
                round(avg_bias, 4),
                round(data["credibility"], 3),
                len(data["scores"]),
            ))

        conn.commit()
        conn.close()

        print(f"[history] Saved run #{run_id}: {topic} (bias={bias_score:+.3f})")
        return run_id

    except Exception as e:
        print(f"[history] Save error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────
# QUERY FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_all_runs(limit: int = 100) -> list:
    """Return all runs ordered by most recent."""
    initialize_database()
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT * FROM runs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"[history] get_all_runs error: {e}")
        return []


def get_runs_for_topic(topic: str) -> list:
    """Return all runs for a specific topic."""
    initialize_database()
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT * FROM runs
            WHERE topic_normalized = ?
            ORDER BY timestamp ASC
        """, (_normalize(topic),))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def get_topic_list() -> list:
    """Return unique topics with run counts and latest bias."""
    initialize_database()
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT
                topic,
                topic_normalized,
                COUNT(*)        AS n_runs,
                AVG(bias_score) AS avg_bias,
                MIN(bias_score) AS min_bias,
                MAX(bias_score) AS max_bias,
                MAX(timestamp)  AS last_run,
                AVG(n_articles) AS avg_articles,
                AVG(mean_credibility) AS avg_credibility
            FROM runs
            GROUP BY topic_normalized
            ORDER BY n_runs DESC, last_run DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"[history] get_topic_list error: {e}")
        return []


def get_bias_timeline(topic: str = None) -> list:
    """
    Get bias scores over time.
    If topic is None, returns all runs.
    """
    initialize_database()
    try:
        conn = _get_connection()
        cur  = conn.cursor()

        if topic:
            cur.execute("""
                SELECT timestamp, date_only, topic, bias_score,
                       weighted_bias, vader_score,
                       n_articles, mean_credibility
                FROM runs
                WHERE topic_normalized = ?
                ORDER BY timestamp ASC
            """, (_normalize(topic),))
        else:
            cur.execute("""
                SELECT timestamp, date_only, topic, bias_score,
                       weighted_bias, vader_score,
                       n_articles, mean_credibility
                FROM runs
                ORDER BY timestamp ASC
            """)

        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def get_source_history(source_name: str) -> list:
    """Get bias scores for a specific source across all topics."""
    initialize_database()
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT sb.topic, sb.bias_score, sb.credibility,
                   sb.n_articles, r.timestamp, r.date_only
            FROM source_bias sb
            JOIN runs r ON sb.run_id = r.id
            WHERE sb.source = ?
            ORDER BY r.timestamp ASC
        """, (source_name,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def get_database_stats() -> dict:
    """Return high-level statistics about the database."""
    initialize_database()
    try:
        conn = _get_connection()
        cur  = conn.cursor()

        cur.execute("SELECT COUNT(*) AS n FROM runs")
        n_runs = cur.fetchone()["n"]

        cur.execute(
            "SELECT COUNT(DISTINCT topic_normalized) AS n FROM runs"
        )
        n_topics = cur.fetchone()["n"]

        cur.execute(
            "SELECT COUNT(DISTINCT source) AS n FROM source_bias"
        )
        n_sources = cur.fetchone()["n"]

        cur.execute(
            "SELECT AVG(bias_score) AS avg FROM runs"
        )
        row = cur.fetchone()
        avg_bias = float(row["avg"]) if row["avg"] is not None else 0.0

        cur.execute(
            "SELECT AVG(mean_credibility) AS avg FROM runs"
        )
        row = cur.fetchone()
        avg_cred = float(row["avg"]) if row["avg"] is not None else 0.60

        cur.execute(
            "SELECT COUNT(*) AS n FROM articles"
        )
        n_articles = cur.fetchone()["n"]

        conn.close()

        return {
            "n_runs":      n_runs,
            "n_topics":    n_topics,
            "n_sources":   n_sources,
            "n_articles":  n_articles,
            "avg_bias":    round(avg_bias, 4),
            "avg_cred":    round(avg_cred, 4),
            "db_path":     DB_PATH,
            "db_size_kb":  round(
                os.path.getsize(DB_PATH) / 1024, 1
            ) if os.path.exists(DB_PATH) else 0,
        }
    except Exception as e:
        print(f"[history] stats error: {e}")
        return {
            "n_runs": 0, "n_topics": 0, "n_sources": 0,
            "n_articles": 0, "avg_bias": 0.0, "avg_cred": 0.60,
        }


def get_recent_runs(n: int = 5) -> list:
    """Return the N most recent runs."""
    initialize_database()
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT id, topic, timestamp, bias_score,
                   bias_direction, n_articles, mean_credibility
            FROM runs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (n,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def delete_run(run_id: int) -> bool:
    """Delete a specific run and its associated records."""
    initialize_database()
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute("DELETE FROM articles    WHERE run_id=?", (run_id,))
        cur.execute("DELETE FROM source_bias WHERE run_id=?", (run_id,))
        cur.execute("DELETE FROM runs        WHERE id=?",     (run_id,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def clear_all_history() -> None:
    """Delete all records from all tables."""
    initialize_database()
    conn = _get_connection()
    cur  = conn.cursor()
    cur.execute("DELETE FROM articles")
    cur.execute("DELETE FROM source_bias")
    cur.execute("DELETE FROM runs")
    conn.commit()
    conn.close()
    print("[history] All history cleared.")


# ─────────────────────────────────────────────────────────────
# CHART DATA BUILDERS
# ─────────────────────────────────────────────────────────────

def build_bias_trend_data(topic: str = None) -> dict:
    """
    Build data for the bias-over-time trend chart.

    Returns:
        dict with timestamps, bias_scores, topics, etc.
    """
    rows = get_bias_timeline(topic)
    if not rows:
        return {"has_data": False}

    return {
        "has_data":      True,
        "timestamps":    [r["timestamp"][:16] for r in rows],
        "dates":         [r["date_only"]      for r in rows],
        "bias_scores":   [float(r["bias_score"])   for r in rows],
        "weighted":      [float(r["weighted_bias"]) for r in rows],
        "vader_scores":  [float(r["vader_score"])   for r in rows],
        "n_articles":    [int(r["n_articles"])       for r in rows],
        "credibilities": [float(r["mean_credibility"]) for r in rows],
        "topics":        [r["topic"]                for r in rows],
        "n_points":      len(rows),
    }


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _normalize(topic: str) -> str:
    return "_".join(topic.lower().split())[:60]