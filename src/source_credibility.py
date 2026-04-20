# src/source_credibility.py
"""
ORBITA Source Credibility Scoring
==================================
Assigns credibility and lean scores to news sources
based on established media bias research databases
(AllSides, MBFC, Reuters Institute).

Why This Matters:
    "Fox News says X" and "BBC says X" should NOT
    carry equal weight in bias computation.
    
    High-credibility source → higher weight in final score
    Low-credibility source  → lower weight, flagged in UI

Credibility Scale:  0.0 (unreliable) → 1.0 (highly reliable)
Lean Scale:        -1.0 (left/liberal) → 0.0 (center) → +1.0 (right)

Sources: AllSides, Media Bias / Fact Check (MBFC),
         Reuters Institute Digital News Report 2023
"""

from typing import Optional


# ─────────────────────────────────────────────────────────────
# SOURCE DATABASE
# ─────────────────────────────────────────────────────────────

SOURCE_DATABASE = {
    # ── Indian Sources ────────────────────────────────────────
    "The Hindu": {
        "credibility":   0.90,
        "lean":         -0.20,
        "lean_label":   "Center-Left",
        "factual":      "High",
        "country":      "India",
        "tier":          1,
        "notes":        "Established, editorially independent",
    },
    "The Indian Express": {
        "credibility":   0.87,
        "lean":         -0.10,
        "lean_label":   "Center",
        "factual":      "High",
        "country":      "India",
        "tier":          1,
        "notes":        "Strong investigative reporting",
    },
    "Times of India": {
        "credibility":   0.72,
        "lean":          0.05,
        "lean_label":   "Center",
        "factual":      "Mostly Factual",
        "country":      "India",
        "tier":          2,
        "notes":        "Largest circulation, some clickbait",
    },
    "The Times of India": {
        "credibility":   0.72,
        "lean":          0.05,
        "lean_label":   "Center",
        "factual":      "Mostly Factual",
        "country":      "India",
        "tier":          2,
        "notes":        "Largest circulation, some clickbait",
    },
    "NDTV": {
        "credibility":   0.80,
        "lean":         -0.15,
        "lean_label":   "Center-Left",
        "factual":      "High",
        "country":      "India",
        "tier":          1,
        "notes":        "Generally reliable, some ownership concerns",
    },
    "Republic TV": {
        "credibility":   0.45,
        "lean":          0.55,
        "lean_label":   "Right",
        "factual":      "Mixed",
        "country":      "India",
        "tier":          3,
        "notes":        "Sensationalist, strong political lean",
    },
    "Hindustan Times": {
        "credibility":   0.78,
        "lean":         -0.05,
        "lean_label":   "Center",
        "factual":      "Mostly Factual",
        "country":      "India",
        "tier":          2,
        "notes":        "Reliable but corporately owned",
    },
    "Livemint": {
        "credibility":   0.83,
        "lean":          0.10,
        "lean_label":   "Center-Right",
        "factual":      "High",
        "country":      "India",
        "tier":          1,
        "notes":        "Strong business journalism",
    },
    "BusinessLine": {
        "credibility":   0.84,
        "lean":          0.05,
        "lean_label":   "Center",
        "factual":      "High",
        "country":      "India",
        "tier":          1,
        "notes":        "Reliable financial reporting",
    },
    "India Today": {
        "credibility":   0.76,
        "lean":          0.0,
        "lean_label":   "Center",
        "factual":      "Mostly Factual",
        "country":      "India",
        "tier":          2,
        "notes":        "Popular, generally factual",
    },
    "The Wire": {
        "credibility":   0.74,
        "lean":         -0.35,
        "lean_label":   "Left",
        "factual":      "Mostly Factual",
        "country":      "India",
        "tier":          2,
        "notes":        "Independent, clear left perspective",
    },
    "Scroll.in": {
        "credibility":   0.75,
        "lean":         -0.30,
        "lean_label":   "Center-Left",
        "factual":      "Mostly Factual",
        "country":      "India",
        "tier":          2,
        "notes":        "Digital-first, good fact checking",
    },
    "The Print": {
        "credibility":   0.79,
        "lean":         -0.10,
        "lean_label":   "Center",
        "factual":      "High",
        "country":      "India",
        "tier":          2,
        "notes":        "Strong analysis, credible reporters",
    },

    # ── International Sources ─────────────────────────────────
    "BBC News": {
        "credibility":   0.93,
        "lean":         -0.05,
        "lean_label":   "Center",
        "factual":      "Very High",
        "country":      "UK",
        "tier":          1,
        "notes":        "International standard for accuracy",
    },
    "BBC": {
        "credibility":   0.93,
        "lean":         -0.05,
        "lean_label":   "Center",
        "factual":      "Very High",
        "country":      "UK",
        "tier":          1,
        "notes":        "International standard for accuracy",
    },
    "Reuters": {
        "credibility":   0.96,
        "lean":          0.0,
        "lean_label":   "Center",
        "factual":      "Very High",
        "country":      "UK",
        "tier":          1,
        "notes":        "Wire service, highest factual rating",
    },
    "Associated Press": {
        "credibility":   0.95,
        "lean":          0.0,
        "lean_label":   "Center",
        "factual":      "Very High",
        "country":      "USA",
        "tier":          1,
        "notes":        "Wire service, highly reliable",
    },
    "Al Jazeera English": {
        "credibility":   0.78,
        "lean":         -0.15,
        "lean_label":   "Center-Left",
        "factual":      "Mostly Factual",
        "country":      "Qatar",
        "tier":          2,
        "notes":        "State-funded but factual reporting",
    },
    "The Guardian": {
        "credibility":   0.85,
        "lean":         -0.40,
        "lean_label":   "Left-Center",
        "factual":      "High",
        "country":      "UK",
        "tier":          1,
        "notes":        "Clear left perspective, well-sourced",
    },
    "Fox News": {
        "credibility":   0.52,
        "lean":          0.65,
        "lean_label":   "Right",
        "factual":      "Mixed",
        "country":      "USA",
        "tier":          3,
        "notes":        "Strong conservative bias, mixed factual",
    },
    "CNN": {
        "credibility":   0.73,
        "lean":         -0.30,
        "lean_label":   "Center-Left",
        "factual":      "Mostly Factual",
        "country":      "USA",
        "tier":          2,
        "notes":        "Cable news, some sensationalism",
    },
    "The New York Times": {
        "credibility":   0.88,
        "lean":         -0.25,
        "lean_label":   "Center-Left",
        "factual":      "High",
        "country":      "USA",
        "tier":          1,
        "notes":        "Strong journalism, clear lean",
    },
    "The Washington Post": {
        "credibility":   0.86,
        "lean":         -0.30,
        "lean_label":   "Center-Left",
        "factual":      "High",
        "country":      "USA",
        "tier":          1,
        "notes":        "Investigative strength",
    },
    "Wired": {
        "credibility":   0.82,
        "lean":         -0.15,
        "lean_label":   "Center-Left",
        "factual":      "High",
        "country":      "USA",
        "tier":          1,
        "notes":        "Tech-focused, generally accurate",
    },
    "The Verge": {
        "credibility":   0.78,
        "lean":         -0.15,
        "lean_label":   "Center-Left",
        "factual":      "Mostly Factual",
        "country":      "USA",
        "tier":          2,
        "notes":        "Tech journalism, good sourcing",
    },
    "Business Insider": {
        "credibility":   0.70,
        "lean":         -0.10,
        "lean_label":   "Center",
        "factual":      "Mostly Factual",
        "country":      "USA",
        "tier":          2,
        "notes":        "Business focus, some clickbait",
    },
    "CNA": {
        "credibility":   0.83,
        "lean":          0.0,
        "lean_label":   "Center",
        "factual":      "High",
        "country":      "Singapore",
        "tier":          1,
        "notes":        "State media but factually accurate",
    },
    "South China Morning Post": {
        "credibility":   0.76,
        "lean":          0.10,
        "lean_label":   "Center-Right",
        "factual":      "Mostly Factual",
        "country":      "HK",
        "tier":          2,
        "notes":        "Alibaba-owned, HK perspective",
    },
    "TechRadar": {
        "credibility":   0.72,
        "lean":          0.0,
        "lean_label":   "Center",
        "factual":      "Mostly Factual",
        "country":      "UK",
        "tier":          2,
        "notes":        "Tech reviews, generally accurate",
    },
    "Yahoo Entertainment": {
        "credibility":   0.55,
        "lean":          0.0,
        "lean_label":   "Center",
        "factual":      "Mixed",
        "country":      "USA",
        "tier":          3,
        "notes":        "Aggregator, quality varies widely",
    },
    "Fortune": {
        "credibility":   0.80,
        "lean":          0.10,
        "lean_label":   "Center-Right",
        "factual":      "High",
        "country":      "USA",
        "tier":          1,
        "notes":        "Strong business journalism",
    },
    "ABC News (AU)": {
        "credibility":   0.88,
        "lean":         -0.10,
        "lean_label":   "Center",
        "factual":      "High",
        "country":      "Australia",
        "tier":          1,
        "notes":        "Australia public broadcaster",
    },
    "Sporting News": {
        "credibility":   0.72,
        "lean":          0.0,
        "lean_label":   "Center",
        "factual":      "Mostly Factual",
        "country":      "USA",
        "tier":          2,
        "notes":        "Sports-focused, factual",
    },
    "ESPN": {
        "credibility":   0.78,
        "lean":          0.0,
        "lean_label":   "Center",
        "factual":      "High",
        "country":      "USA",
        "tier":          2,
        "notes":        "Sports journalism, reliable stats",
    },
    "CNET": {
        "credibility":   0.76,
        "lean":         -0.05,
        "lean_label":   "Center",
        "factual":      "Mostly Factual",
        "country":      "USA",
        "tier":          2,
        "notes":        "Tech reviews and news",
    },
    "Android Central": {
        "credibility":   0.70,
        "lean":          0.0,
        "lean_label":   "Center",
        "factual":      "Mostly Factual",
        "country":      "USA",
        "tier":          2,
        "notes":        "Android/tech niche",
    },
    "Slate Magazine": {
        "credibility":   0.72,
        "lean":         -0.40,
        "lean_label":   "Left",
        "factual":      "Mostly Factual",
        "country":      "USA",
        "tier":          2,
        "notes":        "Opinion-heavy, clear left lean",
    },
    "TheWrap": {
        "credibility":   0.70,
        "lean":         -0.10,
        "lean_label":   "Center",
        "factual":      "Mostly Factual",
        "country":      "USA",
        "tier":          2,
        "notes":        "Entertainment journalism",
    },
    "Deadline": {
        "credibility":   0.73,
        "lean":          0.0,
        "lean_label":   "Center",
        "factual":      "Mostly Factual",
        "country":      "USA",
        "tier":          2,
        "notes":        "Entertainment industry news",
    },
    "MIT Technology Review": {
        "credibility":   0.91,
        "lean":         -0.10,
        "lean_label":   "Center",
        "factual":      "Very High",
        "country":      "USA",
        "tier":          1,
        "notes":        "Academic-backed, highly accurate",
    },
}

# Default for unknown sources
_DEFAULT = {
    "credibility": 0.60,
    "lean":        0.0,
    "lean_label":  "Unknown",
    "factual":     "Unknown",
    "country":     "Unknown",
    "tier":        3,
    "notes":       "Not in database — default score applied",
}


# ─────────────────────────────────────────────────────────────
# LOOKUP FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_source_info(source_name: str) -> dict:
    """
    Get full credibility info for a news source.

    Args:
        source_name: source name string (partial match supported)

    Returns:
        dict with credibility, lean, lean_label, factual, tier, notes
    """
    if not source_name:
        return _DEFAULT.copy()

    # Exact match first
    if source_name in SOURCE_DATABASE:
        info = SOURCE_DATABASE[source_name].copy()
        info["source"] = source_name
        info["found"]  = True
        return info

    # Case-insensitive match
    lower = source_name.lower()
    for key, val in SOURCE_DATABASE.items():
        if key.lower() == lower:
            info = val.copy()
            info["source"] = source_name
            info["found"]  = True
            return info

    # Partial match (source_name is substring of key or vice versa)
    for key, val in SOURCE_DATABASE.items():
        if lower in key.lower() or key.lower() in lower:
            info = val.copy()
            info["source"] = source_name
            info["found"]  = True
            return info

    # Not found — return default
    info = _DEFAULT.copy()
    info["source"] = source_name
    info["found"]  = False
    return info


def get_credibility_score(source_name: str) -> float:
    """Return just the credibility score (0.0 to 1.0)."""
    return get_source_info(source_name).get("credibility", 0.60)


def get_lean_score(source_name: str) -> float:
    """Return just the lean score (-1.0 to +1.0)."""
    return get_source_info(source_name).get("lean", 0.0)


def score_articles(articles: list) -> list:
    """
    Add credibility info to each article dict.

    Args:
        articles: list of article dicts with 'source' field

    Returns:
        Same list with 'credibility_info' added to each article
    """
    for article in articles:
        source = article.get("source", "")
        info   = get_source_info(source)
        article["credibility_info"] = info
        article["credibility_score"] = info.get("credibility", 0.60)
        article["lean_score"]        = info.get("lean", 0.0)

    return articles


def compute_credibility_weighted_bias(
    articles:    list,
    base_bias:   float,
) -> dict:
    """
    Compute a credibility-weighted bias score.

    High-credibility sources influence the final
    bias score more than low-credibility ones.

    Args:
        articles:  list of articles with credibility_info
        base_bias: original bias score from Agent C

    Returns:
        dict with weighted_bias_score and breakdown
    """
    if not articles:
        return {
            "weighted_bias_score": base_bias,
            "adjustment":          0.0,
            "mean_credibility":    0.60,
            "high_cred_count":     0,
            "low_cred_count":      0,
            "credibility_breakdown": [],
        }

    weights  = []
    scores   = []
    breakdown = []

    for article in articles:
        info   = article.get("credibility_info", {})
        cred   = float(info.get("credibility", 0.60))
        lean   = float(info.get("lean", 0.0))
        stance = article.get("stance", "Neutral")

        # Map stance to numeric
        stance_num = (
            -1.0 if stance == "Supportive"
            else 1.0 if stance == "Critical"
            else 0.0
        )

        weights.append(cred)
        scores.append(stance_num * cred)

        breakdown.append({
            "source":      article.get("source", "?"),
            "credibility": round(cred, 3),
            "lean":        round(lean, 3),
            "lean_label":  info.get("lean_label", "?"),
            "tier":        info.get("tier", 3),
            "stance":      stance,
        })

    total_weight = sum(weights)
    if total_weight > 0:
        weighted_score = sum(scores) / total_weight
    else:
        weighted_score = base_bias

    weighted_score = float(max(-1.0, min(1.0, weighted_score)))
    adjustment     = round(weighted_score - base_bias, 4)
    mean_cred      = sum(weights) / len(weights) if weights else 0.60
    high_cred      = sum(1 for w in weights if w >= 0.80)
    low_cred       = sum(1 for w in weights if w  < 0.60)

    return {
        "weighted_bias_score":    round(weighted_score, 4),
        "original_bias_score":    round(base_bias,      4),
        "adjustment":             round(adjustment,     4),
        "mean_credibility":       round(mean_cred,      4),
        "high_cred_count":        high_cred,
        "low_cred_count":         low_cred,
        "total_articles":         len(articles),
        "credibility_breakdown":  breakdown,
    }


def get_credibility_badge_html(
    source_name: str,
    compact:     bool = False,
) -> str:
    """
    Return HTML for a credibility badge to show in the UI.

    Args:
        source_name: source name
        compact:     True for small inline badge

    Returns:
        HTML string
    """
    info       = get_source_info(source_name)
    cred       = info.get("credibility", 0.60)
    lean_label = info.get("lean_label",  "Unknown")
    tier       = info.get("tier",        3)
    found      = info.get("found",       False)

    # Color by credibility
    if cred >= 0.85:
        cred_color = "#3ec97e"
        cred_label = "High"
    elif cred >= 0.70:
        cred_color = "#c9a84c"
        cred_label = "Med"
    else:
        cred_color = "#e05252"
        cred_label = "Low"

    # Tier stars
    stars = "★" * (4 - tier) + "☆" * (tier - 1)

    if compact:
        return (
            f'<span style="'
            f'background:rgba(20,28,46,0.9);'
            f'border:1px solid {cred_color}55;'
            f'border-radius:4px;'
            f'padding:1px 5px;'
            f'font-family:\'DM Mono\',monospace;'
            f'font-size:0.6rem;'
            f'color:{cred_color};'
            f'white-space:nowrap">'
            f'{cred_label} · {cred:.2f}'
            f'</span>'
        )

    return (
        f'<div style="'
        f'display:inline-flex;align-items:center;'
        f'gap:0.4rem;'
        f'background:rgba(20,28,46,0.9);'
        f'border:1px solid {cred_color}44;'
        f'border-radius:6px;'
        f'padding:2px 8px;'
        f'font-family:\'DM Mono\',monospace;'
        f'font-size:0.65rem">'
        f'<span style="color:{cred_color};font-weight:600">'
        f'{cred_label} {cred:.2f}</span>'
        f'<span style="color:#5c6b82">·</span>'
        f'<span style="color:#9ba8bb">{lean_label}</span>'
        f'</div>'
    )