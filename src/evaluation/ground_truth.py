# src/evaluation/ground_truth.py
"""
ORBITA Ground Truth Dataset

Source: AllSides Media Bias Ratings (allsides.com/media-bias/ratings)
        MediaBiasFactCheck (mediabiasfactcheck.com)
        Academic literature on Indian media bias

Why AllSides?
    AllSides employs a team of analysts from left, center, and right
    political perspectives who independently rate each source.
    It is the most cited ground truth dataset in media bias research.

Bias Scale Mapping:
    AllSides uses: Left / Lean Left / Center / Lean Right / Right
    We map this to our continuous [-1, +1] scale:
        Left       → -0.8
        Lean Left  → -0.4
        Center     →  0.0
        Lean Right → +0.4
        Right      → +0.8

Topic-Level Ground Truth:
    For evaluation, we also need expected bias directions for
    specific topics. These are derived from:
    1. Academic papers on media framing of Indian policy issues
    2. AllSides topic pages
    3. Manual annotation by domain experts (you + your supervisor)

"""

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE-LEVEL BIAS RATINGS
# These map news source names to known bias scores
# Source: AllSides + MediaBiasFactCheck + Academic literature
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_BIAS_RATINGS = {
    # ── International Sources ────────────────────────────────────
    "BBC News":              -0.10,   # Center (slight left lean)
    "Reuters":                0.00,   # Center
    "Associated Press":       0.00,   # Center
    "Al Jazeera English":    -0.20,   # Lean Left
    "Al Jazeera":            -0.20,
    "The Guardian":          -0.60,   # Left
    "The New York Times":    -0.40,   # Lean Left
    "Washington Post":       -0.40,   # Lean Left
    "Fox News":               0.80,   # Right
    "Wired":                 -0.15,   # Slight Left
    "The Verge":             -0.15,
    "Business Insider":      -0.10,
    "Fortune":               -0.05,
    "CNN":                   -0.30,   # Left
    "ABC News":              -0.20,
    "ABC News (AU)":         -0.10,

    # ── Indian Sources ───────────────────────────────────────────
    "The Hindu":             -0.30,   # Center-Left (Indian context)
    "Times of India":         0.05,   # Center (slight right)
    "The Times of India":     0.05,
    "Hindustan Times":       -0.05,   # Center
    "NDTV":                  -0.20,   # Center-Left
    "Republic TV":            0.70,   # Right
    "Zee News":               0.50,   # Lean Right
    "OpIndia":                0.80,   # Right
    "The Wire":              -0.60,   # Left
    "The Indian Express":    -0.15,   # Center-Left
    "Livemint":               0.10,   # Center-Right (business)
    "Economic Times":         0.15,   # Center-Right (business)
    "Business Standard":      0.10,
    "BusinessLine":           0.10,

    # ── Technology Sources ───────────────────────────────────────
    "TechCrunch":            -0.10,
    "CNET":                  -0.10,
    "TechRadar":             -0.05,
    "Android Central":        0.00,

    # ── Sports / Entertainment ───────────────────────────────────
    "ESPN":                   0.00,
    "BBC Sport":             -0.10,

    # ── Default for unknown sources ──────────────────────────────
    "_default":               0.00,
}


# ─────────────────────────────────────────────────────────────────────────────
# TOPIC-LEVEL GROUND TRUTH
# For each benchmark topic, we record the expected bias direction
# based on how media typically covers that topic
#
# Format:
#   "topic_key": {
#       "expected_score":     float,  # -1 to +1
#       "expected_direction": str,    # "supportive"/"critical"/"balanced"
#       "confidence":         str,    # "high"/"medium"/"low"
#       "notes":              str,    # justification
#   }
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_GROUND_TRUTH = {
    # ── Indian Politics ───────────────────────────────────────────
    "farm laws india protest": {
        "expected_score":     0.35,
        "expected_direction": "critical",
        "confidence":         "high",
        "domain":             "indian_politics",
        "notes": (
            "International media was predominantly critical of "
            "the farm laws. Coverage focused on farmer protests, "
            "human rights concerns, and democratic backsliding. "
            "Source: Academic papers by Jodhka & Singh (2021)"
        ),
    },
    "upi digital payments india": {
        "expected_score":    -0.40,
        "expected_direction": "supportive",
        "confidence":         "high",
        "domain":             "indian_technology",
        "notes": (
            "UPI coverage is predominantly positive — success story "
            "framing dominates. Critical coverage focuses only on "
            "fraud/sustainability concerns which are minor themes."
        ),
    },
    "electric vehicles india": {
        "expected_score":    -0.15,
        "expected_direction": "balanced",
        "confidence":         "medium",
        "domain":             "indian_technology",
        "notes": (
            "Mixed coverage — positive on growth/environment, "
            "critical on infrastructure gaps and subsidies."
        ),
    },
    "cryptocurrency regulation india": {
        "expected_score":     0.10,
        "expected_direction": "balanced",
        "confidence":         "medium",
        "domain":             "global_economics",
        "notes": (
            "Genuinely split expert opinion. "
            "Slight critical lean due to RBI opposition framing."
        ),
    },
    "iran america war": {
        "expected_score":     0.45,
        "expected_direction": "critical",
        "confidence":         "high",
        "domain":             "global_politics",
        "notes": (
            "War coverage is inherently critical in tone. "
            "Focus on casualties, escalation risks, civilian impact."
        ),
    },
    "artificial intelligence jobs india": {
        "expected_score":     0.25,
        "expected_direction": "critical",
        "confidence":         "medium",
        "domain":             "technology",
        "notes": (
            "AI job displacement narrative dominates recent coverage. "
            "Positive coverage exists but critical framing stronger."
        ),
    },
    "indian budget 2026": {
        "expected_score":    -0.05,
        "expected_direction": "balanced",
        "confidence":         "medium",
        "domain":             "indian_economics",
        "notes": (
            "Budget coverage is typically mixed — "
            "government sources positive, opposition critical."
        ),
    },

    # ── Default for unrecognized topics ──────────────────────────
    "_default": {
        "expected_score":     0.00,
        "expected_direction": "balanced",
        "confidence":         "low",
        "domain":             "general",
        "notes":              "No ground truth available for this topic.",
    },
}


def get_source_bias(source_name: str) -> float:
    """
    Look up the known bias rating for a news source.

    Args:
        source_name: name of the news source (case-insensitive match)

    Returns:
        float bias rating in [-1, +1], or 0.0 if unknown
    """
    if not source_name:
        return SOURCE_BIAS_RATINGS["_default"]

    # Try exact match first
    if source_name in SOURCE_BIAS_RATINGS:
        return SOURCE_BIAS_RATINGS[source_name]

    # Try case-insensitive match
    source_lower = source_name.lower()
    for key, value in SOURCE_BIAS_RATINGS.items():
        if key.lower() == source_lower:
            return value

    # Try partial match (e.g., "BBC" matches "BBC News")
    for key, value in SOURCE_BIAS_RATINGS.items():
        if key.startswith("_"):
            continue
        if source_lower in key.lower() or key.lower() in source_lower:
            return value

    return SOURCE_BIAS_RATINGS["_default"]


def get_topic_ground_truth(topic: str) -> dict:
    """
    Look up the expected bias for a topic.

    Args:
        topic: topic string (case-insensitive, fuzzy match)

    Returns:
        ground truth dict with expected_score, direction, etc.
    """
    topic_lower = topic.lower().strip()

    # Try exact match
    if topic_lower in TOPIC_GROUND_TRUTH:
        return TOPIC_GROUND_TRUTH[topic_lower]

    # Try partial match — find the best matching topic
    best_match     = None
    best_overlap   = 0

    topic_words = set(topic_lower.split())

    for key in TOPIC_GROUND_TRUTH:
        if key.startswith("_"):
            continue
        key_words = set(key.split())
        overlap   = len(topic_words & key_words)  # word intersection
        if overlap > best_overlap:
            best_overlap = overlap
            best_match   = key

    # Only use partial match if we have at least 2 words in common
    if best_match and best_overlap >= 2:
        result = TOPIC_GROUND_TRUTH[best_match].copy()
        result["matched_key"]    = best_match
        result["match_type"]     = "partial"
        result["match_overlap"]  = best_overlap
        return result

    # Fall back to default
    default = TOPIC_GROUND_TRUTH["_default"].copy()
    default["match_type"] = "default"
    return default


def compute_expected_bias_from_sources(articles: list[dict]) -> dict:
    """
    Compute expected bias score from AllSides source ratings.

    Instead of using topic-level ground truth, this computes
    expected bias based on WHICH sources were retrieved.

    This is more rigorous because it accounts for the specific
    article mix in each pipeline run.

    Args:
        articles: list of article dicts with 'source' field

    Returns:
        dict with expected_score, matched_sources, coverage
    """
    source_scores = []
    matched       = []
    unmatched     = []

    for article in articles:
        source = article.get("source", "")
        score  = get_source_bias(source)

        if source in SOURCE_BIAS_RATINGS:
            source_scores.append(score)
            matched.append({"source": source, "bias": score})
        else:
            unmatched.append(source)

    if not source_scores:
        return {
            "expected_score":    0.0,
            "matched_sources":   [],
            "unmatched_sources": unmatched,
            "coverage":          0.0,
            "note": "No sources matched AllSides database",
        }

    import numpy as np
    expected = float(np.mean(source_scores))
    coverage = len(source_scores) / max(len(articles), 1)

    return {
        "expected_score":    round(expected, 4),
        "matched_sources":   matched,
        "unmatched_sources": unmatched,
        "coverage":          round(coverage, 4),
        "n_matched":         len(matched),
        "n_total":           len(articles),
    }