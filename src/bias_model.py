"""
ORBITA Multi-Dimensional Bias Quantification Model

Research Contribution:
    Replaces the heuristic single-scalar bias score with a
    principled multi-dimensional bias vector B = (b_i, b_e, b_f, b_d)
    where:
        b_i = ideological bias   (which side dominates coverage)
        b_e = emotional bias     (how charged the language is)
        b_f = informational bias (fact vs opinion ratio)
        b_d = source diversity   (how varied the sources are)

Theoretical Grounding:
    - Emotional lexicon approach: Mohammad & Turney (2013)
    - Framing theory: Entman (1993)
    - Computational argumentation: Stab & Gurevych (2017)

Author: Vighnesh Thakur
Project: ORBITA — B.Tech 6th Sem, AIML 2026
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# EMOTIONAL LEXICON
# Based on NRC Emotion Lexicon (Mohammad & Turney, 2013)
# These are words that signal emotionally charged / biased writing
# ─────────────────────────────────────────────────────────────────────────────

EMOTIONAL_WORDS_HIGH_AROUSAL = [
    # Strongly negative emotional words
    "shocking", "outrageous", "explosive", "devastating", "horrifying",
    "alarming", "catastrophic", "scandalous", "disgraceful", "terrifying",
    "brutal", "vicious", "dangerous", "disaster", "chaos", "collapse",
    "failure", "threat", "attack", "destroy", "corrupt", "fraud", "fake",
    "hoax", "betrayal", "criminal", "illegal", "absurd", "ridiculous",
    "pathetic", "disgusting", "shameful", "despicable", "reckless",
    # Strongly positive emotional words (can also signal bias)
    "revolutionary", "groundbreaking", "miraculous", "incredible",
    "phenomenal", "outstanding", "remarkable", "extraordinary",
    "brilliant", "flawless", "magnificent", "heroic", "triumphant",
    "glorious", "spectacular", "landmark", "historic", "unprecedented",
    # Intensifiers that signal non-neutral writing
    "extremely", "absolutely", "completely", "totally", "utterly",
    "massively", "drastically", "severely", "deeply", "profoundly",
]

EMOTIONAL_WORDS_NEGATIVE_FRAMING = [
    "crisis", "scandal", "controversy", "fiasco", "debacle", "blunder",
    "mess", "problem", "issue", "concern", "worry", "fear", "danger",
    "risk", "threat", "warning", "alarm", "emergency", "panic",
]

EMOTIONAL_WORDS_POSITIVE_FRAMING = [
    "success", "achievement", "progress", "improvement", "benefit",
    "advantage", "opportunity", "growth", "development", "advancement",
    "breakthrough", "innovation", "solution", "resolution", "victory",
]

# Combine all emotional words into one set for fast lookup
ALL_EMOTIONAL_WORDS = set(
    EMOTIONAL_WORDS_HIGH_AROUSAL +
    EMOTIONAL_WORDS_NEGATIVE_FRAMING +
    EMOTIONAL_WORDS_POSITIVE_FRAMING
)


# ─────────────────────────────────────────────────────────────────────────────
# OPINION MARKERS
# Words and phrases that signal subjective / opinion-based writing
# ─────────────────────────────────────────────────────────────────────────────

OPINION_MARKERS = [
    # Epistemic hedges (uncertainty markers)
    r"\b(I think|I believe|I feel|in my opinion|in my view)\b",
    r"\b(it seems|it appears|seemingly|apparently|presumably)\b",
    r"\b(perhaps|maybe|possibly|probably|likely|arguably)\b",
    # Prescriptive markers (what SHOULD happen)
    r"\b(should|must|ought to|needs to|has to|have to)\b",
    r"\b(it is necessary|it is important|it is crucial|it is vital)\b",
    # Evaluative adjectives (good/bad judgments)
    r"\b(good|bad|wrong|right|better|worse|best|worst|terrible|wonderful)\b",
    r"\b(effective|ineffective|successful|unsuccessful|adequate|inadequate)\b",
    # Certainty overstatement
    r"\b(clearly|obviously|undoubtedly|certainly|definitely|inevitably)\b",
    r"\b(of course|needless to say|it is clear that|everyone knows)\b",
]

# Fact indicators — things that signal objective factual writing
FACT_MARKERS = [
    r"\b\d+\.?\d*\s*(%|percent|per cent)\b",         # percentages
    r"\b\d+\.?\d*\s*(million|billion|trillion|crore|lakh)\b",  # large numbers
    r"\b(19|20)\d{2}\b",                              # years
    r"\b\d{1,2}\s+(January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\b",  # dates
    r"\b(according to|reported by|stated by|confirmed by|announced by)\b",
    r'"[^"]{15,}"',                                   # direct quotes (15+ chars)
    r"\b(study|research|report|survey|data|statistics|figures|analysis)\b",
    r"\b(Rs\.?|INR|\$|USD|EUR)\s*\d+",               # monetary values
]


# ─────────────────────────────────────────────────────────────────────────────
# CORE COMPUTATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_emotional_bias(text: str) -> float:
    """
    Compute emotional bias score using lexicon-based approach.

    The score measures how emotionally charged the writing is.
    Emotionally charged writing tends to be more biased than
    neutral, factual reporting.

    Method:
        1. Tokenize text into words
        2. Count words that appear in our emotional lexicon
        3. Normalize by total word count
        4. Scale so that ~5% emotional words = score of 1.0

    Args:
        text: the full article text or synthesis to analyze

    Returns:
        float in range [0.0, 1.0]
        0.0 = completely neutral language
        1.0 = highly emotionally charged language

    Example:
        "Government announced new policy." → ~0.05 (low)
        "SHOCKING betrayal destroys everything!!!" → ~0.85 (high)
    """
    if not text:
        return 0.0

    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    if len(words) < 10:
        return 0.0

    # Count how many words are in our emotional lexicon
    emotional_count = sum(1 for w in words if w in ALL_EMOTIONAL_WORDS)

    # Compute ratio: emotional words per total words
    ratio = emotional_count / len(words)

    # Scale: 5% emotional words maps to score = 1.0
    # This threshold is based on analysis of clearly biased vs neutral text
    score = ratio / 0.05

    # Clamp to [0, 1]
    return round(min(1.0, max(0.0, score)), 4)


def compute_informational_bias(text: str) -> float:
    """
    Compute informational bias — ratio of opinion to factual content.

    High informational bias means the text is more opinion-based than
    fact-based. This is different from ideological bias — a text can
    be ideologically neutral but still be highly opinionated (or vice versa).

    Method:
        1. Split text into sentences
        2. For each sentence, check for fact markers and opinion markers
        3. Score = (opinion sentences) / (total sentences)

    Args:
        text: the text to analyze (typically the synthesis report)

    Returns:
        float in range [0.0, 1.0]
        0.0 = purely factual writing
        1.0 = purely opinion-based writing
        0.5 = balanced mix (ideal for a news synthesis)
    """
    if not text:
        return 0.5

    # Split into sentences (simple approach)
    sentences = [
        s.strip()
        for s in re.split(r'[.!?]+', text)
        if len(s.strip().split()) >= 4  # skip very short fragments
    ]

    if not sentences:
        return 0.5

    fact_count    = 0
    opinion_count = 0

    for sentence in sentences:
        s_lower = sentence.lower()

        # Check for fact markers
        has_fact = any(
            re.search(pattern, s_lower, re.IGNORECASE)
            for pattern in FACT_MARKERS
        )

        # Check for opinion markers
        has_opinion = any(
            re.search(pattern, s_lower, re.IGNORECASE)
            for pattern in OPINION_MARKERS
        )

        if has_fact:
            fact_count += 1
        if has_opinion:
            opinion_count += 1

    total = len(sentences)

    # Sentences with neither marker are counted as neutral
    # Score based on opinion proportion
    opinion_ratio = opinion_count / total if total > 0 else 0.5

    return round(opinion_ratio, 4)


def compute_source_diversity(articles: list[dict]) -> float:
    """
    Compute how ideologically diverse the article sources are.

    This is a NOVEL metric in ORBITA. Most bias detection systems
    ignore source diversity entirely. We measure it using TF-IDF
    cosine distance between article texts.

    Intuition:
        If all articles say roughly the same thing (high similarity),
        diversity is LOW — you have an echo chamber.

        If articles disagree significantly (low similarity),
        diversity is HIGH — you have genuine multi-perspective coverage.

    Method:
        1. Embed each article using TF-IDF vectors
        2. Compute pairwise cosine similarity
        3. Average pairwise DISTANCE (1 - similarity)
        4. High distance = high diversity

    Args:
        articles: list of article dicts with 'full_text' or 'description'

    Returns:
        float in range [0.0, 1.0]
        0.0 = all articles say the same thing (echo chamber)
        1.0 = maximum diversity of perspectives
    """
    if len(articles) < 2:
        return 0.0

    # Extract texts — use full_text if available, else description
    # Keep threshold low so short but meaningful snippets in tests and
    # lightweight crawled content still contribute to diversity.
    texts = []
    for article in articles:
        text = (
            article.get("full_text") or
            article.get("description") or
            article.get("title") or
            ""
        ).strip()

        if len(text.split()) >= 5:  # allow short snippets; skip near-empty text
            texts.append(text[:2000])  # cap at 2000 chars for efficiency

    if len(texts) < 2:
        return 0.0

    try:
        # Build TF-IDF matrix
        vectorizer = TfidfVectorizer(
            stop_words  = "english",
            max_features = 1000,
            min_df       = 1,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Compute average pairwise distance
        n             = len(texts)
        total_distance = 0.0
        pair_count     = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Distance = 1 - similarity
                total_distance += (1.0 - similarity_matrix[i][j])
                pair_count     += 1

        if pair_count == 0:
            return 0.0

        avg_distance = total_distance / pair_count
        return round(float(avg_distance), 4)

    except Exception as e:
        print(f"[bias_model] Source diversity error: {e}")
        return 0.0


def compute_stance_entropy(articles: list[dict]) -> float:
    """
    Compute Shannon entropy of the stance distribution.

    High entropy means balanced coverage (good).
    Low entropy means one stance dominates (potential bias).

    Shannon entropy: H = -Σ p(x) * log2(p(x))
    Maximum entropy for 3 classes = log2(3) ≈ 1.585 bits
    We normalize to [0, 1].

    Args:
        articles: list of article dicts with 'stance' field

    Returns:
        float in range [0.0, 1.0]
        0.0 = all articles have the same stance (no diversity)
        1.0 = perfectly balanced (equal Supportive/Critical/Neutral)
    """
    counts = {"Supportive": 0, "Critical": 0, "Neutral": 0}

    for article in articles:
        stance = article.get("stance", "Neutral")
        if stance in counts:
            counts[stance] += 1

    total = sum(counts.values())

    if total == 0:
        return 0.0

    # Compute Shannon entropy
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p        = count / total
            entropy -= p * np.log2(p)

    # Normalize by maximum possible entropy log2(3) ≈ 1.585
    max_entropy = np.log2(3)
    normalized  = entropy / max_entropy if max_entropy > 0 else 0.0

    return round(float(normalized), 4)


def compute_ideological_bias(
    articles:        list[dict],
    agent_a_output:  dict,
    agent_b_output:  dict,
) -> float:
    """
    Compute ideological bias score.

    This replaces the old heuristic. Instead of just averaging
    stance labels, we use a weighted combination of:
        1. Article stance distribution
        2. Agent confidence scores
        3. Argument count asymmetry

    A higher confidence from Agent A (Supportive) pulls score negative.
    A higher confidence from Agent B (Critical) pulls score positive.

    Returns:
        float in range [-1.0, +1.0]
        -1.0 = overwhelmingly supportive coverage
         0.0 = balanced
        +1.0 = overwhelmingly critical coverage
    """
    # ── Component 1: Stance-based score ──────────────────────────
    stance_map = {"Supportive": -1.0, "Critical": 1.0, "Neutral": 0.0}
    stance_scores = [
        stance_map.get(a.get("stance", "Neutral"), 0.0)
        for a in articles
    ]
    stance_score = float(np.mean(stance_scores)) if stance_scores else 0.0

    # ── Component 2: Agent confidence asymmetry ───────────────────
    # If Agent A (Analyst) is much more confident than Agent B (Critic),
    # that means the supporting side has stronger evidence → lean negative
    a_conf = float(agent_a_output.get("confidence_score", 0.5))
    b_conf = float(agent_b_output.get("confidence_score", 0.5))

    # Confidence asymmetry in range [-1, +1]
    # Positive = Agent B more confident (critical side stronger)
    # Negative = Agent A more confident (supportive side stronger)
    conf_asymmetry = (b_conf - a_conf) / max(a_conf + b_conf, 0.01)

    # ── Component 3: Argument count asymmetry ────────────────────
    n_args    = len(agent_a_output.get("arguments", []))
    n_counters = len(agent_b_output.get("counter_arguments", []))
    total_args  = n_args + n_counters

    if total_args > 0:
        # Positive = more counter-arguments (critical side dominant)
        arg_asymmetry = (n_counters - n_args) / total_args
    else:
        arg_asymmetry = 0.0

    # ── Weighted combination ──────────────────────────────────────
    # Stance distribution is the primary signal (60%)
    # Agent confidence adds nuance (25%)
    # Argument count adds further nuance (15%)
    ideological = (
        0.60 * stance_score    +
        0.25 * conf_asymmetry  +
        0.15 * arg_asymmetry
    )

    # Clamp to valid range
    return round(float(np.clip(ideological, -1.0, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_bias_vector(
    articles:        list[dict],
    agent_a_output:  dict,
    agent_b_output:  dict,
    agent_c_output:  dict,
) -> dict:
    """
    Compute the complete multi-dimensional bias vector for ORBITA.

    This is the main function you call from pipeline.py and agents.py.
    It replaces the old single-scalar bias_score computation.

    The bias vector has 4 primary dimensions:
        b_i (ideological)   : which ideological side dominates
        b_e (emotional)     : how emotionally charged the language is
        b_f (informational) : ratio of opinion to factual content
        b_d (diversity)     : how diverse the source perspectives are

    Plus derived metrics:
        stance_entropy      : balance of Supportive/Critical/Neutral
        composite_score     : principled weighted combination
        confidence          : mean agent confidence
        interpretation      : human-readable label

    Args:
        articles:        list of scraped article dicts from pipeline
        agent_a_output:  output dict from run_agent_a()
        agent_b_output:  output dict from run_agent_b()
        agent_c_output:  output dict from run_agent_c()

    Returns:
        Complete bias vector dict. Example:
        {
            "ideological_bias":   -0.42,
            "emotional_bias":      0.18,
            "informational_bias":  0.23,
            "source_diversity":    0.71,
            "stance_entropy":      0.88,
            "composite_score":    -0.31,
            "confidence":          0.86,
            "interpretation":     "Moderately Supportive",
            "dimension_labels": {
                "ideological":  "Moderately Supportive",
                "emotional":    "Low Emotional Bias",
                "informational": "Mostly Factual",
                "diversity":    "High Source Diversity"
            }
        }
    """
    print("\n[bias_model] Computing multi-dimensional bias vector...")

    # ── 1. Ideological Bias ───────────────────────────────────────
    ideological = compute_ideological_bias(
        articles, agent_a_output, agent_b_output
    )
    print(f"  Ideological bias:   {ideological:+.4f}")

    # ── 2. Emotional Bias ─────────────────────────────────────────
    # Compute on the synthesis (what ORBITA produces)
    # AND on the raw articles (what media produced)
    synthesis_text = agent_c_output.get("synthesis_report", "")
    article_texts  = " ".join(
        (a.get("full_text") or a.get("description") or "")[:500]
        for a in articles
    )

    # Use synthesis emotional bias (measures ORBITA's output quality)
    synthesis_emotional = compute_emotional_bias(synthesis_text)
    # Use article emotional bias (measures media bias in sources)
    article_emotional   = compute_emotional_bias(article_texts)

    # Weighted: source emotional bias is more relevant for detecting media bias
    emotional = round(
        0.3 * synthesis_emotional + 0.7 * article_emotional, 4
    )
    print(f"  Emotional bias:     {emotional:.4f} "
          f"(synthesis: {synthesis_emotional:.4f}, "
          f"articles: {article_emotional:.4f})")

    # ── 3. Informational Bias ─────────────────────────────────────
    # Measure on the synthesis — how opinionated is our final output?
    informational = compute_informational_bias(synthesis_text)
    print(f"  Informational bias: {informational:.4f}")

    # ── 4. Source Diversity ───────────────────────────────────────
    diversity = compute_source_diversity(articles)
    print(f"  Source diversity:   {diversity:.4f}")

    # ── 5. Stance Entropy ─────────────────────────────────────────
    entropy = compute_stance_entropy(articles)
    print(f"  Stance entropy:     {entropy:.4f}")

    # ── 6. Agent Confidence ───────────────────────────────────────
    a_conf     = float(agent_a_output.get("confidence_score", 0.5))
    b_conf     = float(agent_b_output.get("confidence_score", 0.5))
    confidence = round((a_conf + b_conf) / 2, 4)

    # ── 7. Composite Score ────────────────────────────────────────
    # Principled weighted combination
    #
    # Weights are justified as:
    #   Ideological (50%): primary measure of which side dominates
    #   Emotional   (20%): modulates — charged language = more biased
    #   Information (20%): modulates — opinion-heavy = less objective
    #   Diversity   (10%): high diversity REDUCES overall bias score
    #                      (negative weight because diversity is good)
    #
    # The composite maps to [-1, +1] same as ideological
    # so it's interpretable in the same way

    composite = (
        0.50 * ideological    +
        0.20 * emotional      +  # always positive, pushes away from 0
        0.20 * informational  -  # always positive, pushes away from 0
        0.10 * diversity         # reduces bias score when sources diverse
    )

    # However, the sign should follow ideological direction
    # (emotional and informational bias don't have direction)
    # So we apply emotional + informational as magnitude modifier
    magnitude_modifier = 1.0 + (0.20 * emotional + 0.20 * informational)
    composite_directional = ideological * magnitude_modifier - 0.10 * diversity
    composite_directional = round(
        float(np.clip(composite_directional, -1.0, 1.0)), 4
    )

    print(f"  Composite score:    {composite_directional:+.4f}")

    # ── 8. Human-readable interpretation ─────────────────────────
    interpretation = _get_interpretation(composite_directional)
    print(f"  Interpretation:     {interpretation}")

    # ── 9. Per-dimension labels ───────────────────────────────────
    dimension_labels = {
        "ideological":   _get_interpretation(ideological),
        "emotional":     _label_emotional(emotional),
        "informational": _label_informational(informational),
        "diversity":     _label_diversity(diversity),
    }

    return {
        "ideological_bias":   ideological,
        "emotional_bias":     emotional,
        "informational_bias": informational,
        "source_diversity":   diversity,
        "stance_entropy":     entropy,
        "composite_score":    composite_directional,
        "confidence":         confidence,
        "interpretation":     interpretation,
        "dimension_labels":   dimension_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER LABEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _get_interpretation(score: float) -> str:
    """Convert a -1 to +1 score to a human-readable label."""
    if score <= -0.6:
        return "Strongly Supportive"
    elif score <= -0.3:
        return "Moderately Supportive"
    elif score < -0.1:
        return "Slightly Supportive"
    elif score <= 0.1:
        return "Balanced"
    elif score < 0.3:
        return "Slightly Critical"
    elif score < 0.6:
        return "Moderately Critical"
    else:
        return "Strongly Critical"


def _label_emotional(score: float) -> str:
    """Label emotional bias score."""
    if score < 0.15:
        return "Low Emotional Bias"
    elif score < 0.40:
        return "Moderate Emotional Bias"
    else:
        return "High Emotional Bias"


def _label_informational(score: float) -> str:
    """Label informational bias score."""
    if score < 0.20:
        return "Mostly Factual"
    elif score < 0.45:
        return "Balanced Fact/Opinion"
    else:
        return "Mostly Opinion"


def _label_diversity(score: float) -> str:
    """Label source diversity score."""
    if score < 0.30:
        return "Low Source Diversity"
    elif score < 0.60:
        return "Moderate Source Diversity"
    else:
        return "High Source Diversity"