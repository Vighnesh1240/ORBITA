# src/comparison_engine.py
"""
ORBITA Comparison Engine
========================
Runs ORBITA pipeline on TWO topics simultaneously
and produces a side-by-side comparison report.

Research Value:
    Comparing "India AI Policy" vs "USA AI Policy" reveals:
    - Which gets more positive media coverage
    - Which has more diverse source representation
    - Bias differential between topics
    - Common sources that cover both

This is the feature that makes ORBITA a research PLATFORM
not just a single-shot analysis tool.
"""

import time
from typing import Optional


def run_comparison(
    topic_a:         str,
    topic_b:         str,
    use_demo_cache:  bool = False,
) -> dict:
    """
    Run full ORBITA analysis on two topics and compare results.

    Args:
        topic_a:        First topic string
        topic_b:        Second topic string
        use_demo_cache: If True, try loading from demo cache first

    Returns:
        Comparison result dict with both analyses + diff metrics
    """
    from src.pipeline     import run_pipeline
    from src.demo_manager import DemoManager

    results = {}
    dm      = DemoManager()

    for label, topic in [("A", topic_a), ("B", topic_b)]:
        print(f"\n[comparison] Running Topic {label}: {topic}")
        start = time.time()

        # Try demo cache first if requested
        result = None
        if use_demo_cache:
            result = dm.load(topic)
            if result:
                print(f"  Loaded from demo cache")

        # Run live pipeline if cache miss
        if result is None:
            result = run_pipeline(
                user_input     = topic,
                run_evaluation = False,
                run_nlp        = True,
                run_images     = False,   # skip images for speed
            )

        elapsed = round(time.time() - start, 1)
        print(f"  Completed in {elapsed}s")
        results[label] = result

    # Build comparison
    return _build_comparison(
        topic_a   = topic_a,
        topic_b   = topic_b,
        result_a  = results["A"],
        result_b  = results["B"],
    )


def _build_comparison(
    topic_a:  str,
    topic_b:  str,
    result_a: dict,
    result_b: dict,
) -> dict:
    """Build the structured comparison output."""

    report_a = result_a.get("report", {})
    report_b = result_b.get("report", {})

    bias_a   = float(report_a.get("bias_score", 0.0))
    bias_b   = float(report_b.get("bias_score", 0.0))

    arts_a   = result_a.get("articles", [])
    arts_b   = result_b.get("articles", [])

    sources_a = {a.get("source", "") for a in arts_a}
    sources_b = {a.get("source", "") for a in arts_b}

    common_sources = sources_a & sources_b

    # Stance distributions
    def _stance_dist(articles):
        d = {"Supportive": 0, "Critical": 0, "Neutral": 0}
        for a in articles:
            s = a.get("stance", "Neutral")
            if s in d:
                d[s] += 1
        return d

    stance_a = _stance_dist(arts_a)
    stance_b = _stance_dist(arts_b)

    # NLP comparison
    nlp_a    = result_a.get("nlp_analysis", {})
    nlp_b    = result_b.get("nlp_analysis", {})
    vader_a  = nlp_a.get("corpus_sentiment", {}).get("mean_compound", 0.0)
    vader_b  = nlp_b.get("corpus_sentiment", {}).get("mean_compound", 0.0)

    # Key arguments from each
    args_a   = report_a.get("agent_a", {}).get("arguments",        [])[:3]
    args_b   = report_b.get("agent_a", {}).get("arguments",        [])[:3]
    cnt_a    = report_a.get("agent_b", {}).get("counter_arguments",[])[:3]
    cnt_b    = report_b.get("agent_b", {}).get("counter_arguments",[])[:3]

    # Mean credibility
    from src.source_credibility import get_credibility_score
    cred_a = (
        sum(get_credibility_score(a.get("source","")) for a in arts_a)
        / len(arts_a) if arts_a else 0.60
    )
    cred_b = (
        sum(get_credibility_score(a.get("source","")) for a in arts_b)
        / len(arts_b) if arts_b else 0.60
    )

    # Determine which topic is more positively covered
    if bias_a < bias_b:
        more_positive = topic_a
        bias_delta    = bias_b - bias_a
    elif bias_b < bias_a:
        more_positive = topic_b
        bias_delta    = bias_a - bias_b
    else:
        more_positive = "Equal"
        bias_delta    = 0.0

    return {
        # Topic info
        "topic_a":           topic_a,
        "topic_b":           topic_b,

        # Bias scores
        "bias_a":            round(bias_a, 4),
        "bias_b":            round(bias_b, 4),
        "bias_delta":        round(bias_delta, 4),
        "more_positive":     more_positive,

        # Article counts
        "n_articles_a":      len(arts_a),
        "n_articles_b":      len(arts_b),

        # Stance distributions
        "stance_a":          stance_a,
        "stance_b":          stance_b,

        # Source overlap
        "sources_a":         sorted(sources_a),
        "sources_b":         sorted(sources_b),
        "common_sources":    sorted(common_sources),
        "n_common":          len(common_sources),

        # NLP
        "vader_a":           round(float(vader_a), 4),
        "vader_b":           round(float(vader_b), 4),

        # Credibility
        "mean_credibility_a": round(float(cred_a), 3),
        "mean_credibility_b": round(float(cred_b), 3),

        # Arguments
        "top_args_a":        args_a,
        "top_args_b":        args_b,
        "top_counters_a":    cnt_a,
        "top_counters_b":    cnt_b,

        # Synthesis
        "synthesis_a":       report_a.get("synthesis_report", ""),
        "synthesis_b":       report_b.get("synthesis_report", ""),

        # Raw results for charts
        "result_a":          result_a,
        "result_b":          result_b,

        # Key insight
        "key_insight": _generate_insight(
            topic_a, topic_b,
            bias_a,  bias_b,
            bias_delta, more_positive,
            common_sources,
        ),
    }


def _generate_insight(
    topic_a:      str,
    topic_b:      str,
    bias_a:       float,
    bias_b:       float,
    delta:        float,
    more_positive:str,
    common:       set,
) -> str:
    """Generate a human-readable key insight from comparison."""

    direction_a = (
        "supportively" if bias_a < -0.2
        else "critically" if bias_a > 0.2
        else "neutrally"
    )
    direction_b = (
        "supportively" if bias_b < -0.2
        else "critically" if bias_b > 0.2
        else "neutrally"
    )

    if delta < 0.1:
        return (
            f"Both '{topic_a}' and '{topic_b}' receive "
            f"similar media coverage with minimal bias difference "
            f"(Δ={delta:.2f})."
        )

    insight = (
        f"Media covers '{topic_a}' {direction_a} "
        f"(bias={bias_a:+.2f}) and '{topic_b}' {direction_b} "
        f"(bias={bias_b:+.2f}). "
        f"'{more_positive}' receives more positive coverage "
        f"with a difference of {delta:.2f} points. "
    )

    if common:
        n = len(common)
        insight += (
            f"{n} source{'s' if n > 1 else ''} "
            f"covered both topics: "
            f"{', '.join(list(common)[:3])}"
            f"{'...' if n > 3 else ''}."
        )

    return insight