# src/pipeline.py
"""
ORBITA — Main Pipeline Orchestrator

Coordinates all phases of the ORBITA pipeline:

    Phase 2:   Data Engineering
               Intent decoding (spaCy NER on user query)
               NewsAPI article fetching
               Zero-shot stance classification
               newspaper4k full text scraping
               Semantic deduplication
               Fact density computation

    Phase 2.5: Manual NLP Analysis  [HYBRID SYSTEM]
               VADER sentiment per article
               spaCy NER on article texts
               TF-IDF keyword extraction
               Manual bias score (independent of Gemini)

    Phase 2.6: Image Analysis  [VISUAL BIAS]
               Extract images from articles
               Gemini Vision analysis
               Visual framing detection

    Phase 3:   NLP Core
               Text chunking with overlap
               Gemini embedding generation
               ChromaDB vector store persistence

    Phase 4:   Multi-Agent RAG Synthesis
               Agent A: supporting argument extraction
               Agent B: counter-argument extraction
               Agent C: synthesis + hallucination check
               Post-agent NLP validation

    Evaluation: Formal evaluation framework
               Bias accuracy vs AllSides ground truth
               ROUGE scores, coverage diversity
               Pipeline performance timing

Author: [Your Name]
Project: ORBITA — B.Tech 6th Sem, AIML 2026
"""

import os
import json
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CORE PIPELINE IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

try:
    from .chain_of_thought import ORBITACoT, CoTStepType
except ImportError:
    from chain_of_thought import ORBITACoT, CoTStepType
try:
    from .bias_timeline  import save_bias_entry
    from .fact_density   import compute_fact_density
    from .intent_decoder import decode_intent
    from .news_fetcher   import fetch_articles
    from .stance_filter  import label_all_articles, rebalance_articles
    from .scraper        import scrape_articles
    from .deduplicator   import deduplicate
    from .chunker        import chunk_all_articles
    from .embedder       import embed_chunks
    from .vector_store   import store_chunks, get_collection_stats
    from .agents         import run_all_agents, save_report
    from .config         import DATA_DIR

except ImportError:
    from bias_timeline   import save_bias_entry
    from fact_density    import compute_fact_density
    from intent_decoder  import decode_intent
    from news_fetcher    import fetch_articles
    from stance_filter   import label_all_articles, rebalance_articles
    from scraper         import scrape_articles
    from deduplicator    import deduplicate
    from chunker         import chunk_all_articles
    from embedder        import embed_chunks
    from vector_store    import store_chunks, get_collection_stats
    from agents          import run_all_agents, save_report
    from config          import DATA_DIR


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL MODULE IMPORTS
# Each wrapped in try/except so pipeline works even if module missing
# ─────────────────────────────────────────────────────────────────────────────

# ── Manual NLP Analyzer ───────────────────────────────────────────────────────
_nlp_analyzer_available = False
try:
    from src.nlp_analyzer import run_nlp_analysis, validate_against_gemini
    _nlp_analyzer_available = True
except ImportError:
    try:
        from nlp_analyzer import run_nlp_analysis, validate_against_gemini
        _nlp_analyzer_available = True
    except ImportError:
        pass

if not _nlp_analyzer_available:
    print(
        "[pipeline] Warning: nlp_analyzer not found. "
        "Manual NLP phase will be skipped. "
        "Check src/nlp_analyzer.py exists."
    )

# ── Image Analyzer ────────────────────────────────────────────────────────────
_image_analyzer_available = False
try:
    from src.cnn_image_analyzer import run_image_analysis_pipeline
    _image_analyzer_available = True
except ImportError:
    try:
        from cnn_image_analyzer import run_image_analysis_pipeline
        _image_analyzer_available = True
    except ImportError:
        pass

if not _image_analyzer_available:
    print(
        "[pipeline] Warning: cnn_image_analyzer not found. "
        "Image analysis phase will be skipped."
    )

# ── Evaluation Framework ──────────────────────────────────────────────────────
_evaluator_available = False
try:
    from src.evaluation.evaluator import ORBITAEvaluator
    _evaluator_available = True
except ImportError:
    try:
        from evaluation.evaluator import ORBITAEvaluator
        _evaluator_available = True
    except ImportError:
        pass

if not _evaluator_available:
    print(
        "[pipeline] Warning: evaluation module not found. "
        "Formal evaluation will be skipped."
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_articles(articles: list, topic: str) -> None:
    """
    Save article metadata to disk as JSON.

    Saves lightweight metadata only — not full text —
    so you can review what was fetched without huge files.

    Args:
        articles: list of article dicts from pipeline
        topic:    the topic string (used for filename)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    safe_topic = "".join(
        c if c.isalnum() else "_"
        for c in topic
    )[:40]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = os.path.join(
        DATA_DIR, f"{safe_topic}_{timestamp}.json"
    )

    data = []
    for article in articles:
        data.append({
            "url":          article.get("url"),
            "title":        article.get("title"),
            "source":       article.get("source"),
            "stance":       article.get("stance"),
            "word_count":   len(
                (article.get("full_text") or "").split()
            ),
            "fact_density": article.get("fact_density", 0),
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[pipeline] Article metadata saved → {path}")


def _run_nlp_phase(
    articles: list,
    topic:    str,
) -> tuple:
    """
    Run the Manual NLP Analysis phase (Phase 2.5).

    Separated into its own function for:
    1. Clean error isolation — NLP failure never kills pipeline
    2. Easy timing measurement
    3. Clear logging

    Args:
        articles: list of deduplicated, scraped article dicts
        topic:    topic string for logging

    Returns:
        tuple of (nlp_results dict, nlp_context string, elapsed seconds)
        Returns ({}, "", 0.0) if NLP fails or is unavailable
    """
    if not _nlp_analyzer_available:
        print("\n>>> PHASE 2.5: MANUAL NLP SKIPPED (module not available)")
        return {}, "", 0.0

    print("\n>>> PHASE 2.5: MANUAL NLP ANALYSIS")
    start = time.time()

    try:
        nlp_results = run_nlp_analysis(
            articles          = articles,
            gemini_bias_score = 0.0,
            # Pass 0.0 here — we do NOT have Gemini score yet.
            # We re-run validation AFTER phase 4 with real score.
        )

        nlp_context = nlp_results.get("agent_context", "")
        elapsed     = round(time.time() - start, 2)

        # Print summary
        manual_bias = nlp_results.get("manual_bias", {})
        sentiment   = nlp_results.get("sentiment_summary", {})
        keywords    = nlp_results.get("keyword_analysis", {})
        entities    = nlp_results.get("entity_analysis",  {})

        print(f"\n>>> PHASE 2.5 COMPLETE in {elapsed}s")
        print(f"  Manual bias estimate:  "
              f"{manual_bias.get('manual_bias_score', 0):+.4f}")
        print(f"  Avg VADER compound:    "
              f"{sentiment.get('avg_compound', 0):+.4f}")
        print(f"  Top keyword:           "
              f"{keywords.get('top_keywords', [{}])[0].get('word', 'N/A')}"
              if keywords.get("top_keywords") else
              f"  Top keyword:           N/A"
        )
        print(f"  Top entity:            "
              f"{entities.get('top_entities', [{}])[0].get('text', 'N/A')}"
              if entities.get("top_entities") else
              f"  Top entity:            N/A"
        )
        print(f"  Agent context length:  {len(nlp_context)} chars")

        return nlp_results, nlp_context, elapsed

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"\n[pipeline] Phase 2.5 NLP failed after {elapsed}s: {e}")

        import traceback
        traceback.print_exc()

        return {}, "", elapsed


def _run_image_phase(
    articles:     list,
    topic:        str,
    max_articles: int = 4,
) -> tuple:
    """
    Run the Image Analysis phase (Phase 2.6).

    Args:
        articles:     scraped article dicts with image_urls
        topic:        topic string for logging
        max_articles: max articles to analyze images from

    Returns:
        tuple of (image_results dict, visual_context string, elapsed seconds)
        Returns ({}, "", 0.0) if image analysis fails or is unavailable
    """
    if not _image_analyzer_available:
        print("\n>>> PHASE 2.6: IMAGE ANALYSIS SKIPPED (module not available)")
        return {}, "", 0.0

    print("\n>>> PHASE 2.6: IMAGE ANALYSIS")
    start = time.time()

    try:
        image_results = run_image_analysis_pipeline(
            articles     = articles,
            max_articles = max_articles,
        )

        visual_context = image_results.get("visual_context", "")
        elapsed        = round(time.time() - start, 2)

        print(f"\n>>> PHASE 2.6 COMPLETE in {elapsed}s")
        print(f"  Images analyzed: {image_results.get('total_images', 0)}")
        print(f"  Visual bias:     "
              f"{image_results.get('summary', {}).get('visual_bias_score', 0):+.4f}")

        return image_results, visual_context, elapsed

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"\n[pipeline] Phase 2.6 Image analysis failed: {e}")
        return {}, "", elapsed


def _run_post_agent_nlp_validation(
    nlp_results:      dict,
    gemini_bias_score: float,
) -> dict:
    """
    Re-run NLP validation AFTER agents complete.

    Now that we have the real Gemini bias_score, we can compare
    it against the manual NLP bias score for validation.

    This is the key research contribution:
    "Our manual NLP score and Gemini AI score agree at X%"

    Args:
        nlp_results:       from Phase 2.5
        gemini_bias_score: from report["bias_score"] after Phase 4

    Returns:
        Updated nlp_results with gemini_validation filled in
    """
    if not nlp_results or not _nlp_analyzer_available:
        return nlp_results

    try:
        manual_score = nlp_results.get(
            "manual_bias", {}
        ).get("manual_bias_score", 0.0)

        validation = validate_against_gemini(
            manual_bias_score = manual_score,
            gemini_bias_score = gemini_bias_score,
        )

        nlp_results["gemini_validation"] = validation

        print(
            f"\n[pipeline] NLP Validation Result:"
            f"\n  Manual NLP score:  {manual_score:+.4f}"
            f"\n  Gemini AI score:   {gemini_bias_score:+.4f}"
            f"\n  Absolute diff:     {validation.get('absolute_diff', 0):.4f}"
            f"\n  Agreement:         {validation.get('agreement_level', 'N/A')}"
            f"\n  Direction agrees:  {validation.get('direction_agrees', 'N/A')}"
            f"\n  Note: {validation.get('validation_note', '')}"
        )

        # Also update the agent_context string with validation info
        # so future runs can reference it
        existing_context = nlp_results.get("agent_context", "")
        validation_line  = (
            f"\nGemini vs Manual Validation: "
            f"{validation.get('agreement_level', 'N/A')} "
            f"(diff={validation.get('absolute_diff', 0):.3f}, "
            f"direction={'agrees' if validation.get('direction_agrees') else 'disagrees'})"
        )
        nlp_results["agent_context"] = existing_context + validation_line

        return nlp_results

    except Exception as e:
        print(f"[pipeline] Post-agent validation error (non-critical): {e}")
        return nlp_results


def _run_evaluation_phase(
    result:          dict,
    topic:           str,
    elapsed_seconds: float,
) -> dict:
    """
    Run the formal evaluation framework on pipeline result.

    Wrapped in try/except so evaluation failure never
    crashes the pipeline.

    Args:
        result:          the compiled pipeline result dict
        topic:           topic string
        elapsed_seconds: total pipeline runtime

    Returns:
        evaluation report dict, or {} if evaluation fails
    """
    if not _evaluator_available:
        print("[pipeline] Skipping evaluation — module not available")
        return {}

    try:
        print("\n[pipeline] Running formal evaluation...")
        evaluator   = ORBITAEvaluator()
        eval_report = evaluator.evaluate(
            pipeline_result = result,
            topic           = topic,
            elapsed_seconds = elapsed_seconds,
        )
        evaluator.save_report(eval_report)
        evaluator.print_summary(eval_report)
        return eval_report

    except Exception as e:
        print(f"[pipeline] Evaluation error (non-critical): {e}")
        import traceback
        traceback.print_exc()
        return {}


def _add_nlp_to_report(
    report:      dict,
    nlp_results: dict,
) -> dict:
    """
    Add NLP analysis results to the report dict.

    The report dict is what gets SAVED to reports/ folder
    and RETURNED to app.py.

    We add a lightweight summary (not full data) to keep
    the report JSON file a reasonable size.

    Args:
        report:      existing report dict from run_all_agents()
        nlp_results: full NLP results from Phase 2.5

    Returns:
        Updated report dict with nlp_summary added
    """
    if not nlp_results:
        report["nlp_summary"] = {}
        return report

    # Build a lightweight summary for the report JSON
    sentiment_summary = nlp_results.get("sentiment_summary", {})
    manual_bias       = nlp_results.get("manual_bias",        {})
    validation        = nlp_results.get("gemini_validation",  {})
    keywords          = nlp_results.get("keyword_analysis",   {})
    entities          = nlp_results.get("entity_analysis",    {})

    report["nlp_summary"] = {
        # Sentiment summary
        "avg_vader_compound":   sentiment_summary.get("avg_compound",  0.0),
        "sentiment_distribution": sentiment_summary.get("distribution", {}),

        # Manual bias
        "manual_bias_score":    manual_bias.get("manual_bias_score",  0.0),
        "manual_validation_note": manual_bias.get("validation_note",  ""),

        # Gemini vs Manual validation
        "gemini_validation": {
            "agreement_level":   validation.get("agreement_level",  "N/A"),
            "absolute_diff":     validation.get("absolute_diff",    None),
            "direction_agrees":  validation.get("direction_agrees", None),
            "agreement_score":   validation.get("agreement_score",  None),
        },

        # Top keywords (just names for report)
        "top_keywords": [
            kw["word"]
            for kw in keywords.get("top_keywords", [])[:10]
        ],

        # Top entities (just text + count)
        "top_entities": [
            {
                "text":       e.get("text",  ""),
                "label":      e.get("label", ""),
                "count":      e.get("count",  0),
            }
            for e in entities.get("top_entities", [])[:10]
        ],

        # Libraries used
        "libraries_used": nlp_results.get("libraries_used", {}),
    }

    return report


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    user_input:      str,
    run_evaluation:  bool = True,
    run_nlp:         bool = True,
    run_images:      bool = True,
) -> dict:
    """
    Main ORBITA pipeline — runs all phases end to end.

    Args:
        user_input:      raw user query (topic or URL)
        run_evaluation:  run formal evaluation after pipeline
                         Set False during development to save time
        run_nlp:         run manual NLP analysis (Phase 2.5)
                         Set False to skip VADER/spaCy for speed
        run_images:      run image analysis (Phase 2.6)
                         Set False to skip Gemini Vision

    Returns:
        dict with ALL pipeline results including:
            articles:         list of scraped article dicts
            stats:            ChromaDB collection stats
            report:           full multi-agent report WITH nlp_summary
            topic:            decoded topic string
            intent:           full intent decoder output
            nlp_analysis:     complete NLP analysis data
                              (used by app.py for charts)
            image_analysis:   image analysis results
            visual_context:   text summary of visual analysis
            evaluation:       formal evaluation report
            elapsed_seconds:  total pipeline runtime
            phase_timings:    breakdown per phase
    """

    # ── Start total timer ─────────────────────────────────────────
    pipeline_start = time.time()

    print("\n" + "=" * 65)
    print("ORBITA — Full Pipeline")
    print("=" * 65)
    print(f"Input:       '{user_input}'")
    print(f"Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"NLP enabled:  {run_nlp and _nlp_analyzer_available}")
    print(f"Images:       {run_images and _image_analyzer_available}")
    print(f"Evaluation:   {run_evaluation and _evaluator_available}")
    print("=" * 65)

    # Track per-phase timings for the performance table in your paper
    phase_timings = {}

    # ─────────────────────────────────────────────────────────────
    # PHASE 2 — DATA ENGINEERING
    # ─────────────────────────────────────────────────────────────

    print("\n>>> PHASE 2: DATA ENGINEERING")
    phase2_start = time.time()

    # Step 1: Decode intent ────────────────────────────────────────
    print("\n  [Step 1/5] Decoding intent with spaCy NER...")
    intent = decode_intent(user_input)

    print(f"  Topic:          '{intent['topic']}'")
    print(f"  Entities:        {intent['entities']}")
    print(f"  Keywords:        {intent['keywords']}")
    print(f"  Search queries:  {intent['search_queries']}")

    # Step 2: Fetch articles ───────────────────────────────────────
    print("\n  [Step 2/5] Fetching articles via NewsAPI...")
    articles = fetch_articles(intent["search_queries"])

    if not articles:
        raise RuntimeError(
            "No articles found for this topic.\n"
            "Suggestions:\n"
            "  1. Try a more specific topic\n"
            "  2. Check your NEWS_API_KEY in .env\n"
            "  3. Verify internet connection"
        )

    print(f"  Fetched: {len(articles)} articles")

    # Step 3: Stance classification ────────────────────────────────
    print("\n  [Step 3/5] Zero-shot stance classification (TF-IDF)...")
    articles = label_all_articles(articles)
    articles = rebalance_articles(articles)

    stance_counts = {}
    for a in articles:
        s = a.get("stance", "Neutral")
        stance_counts[s] = stance_counts.get(s, 0) + 1
    print(f"  Stance distribution: {stance_counts}")

    # Step 4: Scraping ─────────────────────────────────────────────
    print("\n  [Step 4/5] Scraping full text via newspaper4k...")
    articles = scrape_articles(articles)

    if len(articles) < 3:
        raise RuntimeError(
            f"Only {len(articles)} articles scraped successfully.\n"
            "Most articles may be paywalled or JavaScript-rendered.\n"
            "Try a different topic with more accessible sources."
        )

    print(f"  Scraped: {len(articles)} articles")

    # Step 5: Deduplication ────────────────────────────────────────
    print("\n  [Step 5/5] Semantic deduplication...")
    articles = deduplicate(articles)
    print(f"  After dedup: {len(articles)} unique articles")

    # Fact density computation ─────────────────────────────────────
    print("\n  Computing fact density scores...")
    for article in articles:
        text = article.get("full_text", "") or ""
        try:
            article["fact_density"] = compute_fact_density(text)
        except Exception:
            article["fact_density"] = 0.0

    # Save article metadata to disk
    save_articles(articles, intent["topic"])

    phase_timings["phase2_data_engineering"] = round(
        time.time() - phase2_start, 2
    )
    print(
        f"\n>>> PHASE 2 COMPLETE — "
        f"{len(articles)} articles, "
        f"{phase_timings['phase2_data_engineering']}s"
    )

    # ─────────────────────────────────────────────────────────────
    # PHASE 2.5 — MANUAL NLP ANALYSIS
    # ─────────────────────────────────────────────────────────────

    nlp_results  = {}
    nlp_context  = ""
    phase25_start = time.time()

    if run_nlp:
        nlp_results, nlp_context, nlp_elapsed = _run_nlp_phase(
            articles = articles,
            topic    = intent["topic"],
        )
        phase_timings["phase2_5_nlp"] = nlp_elapsed
    else:
        print("\n>>> PHASE 2.5: MANUAL NLP SKIPPED (run_nlp=False)")
        phase_timings["phase2_5_nlp"] = 0.0

    # ─────────────────────────────────────────────────────────────
    # PHASE 2.6 — IMAGE ANALYSIS
    # ─────────────────────────────────────────────────────────────

    image_results  = {}
    visual_context = ""

    if run_images:
        image_results, visual_context, img_elapsed = _run_image_phase(
            articles     = articles,
            topic        = intent["topic"],
            max_articles = 4,
        )
        phase_timings["phase2_6_images"] = img_elapsed
    else:
        print("\n>>> PHASE 2.6: IMAGE ANALYSIS SKIPPED (run_images=False)")
        phase_timings["phase2_6_images"] = 0.0

    # ─────────────────────────────────────────────────────────────
    # PHASE 3 — NLP CORE (EMBEDDINGS + VECTOR STORE)
    # ─────────────────────────────────────────────────────────────

    print("\n>>> PHASE 3: EMBEDDINGS & VECTOR STORE")
    phase3_start = time.time()

    # Step 1: Text chunking ────────────────────────────────────────
    print("\n  [Step 1/3] Chunking article text...")
    chunks = chunk_all_articles(articles)
    print(f"  Created: {len(chunks)} chunks")

    # Step 2: Embedding generation ─────────────────────────────────
    print("\n  [Step 2/3] Generating Gemini embeddings...")
    embedded_chunks = embed_chunks(chunks)
    print(f"  Embedded: {len(embedded_chunks)} chunks successfully")

    # Step 3: Store in ChromaDB ────────────────────────────────────
    print("\n  [Step 3/3] Persisting to ChromaDB vector store...")
    store_chunks(embedded_chunks, reset=True)
    stats = get_collection_stats()

    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  By stance:    {stats['by_stance']}")

    phase_timings["phase3_embeddings"] = round(
        time.time() - phase3_start, 2
    )
    print(
        f"\n>>> PHASE 3 COMPLETE — "
        f"{stats['total_chunks']} chunks, "
        f"{phase_timings['phase3_embeddings']}s"
    )

    # ─────────────────────────────────────────────────────────────
    # PHASE 4 — MULTI-AGENT RAG SYNTHESIS
    # ─────────────────────────────────────────────────────────────

    print("\n>>> PHASE 4: MULTI-AGENT RAG SYNTHESIS")
    phase4_start = time.time()

    # Pass BOTH nlp_context AND visual_context to agents
    # This is the hybrid system — agents get manual NLP + visual data
    report = run_all_agents(
        topic          = intent["topic"],
        visual_context = visual_context,   # from Phase 2.6
        nlp_context    = nlp_context,      # from Phase 2.5
    )

    save_report(report, intent["topic"])

    # ── Record to heatmap ──────────────────────────────────
    try:
        from src.heatmap_manager  import HeatmapManager
        hm = HeatmapManager()
        hm.record_run(
            topic    = intent["topic"],
            articles = articles,
            report   = report,
        )
    except Exception as e:
        print(f"[pipeline] Heatmap record error: {e}")

    # ── Save to SQLite history ─────────────────────────────
    try:
        from src.history_tracker import save_run
        elapsed = round(time.time() - pipeline_start, 2)
        run_id = save_run(
            pipeline_result = {
                "report":       report,
                "articles":     articles,
                "topic":        intent["topic"],
                "stats":        stats,
                "nlp_analysis": nlp_results or {},
            },
            elapsed_seconds = elapsed,
            is_demo         = False,
        )
        if run_id:
            print(f"[pipeline] History saved: run #{run_id}")
    except Exception as e:
        print(f"[pipeline] History save error: {e}")

    # ── Add credibility scores to articles ─────────────────
    try:
        from src.source_credibility import score_articles
        articles = score_articles(articles)
    except Exception as e:
        print(f"[pipeline] Credibility scoring error: {e}")

    phase_timings["phase4_agents"] = round(
        time.time() - phase4_start, 2
    )
    print(
        f"\n>>> PHASE 4 COMPLETE — "
        f"{phase_timings['phase4_agents']}s"
    )

    # ─────────────────────────────────────────────────────────────
    # POST-AGENT NLP VALIDATION
    # Now we have real Gemini bias_score — compare with manual NLP
    # ─────────────────────────────────────────────────────────────

    if nlp_results and run_nlp:
        print("\n[pipeline] Running post-agent NLP validation...")
        gemini_bias_score = report.get("bias_score", 0.0)
        nlp_results = _run_post_agent_nlp_validation(
            nlp_results       = nlp_results,
            gemini_bias_score = gemini_bias_score,
        )

    # Add NLP CoT steps and prepend them to agent chain-of-thought
    if nlp_results and report.get("chain_of_thought") is not None:
        try:
            from src.chain_of_thought import ORBITACoT, CoTStepType
        except ImportError:
            from chain_of_thought import ORBITACoT, CoTStepType

        _ = CoTStepType

        nlp_cot = ORBITACoT(topic=intent["topic"])

        per_article = nlp_results.get(
            "per_article_sentiment", []
        )
        sent_summary = nlp_results.get("sentiment_summary", {})

        if per_article:
            nlp_cot.add_sentiment_step(
                per_article  = per_article,
                avg_compound = sent_summary.get("avg_compound", 0),
            )

        entity_data  = nlp_results.get("entity_analysis", {})
        top_entities = entity_data.get("top_entities", [])
        if top_entities:
            nlp_cot.add_entity_step(
                top_entities = top_entities,
                n_total      = len(top_entities),
            )

        kw_data = nlp_results.get("keyword_analysis", {})
        top_kws = kw_data.get("top_keywords", [])
        if top_kws:
            nlp_cot.add_keyword_step(top_keywords=top_kws)

        validation = nlp_results.get("gemini_validation", {})
        manual     = nlp_results.get("manual_bias", {})
        if validation and manual:
            nlp_cot.add_validation_step(
                manual_score = manual.get("manual_bias_score", 0),
                gemini_score = report.get("bias_score", 0),
                agreement    = validation.get("agreement_level", "N/A"),
                diff         = validation.get("absolute_diff", 0),
            )

        existing_chain = report.get("chain_of_thought", [])
        report["chain_of_thought"] = (
            nlp_cot.get_chain() + existing_chain
        )

    # ─────────────────────────────────────────────────────────────
    # ADD NLP RESULTS TO REPORT DICT
    # This ensures NLP data is saved in reports/ JSON files
    # and available when result is loaded from cache
    # ─────────────────────────────────────────────────────────────

    report = _add_nlp_to_report(report, nlp_results)

    # ─────────────────────────────────────────────────────────────
    # SAVE BIAS TIMELINE ENTRY
    # ─────────────────────────────────────────────────────────────

    try:
        bias_score = report.get("bias_score", 0.0)
        save_bias_entry(
            topic      = intent["topic"],
            bias_score = bias_score,
        )
    except Exception as e:
        print(f"[pipeline] Timeline save error (non-critical): {e}")

    # ─────────────────────────────────────────────────────────────
    # COMPUTE TOTAL ELAPSED TIME
    # ─────────────────────────────────────────────────────────────

    pipeline_elapsed = round(time.time() - pipeline_start, 2)
    phase_timings["total"] = pipeline_elapsed

    # ─────────────────────────────────────────────────────────────
    # COMPILE FINAL RESULT DICT
    # This is what app.py, cache_manager.py, and evaluator receive
    # ─────────────────────────────────────────────────────────────

    result = {
        # Core results
        "articles":         articles,
        "stats":            stats,
        "report":           report,
        "topic":            intent["topic"],
        "intent":           intent,

        # NLP analysis — FULL data for app.py charts
        # This is separate from report["nlp_summary"]
        # report["nlp_summary"] = lightweight for JSON saving
        # result["nlp_analysis"] = full data for live charts
        "nlp_analysis":     nlp_results,

        # Image analysis
        "image_analysis":   image_results,
        "visual_context":   visual_context,

        # Performance metrics
        "elapsed_seconds":  pipeline_elapsed,
        "phase_timings":    phase_timings,
    }

    # ─────────────────────────────────────────────────────────────
    # FORMAL EVALUATION (OPTIONAL)
    # ─────────────────────────────────────────────────────────────

    if run_evaluation:
        eval_report = _run_evaluation_phase(
            result          = result,
            topic           = intent["topic"],
            elapsed_seconds = pipeline_elapsed,
        )
        result["evaluation"] = eval_report
    else:
        print("\n[pipeline] Evaluation skipped (run_evaluation=False)")
        result["evaluation"] = {}

    # ─────────────────────────────────────────────────────────────
    # FINAL SUMMARY PRINT
    # ─────────────────────────────────────────────────────────────

    _print_pipeline_summary(result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def _print_pipeline_summary(result: dict) -> None:
    """
    Print a formatted summary of all pipeline results.

    Shows all key metrics in one clean block.
    Useful for debugging and for recording in lab notebook.
    """
    report       = result.get("report",        {})
    nlp_analysis = result.get("nlp_analysis",  {})
    bias_vector  = report.get("bias_vector",   {})
    phase_times  = result.get("phase_timings", {})
    evaluation   = result.get("evaluation",    {})

    print("\n" + "=" * 65)
    print("  ORBITA — Pipeline Complete")
    print("=" * 65)
    print(f"  Topic:              '{result.get('topic', 'Unknown')}'")
    print(f"  Articles:            {len(result.get('articles', []))}")
    print(f"  Chunks in DB:        {result.get('stats', {}).get('total_chunks', 0)}")

    # Bias scores
    print(f"\n  ── Bias Scores ──────────────────────────────")
    print(f"  Gemini composite:    {report.get('bias_score', 0):+.4f}")
    if bias_vector:
        print(f"  Ideological:         {bias_vector.get('ideological_bias', 0):+.4f}")
        print(f"  Emotional:           {bias_vector.get('emotional_bias',   0):.4f}")
        print(f"  Source diversity:    {bias_vector.get('source_diversity', 0):.4f}")
        print(f"  Interpretation:      {bias_vector.get('interpretation', 'N/A')}")

    # Manual NLP
    if nlp_analysis:
        manual = nlp_analysis.get("manual_bias", {})
        sent   = nlp_analysis.get("sentiment_summary", {})
        val    = nlp_analysis.get("gemini_validation", {})

        print(f"\n  ── Manual NLP ───────────────────────────────")
        print(f"  Manual bias score:   {manual.get('manual_bias_score', 0):+.4f}")
        print(f"  Avg VADER compound:  {sent.get('avg_compound', 0):+.4f}")

        if val:
            print(
                f"  NLP vs AI:           "
                f"{val.get('agreement_level', 'N/A')} "
                f"(diff={val.get('absolute_diff', 0):.4f})"
            )

    # Phase timings
    print(f"\n  ── Phase Timings ────────────────────────────")
    print(f"  Phase 2  (Data):     {phase_times.get('phase2_data_engineering', 0):.1f}s")
    print(f"  Phase 2.5 (NLP):     {phase_times.get('phase2_5_nlp', 0):.1f}s")
    print(f"  Phase 2.6 (Images):  {phase_times.get('phase2_6_images', 0):.1f}s")
    print(f"  Phase 3  (Embed):    {phase_times.get('phase3_embeddings', 0):.1f}s")
    print(f"  Phase 4  (Agents):   {phase_times.get('phase4_agents', 0):.1f}s")
    print(f"  Total:               {phase_times.get('total', 0):.1f}s")

    # Evaluation
    if evaluation:
        ov = evaluation.get("overall_score", {})
        print(f"\n  ── Evaluation ───────────────────────────────")
        print(
            f"  Quality score:       "
            f"{ov.get('score', 0):.3f} ({ov.get('grade', 'N/A')})"
        )

    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run the pipeline directly from command line.

    Usage:
        python src/pipeline.py
    """
    print("ORBITA — Direct Pipeline Runner")
    print("─" * 40)
    print("Example topics:")
    print("  Farm Laws India protest")
    print("  UPI digital payments India")
    print("  Electric Vehicles India policy")
    print("  Cryptocurrency regulation India")
    print("─" * 40)

    user_input = input(
        "\nEnter topic (or press Enter for default): "
    ).strip()

    if not user_input:
        user_input = "Cryptocurrency regulation India"
        print(f"Using default: '{user_input}'")

    # Ask for options
    eval_input = input(
        "Run formal evaluation? [Y/n]: "
    ).strip().lower()
    run_eval = eval_input not in ("n", "no")

    nlp_input = input(
        "Run manual NLP analysis? [Y/n]: "
    ).strip().lower()
    run_nlp_flag = nlp_input not in ("n", "no")

    img_input = input(
        "Run image analysis? [y/N]: "
    ).strip().lower()
    run_img_flag = img_input in ("y", "yes")

    try:
        result = run_pipeline(
            user_input     = user_input,
            run_evaluation = run_eval,
            run_nlp        = run_nlp_flag,
            run_images     = run_img_flag,
        )

        print(f"\n[Done] Pipeline completed successfully.")
        print(f"  Topic:   {result['topic']}")
        print(f"  Time:    {result['elapsed_seconds']}s")
        print(f"  Reports: check reports/ folder")
        if run_eval:
            print(f"  Evals:   check evaluation_results/ folder")

    except RuntimeError as e:
        print(f"\n[Error] Pipeline failed: {e}")

    except KeyboardInterrupt:
        print("\n[Cancelled] Pipeline interrupted.")