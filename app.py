import os
import sys
import json
import streamlit as st
from src.report_generator import generate_pdf_report, get_pdf_filename
from src.ui.timeline_chart import build_bias_timeline
from src.demo_manager import DemoManager, DEMO_TOPICS
from src.live_meter   import LiveBiasMeter
from src.ui.charts import (
    build_bias_spectrum_graph,
    build_confidence_gauge,
    build_stance_distribution_chart,
    build_word_count_chart,
    build_bias_radar_chart,
    build_bias_breakdown_bars,
    # ADD THESE NEW IMPORTS:
    build_sentiment_bar_chart,
    build_word_cloud_chart,
    build_word_cloud_matplotlib,
    build_entity_frequency_chart,
    build_sentiment_timeline_chart,
    _empty_chart,
)
 
sys.path.insert(0, os.path.dirname(__file__))
 
st.set_page_config(
    page_title     = "ORBITA",
    page_icon      = "🔭",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)
 
 
def load_css():
    path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
load_css()
 
# ── Imports ───────────────────────────────────────────────────────────────────
from src.intent_decoder import decode_intent
from src.news_fetcher   import fetch_articles
from src.stance_filter  import label_all_articles, rebalance_articles
from src.scraper        import scrape_articles
from src.deduplicator   import deduplicate
from src.chunker        import chunk_all_articles
from src.embedder       import embed_chunks
from src.vector_store   import store_chunks, get_collection_stats
from src.agent_a        import run_agent_a
from src.agent_b        import run_agent_b
from src.agent_c        import run_agent_c
from src.agent_c        import _extract_real_citations
import src.agents
from src.cache_manager  import (
    get_cached_result, save_to_cache,
    list_cached_topics, clear_cache,
)

# ── Manual NLP Analyzer (optional) ────────────────────────────────────────────
_nlp_analyzer_available = False
try:
    from src.nlp_analyzer import run_nlp_analysis, validate_against_gemini
    _nlp_analyzer_available = True
except ImportError:
    pass

# ── Image Analyzer (optional) ────────────────────────────────────────────────
_image_analyzer_available = False
try:
    from src.cnn_image_analyzer import run_image_analysis_pipeline
    _image_analyzer_available = True
except ImportError:
    try:
        from cnn_image_analyzer import run_image_analysis_pipeline  # type: ignore[import-not-found]
        _image_analyzer_available = True
    except ImportError:
        try:
            from src.image_analyzer import run_image_analysis_pipeline  # type: ignore[import-not-found]
            _image_analyzer_available = True
        except ImportError:
            try:
                from image_analyzer import run_image_analysis_pipeline  # type: ignore[import-not-found]
                _image_analyzer_available = True
            except ImportError:
                pass
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def _section(icon: str, title: str):
    st.markdown(
        f'<div class="orb-section">'
        f'<div class="orb-section-line"></div>'
        f'<div class="orb-section-title">{icon}&nbsp; {title}</div>'
        f'<div class="orb-section-line"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
 
 
def _confidence_badge(score: float, label: str):
    if score >= 0.75:
        dot_color = "#3ec97e"
    elif score >= 0.5:
        dot_color = "#c9a84c"
    else:
        dot_color = "#e05252"
 
    st.markdown(
        f'<div class="orb-confidence">'
        f'<div class="orb-confidence-dot" '
        f'style="background:{dot_color}"></div>'
        f'{label} &nbsp;·&nbsp; confidence <b style="color:{dot_color}">'
        f'{score:.2f}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_demo_panel():
    """
    Render the Demo Mode topic selector panel in sidebar.
    Shows available pre-cached topics with bias scores.
    """
    dm     = DemoManager()
    topics = dm.get_all_topics_with_status()

    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;'
        'font-size:0.62rem;letter-spacing:2px;'
        'color:#c9a84c;text-transform:uppercase;'
        'margin:0.5rem 0 0.4rem">Pre-loaded Topics</div>',
        unsafe_allow_html=True,
    )

    for t in topics:
        is_avail = t["available"]
        icon     = t["icon"]
        name     = t["name"]
        desc     = t["description"][:35]

        if is_avail:
            # Clickable card for available topics
            if st.button(
                f"{icon}  {name}",
                key              = f"demo_{name}",
                use_container_width = True,
                help             = desc,
            ):
                # Load cached result
                with st.spinner("Loading cached analysis..."):
                    result = dm.load(name)

                if result:
                    try:
                        from src.heatmap_manager import HeatmapManager
                        from src.history_tracker import save_run

                        hm = HeatmapManager()
                        hm.record_run(
                            topic    = result.get("topic", name),
                            articles = result.get("articles", []),
                            report   = result.get("report", {}),
                        )

                        save_run(
                            pipeline_result = {
                                "report":       result.get("report", {}),
                                "articles":     result.get("articles", []),
                                "topic":        result.get("topic", name),
                                "stats":        result.get("stats", {}),
                                "nlp_analysis": result.get("nlp_analysis", {}),
                            },
                            elapsed_seconds = 0.0,
                            is_demo         = True,
                        )
                    except Exception:
                        pass

                    st.session_state["results"]    = result
                    st.session_state["from_cache"] = True
                    st.session_state["last_topic"] = name
                    st.session_state["is_demo"]    = True
                    st.session_state["error"]      = None
                    st.rerun()
                else:
                    st.error(f"Cache missing: {name}")
        else:
            # Greyed-out card for uncached topics
            st.markdown(
                f'<div style="'
                f'opacity:0.35;'
                f'font-family:\'DM Mono\',monospace;'
                f'font-size:0.78rem;'
                f'color:#5c6b82;'
                f'padding:0.35rem 0.5rem;'
                f'margin-bottom:0.25rem">'
                f'{icon}  {name}'
                f'<span style="float:right;font-size:0.6rem">'
                f'not cached</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Instructions
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;'
        'font-size:0.6rem;color:#3d4d60;'
        'margin-top:0.8rem;line-height:1.7">'
        'To add topics:<br>'
        'python demo_cache/create_demo_cache.py'
        '</div>',
        unsafe_allow_html=True,
    )
 
 
# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    for key, val in {
        "results":    None,
        "running":    False,
        "last_topic": "",
        "error":      None,
        "from_cache": False,
        "collection_name": "",
        "pdf_bytes":  None,
        "demo_mode":  False,
        "is_demo":    False,
        "show_home":  True,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val
 
 
# ── Pipeline ──────────────────────────────────────────────────────────────────
# app.py — REPLACE existing run_pipeline_with_progress() with:

def run_pipeline_with_progress(
    user_input:      str,
    collection_name: str = "",
) -> dict:
    """
    Run the full ORBITA pipeline with live bias meter UI.
    Replaces the old spinner-based progress function.
    All existing functionality preserved — meter is purely additive.
    """
    # Initialize live meter
    meter = LiveBiasMeter()
    meter.start(user_input)

    try:
        # ── PHASE: Intent ──────────────────────────────────
        meter.phase_complete("start")

        from src.intent_decoder import decode_intent
        intent = decode_intent(user_input)
        meter.phase_complete(
            "intent",
            n_queries = len(intent.get("search_queries", [])),
        )

        # ── PHASE: Fetch Articles ──────────────────────────
        from src.news_fetcher import fetch_articles
        articles = fetch_articles(intent["search_queries"])

        if not articles:
            meter.error("fetching", "No articles found")
            raise RuntimeError(
                "No articles found. "
                "Try a broader topic or check NEWS_API_KEY."
            )
        meter.phase_complete("fetching", n_articles=len(articles))

        # ── PHASE: Stance Classification ───────────────────
        from src.stance_filter import (
            label_all_articles, rebalance_articles
        )
        articles = label_all_articles(articles)
        articles = rebalance_articles(articles)
        meter.phase_complete("stance", n_articles=len(articles))

        # ── PHASE: Scraping ────────────────────────────────
        from src.scraper import scrape_articles
        articles = scrape_articles(articles)

        if len(articles) < 3:
            meter.error(
                "scraping",
                f"Only {len(articles)} articles scraped"
            )
            raise RuntimeError(
                f"Only {len(articles)} articles scraped."
                " Try a different topic."
            )
        meter.phase_complete("scraping", n_articles=len(articles))

        # ── PHASE: Deduplication ───────────────────────────
        from src.deduplicator import deduplicate
        articles = deduplicate(articles)

        # ── PHASE: NLP Analysis ────────────────────────────
        nlp_results = None
        nlp_context = ""
        try:
            from src.nlp_analyzer import run_nlp_analysis
            # Pass 0.0 here: Gemini bias score is not available yet.
            # Passing user_input (string) causes type errors in NLP validation.
            nlp_results = run_nlp_analysis(
                articles          = articles,
                gemini_bias_score = 0.0,
            )
            vader_avg   = nlp_results.get(
                "sentiment_summary", {}
            ).get("avg_compound", 0.0)

            from src.nlp_analyzer import build_nlp_context_for_agents
            nlp_context = build_nlp_context_for_agents(nlp_results)

            meter.phase_complete(
                "nlp",
                vader_score = vader_avg,
                extra       = (
                    f"{len(nlp_results.get('per_article_sentiment', []))} "
                    "articles analyzed"
                ),
            )
        except Exception as e:
            meter.error("nlp", f"NLP skipped: {str(e)[:40]}")
            nlp_results = {}
            nlp_context = ""

        # ── PHASE: CNN Image Analysis ──────────────────────
        image_result   = {}
        visual_context = ""
        try:
            from src.cnn_image_analyzer import run_image_analysis_pipeline
            image_result   = run_image_analysis_pipeline(
                articles, max_articles=5
            )
            visual_context = image_result.get("visual_context", "")
            n_img = image_result.get("total_images", 0)
            meter.phase_complete("cnn", n_images=n_img)
        except Exception as e:
            meter.error("cnn", f"Image analysis skipped: {str(e)[:40]}")

        # ── PHASE: Chunking ────────────────────────────────
        from src.chunker import chunk_all_articles
        chunks = chunk_all_articles(articles)

        # ── PHASE: Embedding ───────────────────────────────
        from src.embedder import embed_chunks
        embedded = embed_chunks(chunks)
        meter.phase_complete("embedding", n_chunks=len(embedded))

        # ── PHASE: ChromaDB ────────────────────────────────
        from src.vector_store import store_chunks, get_collection_stats
        store_chunks(embedded, reset=True)
        stats = get_collection_stats()
        meter.phase_complete(
            "chromadb",
            extra=(
                f"{stats['total_chunks']} chunks | "
                f"S:{stats['by_stance'].get('Supportive',0)} "
                f"C:{stats['by_stance'].get('Critical',0)} "
                f"N:{stats['by_stance'].get('Neutral',0)}"
            ),
        )

        # ── PHASE: Agent A ─────────────────────────────────
        from src.agent_a import run_agent_a
        agent_a = run_agent_a(
            topic          = intent["topic"],
            visual_context = visual_context,
            nlp_context    = nlp_context,
        )
        meter.phase_complete(
            "agent_a",
            n_args = len(agent_a.get("arguments", [])),
        )

        # ── PHASE: Agent B ─────────────────────────────────
        from src.agent_b import run_agent_b
        agent_b = run_agent_b(
            topic          = intent["topic"],
            visual_context = visual_context,
            nlp_context    = nlp_context,
        )
        meter.phase_complete(
            "agent_b",
            n_counters = len(agent_b.get("counter_arguments", [])),
        )

        # ── PHASE: Agent C ─────────────────────────────────
        from src.agent_c import run_agent_c
        agent_c = run_agent_c(
            topic          = intent["topic"],
            agent_a_output = agent_a,
            agent_b_output = agent_b,
            visual_context = visual_context,
            nlp_context    = nlp_context,
        )
        meter.phase_complete(
            "agent_c",
            bias_score = agent_c.get("bias_score", 0.0),
        )

        # ── PHASE: Save Report ─────────────────────────────
        from src.agents import save_report
        report = {
            "topic":                   intent["topic"],
            "bias_score":              agent_c.get("bias_score", 0.0),
            "synthesis_report":        agent_c.get("synthesis_report", ""),
            "loaded_language_removed": agent_c.get("loaded_language_removed", []),
            "key_agreements":          agent_c.get("key_agreements", []),
            "key_disagreements":       agent_c.get("key_disagreements", []),
            "source_citations":        agent_c.get("source_citations", []),
            "hallucination_flags":     agent_c.get("hallucination_flags", []),
            "agent_a": agent_a,
            "agent_b": agent_b,
            "agent_c": agent_c,
        }
        save_report(report, intent["topic"])

        # Persist run for Heatmap and History pages.
        # These pages read from dedicated stores, not from Streamlit session state.
        try:
            from src.heatmap_manager import HeatmapManager
            hm = HeatmapManager()
            hm.record_run(
                topic    = intent["topic"],
                articles = articles,
                report   = report,
            )
        except Exception as e:
            meter.error("saving", f"Heatmap save skipped: {str(e)[:40]}")

        try:
            from src.history_tracker import save_run
            run_id = save_run(
                pipeline_result = {
                    "report":       report,
                    "articles":     articles,
                    "topic":        intent["topic"],
                    "stats":        stats,
                    "nlp_analysis": nlp_results or {},
                },
                elapsed_seconds = float(getattr(meter, "elapsed_seconds", 0.0) or 0.0),
                is_demo         = bool(st.session_state.get("is_demo", False)),
            )
            if not run_id:
                meter.error("saving", "History save skipped")
        except Exception as e:
            meter.error("saving", f"History save skipped: {str(e)[:40]}")

        result = {
            "report":        report,
            "articles":      articles,
            "stats":         stats,
            "topic":         intent["topic"],
            "intent":        intent,
            "nlp_analysis":  nlp_results,
            "image_analysis":image_result,
        }

        # Save to regular cache too
        from src.cache_manager import save_to_cache
        save_to_cache(intent["topic"], result)

        meter.phase_complete("saving")

        # ── FINISH ─────────────────────────────────────────
        meter.finish(bias_score=report["bias_score"])

        return result

    except Exception as e:
        meter.error("start", str(e)[:60])
        raise
 
 
# ── Render ────────────────────────────────────────────────────────────────────
def render_results(results: dict, from_cache: bool = False):
    report   = results["report"]
    articles = results["articles"]
    stats    = results.get("stats", {})
    topic    = results["topic"]
    agent_a  = report.get("agent_a", {})
    agent_b  = report.get("agent_b", {})
    bias     = report.get("bias_score", 0.0)
 
    # Cache notice
    if from_cache:
        st.markdown(
            '<div class="orb-cache">⚡ &nbsp;Loaded from cache</div>',
            unsafe_allow_html=True,
        )
 
    # ── Metrics ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(5)
    metrics = [
        (len(articles),                             "Articles"),
        (stats.get("total_chunks", 0),              "Chunks"),
        (len(agent_a.get("arguments", [])),         "Pro Args"),
        (len(agent_b.get("counter_arguments", [])), "Counter Args"),
        (len(report.get("source_citations", [])),   "Sources Cited"),
    ]
    for col, (val, lbl) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="orb-metric">'
                f'<div class="orb-metric-val">{val}</div>'
                f'<div class="orb-metric-label">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
 
    # ── Bias Spectrum ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("◈", "Bias Spectrum Analysis")
 
    col_graph, col_score = st.columns([3, 1])
 
    with col_graph:
        fig = build_bias_spectrum_graph(
            articles=articles, bias_score=bias, topic=topic
        )
        st.plotly_chart(fig, use_container_width=True)
 
    with col_score:
        # Bias score block
        if bias < -0.3:
            num_color  = "#3ec97e"
            bias_label = "LEANS SUPPORTIVE"
            fill_left  = 50 - abs(bias) * 40
            fill_width = abs(bias) * 40
            fill_color = "#3ec97e"
        elif bias > 0.3:
            num_color  = "#e05252"
            bias_label = "LEANS CRITICAL"
            fill_left  = 50
            fill_width = abs(bias) * 40
            fill_color = "#e05252"
        else:
            num_color  = "#5b9cf6"
            bias_label = "BALANCED"
            fill_left  = 45
            fill_width = 10
            fill_color = "#5b9cf6"
 
        st.markdown(
            f'<div class="orb-bias-block">'
            f'<div class="orb-bias-num" style="color:{num_color}">'
            f'{bias:+.2f}</div>'
            f'<div class="orb-bias-bar">'
            f'<div class="orb-bias-fill" style="'
            f'left:{fill_left}%;width:{fill_width}%;'
            f'background:{fill_color};"></div>'
            f'</div>'
            f'<div class="orb-bias-label">{bias_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        fig2 = build_confidence_gauge(
            a_score=agent_a.get("confidence_score", 0.0),
            b_score=agent_b.get("confidence_score", 0.0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Multi-Dimensional Bias Vector ─────────────────────────────
    bias_vector = report.get("bias_vector", {})

    if bias_vector:
        st.markdown("<br>", unsafe_allow_html=True)
        _section("◈", "Multi-Dimensional Bias Analysis")

        from src.ui.charts import (
            build_bias_radar_chart,
            build_bias_breakdown_bars,
        )

        col_radar, col_bars, col_labels = st.columns([2, 2, 1])

        with col_radar:
            fig_radar = build_bias_radar_chart(bias_vector)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_bars:
            fig_bars = build_bias_breakdown_bars(bias_vector)
            st.plotly_chart(fig_bars, use_container_width=True)

        with col_labels:
            st.markdown("<br>", unsafe_allow_html=True)
            dim_labels = bias_vector.get("dimension_labels", {})

            for dim, label in dim_labels.items():
                # Choose color based on label content.
                if "Low" in label or "Factual" in label or ("High" in label and "Diversity" in label):
                    color = "#3ec97e"
                elif "High" in label or "Opinion" in label:
                    color = "#e05252"
                else:
                    color = "#c9a84c"

                st.markdown(
                    f'<div style="margin-bottom:0.8rem">'
                    f'<div style="font-family:\'DM Mono\',monospace;'
                    f'font-size:0.62rem;letter-spacing:1px;'
                    f'color:var(--text-3,#5c6b82);text-transform:uppercase;'
                    f'margin-bottom:0.2rem">{dim}</div>'
                    f'<div style="font-size:0.78rem;color:{color};'
                    f'font-weight:500">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    col_donut, col_words = st.columns(2)
    with col_donut:
        st.plotly_chart(
            build_stance_distribution_chart(articles),
            use_container_width=True,
        )
    with col_words:
        st.plotly_chart(
            build_word_count_chart(articles),
            use_container_width=True,
        )

    # ── Image Analysis Method Badge ──────────────────────────────
    image_analysis = results.get("image_analysis", {})
    if image_analysis:
        cnn_available = image_analysis.get("cnn_available", False)
        summary = image_analysis.get("summary", {})

        # Show CNN badge
        method_badge = (
            "CNN (ResNet-50) + Color Analysis"
            if cnn_available
            else "Color Analysis Only"
        )
        st.caption(f"Image Analysis Method: {method_badge}")

        # ── NLP Analysis Section ───────────────────────────────────────
    # This displays results from the manual NLP analyzer
    # (VADER sentiment, spaCy entities, TF-IDF keywords)
    nlp_analysis = results.get("nlp_analysis", {})

    if nlp_analysis and isinstance(nlp_analysis, dict):
        st.markdown("<br>", unsafe_allow_html=True)
        _section("◎", "Manual NLP Analysis — Independent AI Validation")

        st.caption(
            "Results computed using VADER sentiment analysis, "
            "spaCy Named Entity Recognition, and TF-IDF keyword "
            "extraction — no Gemini AI involved in this section."
        )

        # ── NLP Summary Metrics ───────────────────────────────────
        sentiment_summary = nlp_analysis.get("sentiment_summary", {})
        manual_bias_data  = nlp_analysis.get("manual_bias",       {})
        validation_data   = nlp_analysis.get("gemini_validation",  {})

        # Show 4 quick metric cards
        m1, m2, m3, m4 = st.columns(4)

        avg_compound = sentiment_summary.get("avg_compound", 0.0)
        dist         = sentiment_summary.get("distribution", {})
        manual_score = manual_bias_data.get("manual_bias_score", 0.0)
        agreement    = validation_data.get("agreement_level", "N/A")
        agreement_score = validation_data.get("agreement_score", 0.0)

        # Color for avg compound
        compound_color = (
            "#3ec97e" if avg_compound > 0.05
            else "#e05252" if avg_compound < -0.05
            else "#5b9cf6"
        )

        with m1:
            st.markdown(
                f'<div class="orb-metric">'
                f'<div class="orb-metric-val" '
                f'style="color:{compound_color}">'
                f'{avg_compound:+.3f}</div>'
                f'<div class="orb-metric-label">Avg VADER Score</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with m2:
            manual_color = (
                "#3ec97e" if manual_score < -0.1
                else "#e05252" if manual_score > 0.1
                else "#5b9cf6"
            )
            st.markdown(
                f'<div class="orb-metric">'
                f'<div class="orb-metric-val" '
                f'style="color:{manual_color}">'
                f'{manual_score:+.3f}</div>'
                f'<div class="orb-metric-label">Manual Bias Score</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with m3:
            pos = dist.get("positive", 0)
            neg = dist.get("negative", 0)
            neu = dist.get("neutral",  0)
            st.markdown(
                f'<div class="orb-metric">'
                f'<div class="orb-metric-val" '
                f'style="font-size:1.1rem;color:var(--gold)">'
                f'<span style="color:#3ec97e">{pos}↑</span> '
                f'<span style="color:#e05252">{neg}↓</span> '
                f'<span style="color:#5b9cf6">{neu}→</span>'
                f'</div>'
                f'<div class="orb-metric-label">Pos / Neg / Neu</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with m4:
            agree_color = (
                "#3ec97e" if "Strong" in agreement
                else "#c9a84c" if "Moderate" in agreement
                else "#e05252"
            )
            st.markdown(
                f'<div class="orb-metric">'
                f'<div class="orb-metric-val" '
                f'style="color:{agree_color};font-size:1rem">'
                f'{agreement.split()[0] if agreement != "N/A" else "N/A"}'
                f'</div>'
                f'<div class="orb-metric-label">NLP vs AI Agreement</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Validation Banner ─────────────────────────────────────
        if validation_data:
            val_note = validation_data.get("validation_note", "")
            diff     = validation_data.get("absolute_diff", 0)
            dir_ok   = validation_data.get("direction_agrees", True)

            if "Strong" in agreement:
                banner_color = "rgba(62,201,126,0.1)"
                border_color = "rgba(62,201,126,0.3)"
                icon         = "✓"
                text_color   = "#3ec97e"
            elif "Moderate" in agreement:
                banner_color = "rgba(201,168,76,0.1)"
                border_color = "rgba(201,168,76,0.3)"
                icon         = "~"
                text_color   = "#c9a84c"
            else:
                banner_color = "rgba(224,82,82,0.1)"
                border_color = "rgba(224,82,82,0.3)"
                icon         = "!"
                text_color   = "#e05252"

            st.markdown(
                f'<div style="background:{banner_color};'
                f'border:1px solid {border_color};'
                f'border-radius:8px;padding:0.8rem 1.2rem;'
                f'margin:0.8rem 0;font-family:\'DM Mono\',monospace;'
                f'font-size:0.82rem;color:{text_color}">'
                f'<b>{icon} Validation:</b> {val_note} &nbsp;|&nbsp; '
                f'Absolute difference: {diff:.4f} &nbsp;|&nbsp; '
                f'Direction: {"agrees ✓" if dir_ok else "disagrees ✗"}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Chart Row 1: Sentiment Bar + Timeline ─────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        per_article = nlp_analysis.get("per_article_sentiment", [])

        col_sent, col_timeline = st.columns(2)

        with col_sent:
            if per_article:
                fig_sent = build_sentiment_bar_chart(
                    per_article_sentiment = per_article,
                    topic                 = topic,
                )
                st.plotly_chart(
                    fig_sent,
                    use_container_width = True,
                )
            else:
                st.plotly_chart(
                    _empty_chart("Sentiment data not available"),
                    use_container_width=True,
                )

        with col_timeline:
            if per_article:
                fig_timeline = build_sentiment_timeline_chart(
                    per_article_sentiment = per_article,
                    topic                 = topic,
                )
                st.plotly_chart(
                    fig_timeline,
                    use_container_width = True,
                )
            else:
                st.plotly_chart(
                    _empty_chart("Sentiment timeline not available"),
                    use_container_width=True,
                )

        # ── Chart Row 2: Word Cloud + Entity Chart ────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        keyword_analysis = nlp_analysis.get("keyword_analysis", {})
        word_frequencies = keyword_analysis.get("word_frequencies", {})
        entity_analysis  = nlp_analysis.get("entity_analysis",     {})

        col_wc, col_ent = st.columns(2)

        with col_wc:
            st.markdown(
                '<div class="orb-section-title" '
                'style="margin-bottom:0.5rem">'
                '☁ &nbsp;TF-IDF Word Cloud</div>',
                unsafe_allow_html=True,
            )

            if word_frequencies:
                # Try proper word cloud first
                mpl_fig = build_word_cloud_matplotlib(
                    word_frequencies = word_frequencies,
                    topic            = topic,
                )

                if mpl_fig is not None:
                    # Display matplotlib figure in Streamlit
                    st.pyplot(mpl_fig, use_container_width=True)
                    import matplotlib.pyplot as plt
                    plt.close(mpl_fig)     # Free memory
                else:
                    # Fallback to Plotly scatter word cloud
                    fig_wc = build_word_cloud_chart(
                        word_frequencies = word_frequencies,
                        topic            = topic,
                    )
                    st.plotly_chart(
                        fig_wc,
                        use_container_width = True,
                    )
            else:
                st.plotly_chart(
                    _empty_chart("Keyword data not available"),
                    use_container_width=True,
                )

        with col_ent:
            st.markdown(
                '<div class="orb-section-title" '
                'style="margin-bottom:0.5rem">'
                '◉ &nbsp;Named Entity Frequency (spaCy NER)</div>',
                unsafe_allow_html=True,
            )

            if entity_analysis:
                fig_ent = build_entity_frequency_chart(
                    entity_analysis = entity_analysis,
                    topic           = topic,
                )
                st.plotly_chart(
                    fig_ent,
                    use_container_width = True,
                )
            else:
                st.plotly_chart(
                    _empty_chart("Entity data not available"),
                    use_container_width=True,
                )

        # ── Top Keywords Table ────────────────────────────────────
        top_keywords = keyword_analysis.get("top_keywords", [])
        if top_keywords:
            with st.expander(
                f"View all {len(top_keywords)} TF-IDF keywords"
            ):
                for rank, item in enumerate(top_keywords, start=1):
                    word = item.get("word", "?")
                    score = item.get("score", 0.0)
                    kw_color = (
                        "#c9a84c" if rank <= 5
                        else "#9ba8bb" if rank <= 10
                        else "#5c6b82"
                    )
                    st.markdown(
                        f'<div style="font-family:\'DM Mono\','
                        f'monospace;font-size:0.78rem;'
                        f'color:{kw_color};padding:0.2rem 0">'
                        f'#{rank} {word} '
                        f'<span style="color:#5c6b82">'
                        f'({score:.3f})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Top Entities Section ──────────────────────────────────
        top_entities = entity_analysis.get("top_entities", [])
        if top_entities:
            with st.expander(
                f"View all {len(top_entities)} named entities"
            ):
                type_colors = {
                    "PERSON": "#3ec97e",
                    "ORG":    "#c9a84c",
                    "GPE":    "#5b9cf6",
                    "LOC":    "#5b9cf6",
                    "LAW":    "#e05252",
                    "EVENT":  "#9b59b6",
                    "NORP":   "#e67e22",
                }
                cols2 = st.columns(3)
                for i, ent in enumerate(top_entities[:18]):
                    with cols2[i % 3]:
                        ent_color = type_colors.get(
                            ent.get("label", ""), "#9ba8bb"
                        )
                        st.markdown(
                            f'<div style="font-family:\'DM Mono\','
                            f'monospace;font-size:0.78rem;'
                            f'padding:0.2rem 0">'
                            f'<span style="color:{ent_color}">'
                            f'{ent["text"]}</span> '
                            f'<span style="color:#5c6b82">'
                            f'({ent.get("label_name","?")},'
                            f' {ent.get("count",0)}x)</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    elif not nlp_analysis:
        # NLP was not run — show informational message
        st.markdown("<br>", unsafe_allow_html=True)
        _section("◎", "Manual NLP Analysis")
        st.info(
            "Manual NLP analysis not available. "
            "Ensure src/nlp_analyzer.py is installed and "
            "vaderSentiment, spaCy are installed."
        )    
    # ADD after bias spectrum, before argumentative breakdown

    # ── Argumentative Breakdown ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("⊕", "Deep Argumentative Breakdown")
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        a_score = agent_a.get("confidence_score", 0.0)
        _confidence_badge(a_score, "Agent A — Analyst")
        arguments = agent_a.get("arguments", [])
        if arguments:
            for arg in arguments:
                st.markdown(
                    f'<div class="orb-arg pro">'
                    f'<div class="orb-arg-icon">+</div>'
                    f'<div>{arg}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="orb-arg neutral">'
                '<div class="orb-arg-icon">?</div>'
                '<div>No supporting arguments found for this topic.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
 
        evidence_a = agent_a.get("evidence", [])
        if evidence_a:
            with st.expander("Evidence & data points"):
                for ev in evidence_a:
                    st.markdown(
                        f'<div class="orb-lang">{ev}</div>',
                        unsafe_allow_html=True,
                    )
 
    with col_b:
        b_score = agent_b.get("confidence_score", 0.0)
        _confidence_badge(b_score, "Agent B — Critic")
        counters = agent_b.get("counter_arguments", [])
        if counters:
            for arg in counters:
                st.markdown(
                    f'<div class="orb-arg con">'
                    f'<div class="orb-arg-icon">−</div>'
                    f'<div>{arg}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="orb-arg neutral">'
                '<div class="orb-arg-icon">?</div>'
                '<div>No counter-arguments found for this topic.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
 
        evidence_b = agent_b.get("evidence", [])
        if evidence_b:
            with st.expander("Critical evidence"):
                for ev in evidence_b:
                    st.markdown(
                        f'<div class="orb-lang">{ev}</div>',
                        unsafe_allow_html=True,
                    )

    from src.ui.debate_viz import render_debate_board

    st.markdown("<br>", unsafe_allow_html=True)
    _section("⚔", "Agent Debate Board")
    st.caption(
        "Watch the three agents debate — "
        "Agent A argues · Agent B counters · "
        "Agent C arbitrates a neutral verdict"
    )

    render_debate_board(report)
 
    # ── Unbiased Synthesis ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("⚖", "Unbiased 360° Synthesis")
    st.caption(
        "Agent C — every claim cross-referenced against "
        "source excerpts · loaded language removed · hallucination checked"
    )
 
    synthesis = report.get("synthesis_report", "").strip()
    if synthesis and len(synthesis) > 50:
        st.markdown(
            f'<div class="orb-synthesis">'
            f'<div class="orb-synthesis-text">{synthesis}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Synthesis could not be generated for this topic.")
 
    # Agreements / disagreements
    agreements    = report.get("key_agreements", [])
    disagreements = report.get("key_disagreements", [])
    if agreements or disagreements:
        col_ag, col_di = st.columns(2)
        with col_ag:
            if agreements:
                st.markdown(
                    '<div class="orb-section-title" style="margin:1rem 0 0.6rem">'
                    '∩ &nbsp;Points of Agreement</div>',
                    unsafe_allow_html=True,
                )
                for item in agreements:
                    st.markdown(
                        f'<div class="orb-agree">'
                        f'<span class="orb-agree-sym">~</span>'
                        f'{item}</div>',
                        unsafe_allow_html=True,
                    )
        with col_di:
            if disagreements:
                st.markdown(
                    '<div class="orb-section-title" style="margin:1rem 0 0.6rem">'
                    '≠ &nbsp;Core Disagreements</div>',
                    unsafe_allow_html=True,
                )
                for item in disagreements:
                    st.markdown(
                        f'<div class="orb-agree">'
                        f'<span class="orb-agree-sym" style="color:#e05252">×</span>'
                        f'{item}</div>',
                        unsafe_allow_html=True,
                    )
 
    # ── Quality & Hallucination ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("◉", "Analysis Quality")
 
    col_q1, col_q2 = st.columns(2)
 
    with col_q1:
        st.markdown(
            '<div class="orb-section-title" style="margin-bottom:0.7rem">'
            '✦ &nbsp;Loaded Language Neutralised</div>',
            unsafe_allow_html=True,
        )
        removed = report.get("loaded_language_removed", [])
        if removed:
            for phrase in removed:
                st.markdown(
                    f'<div class="orb-lang">{phrase}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="orb-arg neutral" style="font-size:0.83rem">'
                '<div class="orb-arg-icon">✓</div>'
                '<div>No significantly biased language detected.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
 
    with col_q2:
        st.markdown(
            '<div class="orb-section-title" style="margin-bottom:0.7rem">'
            '⚑ &nbsp;Hallucination Check</div>',
            unsafe_allow_html=True,
        )
        flags = report.get("hallucination_flags", [])
        if flags:
            for flag in flags:
                st.markdown(
                    f'<div class="orb-flag">⚑ &nbsp;{flag}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="orb-arg pro" style="font-size:0.83rem">'
                '<div class="orb-arg-icon">✓</div>'
                '<div>All claims verified against source excerpts.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
 
    # ── Source Transparency ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("⊟", f"Source Transparency — {len(articles)} Articles")
 
    stance_pill = {
        "Supportive": '<span class="orb-stance-pill pill-sup">Supportive</span>',
        "Critical":   '<span class="orb-stance-pill pill-crit">Critical</span>',
        "Neutral":    '<span class="orb-stance-pill pill-neu">Neutral</span>',
    }

    from src.source_credibility import (
        get_credibility_badge_html,
        compute_credibility_weighted_bias,
    )

    # Local palette for credibility summary row
    GREEN  = "#3ec97e"
    GOLD   = "#c9a84c"
    TEXT_3 = "#5c6b82"

    # Credibility-weighted bias
    cred_result = compute_credibility_weighted_bias(
        articles, report.get("bias_score", 0.0)
    )
    w_bias = cred_result.get("weighted_bias_score", 0.0)
    m_cred = cred_result.get("mean_credibility", 0.60)
    adj    = cred_result.get("adjustment", 0.0)

    # Show credibility summary
    adj_color = GREEN if abs(adj) < 0.05 else GOLD
    st.markdown(
        f'<div style="display:flex;gap:1.5rem;'
        f'margin-bottom:0.8rem;'
        f'font-family:\'DM Mono\',monospace;font-size:0.7rem">'
        f'<span style="color:{TEXT_3}">Mean Credibility: '
        f'<b style="color:{GOLD}">{m_cred:.2f}</b></span>'
        f'<span style="color:{TEXT_3}">Weighted Bias: '
        f'<b style="color:{adj_color}">{w_bias:+.3f}</b></span>'
        f'<span style="color:{TEXT_3}">Adjustment: '
        f'<b style="color:{adj_color}">{adj:+.3f}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Article rows with credibility badge
    for i, article in enumerate(articles, 1):
        title  = (article.get("title")  or "Unknown")[:70]
        source = (article.get("source") or "Unknown")
        stance = article.get("stance",  "Neutral")
        url    = article.get("url",     "#")
        words  = len((article.get("full_text") or "").split())
        pill   = stance_pill.get(stance, stance_pill["Neutral"])
        badge  = get_credibility_badge_html(source, compact=True)

        col_info, col_stance, col_link = st.columns([6, 2, 1])
        with col_info:
            st.markdown(
                f'<div class="orb-source-title">'
                f'<span style="color:var(--text-3);font-family:\'DM Mono\','
                f'monospace;font-size:0.7rem">{i:02d}&nbsp;</span>'
                f'{title}</div>'
                f'<div class="orb-source-meta">'
                f'{source}&nbsp;·&nbsp;{words:,} words'
                f'&nbsp;&nbsp;{badge}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_stance:
            st.markdown(
                f'<div style="padding-top:0.3rem">{pill}</div>',
                unsafe_allow_html=True,
            )
        with col_link:
            st.link_button("↗", url, use_container_width=True)

        if i < len(articles):
            st.markdown(
                '<hr style="margin:0.3rem 0;'
                'border-top:1px solid var(--border-dim)">',
                unsafe_allow_html=True,
            )
 
    # ── Download ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("◎", "Export Report")

    # Collection name display
    coll_name = st.session_state.get("collection_name", "")
    if coll_name:
        st.markdown(
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.72rem;color:var(--gold,#c9a84c);'
            f'margin-bottom:0.8rem">'
            f'Session: {coll_name}</div>',
            unsafe_allow_html=True,
        )

    col_pdf, col_json = st.columns(2)

    # ── PDF Download ──────────────────────────────────────────────
    with col_pdf:
        st.markdown(
            '<div style="font-family:\'DM Mono\',monospace;'
            'font-size:0.72rem;color:var(--text-3,#5c6b82);'
            'margin-bottom:0.4rem">PDF REPORT</div>',
            unsafe_allow_html=True,
        )

        # Generate PDF button
        if st.button(
            "Generate PDF Report",
            key                 = "gen_pdf_btn",
            use_container_width = True,
        ):
            with st.spinner("Generating PDF..."):
                try:
                    pdf_bytes = generate_pdf_report(
                        pipeline_result = results,
                        collection_name = coll_name,
                    )

                    if pdf_bytes:
                        st.session_state["pdf_bytes"] = pdf_bytes
                        st.success(
                            f"PDF ready! "
                            f"{len(pdf_bytes):,} bytes"
                        )
                    else:
                        st.error(
                            "PDF generation failed. "
                            "Install fpdf2: pip install fpdf2"
                        )
                except Exception as e:
                    st.error(f"PDF error: {e}")

        # Download button appears after generation
        pdf_bytes = st.session_state.get("pdf_bytes")
        if pdf_bytes:
            pdf_filename = get_pdf_filename(topic, coll_name)
            st.download_button(
                label               = "⬇  Download PDF",
                data                = pdf_bytes,
                file_name           = pdf_filename,
                mime                = "application/pdf",
                use_container_width = True,
                key                 = "download_pdf_btn",
            )

    # ── JSON Download ─────────────────────────────────────────────
    with col_json:
        st.markdown(
            '<div style="font-family:\'DM Mono\',monospace;'
            'font-size:0.72rem;color:var(--text-3,#5c6b82);'
            'margin-bottom:0.4rem">JSON DATA</div>',
            unsafe_allow_html=True,
        )

        clean = {
            k: v for k, v in report.items()
            if k not in ("agent_a", "agent_b", "agent_c")
        }
        clean["supporting_arguments"] = agent_a.get("arguments",         [])
        clean["counter_arguments"]    = agent_b.get("counter_arguments", [])

        st.download_button(
            label               = "⬇  Download JSON",
            data                = json.dumps(
                report, indent=2, ensure_ascii=False
            ),
            file_name           = (
                f"orbita_{topic[:25].replace(' ', '_')}.json"
            ),
            mime                = "application/json",
            use_container_width = True,
            key                 = "download_json_btn",
        )
 
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="padding:1rem 0 0.5rem">'
            '<div style="font-family:\'Playfair Display\',serif;'
            'font-size:1.5rem;font-weight:900;'
            'color:#f0ebe0;letter-spacing:4px">ORBITA</div>'
            '<div style="font-family:\'DM Mono\',monospace;'
            'font-size:0.6rem;letter-spacing:2px;'
            'color:#5c6b82;text-transform:uppercase;margin-top:0.2rem">'
            'Bias Analysis Engine</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<hr style="border-top:1px solid rgba(201,168,76,0.18);margin:0.5rem 0">',
            unsafe_allow_html=True,
        )
 
        # History
        cached = list_cached_topics()
        if cached:
            st.markdown(
                '<div style="font-family:\'DM Mono\',monospace;'
                'font-size:0.65rem;letter-spacing:2px;'
                'color:var(--gold,#c9a84c);text-transform:uppercase;'
                'margin-bottom:0.6rem">Recent Analyses</div>',
                unsafe_allow_html=True,
            )
            for item in cached:
                bias   = item["bias_score"]
                symbol = "↙" if bias < -0.2 else "↗" if bias > 0.2 else "→"
                color  = "#3ec97e" if bias < -0.2 else "#e05252" if bias > 0.2 else "#5b9cf6"
                age    = item["age_mins"]
                label  = item["topic"][:26] + ("…" if len(item["topic"]) > 26 else "")
 
                if st.button(
                    f"{label}",
                    key=f"h_{item['cache_key']}",
                    use_container_width=True,
                ):
                    cached_result = get_cached_result(item["topic"])
                    if cached_result:
                        st.session_state.results    = cached_result
                        st.session_state.from_cache = True
                        st.session_state.last_topic = item["topic"]
                        st.session_state.is_demo    = False
                        st.rerun()
 
                st.markdown(
                    f'<div style="font-family:\'DM Mono\',monospace;'
                    f'font-size:0.62rem;color:{color};'
                    f'margin:-0.5rem 0 0.4rem 0.2rem">'
                    f'{symbol} {bias:+.2f} &nbsp;·&nbsp; '
                    f'{item["n_articles"]} articles &nbsp;·&nbsp; '
                    f'{age}m ago</div>',
                    unsafe_allow_html=True,
                )
 
            st.markdown("")
            if st.button("Clear History", use_container_width=True):
                n = clear_cache()
                st.success(f"Cleared {n} item(s).")
                st.rerun()
 
            st.markdown(
                '<hr style="border-top:1px solid rgba(255,255,255,0.06);'
                'margin:0.8rem 0">',
                unsafe_allow_html=True,
            )

        # Demo Mode
        demo_on = st.toggle(
            "⚡  Demo Mode",
            value = st.session_state.get("demo_mode", False),
            help  = (
                "Load pre-cached results instantly. "
                "Perfect for presentations."
            ),
        )
        st.session_state["demo_mode"] = demo_on
        st.session_state["is_demo"] = demo_on

        if demo_on:
            _render_demo_panel()

            st.markdown(
                '<hr style="border-top:1px solid rgba(255,255,255,0.06);'
                'margin:0.8rem 0">',
                unsafe_allow_html=True,
            )
 
        
        with st.expander("How it works"):
            st.markdown(
                '<div style="font-size:0.83rem;line-height:1.7;'
                'color:#9ba8bb">'
                '<b style="color:#c9a84c">Pipeline</b><br>'
                '① spaCy NER → intent<br>'
                '② NewsAPI → diverse articles<br>'
                '③ Zero-shot stance labelling<br>'
                '④ newspaper4k → full text<br>'
                '⑤ Gemini embeddings + ChromaDB<br>'
                '⑥ Agent A (Analyst) RAG<br>'
                '⑦ Agent B (Critic) RAG<br>'
                '⑧ Agent C synthesis + hallucination check<br><br>'
                '<b style="color:#c9a84c">Bias Score</b><br>'
                '−1.0 = fully supportive<br>'
                '&nbsp;0.0 = balanced<br>'
                '+1.0 = fully critical'
                '</div>',
                unsafe_allow_html=True,
            )
 
        st.markdown(
            '<hr style="border-top:1px solid rgba(255,255,255,0.06);'
            'margin:0.8rem 0">'
            '<div style="font-family:\'DM Mono\',monospace;'
            'font-size:0.6rem;color:#3d4d60;line-height:1.8">'
            'B.Tech 6th Sem · AIML 2026<br>'
            'JNGEC Sunder Nagar, Mandi, HP'
            '</div>',
            unsafe_allow_html=True,
        )
 
def render_cot_sidebar(results: dict) -> None:
    """
    Render the Chain of Thought reasoning panel.
    
    Displayed as a collapsible sidebar with a timeline UI.
    Each step shows the reasoning behind ORBITA's decisions.
    """
    report = results.get("report", {})
    chain  = report.get("chain_of_thought", [])

    if not chain:
        return

    with st.sidebar:
        st.markdown(
            '<div style="margin-top:1rem">'
            '<div style="font-family:\'Playfair Display\',serif;'
            'font-size:1.1rem;font-weight:900;color:#f0ebe0;'
            'letter-spacing:2px">Chain of Thought</div>'
            '<div style="font-family:\'DM Mono\',monospace;'
            'font-size:0.58rem;letter-spacing:2px;'
            'color:#5c6b82;text-transform:uppercase;'
            'margin-top:0.2rem">'
            f'{len(chain)} reasoning steps'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<hr style="border-top:1px solid rgba(201,168,76,0.2);'
            'margin:0.6rem 0">',
            unsafe_allow_html=True,
        )

        # Summary metrics
        cot_summary = report.get("cot_summary", {})
        if cot_summary:
            breakdown = cot_summary.get("step_breakdown", {})
            cols = st.columns(2)

            metric_pairs = list(breakdown.items())[:4]
            for i, (stype, count) in enumerate(metric_pairs):
                with cols[i % 2]:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.03);'
                        f'border:1px solid rgba(255,255,255,0.06);'
                        f'border-radius:6px;padding:0.4rem 0.5rem;'
                        f'margin-bottom:0.4rem;text-align:center">'
                        f'<div style="font-family:\'Playfair Display\','
                        f'serif;font-size:1.2rem;color:#c9a84c">'
                        f'{count}</div>'
                        f'<div style="font-family:\'DM Mono\',monospace;'
                        f'font-size:0.58rem;color:#5c6b82;'
                        f'text-transform:uppercase">{stype[:10]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown(
            '<hr style="border-top:1px solid rgba(255,255,255,0.06);'
            'margin:0.5rem 0">',
            unsafe_allow_html=True,
        )

        # Step type color map
        COLORS = {
            "pipeline":   "#c9a84c",
            "retrieval":  "#5b9cf6",
            "nlp":        "#9b59b6",
            "sentiment":  "#3ec97e",
            "entity":     "#e67e22",
            "keyword":    "#1abc9c",
            "argument":   "#c9a84c",
            "validation": "#3ec97e",
            "decision":   "#e05252",
            "synthesis":  "#5b9cf6",
            "image":      "#fd79a8",
            "error":      "#e05252",
        }

        ICONS = {
            "pipeline":   "🔄",
            "retrieval":  "🔍",
            "nlp":        "📊",
            "sentiment":  "💭",
            "entity":     "👤",
            "keyword":    "🔑",
            "argument":   "⚖️",
            "validation": "✅",
            "decision":   "🎯",
            "synthesis":  "📝",
            "image":      "🖼️",
            "error":      "⚠️",
        }

        # Render each step as a timeline item
        for i, step in enumerate(chain):
            stype      = step.get("step_type", "pipeline")
            title      = step.get("title",     "Step")
            detail     = step.get("detail",    "")
            evidence   = step.get("evidence",  [])
            confidence = step.get("confidence", 0.0)
            score      = step.get("score")
            agent      = step.get("agent",     "")
            phase      = step.get("phase",     "")
            elapsed    = step.get("elapsed_ms", 0)

            color = COLORS.get(stype, "#9ba8bb")
            icon  = ICONS.get(stype, "•")

            # Timeline connector line
            if i < len(chain) - 1:
                connector_style = (
                    f"border-left:2px solid "
                    f"rgba({_hex_to_rgb(color)},0.3);"
                    f"margin-left:0.8rem;"
                    f"padding-left:0.8rem;"
                    f"padding-bottom:0.3rem"
                )
            else:
                connector_style = ""

            # Step card
            st.markdown(
                f'<div style="{connector_style}">'
                f'<div style="display:flex;align-items:flex-start;'
                f'gap:0.5rem;margin-bottom:0.2rem">'

                # Circle icon
                f'<div style="width:22px;height:22px;'
                f'border-radius:50%;background:{color}22;'
                f'border:1.5px solid {color};'
                f'display:flex;align-items:center;'
                f'justify-content:center;flex-shrink:0;'
                f'font-size:10px">{icon}</div>'

                # Content
                f'<div style="flex:1;min-width:0">'

                # Title
                f'<div style="font-family:\'DM Sans\',sans-serif;'
                f'font-size:0.75rem;font-weight:600;'
                f'color:#f0ebe0;line-height:1.3;'
                f'word-break:break-word">'
                f'{title[:55]}'
                f'{"..." if len(title) > 55 else ""}'
                f'</div>'

                # Meta row
                f'<div style="font-family:\'DM Mono\',monospace;'
                f'font-size:0.60rem;color:#5c6b82;'
                f'margin-top:0.15rem">'
                f'{phase}'
                f'{" · " + agent if agent else ""}'
                f'{" · " + str(elapsed)+"ms" if elapsed else ""}'
                f'</div>'
                f'</div>'
                f'</div>'

                # Detail (expandable via expander below)
                f'</div>',
                unsafe_allow_html=True,
            )

            # Detail expander
            if detail or evidence:
                with st.expander("", expanded=False):
                    if detail:
                        st.markdown(
                            f'<div style="font-family:\'DM Sans\','
                            f'sans-serif;font-size:0.75rem;'
                            f'color:#9ba8bb;line-height:1.6;'
                            f'white-space:pre-line">'
                            f'{detail[:400]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    if evidence:
                        st.markdown(
                            '<div style="margin-top:0.5rem">',
                            unsafe_allow_html=True,
                        )
                        for ev in evidence[:5]:
                            st.markdown(
                                f'<div style="font-family:\'DM Mono\','
                                f'monospace;font-size:0.68rem;'
                                f'color:{color};padding:0.15rem 0;'
                                f'border-left:2px solid {color}44;'
                                f'padding-left:0.4rem;'
                                f'margin-bottom:0.2rem">'
                                f'{str(ev)[:80]}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        st.markdown("</div>", unsafe_allow_html=True)

                    if confidence > 0:
                        st.markdown(
                            f'<div style="font-family:\'DM Mono\','
                            f'monospace;font-size:0.65rem;'
                            f'color:#5c6b82;margin-top:0.4rem">'
                            f'Confidence: '
                            f'<span style="color:{color}">'
                            f'{confidence:.2f}</span>'
                            f'{" | Score: " + f"{score:+.4f}" if score is not None else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # Download CoT JSON
        st.markdown(
            '<hr style="border-top:1px solid rgba(255,255,255,0.06);'
            'margin:0.8rem 0">',
            unsafe_allow_html=True,
        )
        st.download_button(
            label               = "⬇ Download Chain of Thought",
            data                = json.dumps(
                {"chain": chain, "summary": cot_summary},
                indent=2, ensure_ascii=False,
            ),
            file_name           = (
                f"orbita_cot_"
                f"{results.get('topic','')[:20].replace(' ','_')}"
                f".json"
            ),
            mime                = "application/json",
            use_container_width = True,
            key                 = "download_cot_btn",
        )
def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to 'R,G,B' string for CSS rgba()."""
    hex_color = hex_color.lstrip("#")
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"{r},{g},{b}"
    except Exception:
        return "201,168,76"        

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session()
    render_sidebar()

    # Home-first flow: show a landing screen before the analysis UI.
    if st.session_state.get("show_home", True):
        st.markdown(
            '<div class="orb-header">'
            '<div class="orb-wordmark">ORB<span>I</span>TA</div>'
            '<div class="orb-rule"></div>'
            '<div class="orb-tagline">'
            'Objective Reasoning &amp; Bias Interpretation Tool for Analysis'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="orb-empty">'
            '<div class="orb-empty-glyph">🏠</div>'
            '<div class="orb-empty-title">Welcome to ORBITA</div>'
            '<div class="orb-empty-sub">'
            'Start from this home page, then open the analysis workspace '
            'when you are ready to run a topic.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([1, 1.4, 1])
        with c2:
            if st.button(
                "Open Analysis Workspace",
                type="primary",
                use_container_width=True,
                key="open_analysis_workspace_btn",
            ):
                st.session_state["show_home"] = False
                st.rerun()

        return
 
    # Header
    st.markdown(
        '<div class="orb-header">'
        '<div class="orb-wordmark">ORB<span>I</span>TA</div>'
        '<div class="orb-rule"></div>'
        '<div class="orb-tagline">'
        'Objective Reasoning &amp; Bias Interpretation Tool for Analysis'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Demo Mode banner
    if st.session_state.get("demo_mode", False):
        dm    = DemoManager()
        stats = dm.get_stats()
        st.markdown(
            f'<div style="'
            f'background:rgba(201,168,76,0.08);'
            f'border:1px solid rgba(201,168,76,0.3);'
            f'border-radius:10px;'
            f'padding:0.6rem 1.2rem;'
            f'margin-bottom:0.8rem;'
            f'display:flex;align-items:center;gap:0.8rem">'

            f'<span style="font-size:1rem">⚡</span>'

            f'<div style="flex:1">'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.7rem;letter-spacing:1.5px;'
            f'color:#c9a84c;text-transform:uppercase">'
            f'Demo Mode Active</div>'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.62rem;color:#5c6b82;margin-top:0.1rem">'
            f'{stats["total_cached"]} pre-loaded topics available '
            f'— select from sidebar for instant results'
            f'</div>'
            f'</div>'

            f'</div>',
            unsafe_allow_html=True,
        )
 
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Only show search when NOT in demo mode ─────────────
    _demo_active = st.session_state.get("demo_mode", False)

    if not _demo_active:
        # ── 3-column search row (topic | session | button) ─
        col_in, col_session, col_btn = st.columns([4, 2, 1])
        with col_in:
            user_input = st.text_input(
                label            = "topic",
                placeholder      = "Enter a news topic — e.g.  Farm Laws India",
                label_visibility = "collapsed",
                key              = "topic_input",
            )
        with col_session:
            collection_name = st.text_input(
                label            = "session",
                placeholder      = "Session name (optional)",
                label_visibility = "collapsed",
                key              = "collection_input",
            )
        with col_btn:
            clicked = st.button(
                "Analyse →",
                type                = "primary",
                use_container_width = True,
                disabled            = st.session_state.running,
            )

        # Chips — only shown when NOT in demo mode
        st.markdown(
            '<div class="orb-chips">'
            '<div class="orb-chip">Farm Laws India</div>'
            '<div class="orb-chip">Crypto Regulation</div>'
            '<div class="orb-chip">AI Policy India</div>'
            '<div class="orb-chip">UPI Digital Payments</div>'
            '<div class="orb-chip">Electric Vehicles India</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Demo mode active — no search, no chips
        # Variables must still exist to avoid NameError below
        user_input      = ""
        collection_name = ""
        clicked         = False
 
    # Run pipeline on click
    if clicked and user_input.strip():
        topic = user_input.strip()
        # ADD: save collection name to session state
        st.session_state.collection_name = (
            collection_name.strip()
            if collection_name and collection_name.strip()
            else ""
        )
 
        # Check cache first
        cached = get_cached_result(topic)
        if cached:
            try:
                from src.heatmap_manager import HeatmapManager
                from src.history_tracker import save_run

                hm = HeatmapManager()
                hm.record_run(
                    topic    = cached.get("topic", topic),
                    articles = cached.get("articles", []),
                    report   = cached.get("report", {}),
                )

                save_run(
                    pipeline_result = {
                        "report":       cached.get("report", {}),
                        "articles":     cached.get("articles", []),
                        "topic":        cached.get("topic", topic),
                        "stats":        cached.get("stats", {}),
                        "nlp_analysis": cached.get("nlp_analysis", {}),
                    },
                    elapsed_seconds = 0.0,
                    is_demo         = False,
                )
            except Exception:
                pass

            st.session_state.results    = cached
            st.session_state.from_cache = True
            st.session_state.last_topic = topic
            st.session_state.is_demo    = False
            st.session_state.error      = None
            st.rerun()
 
        st.session_state.running    = True
        st.session_state.results    = None
        st.session_state.error      = None
        st.session_state.from_cache = False
        st.session_state.last_topic = topic
        st.session_state.is_demo    = False
        st.session_state.pdf_bytes       = None
 
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="orb-analysing">'
            f'<div class="orb-analysing-sub">Analysing</div>'
            f'<div class="orb-analysing-topic">{topic}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "This takes 2–4 minutes on the free Gemini tier. "
            "Do not close this tab."
        )
 
        try:
            results = run_pipeline_with_progress(topic)
            st.session_state.results = results
            st.session_state.running = False
            st.rerun()
        except Exception as e:
            st.session_state.error   = str(e)
            st.session_state.running = False
            st.error(f"**Pipeline failed:** {e}")
 
    elif clicked and not user_input.strip():
        st.warning("Please enter a topic before clicking Analyse.")
 
    # Show error state
    if st.session_state.error and not st.session_state.running:
        st.markdown("<br>", unsafe_allow_html=True)
        st.error(f"**Last run failed:** {st.session_state.error}")
        if st.button("Dismiss"):
            st.session_state.error = None
            st.rerun()
 
    # Show results
    if st.session_state.results:
        st.markdown("<br>", unsafe_allow_html=True)

        # Demo badge on results
        if st.session_state.get("is_demo") and st.session_state.get("results"):
            topic_name = st.session_state.get("last_topic", "")
            dm         = DemoManager()
            config     = DEMO_TOPICS.get(topic_name, {})

            st.markdown(
                f'<div style="'
                f'background:rgba(91,156,246,0.08);'
                f'border:1px solid rgba(91,156,246,0.25);'
                f'border-radius:8px;padding:0.5rem 1rem;'
                f'margin-bottom:0.6rem;'
                f'display:flex;align-items:center;gap:0.6rem">'
                f'<span style="color:#5b9cf6;font-size:0.9rem">'
                f'{config.get("icon","◉")}</span>'
                f'<div style="font-family:\'DM Mono\',monospace;'
                f'font-size:0.68rem">'
                f'<span style="color:#9ba8bb">Demo — </span>'
                f'<span style="color:#f0ebe0">{topic_name}</span>'
                f'<span style="color:#5c6b82"> · '
                f'{config.get("description","")[:45]}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        render_results(
            st.session_state.results,
            from_cache=st.session_state.from_cache,
        )
        render_cot_sidebar(st.session_state.results)
    
    if st.session_state.last_topic:
        timeline_fig = build_bias_timeline(st.session_state.last_topic)

        if timeline_fig:

            st.markdown(
                "### 📈 Bias Over Time"
            )

            st.plotly_chart(
                timeline_fig,
                use_container_width=True
            )
    # Empty state
    elif not st.session_state.running and not st.session_state.error:
        st.markdown(
            '<div class="orb-empty">'
            '<div class="orb-empty-glyph">🔭</div>'
            '<div class="orb-empty-title">Enter a topic to begin</div>'
            '<div class="orb-empty-sub">'
            'ORBITA retrieves diverse news sources across the ideological '
            'spectrum, mines arguments from all sides, and produces an '
            'unbiased 360° synthesis with a quantified bias score.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
 
 
if __name__ == "__main__":
    main()