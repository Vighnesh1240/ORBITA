import os
import sys
import json
import streamlit as st
from src.ui.timeline_chart import build_bias_timeline
 
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
from src.agents         import save_report
from src.cache_manager  import (
    get_cached_result, save_to_cache,
    list_cached_topics, clear_cache,
)
from src.ui.charts import (
    build_bias_spectrum_graph,
    build_confidence_gauge,
    build_stance_distribution_chart,
    build_word_count_chart,
)
 
 
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
 
 
# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    for key, val in {
        "results":    None,
        "running":    False,
        "last_topic": "",
        "error":      None,
        "from_cache": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val
 
 
# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline_with_progress(user_input: str) -> dict:
    progress   = st.progress(0)
    log_box    = st.empty()
    done_steps = []
 
    def step(msg: str, pct: int, done: bool = False):
        progress.progress(pct)
        if done:
            done_steps.append(f'<span class="log-ok">✓&nbsp; {msg}</span>')
        log_box.markdown(
            '<div class="orb-log">'
            + "<br>".join(done_steps)
            + ("" if done
               else f'<br><span class="log-run">◈&nbsp; {msg}</span>')
            + "</div>",
            unsafe_allow_html=True,
        )
 
    try:
        step("Decoding intent with spaCy NER", 4)
        intent = decode_intent(user_input)
        step(
            f"Intent decoded &mdash; "
            f"{len(intent['search_queries'])} search queries generated",
            8, done=True,
        )
 
        step("Fetching news articles via NewsAPI", 9)
        articles = fetch_articles(intent["search_queries"])
        if not articles:
            raise RuntimeError(
                "No articles found. Try a broader topic or check NEWS_API_KEY."
            )
        step(f"{len(articles)} articles fetched", 20, done=True)
 
        step("Zero-shot stance classification", 21)
        articles = label_all_articles(articles)
        articles = rebalance_articles(articles)
        step("Stances classified and rebalanced", 30, done=True)
 
        step("Scraping full article text via newspaper4k", 31)
        articles = scrape_articles(articles)
        if len(articles) < 3:
            raise RuntimeError(
                f"Only {len(articles)} articles scraped. Try a different topic."
            )
        step(f"{len(articles)} articles scraped successfully", 43, done=True)
 
        step("Deduplication pass", 44)
        articles = deduplicate(articles)
        step(f"{len(articles)} unique articles remain", 49, done=True)
 
        step("Chunking article text", 50)
        chunks = chunk_all_articles(articles)
        step(f"{len(chunks)} text chunks created", 55, done=True)
 
        step("Generating Gemini text embeddings", 56)
        embedded = embed_chunks(chunks)
        step(f"{len(embedded)} embeddings generated", 66, done=True)
 
        step("Persisting to ChromaDB vector store", 67)
        store_chunks(embedded, reset=True)
        stats = get_collection_stats()
        step(
            f"{stats['total_chunks']} chunks stored &nbsp;"
            f"[S:{stats['by_stance'].get('Supportive',0)} &nbsp;"
            f"C:{stats['by_stance'].get('Critical',0)} &nbsp;"
            f"N:{stats['by_stance'].get('Neutral',0)}]",
            72, done=True,
        )
 
        step("Agent A (Analyst) — RAG retrieval + argument extraction", 73)
        agent_a = run_agent_a(intent["topic"])
        step(
            f"Agent A &mdash; "
            f"{len(agent_a.get('arguments',[]))} supporting arguments",
            82, done=True,
        )
 
        step("Agent B (Critic) — RAG retrieval + counter-argument extraction", 83)
        agent_b = run_agent_b(intent["topic"])
        step(
            f"Agent B &mdash; "
            f"{len(agent_b.get('counter_arguments',[]))} counter-arguments",
            89, done=True,
        )
 
        step("Agent C (Arbitrator) — synthesis + hallucination check", 90)
        agent_c = run_agent_c(intent["topic"], agent_a, agent_b)
        synthesis_words = len(
            (agent_c.get("synthesis_report") or "").split()
        )
        step(
            f"Agent C &mdash; {synthesis_words}-word synthesis, "
            f"bias score {agent_c.get('bias_score', 0.0):+.2f}",
            97, done=True,
        )
 
        step("Saving report to disk", 98)
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
        result = {
            "report":   report,
            "articles": articles,
            "stats":    stats,
            "topic":    intent["topic"],
            "intent":   intent,
        }
        save_to_cache(intent["topic"], result)
 
        progress.progress(100)
        log_box.empty()
        progress.empty()
        return result
 
    except Exception as e:
        progress.empty()
        log_box.empty()
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
 
    for i, article in enumerate(articles, 1):
        title  = (article.get("title")  or "Unknown Title")[:75]
        source = (article.get("source") or "Unknown")
        stance = article.get("stance",  "Neutral")
        url    = article.get("url",     "#")
        words  = len((article.get("full_text") or "").split())
        pill   = stance_pill.get(stance, stance_pill["Neutral"])
 
        col_info, col_stance, col_link = st.columns([6, 2, 1])
        with col_info:
            st.markdown(
                f'<div class="orb-source-title">'
                f'<span style="color:var(--text-3);font-family:\'DM Mono\','
                f'monospace;font-size:0.7rem">{i:02d}&nbsp;</span>'
                f'{title}</div>'
                f'<div class="orb-source-meta">'
                f'{source}&nbsp;·&nbsp;{words:,} words</div>',
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
    _section("◎", "Export")
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        clean = {
            k: v for k, v in report.items()
            if k not in ("agent_a", "agent_b", "agent_c")
        }
        clean["supporting_arguments"] = agent_a.get("arguments", [])
        clean["counter_arguments"]    = agent_b.get("counter_arguments", [])
        st.download_button(
            label    = "⬇  Download Full Report (JSON)",
            data     = json.dumps(report, indent=2, ensure_ascii=False),
            file_name = f"orbita_{topic[:25].replace(' ','_')}.json",
            mime     = "application/json",
            use_container_width = True,
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
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session()
    render_sidebar()
 
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
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Search row
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_input = st.text_input(
            label            = "topic",
            placeholder      = "Enter a news topic — e.g.  Farm Laws India",
            label_visibility = "collapsed",
            key              = "topic_input",
        )
    with col_btn:
        clicked = st.button(
            "Analyse →",
            type                = "primary",
            use_container_width = True,
            disabled            = st.session_state.running,
        )
 
    # Example chips (display only)
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
 
    # Run pipeline on click
    if clicked and user_input.strip():
        topic = user_input.strip()
 
        # Check cache first
        cached = get_cached_result(topic)
        if cached:
            st.session_state.results    = cached
            st.session_state.from_cache = True
            st.session_state.last_topic = topic
            st.session_state.error      = None
            st.rerun()
 
        st.session_state.running    = True
        st.session_state.results    = None
        st.session_state.error      = None
        st.session_state.from_cache = False
        st.session_state.last_topic = topic
 
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
        render_results(
            st.session_state.results,
            from_cache=st.session_state.from_cache,
        )
    
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