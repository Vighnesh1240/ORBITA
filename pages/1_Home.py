# pages/1_Home.py
"""
ORBITA Landing Page
Official product homepage shown before analysis.
"""

import sys
import os
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title = "ORBITA — Home",
    page_icon  = "🔭",
    layout     = "wide",
)

def _load_css():
    path = os.path.join(
        os.path.dirname(__file__), "..", "assets", "style.css"
    )
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )

_load_css()

from src.history_tracker import get_database_stats

# ── Colors ────────────────────────────────────────────────────
GOLD      = "#c9a84c"
GREEN     = "#3ec97e"
RED       = "#e05252"
BLUE      = "#5b9cf6"
TEXT_1    = "#f0ebe0"
TEXT_2    = "#9ba8bb"
TEXT_3    = "#5c6b82"
NAVY_CARD = "#141c2e"
BORDER    = "rgba(201,168,76,0.18)"

stats = get_database_stats()

# ─────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;padding:3rem 2rem 2rem">'

    # Big logo
    '<div style="font-family:\'Playfair Display\',serif;'
    'font-size:5.5rem;font-weight:900;color:#f0ebe0;'
    'letter-spacing:18px;line-height:1;'
    'animation:fadeInUp 0.8s ease">'
    'ORB<span style="color:#c9a84c">I</span>TA'
    '</div>'

    # Gold rule
    '<div style="width:100px;height:2px;'
    'background:linear-gradient(90deg,transparent,#c9a84c,transparent);'
    'margin:1.2rem auto 1rem"></div>'

    # Full name
    '<div style="font-family:\'DM Mono\',monospace;'
    'font-size:0.72rem;letter-spacing:3px;'
    'color:#9ba8bb;text-transform:uppercase;'
    'margin-bottom:1.5rem">'
    'Objective Reasoning &amp; Bias Interpretation Tool for Analysis'
    '</div>'

    # Tagline
    '<div style="font-family:\'DM Sans\',sans-serif;'
    'font-size:1.15rem;color:#f0ebe0;'
    'max-width:600px;margin:0 auto 2rem;'
    'line-height:1.7;font-weight:300">'
    'An autonomous multi-agent AI framework that quantifies '
    'ideological bias in news media — combining '
    '<span style="color:#3ec97e">NLP</span>, '
    '<span style="color:#5b9cf6">RAG</span>, '
    '<span style="color:#c9a84c">CNN</span>, and '
    '<span style="color:#e05252">multi-agent debate</span> '
    'into a single pipeline.'
    '</div>'

    # CTA button
    '</div>',
    unsafe_allow_html=True,
)

# CTA
col_l, col_btn, col_r = st.columns([3, 2, 3])
with col_btn:
    if st.button(
        "🔭  Start Analysis →",
        type                = "primary",
        use_container_width = True,
    ):
        st.session_state["entered_from_home"] = True
        st.switch_page("app.py")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LIVE STATS BAR
# ─────────────────────────────────────────────────────────────
n_runs    = stats.get("n_runs",     0)
n_topics  = stats.get("n_topics",   0)
n_sources = stats.get("n_sources",  0)
n_art     = stats.get("n_articles", 0)

live_stats = [
    (str(n_runs)    if n_runs    else "—", "Analyses Run"),
    (str(n_topics)  if n_topics  else "—", "Topics Tracked"),
    (str(n_sources) if n_sources else "—", "Sources Indexed"),
    (str(n_art)     if n_art     else "—", "Articles Processed"),
    ("35+",                                "Credibility Ratings"),
    ("4D",                                 "Bias Dimensions"),
]

cols = st.columns(len(live_stats))
for col, (val, label) in zip(cols, live_stats):
    with col:
        st.markdown(
            f'<div class="orb-metric">'
            f'<div class="orb-metric-val">{val}</div>'
            f'<div class="orb-metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-family:\'DM Mono\',monospace;'
    'font-size:0.65rem;letter-spacing:2.5px;'
    'color:#c9a84c;text-transform:uppercase;'
    'text-align:center;margin-bottom:1.2rem">'
    '◈ &nbsp; How ORBITA Works &nbsp; ◈'
    '</div>',
    unsafe_allow_html=True,
)

pipeline_steps = [
    ("🔬", "Intent Decode",    "spaCy NER extracts topic entities and generates targeted search queries"),
    ("📡", "News Fetch",       "NewsAPI retrieves 10-15 diverse articles across ideological sources"),
    ("🏷️", "Stance Classify", "Zero-shot classifier labels each article: Supportive / Critical / Neutral"),
    ("🧠", "NLP Analysis",     "VADER sentiment + spaCy entities + TF-IDF keywords computed locally"),
    ("🖼️", "CNN Vision",      "ResNet-50 analyzes article images for visual sentiment and framing"),
    ("⚛️",  "Embed + Store",   "Gemini embeddings stored in ChromaDB vector database for semantic RAG"),
    ("⚖️", "Agent Debate",     "Agent A argues · Agent B counters · Agent C synthesizes and arbitrates"),
    ("📊", "Bias Vector",      "4-dimensional bias score: ideological · emotional · informational · diversity"),
]

n_cols = 4
for i in range(0, len(pipeline_steps), n_cols):
    row   = pipeline_steps[i:i + n_cols]
    cols  = st.columns(n_cols)
    for col, (icon, title, desc) in zip(cols, row):
        step_n = i + pipeline_steps.index(
            next(s for s in pipeline_steps if s[1] == title)
        ) + 1 if False else i + 1 + pipeline_steps[i:i+n_cols].index(
            (icon, title, desc)
        )
        with col:
            st.markdown(
                f'<div style="background:{NAVY_CARD};'
                f'border:1px solid rgba(255,255,255,0.05);'
                f'border-radius:10px;padding:1rem;'
                f'margin-bottom:0.6rem;height:130px;'
                f'animation:fadeInUp 0.4s ease both;'
                f'transition:all 0.2s">'

                f'<div style="display:flex;align-items:center;'
                f'gap:0.5rem;margin-bottom:0.5rem">'
                f'<span style="font-size:1.2rem">{icon}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;'
                f'font-size:0.58rem;letter-spacing:2px;'
                f'color:{GOLD};text-transform:uppercase">'
                f'{title}</span>'
                f'</div>'

                f'<div style="font-family:\'DM Sans\',sans-serif;'
                f'font-size:0.78rem;color:{TEXT_2};'
                f'line-height:1.55">'
                f'{desc}'
                f'</div>'

                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FEATURE CARDS
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-family:\'DM Mono\',monospace;'
    'font-size:0.65rem;letter-spacing:2.5px;'
    'color:#c9a84c;text-transform:uppercase;'
    'text-align:center;margin-bottom:1.2rem">'
    '◈ &nbsp; Key Features &nbsp; ◈'
    '</div>',
    unsafe_allow_html=True,
)

features = [
    (GREEN, "⊕",  "Multi-Agent Debate",
     "Three specialized Gemini agents debate each topic — Analyst argues, Critic counters, Arbitrator synthesizes"),
    (BLUE,  "⚛️", "RAG Architecture",
     "ChromaDB vector store enables semantic retrieval — agents ground every claim in source documents"),
    (GOLD,  "📐", "4D Bias Vector",
     "Ideological · Emotional · Informational · Source Diversity — four dimensions, not a single scalar"),
    (RED,   "🖼️", "CNN Visual Analysis",
     "ResNet-50 analyzes article images locally — no API dependency, fully reproducible"),
    (GREEN, "🔍", "NLP Validation",
     "VADER sentiment cross-validates Gemini output — if they disagree by >0.3, a flag is raised"),
    (BLUE,  "📈", "Longitudinal Tracking",
     "SQLite records every run — track how media bias on a topic shifts over days, weeks, months"),
    (GOLD,  "⚖️", "Comparison Mode",
     "Analyze two topics side by side — compare international coverage of the same event"),
    (RED,   "🗺️", "Bias Heatmap",
     "Sources × Topics matrix — see at a glance which sources are biased and on which topics"),
]

for i in range(0, len(features), 4):
    row  = features[i:i + 4]
    cols = st.columns(4)
    for col, (color, icon, title, desc) in zip(cols, row):
        with col:
            st.markdown(
                f'<div style="background:{NAVY_CARD};'
                f'border:1px solid {color}22;'
                f'border-top:3px solid {color};'
                f'border-radius:10px;padding:1.1rem;'
                f'margin-bottom:0.6rem;'
                f'height:150px;'
                f'transition:all 0.2s">'

                f'<div style="font-size:1.3rem;'
                f'margin-bottom:0.4rem">{icon}</div>'

                f'<div style="font-family:\'DM Mono\',monospace;'
                f'font-size:0.62rem;letter-spacing:1.5px;'
                f'color:{color};text-transform:uppercase;'
                f'margin-bottom:0.4rem">{title}</div>'

                f'<div style="font-family:\'DM Sans\',sans-serif;'
                f'font-size:0.75rem;color:{TEXT_3};'
                f'line-height:1.55">{desc}</div>'

                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RESEARCH CONTEXT
# ─────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="background:{NAVY_CARD};'
    f'border:1px solid {BORDER};'
    f'border-radius:14px;padding:2rem;'
    f'max-width:800px;margin:0 auto 2rem">'

    f'<div style="font-family:\'DM Mono\',monospace;'
    f'font-size:0.62rem;letter-spacing:2px;'
    f'color:{GOLD};text-transform:uppercase;'
    f'margin-bottom:0.8rem">Research Context</div>'

    f'<div style="font-family:\'DM Sans\',sans-serif;'
    f'font-size:0.88rem;color:{TEXT_2};line-height:1.8">'
    f'ORBITA addresses a critical gap in media literacy tools: '
    f'existing bias detectors (AllSides, MBFC) produce a single '
    f'left/right score. ORBITA produces a '
    f'<b style="color:{GOLD}">4-dimensional bias vector</b> '
    f'validated by both NLP and LLM, with source credibility '
    f'weighting and visual framing analysis. The multi-agent '
    f'debate architecture ensures every conclusion is reached '
    f'through structured argumentation, not a single model\'s opinion.'
    f'</div>'

    f'<div style="font-family:\'DM Mono\',monospace;'
    f'font-size:0.65rem;color:{TEXT_3};margin-top:1rem">'
    f'B.Tech 6th Sem · AIML 2026 · '
    f'JNGEC Sunder Nagar, Mandi, HP'
    f'</div>'

    f'</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-family:\'DM Mono\',monospace;'
    'font-size:0.65rem;letter-spacing:2.5px;'
    'color:#c9a84c;text-transform:uppercase;'
    'text-align:center;margin-bottom:1rem">'
    '◈ &nbsp; Navigate &nbsp; ◈'
    '</div>',
    unsafe_allow_html=True,
)

nav_cols = st.columns(5)
nav_items = [
    ("🔭", "Analysis",   "app.py",              "Run live analysis"),
    ("⚖️", "Compare",    "pages/2_Compare.py",  "Two topics side by side"),
    ("🗺️", "Heatmap",   "pages/3_Heatmap.py",  "Source × topic matrix"),
    ("📈", "History",    "pages/4_History.py",  "Longitudinal tracking"),
    ("🏠", "Home",       "pages/1_Home.py",     "This page"),
]

for col, (icon, label, page, desc) in zip(nav_cols, nav_items):
    with col:
        st.markdown(
            f'<div style="text-align:center;'
            f'background:{NAVY_CARD};'
            f'border:1px solid rgba(255,255,255,0.05);'
            f'border-radius:10px;padding:1rem 0.5rem;'
            f'margin-bottom:0.4rem">'
            f'<div style="font-size:1.5rem;'
            f'margin-bottom:0.3rem">{icon}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.65rem;color:{GOLD};'
            f'margin-bottom:0.2rem">{label}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.58rem;color:{TEXT_3}">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )