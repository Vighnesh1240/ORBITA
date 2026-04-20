# pages/3_Heatmap.py
"""
ORBITA Bias Heatmap — Sources × Topics Matrix
"""

import sys
import os
import streamlit as st
import json

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    def get_script_run_ctx():
        return None

if __name__ == "__main__" and get_script_run_ctx() is None:
    print(
        "This page must be run with: streamlit run pages/3_Heatmap.py"
    )
    raise SystemExit(0)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title = "ORBITA — Heatmap",
    page_icon  = "🗺️",
    layout     = "wide",
)

def _load_css():
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()

from src.heatmap_manager          import HeatmapManager
from src.ui.comparison_charts     import build_bias_heatmap_chart
from src.source_credibility       import (
    get_source_info,
    get_credibility_badge_html,
    SOURCE_DATABASE,
)

GOLD      = "#c9a84c"
GREEN     = "#3ec97e"
RED       = "#e05252"
BLUE      = "#5b9cf6"
TEXT_1    = "#f0ebe0"
TEXT_2    = "#9ba8bb"
TEXT_3    = "#5c6b82"
NAVY_CARD = "#141c2e"
BORDER    = "rgba(201,168,76,0.18)"


def _section(icon, title):
    st.markdown(
        f'<div class="orb-section">'
        f'<div class="orb-section-line"></div>'
        f'<div class="orb-section-title">{icon}&nbsp; {title}</div>'
        f'<div class="orb-section-line"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────
st.markdown(
    '<div style="padding:1.5rem 0 1rem;text-align:center">'
    '<div style="font-family:\'Playfair Display\',serif;'
    'font-size:2.8rem;font-weight:900;color:#f0ebe0;'
    'letter-spacing:8px">ORB<span style="color:#c9a84c">I</span>TA</div>'
    '<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
    'letter-spacing:3px;color:#5c6b82;text-transform:uppercase;'
    'margin-top:0.3rem">Bias Heatmap — Sources × Topics Matrix</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Heatmap Data ──────────────────────────────────────────────
hm      = HeatmapManager()
stats   = hm.get_stats()
matrix  = hm.get_matrix(max_topics=10, max_sources=14)

# Stats bar
s1, s2, s3 = st.columns(3)
for col, val, label in [
    (s1, stats["n_topics"],  "Topics Analyzed"),
    (s2, stats["n_sources"], "Sources Tracked"),
    (s3, len(SOURCE_DATABASE), "Sources in Database"),
]:
    with col:
        st.markdown(
            f'<div class="orb-metric">'
            f'<div class="orb-metric-val">{val}</div>'
            f'<div class="orb-metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Heatmap ──────────────────────────────────────────────
_section("🗺", "Source × Topic Bias Matrix")

st.caption(
    "Each cell shows the average bias score for a source on that topic. "
    "Green = supportive framing · Red = critical framing · "
    "White = neutral. Run more topics to fill the matrix."
)

if not matrix.get("has_data"):
    st.markdown(
        '<div style="text-align:center;padding:3rem 2rem;'
        'color:#5c6b82">'
        '<div style="font-size:2.5rem;opacity:0.4;'
        'margin-bottom:1rem">🗺️</div>'
        '<div style="font-family:\'Playfair Display\',serif;'
        'font-size:1.2rem;color:#9ba8bb;margin-bottom:0.5rem">'
        'No heatmap data yet</div>'
        '<div style="font-family:\'DM Mono\',monospace;'
        'font-size:0.8rem;line-height:1.8">'
        'Run analyses on the main page to populate this matrix.<br>'
        'Each analysis automatically records bias scores per source.<br>'
        'After 3+ topics, meaningful patterns will emerge.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.plotly_chart(
        build_bias_heatmap_chart(matrix),
        use_container_width=True,
    )

    # Topic list
    st.markdown("<br>", unsafe_allow_html=True)
    _section("◉", "Topics in Matrix")

    topics = hm.get_topic_list()
    if topics:
        t_cols = st.columns(min(4, len(topics)))
        for i, t in enumerate(topics[:8]):
            with t_cols[i % len(t_cols)]:
                bias  = t.get("last_bias", 0.0)
                color = (
                    GREEN if bias < -0.2
                    else RED if bias > 0.2
                    else BLUE
                )
                st.markdown(
                    f'<div class="orb-metric" style="text-align:left;'
                    f'padding:0.8rem 1rem">'
                    f'<div style="font-family:\'DM Sans\',sans-serif;'
                    f'font-size:0.8rem;color:{TEXT_1};'
                    f'margin-bottom:0.3rem">{t["display"][:28]}</div>'
                    f'<div style="font-family:\'DM Mono\',monospace;'
                    f'font-size:0.9rem;color:{color};font-weight:700">'
                    f'{bias:+.3f}</div>'
                    f'<div class="orb-metric-label">'
                    f'{t["n_runs"]} run(s)</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ── Source Credibility Database ───────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_section("◈", f"Source Credibility Database ({len(SOURCE_DATABASE)} sources)")

st.caption(
    "Based on AllSides, Media Bias / Fact Check, "
    "and Reuters Institute Digital News Report 2023."
)

# Filter controls
search_src = st.text_input(
    "Search sources",
    placeholder      = "Type source name...",
    label_visibility = "collapsed",
)

col_country, col_tier = st.columns(2)
with col_country:
    all_countries = sorted(set(
        v.get("country", "Unknown")
        for v in SOURCE_DATABASE.values()
    ))
    sel_country = st.selectbox(
        "Country",
        ["All"] + all_countries,
    )

with col_tier:
    sel_tier = st.selectbox(
        "Tier",
        ["All Tiers", "Tier 1 (High)", "Tier 2 (Medium)", "Tier 3 (Low)"],
    )

# Filter database
filtered = {}
for name, info in SOURCE_DATABASE.items():
    if search_src and search_src.lower() not in name.lower():
        continue
    if sel_country != "All" and info.get("country") != sel_country:
        continue
    tier = info.get("tier", 3)
    if sel_tier == "Tier 1 (High)"   and tier != 1: continue
    if sel_tier == "Tier 2 (Medium)" and tier != 2: continue
    if sel_tier == "Tier 3 (Low)"    and tier != 3: continue
    filtered[name] = info

st.markdown(
    f'<div style="font-family:\'DM Mono\',monospace;'
    f'font-size:0.62rem;color:{TEXT_3};'
    f'margin-bottom:0.6rem">'
    f'Showing {len(filtered)} sources</div>',
    unsafe_allow_html=True,
)

# Table header
st.markdown(
    f'<div style="display:grid;'
    f'grid-template-columns:2fr 1fr 1fr 1fr 2fr;'
    f'gap:0.5rem;padding:0.4rem 0.8rem;'
    f'font-family:\'DM Mono\',monospace;'
    f'font-size:0.6rem;letter-spacing:1.5px;'
    f'color:{TEXT_3};text-transform:uppercase;'
    f'border-bottom:1px solid rgba(255,255,255,0.08);'
    f'margin-bottom:0.3rem">'
    f'<div>Source</div>'
    f'<div>Credibility</div>'
    f'<div>Lean</div>'
    f'<div>Tier</div>'
    f'<div>Notes</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# Table rows
for name, info in sorted(filtered.items(),
                         key=lambda x: x[1]["credibility"],
                         reverse=True):
    cred  = info.get("credibility", 0.60)
    lean  = info.get("lean_label",  "Unknown")
    tier  = info.get("tier",        3)
    notes = info.get("notes",       "")
    country = info.get("country",   "")

    cred_color = (
        GREEN if cred >= 0.85
        else GOLD if cred >= 0.70
        else RED
    )
    tier_stars = "★" * (4 - tier) + "☆" * (tier - 1)

    st.markdown(
        f'<div style="display:grid;'
        f'grid-template-columns:2fr 1fr 1fr 1fr 2fr;'
        f'gap:0.5rem;padding:0.5rem 0.8rem;'
        f'background:{NAVY_CARD};'
        f'border:1px solid rgba(255,255,255,0.04);'
        f'border-radius:8px;margin-bottom:0.3rem;'
        f'transition:border-color 0.2s">'

        f'<div style="font-family:\'DM Sans\',sans-serif;'
        f'font-size:0.83rem;color:{TEXT_1};font-weight:500">'
        f'{name}'
        f'<span style="font-family:\'DM Mono\',monospace;'
        f'font-size:0.6rem;color:{TEXT_3};margin-left:0.4rem">'
        f'{country}</span>'
        f'</div>'

        f'<div style="font-family:\'DM Mono\',monospace;'
        f'font-size:0.8rem;color:{cred_color};font-weight:600">'
        f'{cred:.2f}</div>'

        f'<div style="font-family:\'DM Mono\',monospace;'
        f'font-size:0.72rem;color:{TEXT_2}">{lean}</div>'

        f'<div style="font-family:\'DM Mono\',monospace;'
        f'font-size:0.72rem;color:{GOLD}">{tier_stars}</div>'

        f'<div style="font-family:\'DM Mono\',monospace;'
        f'font-size:0.68rem;color:{TEXT_3};'
        f'line-height:1.4">{notes[:50]}</div>'

        f'</div>',
        unsafe_allow_html=True,
    )

# ── Clear button ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if matrix.get("has_data"):
    if st.button("🗑️ Clear Heatmap Data", type="secondary"):
        hm.clear()
        st.success("Heatmap data cleared.")
        st.rerun()