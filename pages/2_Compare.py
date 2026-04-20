# pages/2_Compare.py
"""
ORBITA Compare Mode — Two-Topic Side-by-Side Analysis
Streamlit multi-page app page.
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
        "This page must be run with: streamlit run pages/2_Compare.py"
    )
    raise SystemExit(0)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title = "ORBITA — Compare",
    page_icon  = "⚖️",
    layout     = "wide",
)

# Load CSS
def _load_css():
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()

from src.comparison_engine import run_comparison
from src.demo_manager      import DemoManager, DEMO_TOPICS
from src.ui.comparison_charts import (
    build_bias_comparison_bar,
    build_stance_comparison_chart,
    build_metric_comparison_radar,
)
from src.source_credibility import get_credibility_badge_html

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
COLOR_A   = "#5b9cf6"
COLOR_B   = "#e8c96a"


def _section(icon, title):
    st.markdown(
        f'<div class="orb-section">'
        f'<div class="orb-section-line"></div>'
        f'<div class="orb-section-title">{icon}&nbsp; {title}</div>'
        f'<div class="orb-section-line"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _bias_direction(score):
    if score < -0.2:
        return "Leans Supportive", GREEN
    elif score > 0.2:
        return "Leans Critical", RED
    return "Balanced", BLUE


# ── Header ────────────────────────────────────────────────────
st.markdown(
    '<div style="padding:1.5rem 0 1rem;text-align:center">'
    '<div style="font-family:\'Playfair Display\',serif;'
    'font-size:2.8rem;font-weight:900;color:#f0ebe0;'
    'letter-spacing:8px">ORB<span style="color:#c9a84c">I</span>TA</div>'
    '<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
    'letter-spacing:3px;color:#5c6b82;text-transform:uppercase;'
    'margin-top:0.3rem">Comparison Mode — Two Topics Side by Side</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Topic Input ───────────────────────────────────────────────
st.markdown(
    '<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
    'letter-spacing:2px;color:#c9a84c;text-transform:uppercase;'
    'margin-bottom:0.6rem">Enter Two Topics to Compare</div>',
    unsafe_allow_html=True,
)

col_a, col_vs, col_b, col_btn = st.columns([4, 1, 4, 2])

with col_a:
    topic_a = st.text_input(
        "Topic A",
        placeholder      = "e.g.  India AI Policy",
        label_visibility = "collapsed",
        key              = "cmp_topic_a",
    )

with col_vs:
    st.markdown(
        '<div style="text-align:center;padding-top:0.6rem;'
        'font-family:\'Playfair Display\',serif;font-size:1.4rem;'
        'color:#c9a84c">vs</div>',
        unsafe_allow_html=True,
    )

with col_b:
    topic_b = st.text_input(
        "Topic B",
        placeholder      = "e.g.  USA AI Policy",
        label_visibility = "collapsed",
        key              = "cmp_topic_b",
    )

with col_btn:
    use_cache = st.checkbox("Use Demo Cache", value=True)
    compare_clicked = st.button(
        "Compare →",
        type                = "primary",
        use_container_width = True,
    )

# Quick topic suggestions
dm            = DemoManager()
cached_topics = [t["name"] for t in dm.get_available_topics()]

if cached_topics:
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
        'color:#5c6b82;margin-top:0.4rem">'
        'Pre-cached topics (instant): '
        + " · ".join(
            f'<span style="color:#9ba8bb">{t}</span>'
            for t in cached_topics[:5]
        )
        + '</div>',
        unsafe_allow_html=True,
    )

# ── Run Comparison ────────────────────────────────────────────
if compare_clicked:
    if not topic_a.strip() or not topic_b.strip():
        st.warning("Please enter both topics.")
        st.stop()

    if topic_a.strip().lower() == topic_b.strip().lower():
        st.warning("Topics must be different.")
        st.stop()

    st.session_state["comparison_result"] = None

    with st.status(
        f"Comparing '{topic_a}' vs '{topic_b}'...",
        expanded=True,
    ) as status:
        st.write(f"▶ Analyzing Topic A: {topic_a}")
        st.write(f"▶ Analyzing Topic B: {topic_b}")
        st.write("⚠️ This may take 3-6 minutes without cache...")

        try:
            result = run_comparison(
                topic_a        = topic_a.strip(),
                topic_b        = topic_b.strip(),
                use_demo_cache = use_cache,
            )
            st.session_state["comparison_result"] = result
            status.update(
                label    = "Comparison complete!",
                state    = "complete",
                expanded = False,
            )
        except Exception as e:
            status.update(
                label = f"Failed: {e}",
                state = "error",
            )
            st.error(str(e))
            st.stop()

# ── Display Results ───────────────────────────────────────────
comparison = st.session_state.get("comparison_result")

if not comparison:
    st.markdown(
        '<div style="text-align:center;padding:4rem 2rem;'
        'color:#5c6b82">'
        '<div style="font-size:2.5rem;opacity:0.4;'
        'margin-bottom:1rem">⚖️</div>'
        '<div style="font-family:\'Playfair Display\',serif;'
        'font-size:1.3rem;color:#9ba8bb;margin-bottom:0.5rem">'
        'Enter two topics to compare</div>'
        '<div style="font-family:\'DM Mono\',monospace;'
        'font-size:0.8rem">Side-by-side bias analysis</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# From here comparison dict is available
t_a     = comparison["topic_a"]
t_b     = comparison["topic_b"]
bias_a  = comparison["bias_a"]
bias_b  = comparison["bias_b"]
delta   = comparison["bias_delta"]

dir_a, col_a_c = _bias_direction(bias_a)
dir_b, col_b_c = _bias_direction(bias_b)

# ── KEY INSIGHT BANNER ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f'<div style="background:{NAVY_CARD};'
    f'border:1px solid {GOLD}44;'
    f'border-left:4px solid {GOLD};'
    f'border-radius:12px;padding:1rem 1.4rem;'
    f'margin-bottom:1rem">'
    f'<div style="font-family:\'DM Mono\',monospace;'
    f'font-size:0.62rem;letter-spacing:2px;color:{GOLD};'
    f'text-transform:uppercase;margin-bottom:0.4rem">'
    f'◈ Key Insight</div>'
    f'<div style="font-family:\'DM Sans\',sans-serif;'
    f'font-size:0.92rem;color:{TEXT_1};line-height:1.6">'
    f'{comparison["key_insight"]}'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── TOPIC HEADER CARDS ────────────────────────────────────────
col1, col_mid, col2 = st.columns([5, 1, 5])

for col, topic, bias, direction, color, bg_color in [
    (col1, t_a, bias_a, dir_a, col_a_c, COLOR_A),
    (col2, t_b, bias_b, dir_b, col_b_c, COLOR_B),
]:
    with col:
        st.markdown(
            f'<div style="background:{NAVY_CARD};'
            f'border:1px solid {bg_color}33;'
            f'border-top:4px solid {bg_color};'
            f'border-radius:12px;padding:1.2rem 1.4rem">'

            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.6rem;letter-spacing:2px;'
            f'color:{bg_color};text-transform:uppercase;'
            f'margin-bottom:0.4rem">Topic</div>'

            f'<div style="font-family:\'Playfair Display\',serif;'
            f'font-size:1.1rem;color:{TEXT_1};'
            f'margin-bottom:0.8rem">{topic}</div>'

            f'<div style="display:flex;align-items:baseline;gap:0.6rem">'
            f'<div style="font-family:\'Playfair Display\',serif;'
            f'font-size:2.4rem;font-weight:700;color:{color}">'
            f'{bias:+.3f}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.65rem;color:{TEXT_3};'
            f'text-transform:uppercase;letter-spacing:1px">'
            f'{direction}</div>'
            f'</div>'

            f'</div>',
            unsafe_allow_html=True,
        )

with col_mid:
    delta_color = GREEN if delta < 0.2 else RED
    st.markdown(
        f'<div style="text-align:center;padding-top:1.5rem">'
        f'<div style="font-family:\'DM Mono\',monospace;'
        f'font-size:0.58rem;color:{TEXT_3};'
        f'text-transform:uppercase">Δ delta</div>'
        f'<div style="font-family:\'Playfair Display\',serif;'
        f'font-size:1.4rem;color:{delta_color};font-weight:700">'
        f'{delta:.3f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── SUMMARY METRICS ───────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_section("◉", "Comparison Metrics")

m1, m2, m3, m4, m5, m6 = st.columns(6)
metrics = [
    ("Articles A",    comparison["n_articles_a"],                  ""),
    ("Articles B",    comparison["n_articles_b"],                  ""),
    ("Common Sources",comparison["n_common"],                      "sources cover both"),
    ("Cred. A",       f'{comparison["mean_credibility_a"]:.2f}',   "/ 1.0"),
    ("Cred. B",       f'{comparison["mean_credibility_b"]:.2f}',   "/ 1.0"),
    ("Bias Δ",        f'{delta:.3f}',                              "difference"),
]
for col, (label, val, sub) in zip(
    [m1, m2, m3, m4, m5, m6], metrics
):
    with col:
        st.markdown(
            f'<div class="orb-metric">'
            f'<div class="orb-metric-val">{val}</div>'
            f'<div class="orb-metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── CHARTS ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_section("◈", "Visual Comparison")

ch1, ch2 = st.columns(2)
with ch1:
    st.plotly_chart(
        build_bias_comparison_bar(comparison),
        use_container_width=True,
    )
with ch2:
    st.plotly_chart(
        build_stance_comparison_chart(comparison),
        use_container_width=True,
    )

st.plotly_chart(
    build_metric_comparison_radar(comparison),
    use_container_width=True,
)

# ── ARGUMENTS SIDE BY SIDE ────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_section("⊕", "Argument Comparison")

arg_col_a, arg_col_b = st.columns(2)

for col, topic, args, counters, color in [
    (arg_col_a, t_a, comparison["top_args_a"],
     comparison["top_counters_a"], COLOR_A),
    (arg_col_b, t_b, comparison["top_args_b"],
     comparison["top_counters_b"], COLOR_B),
]:
    with col:
        st.markdown(
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.65rem;letter-spacing:2px;color:{color};'
            f'text-transform:uppercase;margin-bottom:0.6rem">'
            f'{topic[:35]}</div>',
            unsafe_allow_html=True,
        )
        for arg in args:
            st.markdown(
                f'<div class="orb-arg pro" style="'
                f'border-color:{color}33;margin-bottom:0.4rem">'
                f'<div class="orb-arg-icon" style="'
                f'background:{color}22;color:{color}">+</div>'
                f'<div style="font-size:0.83rem">{arg[:120]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── SOURCE OVERLAP ────────────────────────────────────────────
if comparison["common_sources"]:
    st.markdown("<br>", unsafe_allow_html=True)
    _section("⊟", f"Common Sources ({comparison['n_common']})")

    cols = st.columns(min(4, comparison["n_common"]))
    for i, src in enumerate(comparison["common_sources"][:8]):
        with cols[i % len(cols)]:
            badge = get_credibility_badge_html(src, compact=False)
            st.markdown(
                f'<div style="margin-bottom:0.4rem">'
                f'<div style="font-family:\'DM Sans\',sans-serif;'
                f'font-size:0.83rem;color:{TEXT_1};'
                f'margin-bottom:0.2rem">{src}</div>'
                f'{badge}'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── SYNTHESIS COMPARISON ──────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_section("⚖", "Synthesis Comparison")

syn_a, syn_b = st.columns(2)
for col, topic, synth, color in [
    (syn_a, t_a, comparison["synthesis_a"], COLOR_A),
    (syn_b, t_b, comparison["synthesis_b"], COLOR_B),
]:
    with col:
        st.markdown(
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.62rem;letter-spacing:2px;color:{color};'
            f'text-transform:uppercase;margin-bottom:0.5rem">'
            f'{topic[:30]}</div>'
            f'<div style="background:{NAVY_CARD};'
            f'border:1px solid {color}22;border-radius:10px;'
            f'padding:1rem;font-size:0.84rem;'
            f'line-height:1.7;color:{TEXT_2};'
            f'max-height:320px;overflow-y:auto;overflow-x:hidden;'
            f'white-space:normal;overflow-wrap:anywhere">'
            f'{synth}'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── DOWNLOAD ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_section("◎", "Export")

dl_data = {
    "topic_a":      t_a,
    "topic_b":      t_b,
    "bias_a":       bias_a,
    "bias_b":       bias_b,
    "bias_delta":   delta,
    "key_insight":  comparison["key_insight"],
    "stance_a":     comparison["stance_a"],
    "stance_b":     comparison["stance_b"],
    "common_sources": comparison["common_sources"],
}
st.download_button(
    label    = "⬇  Download Comparison JSON",
    data     = json.dumps(dl_data, indent=2),
    file_name= f"orbita_compare_{t_a[:15]}_{t_b[:15]}.json"
              .replace(" ", "_"),
    mime     = "application/json",
)