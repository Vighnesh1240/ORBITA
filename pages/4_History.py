# pages/4_History.py
"""
ORBITA Historical Analysis — SQLite-backed time series
"""

import sys
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title = "ORBITA — History",
    page_icon  = "📈",
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

from src.history_tracker import (
    get_all_runs,
    get_topic_list,
    get_bias_timeline,
    get_database_stats,
    get_recent_runs,
    delete_run,
    clear_all_history,
    build_bias_trend_data,
)

# ── Colors ────────────────────────────────────────────────────
GOLD      = "#c9a84c"
GREEN     = "#3ec97e"
RED       = "#e05252"
BLUE      = "#5b9cf6"
TEXT_1    = "#f0ebe0"
TEXT_2    = "#9ba8bb"
TEXT_3    = "#5c6b82"
NAVY_CARD = "#141c2e"
NAVY_2    = "#111827"
BORDER    = "rgba(201,168,76,0.18)"
BORDER_DIM= "rgba(255,255,255,0.06)"

_BASE = dict(
    paper_bgcolor = NAVY_CARD,
    plot_bgcolor  = NAVY_CARD,
    font          = dict(
        family="DM Mono, monospace", color=TEXT_2
    ),
    hoverlabel = dict(
        bgcolor    = NAVY_2,
        bordercolor= GOLD,
        font_size  = 11,
        font_family= "DM Mono, monospace",
    ),
    margin = dict(l=16, r=16, t=50, b=16),
)


def _section(icon, title):
    st.markdown(
        f'<div class="orb-section">'
        f'<div class="orb-section-line"></div>'
        f'<div class="orb-section-title">'
        f'{icon}&nbsp; {title}</div>'
        f'<div class="orb-section-line"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _bias_color(score):
    if score < -0.2:
        return GREEN
    elif score > 0.2:
        return RED
    return BLUE


# ── Header ────────────────────────────────────────────────────
st.markdown(
    '<div style="padding:1.5rem 0 1rem;text-align:center">'
    '<div style="font-family:\'Playfair Display\',serif;'
    'font-size:2.8rem;font-weight:900;color:#f0ebe0;'
    'letter-spacing:8px">'
    'ORB<span style="color:#c9a84c">I</span>TA</div>'
    '<div style="font-family:\'DM Mono\',monospace;'
    'font-size:0.65rem;letter-spacing:3px;color:#5c6b82;'
    'text-transform:uppercase;margin-top:0.3rem">'
    'Historical Bias Tracker — Longitudinal Analysis</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Database Stats ────────────────────────────────────────────
stats = get_database_stats()

s1, s2, s3, s4, s5 = st.columns(5)
for col, val, label in [
    (s1, stats["n_runs"],     "Total Runs"),
    (s2, stats["n_topics"],   "Topics Tracked"),
    (s3, stats["n_sources"],  "Sources Seen"),
    (s4, stats["n_articles"], "Articles Stored"),
    (s5, f'{stats["db_size_kb"]:.0f} KB', "Database Size"),
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

if stats["n_runs"] == 0:
    st.markdown(
        '<div style="text-align:center;padding:4rem 2rem;'
        'color:#5c6b82">'
        '<div style="font-size:2.5rem;opacity:0.4;'
        'margin-bottom:1rem">📈</div>'
        '<div style="font-family:\'Playfair Display\',serif;'
        'font-size:1.3rem;color:#9ba8bb;margin-bottom:0.5rem">'
        'No history yet</div>'
        '<div style="font-family:\'DM Mono\',monospace;'
        'font-size:0.8rem;line-height:1.8">'
        'Every analysis you run is automatically saved here.<br>'
        'Run a few topics on the main page to see trends.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Topic Selector ────────────────────────────────────────────
_section("◉", "Topic Filter")

topics_db  = get_topic_list()
topic_names= ["All Topics"] + [t["topic"] for t in topics_db]

sel_topic = st.selectbox(
    "Select topic",
    topic_names,
    label_visibility = "collapsed",
)

filter_topic = None if sel_topic == "All Topics" else sel_topic

# ── Bias Timeline Chart ───────────────────────────────────────
_section("📈", "Bias Score Over Time")

trend = build_bias_trend_data(filter_topic)

if not trend.get("has_data"):
    st.info("No data for selected topic.")
else:
    fig = go.Figure()

    # Background zones
    for y0, y1, color in [
        (-1.0, -0.2, "rgba(62,201,126,0.04)"),
        (-0.2,  0.2, "rgba(91,156,246,0.03)"),
        ( 0.2,  1.0, "rgba(224,82,82,0.04)"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0)

    # Zero line
    fig.add_hline(
        y=0,
        line_color = TEXT_3,
        line_width = 1,
        line_dash  = "dot",
    )

    xs = trend["timestamps"]

    # Main bias line
    bias_colors = [_bias_color(b) for b in trend["bias_scores"]]
    fig.add_trace(go.Scatter(
        x    = xs,
        y    = trend["bias_scores"],
        mode = "lines+markers",
        name = "Bias Score",
        line = dict(color=GOLD, width=2.5),
        marker = dict(
            color  = bias_colors,
            size   = 10,
            symbol = "circle",
            line   = dict(color=NAVY_CARD, width=2),
        ),
        customdata = list(zip(
            trend["topics"],
            trend["n_articles"],
            trend["credibilities"],
        )),
        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Bias: <b>%{y:+.4f}</b><br>"
            "Articles: %{customdata[1]}<br>"
            "Credibility: %{customdata[2]:.2f}<br>"
            "Time: %{x}"
            "<extra></extra>"
        ),
    ))

    # Weighted bias line
    fig.add_trace(go.Scatter(
        x    = xs,
        y    = trend["weighted"],
        mode = "lines",
        name = "Weighted Bias",
        line = dict(
            color = BLUE,
            width = 1.5,
            dash  = "dash",
        ),
        hovertemplate = (
            "Weighted: <b>%{y:+.4f}</b>"
            "<extra></extra>"
        ),
    ))

    # VADER line
    fig.add_trace(go.Scatter(
        x    = xs,
        y    = trend["vader_scores"],
        mode = "lines",
        name = "VADER Sentiment",
        line = dict(
            color = GREEN,
            width = 1,
            dash  = "dot",
        ),
        opacity          = 0.7,
        hovertemplate    = (
            "VADER: <b>%{y:+.4f}</b>"
            "<extra></extra>"
        ),
    ))

    layout_base = dict(_BASE)
    layout_base["margin"] = dict(l=50, r=20, t=50, b=70)

    fig.update_layout(
        **layout_base,
        title = dict(
            text  = (
                f"Bias Over Time — {sel_topic}"
                if sel_topic != "All Topics"
                else "Bias Scores — All Topics"
            ),
            font  = dict(size=12, color=TEXT_1,
                        family="DM Mono, monospace"),
            x=0.5, xanchor="center",
        ),
        xaxis = dict(
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            tickfont  = dict(size=8, color=TEXT_3,
                            family="DM Mono, monospace"),
            tickangle = -30,
        ),
        yaxis = dict(
            range     = [-1.1, 1.1],
            tickvals  = [-1,-0.5,0,0.5,1],
            ticktext  = ["−1.0","−0.5","0","+0.5","+1.0"],
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            zeroline  = False,
            tickfont  = dict(size=9, color=TEXT_3,
                            family="DM Mono, monospace"),
        ),
        legend = dict(
            orientation = "h",
            yanchor="bottom", y=-0.25,
            xanchor="center", x=0.5,
            font=dict(size=9, color=TEXT_2,
                     family="DM Mono, monospace"),
            bgcolor     = "rgba(20,28,46,0.8)",
            bordercolor = BORDER,
            borderwidth = 1,
        ),
        height     = 400,
        transition = dict(duration=500, easing="cubic-in-out"),
    )

    st.plotly_chart(fig, use_container_width=True)

# ── Topic Summary Cards ───────────────────────────────────────
_section("◈", "Topic Summary")

topics_db = get_topic_list()

if topics_db:
    cols_per_row = 3
    for i in range(0, min(9, len(topics_db)), cols_per_row):
        row_topics = topics_db[i:i + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, t in zip(cols, row_topics):
            avg  = float(t.get("avg_bias", 0.0))
            mn   = float(t.get("min_bias", 0.0))
            mx   = float(t.get("max_bias", 0.0))
            n    = int(t.get("n_runs",    0))
            cred = float(t.get("avg_credibility", 0.60))
            color = _bias_color(avg)

            # Trend indicator
            runs_for = get_bias_timeline(t["topic"])
            if len(runs_for) >= 2:
                first = float(runs_for[0]["bias_score"])
                last  = float(runs_for[-1]["bias_score"])
                trend_sym = "↑" if last > first + 0.05 else (
                    "↓" if last < first - 0.05 else "→"
                )
                trend_col = RED if trend_sym == "↑" else (
                    GREEN if trend_sym == "↓" else TEXT_3
                )
            else:
                trend_sym = "—"
                trend_col = TEXT_3

            with col:
                st.markdown(
                    f'<div style="background:{NAVY_CARD};'
                    f'border:1px solid {color}33;'
                    f'border-top:3px solid {color};'
                    f'border-radius:10px;padding:1rem;'
                    f'margin-bottom:0.6rem;'
                    f'transition:all 0.2s">'

                    f'<div style="font-family:\'DM Sans\',sans-serif;'
                    f'font-size:0.83rem;color:{TEXT_1};'
                    f'font-weight:500;margin-bottom:0.5rem">'
                    f'{t["topic"][:35]}</div>'

                    f'<div style="display:flex;align-items:baseline;'
                    f'gap:0.5rem;margin-bottom:0.4rem">'
                    f'<div style="font-family:\'Playfair Display\',serif;'
                    f'font-size:1.6rem;font-weight:700;color:{color}">'
                    f'{avg:+.3f}</div>'
                    f'<div style="font-family:\'DM Mono\',monospace;'
                    f'font-size:0.75rem;color:{trend_col}">'
                    f'{trend_sym}</div>'
                    f'</div>'

                    f'<div style="font-family:\'DM Mono\',monospace;'
                    f'font-size:0.62rem;color:{TEXT_3};line-height:1.8">'
                    f'runs: {n} &nbsp;·&nbsp; '
                    f'range: {mn:+.2f} → {mx:+.2f}<br>'
                    f'credibility: {cred:.2f}'
                    f'</div>'

                    f'</div>',
                    unsafe_allow_html=True,
                )

# ── Full Run Table ────────────────────────────────────────────
_section("⊟", "All Runs")

all_runs = get_all_runs(limit=50)

if all_runs:
    # Header row
    st.markdown(
        f'<div style="display:grid;'
        f'grid-template-columns:2fr 1fr 0.8fr 0.8fr 0.8fr 0.6fr;'
        f'gap:0.4rem;padding:0.4rem 0.8rem;'
        f'font-family:\'DM Mono\',monospace;'
        f'font-size:0.58rem;letter-spacing:1.5px;'
        f'color:{TEXT_3};text-transform:uppercase;'
        f'border-bottom:1px solid rgba(255,255,255,0.08);'
        f'margin-bottom:0.3rem">'
        f'<div>Topic</div>'
        f'<div>Timestamp</div>'
        f'<div>Bias</div>'
        f'<div>Articles</div>'
        f'<div>Credibility</div>'
        f'<div>Direction</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for run in all_runs:
        bias    = float(run.get("bias_score", 0.0))
        color   = _bias_color(bias)
        ts      = run.get("timestamp", "")[:16]
        topic   = run.get("topic", "")[:35]
        n_art   = int(run.get("n_articles",       0))
        cred    = float(run.get("mean_credibility", 0.60))
        direc   = run.get("bias_direction",       "balanced")
        run_id  = run.get("id", 0)

        st.markdown(
            f'<div style="display:grid;'
            f'grid-template-columns:2fr 1fr 0.8fr 0.8fr 0.8fr 0.6fr;'
            f'gap:0.4rem;padding:0.5rem 0.8rem;'
            f'background:{NAVY_CARD};'
            f'border:1px solid rgba(255,255,255,0.04);'
            f'border-radius:6px;margin-bottom:0.25rem">'

            f'<div style="font-family:\'DM Sans\',sans-serif;'
            f'font-size:0.78rem;color:{TEXT_1}">{topic}</div>'

            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.68rem;color:{TEXT_3}">{ts}</div>'

            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.82rem;color:{color};font-weight:600">'
            f'{bias:+.3f}</div>'

            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.72rem;color:{TEXT_2}">{n_art}</div>'

            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.72rem;color:{TEXT_2}">{cred:.2f}</div>'

            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.65rem;color:{color};'
            f'text-transform:capitalize">{direc[:8]}</div>'

            f'</div>',
            unsafe_allow_html=True,
        )

# ── Danger zone ───────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_section("⚠", "Database Management")

d1, d2 = st.columns(2)
with d1:
    st.caption(
        f"Database: {stats['db_path']}"
    )
with d2:
    if st.button(
        "🗑️ Clear All History",
        type="secondary",
    ):
        clear_all_history()
        st.success("All history cleared.")
        st.rerun()