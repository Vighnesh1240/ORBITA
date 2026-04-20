# src/ui/comparison_charts.py
"""
ORBITA Comparison Charts
All charts for the two-topic comparison view.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Colors
NAVY      = "#0a0f1e"
NAVY_2    = "#111827"
NAVY_CARD = "#141c2e"
GOLD      = "#c9a84c"
RED       = "#e05252"
GREEN     = "#3ec97e"
BLUE      = "#5b9cf6"
PURPLE    = "#9b59b6"
TEXT_1    = "#f0ebe0"
TEXT_2    = "#9ba8bb"
TEXT_3    = "#5c6b82"
BORDER    = "rgba(201,168,76,0.18)"

COLOR_A   = "#5b9cf6"   # Topic A — Blue
COLOR_B   = "#e8c96a"   # Topic B — Gold

_BASE = dict(
    paper_bgcolor = NAVY_CARD,
    plot_bgcolor  = NAVY_CARD,
    font          = dict(family="DM Mono, monospace", color=TEXT_2),
    hoverlabel    = dict(
        bgcolor    = NAVY_2,
        bordercolor= GOLD,
        font_size  = 11,
        font_family= "DM Mono, monospace",
    ),
    margin = dict(l=16, r=16, t=50, b=16),
)


def build_bias_comparison_bar(comparison: dict) -> go.Figure:
    """
    Side-by-side bias score bars for two topics.
    The primary comparison chart.
    """
    topic_a  = comparison["topic_a"][:30]
    topic_b  = comparison["topic_b"][:30]
    bias_a   = comparison["bias_a"]
    bias_b   = comparison["bias_b"]

    color_a  = (
        GREEN if bias_a < -0.2
        else RED if bias_a > 0.2
        else BLUE
    )
    color_b  = (
        GREEN if bias_b < -0.2
        else RED if bias_b > 0.2
        else BLUE
    )

    fig = go.Figure()

    for label, bias, color in [
        (topic_a, bias_a, color_a),
        (topic_b, bias_b, color_b),
    ]:
        direction = (
            "Supportive" if bias < -0.2
            else "Critical" if bias > 0.2
            else "Balanced"
        )
        fig.add_trace(go.Bar(
            x            = [label],
            y            = [bias],
            name         = label,
            marker       = dict(
                color        = color,
                opacity      = 0.88,
                line         = dict(color=NAVY_CARD, width=1),
                cornerradius = 6,
            ),
            text         = [f"{bias:+.3f}\n{direction}"],
            textposition = "outside",
            textfont     = dict(
                size   = 10,
                color  = color,
                family = "DM Mono, monospace",
            ),
            hovertemplate= (
                f"<b>{label}</b><br>"
                f"Bias: <b>{bias:+.4f}</b><br>"
                f"Direction: {direction}"
                f"<extra></extra>"
            ),
            width = 0.35,
        ))

    # Zero line
    fig.add_hline(
        y=0,
        line_color = TEXT_3,
        line_width = 1.5,
        line_dash  = "dot",
    )

    # Bias zones
    for y0, y1, color, label in [
        (-1.0, -0.2, "rgba(62,201,126,0.05)",  "Supportive Zone"),
        (-0.2,  0.2, "rgba(91,156,246,0.05)",  "Balanced Zone"),
        ( 0.2,  1.0, "rgba(224,82,82,0.05)",   "Critical Zone"),
    ]:
        fig.add_hrect(
            y0=y0, y1=y1,
            fillcolor = color,
            line_width= 0,
        )

    layout_base = dict(_BASE)
    layout_base["margin"] = dict(l=130, r=90, t=60, b=80)
    layout_base["hoverlabel"] = dict(
        bgcolor    = NAVY_2,
        bordercolor= GOLD,
        font_size  = 11,
        font_family= "DM Mono, monospace",
    )

    fig.update_layout(
        **layout_base,
        title = dict(
            text    = "Bias Score Comparison",
            font    = dict(size=12, color=TEXT_1,
                          family="DM Mono, monospace"),
            x=0.5, xanchor="center",
        ),
        yaxis = dict(
            range     = [-1.2, 1.2],
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            zeroline  = False,
            tickvals  = [-1, -0.5, 0, 0.5, 1],
            ticktext  = ["−1.0","−0.5","0","+0.5","+1.0"],
            tickfont  = dict(size=9, color=TEXT_3,
                            family="DM Mono, monospace"),
        ),
        xaxis = dict(
            showgrid = False,
            tickfont = dict(size=9, color=TEXT_2,
                           family="DM Mono, monospace"),
        ),
        showlegend = False,
        bargap     = 0.6,
        height     = 380,
        transition = dict(duration=600, easing="cubic-in-out"),
    )

    return fig


def build_stance_comparison_chart(comparison: dict) -> go.Figure:
    """Grouped bars showing stance distribution for both topics."""
    topic_a  = comparison["topic_a"][:25]
    topic_b  = comparison["topic_b"][:25]
    stance_a = comparison["stance_a"]
    stance_b = comparison["stance_b"]

    stances = ["Supportive", "Critical", "Neutral"]
    colors  = [GREEN, RED, BLUE]

    fig = go.Figure()

    for stance, color in zip(stances, colors):
        val_a = stance_a.get(stance, 0)
        val_b = stance_b.get(stance, 0)

        fig.add_trace(go.Bar(
            name         = stance,
            x            = [topic_a, topic_b],
            y            = [val_a, val_b],
            marker       = dict(
                color        = color,
                opacity      = 0.82,
                line         = dict(color=NAVY_CARD, width=0.5),
                cornerradius = 4,
            ),
            text         = [val_a, val_b],
            textposition = "auto",
            textfont     = dict(size=9, color=TEXT_1,
                               family="DM Mono, monospace"),
            hovertemplate= (
                f"<b>%{{x}}</b><br>"
                f"{stance}: <b>%{{y}}</b>"
                f"<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_BASE,
        title = dict(
            text    = "Stance Distribution",
            font    = dict(size=12, color=TEXT_1,
                          family="DM Mono, monospace"),
            x=0.5, xanchor="center",
        ),
        barmode = "group",
        xaxis   = dict(
            showgrid = False,
            tickfont = dict(size=9, color=TEXT_2,
                           family="DM Mono, monospace"),
        ),
        yaxis   = dict(
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            zeroline  = False,
            tickfont  = dict(size=9, color=TEXT_3,
                            family="DM Mono, monospace"),
        ),
        legend  = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = -0.22,
            xanchor     = "center",
            x           = 0.5,
            font        = dict(size=9, color=TEXT_2,
                              family="DM Mono, monospace"),
            bgcolor     = "rgba(20,28,46,0.8)",
            bordercolor = BORDER,
            borderwidth = 1,
        ),
        height  = 320,
    )

    return fig


def build_metric_comparison_radar(comparison: dict) -> go.Figure:
    """Radar chart comparing key metrics of both topics."""
    topic_a = comparison["topic_a"][:20]
    topic_b = comparison["topic_b"][:20]

    def _hex_to_rgba(hex_color: str, alpha: float = 0.10) -> str:
        """Convert '#RRGGBB' to Plotly-compatible 'rgba(r,g,b,a)' string."""
        c = (hex_color or "").strip().lstrip("#")
        if len(c) != 6:
            return f"rgba(91,156,246,{alpha})"
        try:
            r = int(c[0:2], 16)
            g = int(c[2:4], 16)
            b = int(c[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        except ValueError:
            return f"rgba(91,156,246,{alpha})"

    # Normalize metrics to 0-1
    def _norm(val, min_v, max_v):
        r = max_v - min_v
        if r == 0:
            return 0.5
        return max(0.0, min(1.0, (val - min_v) / r))

    # Metrics: articles, credibility, vader positive, sources
    n_a = comparison.get("n_articles_a",     0)
    n_b = comparison.get("n_articles_b",     0)
    c_a = comparison.get("mean_credibility_a",0.6)
    c_b = comparison.get("mean_credibility_b",0.6)
    v_a = comparison.get("vader_a",          0.0)
    v_b = comparison.get("vader_b",          0.0)
    s_a = len(comparison.get("sources_a",    []))
    s_b = len(comparison.get("sources_b",    []))

    max_n = max(n_a, n_b, 1)
    max_s = max(s_a, s_b, 1)

    categories   = ["Articles", "Credibility", "Sentiment", "Sources"]
    categories_c = categories + [categories[0]]

    vals_a = [
        n_a / max_n,
        c_a,
        _norm(v_a, -1, 1),
        s_a / max_s,
    ]
    vals_b = [
        n_b / max_n,
        c_b,
        _norm(v_b, -1, 1),
        s_b / max_s,
    ]

    vals_a_c = vals_a + [vals_a[0]]
    vals_b_c = vals_b + [vals_b[0]]

    fig = go.Figure()

    for vals, color, name in [
        (vals_a_c, COLOR_A, topic_a),
        (vals_b_c, COLOR_B, topic_b),
    ]:
        fig.add_trace(go.Scatterpolar(
            r         = vals,
            theta     = categories_c,
            fill      = "toself",
            fillcolor = _hex_to_rgba(color, 0.10),
            line      = dict(color=color, width=2),
            name      = name,
            hovertemplate = (
                "<b>%{theta}</b><br>"
                "Score: <b>%{r:.3f}</b>"
                "<extra></extra>"
            ),
            marker    = dict(color=color, size=5),
        ))

    layout_base = dict(_BASE)
    layout_base["margin"] = dict(l=40, r=40, t=55, b=20)

    fig.update_layout(
        **layout_base,
        polar = dict(
            radialaxis  = dict(
                visible   = True,
                range     = [0, 1],
                tickfont  = dict(size=7, color=TEXT_3,
                                family="DM Mono, monospace"),
                gridcolor = "rgba(255,255,255,0.06)",
                linecolor = "rgba(255,255,255,0.06)",
            ),
            angularaxis = dict(
                tickfont  = dict(size=9, color=TEXT_2,
                                family="DM Mono, monospace"),
                gridcolor = "rgba(255,255,255,0.06)",
                linecolor = "rgba(255,255,255,0.06)",
            ),
            bgcolor = NAVY_CARD,
        ),
        title  = dict(
            text    = "Multi-Metric Comparison",
            font    = dict(size=12, color=TEXT_1,
                          family="DM Mono, monospace"),
            x=0.5, xanchor="center",
        ),
        legend = dict(
            font    = dict(size=9, color=TEXT_2,
                          family="DM Mono, monospace"),
            bgcolor     = "rgba(20,28,46,0.8)",
            bordercolor = BORDER,
            borderwidth = 1,
        ),
        height = 340,
    )

    return fig


def build_bias_heatmap_chart(matrix_data: dict) -> go.Figure:
    """
    The signature ORBITA heatmap: Sources × Topics.
    Red = critical bias, Blue = supportive, White = neutral.
    """
    if not matrix_data.get("has_data"):
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=(
                "No heatmap data yet.<br>"
                "Run analyses on multiple topics to populate."
            ),
            showarrow = False,
            font      = dict(size=12, color=TEXT_3,
                            family="DM Mono, monospace"),
        )
        fig.update_layout(
            **_BASE, height=300,
            xaxis=dict(showgrid=False,showticklabels=False),
            yaxis=dict(showgrid=False,showticklabels=False),
        )
        return fig

    topics  = matrix_data["topics"]
    sources = matrix_data["sources"]
    matrix  = matrix_data["matrix"]

    # Replace None with NaN for display
    z = []
    for row in matrix:
        z.append([
            v if v is not None else float("nan")
            for v in row
        ])

    # Custom colorscale: Blue → White → Red
    # Blue = supportive (negative bias)
    # Red  = critical   (positive bias)
    colorscale = [
        [0.00, "#3ec97e"],   # -1.0 fully supportive = green
        [0.25, "#5b9cf6"],   # -0.5 supportive = blue
        [0.50, "#f0ebe0"],   # 0.0  neutral = light cream
        [0.75, "#e05252"],   # +0.5 critical = red
        [1.00, "#8b0000"],   # +1.0 fully critical = dark red
    ]

    fig = go.Figure(go.Heatmap(
        z          = z,
        x          = topics,
        y          = sources,
        colorscale = colorscale,
        zmin       = -1.0,
        zmax       =  1.0,
        zmid       =  0.0,
        text       = [
            [
                f"{v:+.2f}" if not (v != v) else "N/A"
                for v in row
            ]
            for row in z
        ],
        texttemplate = "%{text}",
        textfont     = dict(
            size   = 9,
            family = "DM Mono, monospace",
        ),
        hoverongaps      = False,
        colorbar         = dict(
            title      = dict(
                text   = "Bias",
                font   = dict(size=9, color=TEXT_3,
                             family="DM Mono, monospace"),
            ),
            tickvals   = [-1, -0.5, 0, 0.5, 1],
            ticktext   = ["−1.0","−0.5","0","+0.5","+1.0"],
            tickfont   = dict(size=8, color=TEXT_3,
                             family="DM Mono, monospace"),
            bgcolor    = NAVY_CARD,
            bordercolor= BORDER,
            borderwidth= 1,
            len        = 0.8,
        ),
        hovertemplate = (
            "<b>%{y}</b><br>"
            "Topic: <b>%{x}</b><br>"
            "Bias: <b>%{z:+.3f}</b>"
            "<extra></extra>"
        ),
    ))

    n_s  = len(sources)
    h    = max(300, n_s * 38 + 100)

    fig.update_layout(
        **_BASE,
        title  = dict(
            text    = (
                "📊 Source × Topic Bias Heatmap "
                "<span style='font-size:0.7em;color:#5c6b82'>"
                " — green=supportive · red=critical · "
                "white=neutral</span>"
            ),
            font    = dict(size=12, color=TEXT_1,
                          family="DM Mono, monospace"),
            x=0.0, xanchor="left",
        ),
        xaxis  = dict(
            tickangle = -35,
            tickfont  = dict(size=9, color=TEXT_2,
                            family="DM Mono, monospace"),
            showgrid  = False,
        ),
        yaxis  = dict(
            tickfont  = dict(size=9, color=TEXT_2,
                            family="DM Mono, monospace"),
            showgrid  = False,
            autorange = "reversed",
        ),
        height = h,
    )

    return fig