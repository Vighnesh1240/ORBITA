# src/ui/charts.py
"""
ORBITA Enhanced Charts
Dark editorial theme — navy + gold + animated + interactive
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import random
from collections import Counter

# ── Color Palette ─────────────────────────────────────────────
NAVY        = "#0a0f1e"
NAVY_2      = "#111827"
NAVY_3      = "#1a2236"
NAVY_CARD   = "#141c2e"
GOLD        = "#c9a84c"
GOLD_L      = "#e8c96a"
GOLD_DIM    = "rgba(201,168,76,0.15)"
RED         = "#e05252"
RED_DIM     = "rgba(224,82,82,0.12)"
GREEN       = "#3ec97e"
GREEN_DIM   = "rgba(62,201,126,0.12)"
BLUE        = "#5b9cf6"
BLUE_DIM    = "rgba(91,156,246,0.12)"
PURPLE      = "#9b59b6"
TEAL        = "#1abc9c"
ORANGE      = "#e67e22"
TEXT_1      = "#f0ebe0"
TEXT_2      = "#9ba8bb"
TEXT_3      = "#5c6b82"
BORDER      = "rgba(201,168,76,0.18)"
BORDER_DIM  = "rgba(255,255,255,0.06)"

STANCE_COLOR = {
    "Supportive": GREEN,
    "Critical":   RED,
    "Neutral":    BLUE,
}

# ── Base Layout (applied to all charts) ──────────────────────
_LAYOUT_BASE = dict(
    paper_bgcolor = NAVY_CARD,
    plot_bgcolor  = NAVY_CARD,
    font          = dict(
        family = "DM Mono, monospace",
        color  = TEXT_2,
    ),
)


def _apply_dark_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply consistent dark theme + animation to any figure."""
    fig.update_layout(
        **_LAYOUT_BASE,
        height    = height,
        margin    = dict(l=16, r=16, t=50, b=16),
        hoverlabel= dict(
            bgcolor    = NAVY_2,
            bordercolor= GOLD,
            font_size  = 11,
            font_family= "DM Mono, monospace",
        ),
        # Smooth transition animation
        transition = dict(
            duration = 500,
            easing   = "cubic-in-out",
        ),
        xaxis = dict(
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            zeroline  = False,
            tickfont  = dict(
                size   = 9,
                color  = TEXT_3,
                family = "DM Mono, monospace",
            ),
            linecolor = BORDER_DIM,
        ),
        yaxis = dict(
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            zeroline  = False,
            tickfont  = dict(
                size   = 9,
                color  = TEXT_3,
                family = "DM Mono, monospace",
            ),
            linecolor = BORDER_DIM,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 1. BIAS SPECTRUM GRAPH (Enhanced)
# ─────────────────────────────────────────────────────────────

def build_bias_spectrum_graph(
    articles:   list,
    bias_score: float,
    topic:      str,
) -> go.Figure:
    """
    Enhanced bias spectrum with animated entry,
    glowing score marker, and rich hover cards.
    """
    if not isinstance(articles, list):
        articles = []
    if not isinstance(bias_score, (int, float)):
        bias_score = 0.0

    random.seed(42)
    stance_base = {
        "Supportive": -0.62,
        "Neutral":     0.0,
        "Critical":    0.62,
    }

    fig = go.Figure()

    # ── Gradient background zones ─────────────────────────────
    for x0, x1, color, label in [
        (-1.0, -0.15, "rgba(62,201,126,0.06)",  ""),
        (-0.15, 0.15, "rgba(91,156,246,0.04)",  ""),
        ( 0.15,  1.0, "rgba(224,82,82,0.06)",   ""),
    ]:
        fig.add_shape(
            type      = "rect",
            x0=x0, x1=x1, y0=0, y1=1,
            fillcolor = color,
            line_width= 0,
            layer     = "below",
        )

    # Zone boundary lines
    for x in [-0.15, 0.15]:
        fig.add_vline(
            x          = x,
            line_color = "rgba(255,255,255,0.06)",
            line_width = 1,
            line_dash  = "dot",
        )

    # Zone labels
    for x, label, color in [
        (-0.57, "SUPPORTIVE", GREEN),
        ( 0.00, "BALANCED",   BLUE),
        ( 0.57, "CRITICAL",   RED),
    ]:
        fig.add_annotation(
            x=x, y=0.96, text=label,
            showarrow = False,
            font      = dict(
                size   = 7,
                color  = color,
                family = "DM Mono, monospace",
            ),
            xanchor = "center",
        )

    # Center line
    fig.add_vline(
        x          = 0,
        line_color = "rgba(255,255,255,0.08)",
        line_width = 1,
    )

    # ── Article dots ──────────────────────────────────────────
    for stance in ["Supportive", "Critical", "Neutral"]:
        group = [a for a in articles if a.get("stance") == stance]
        if not group:
            continue

        xs, ys, titles, sources = [], [], [], []
        for a in group:
            base = stance_base.get(stance, 0.0)
            xs.append(max(-0.92, min(0.92,
                base + random.uniform(-0.12, 0.12))))
            ys.append(random.uniform(0.25, 0.75))
            titles.append((a.get("title",  "") or "")[:55])
            sources.append((a.get("source", "") or "Unknown"))

        color = STANCE_COLOR[stance]

        fig.add_trace(go.Scatter(
            x    = xs,
            y    = ys,
            mode = "markers",
            name = stance,
            marker = dict(
                color   = color,
                size    = 14,
                opacity = 0.85,
                line    = dict(color=NAVY, width=2),
                symbol  = "circle",
            ),
            customdata    = list(zip(titles, sources)),
            hovertemplate = (
                f"<b>%{{customdata[0]}}</b><br>"
                f"<span style='color:{TEXT_3}'>%{{customdata[1]}}</span><br>"
                f"<b style='color:{color}'>{stance}</b>"
                f"<extra></extra>"
            ),
        ))

    # ── Bias Score Marker (glowing diamond) ───────────────────
    bias_color = (
        GREEN if bias_score < -0.2
        else RED if bias_score > 0.2
        else BLUE
    )

    # Outer glow ring
    fig.add_trace(go.Scatter(
        x    = [bias_score],
        y    = [0.12],
        mode = "markers",
        name = "",
        marker = dict(
            color   = bias_color,
            size    = 42,
            opacity = 0.12,
            symbol  = "diamond",
        ),
        hoverinfo  = "skip",
        showlegend = False,
    ))

    # Middle ring
    fig.add_trace(go.Scatter(
        x    = [bias_score],
        y    = [0.12],
        mode = "markers",
        name = "",
        marker = dict(
            color   = bias_color,
            size    = 30,
            opacity = 0.20,
            symbol  = "diamond",
        ),
        hoverinfo  = "skip",
        showlegend = False,
    ))

    # Core diamond
    text_side = "right" if bias_score <= 0.4 else "left"
    text_str  = (
        f"  {bias_score:+.3f}"
        if text_side == "right"
        else f"{bias_score:+.3f}  "
    )

    fig.add_trace(go.Scatter(
        x    = [bias_score],
        y    = [0.12],
        mode = "markers+text",
        name = "Bias Score",
        marker = dict(
            color  = bias_color,
            size   = 20,
            symbol = "diamond",
            line   = dict(color=NAVY_CARD, width=2),
        ),
        text         = [text_str],
        textfont     = dict(
            size   = 12,
            color  = bias_color,
            family = "DM Mono, monospace",
        ),
        textposition = f"middle {text_side}",
        hovertemplate= (
            f"<b>Overall Bias Score</b><br>"
            f"<b style='color:{bias_color}'>{bias_score:+.4f}</b>"
            f"<extra></extra>"
        ),
    ))

    # ── Axis Labels ───────────────────────────────────────────
    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = (
                f"<b style='color:{TEXT_1}'>"
                f"{topic[:50]}</b>"
            ),
            font    = dict(
                size   = 12,
                family = "DM Mono, monospace",
                color  = TEXT_1,
            ),
            x=0.5, xanchor="center", y=0.97,
        ),
        xaxis = dict(
            range     = [-1.05, 1.05],
            tickvals  = [-1, -0.5, 0, 0.5, 1],
            ticktext  = ["−1.0", "−0.5", "0", "+0.5", "+1.0"],
            tickfont  = dict(
                size   = 9,
                family = "DM Mono, monospace",
                color  = TEXT_3,
            ),
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            zeroline  = False,
        ),
        yaxis = dict(
            showticklabels = False,
            showgrid       = False,
            zeroline       = False,
            range          = [0, 1],
        ),
        legend = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = -0.25,
            xanchor     = "center",
            x           = 0.5,
            font        = dict(
                size   = 9,
                family = "DM Mono, monospace",
                color  = TEXT_2,
            ),
            bgcolor     = "rgba(20,28,46,0.9)",
            bordercolor = BORDER,
            borderwidth = 1,
        ),
        height     = 360,
        margin     = dict(l=16, r=16, t=46, b=60),
        hovermode  = "closest",
        transition = dict(duration=600, easing="cubic-in-out"),
        hoverlabel = dict(
            bgcolor    = NAVY_2,
            bordercolor= GOLD,
            font_size  = 11,
            font_family= "DM Mono, monospace",
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────
# 2. CONFIDENCE GAUGE (Enhanced)
# ─────────────────────────────────────────────────────────────

def build_confidence_gauge(
    a_score: float,
    b_score: float,
) -> go.Figure:
    """Animated gauge charts for agent confidence."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[
            {"type": "indicator"},
            {"type": "indicator"},
        ]],
        subplot_titles=["Agent A", "Agent B"],
    )

    for i, (score, label, color) in enumerate([
        (a_score, "Agent A", GREEN),
        (b_score, "Agent B", RED),
    ], 1):
        fig.add_trace(
            go.Indicator(
                mode  = "gauge+number",
                value = score * 100,
                number= dict(
                    suffix   = "%",
                    font     = dict(
                        size   = 20,
                        color  = color,
                        family = "Playfair Display, serif",
                    ),
                ),
                gauge = dict(
                    axis = dict(
                        range    = [0, 100],
                        tickfont = dict(
                            size  = 8,
                            color = TEXT_3,
                        ),
                        tickcolor= TEXT_3,
                    ),
                    bar = dict(
                        color     = color,
                        thickness = 0.25,
                    ),
                    bgcolor    = NAVY_3,
                    borderwidth= 0,
                    steps = [
                        dict(range=[0,  40], color="rgba(224,82,82,0.08)"),
                        dict(range=[40, 70], color="rgba(201,168,76,0.08)"),
                        dict(range=[70,100], color="rgba(62,201,126,0.08)"),
                    ],
                    threshold = dict(
                        line  = dict(color=GOLD, width=2),
                        value = 70,
                    ),
                ),
            ),
            row=1, col=i,
        )

    gauge_layout = {
        **_LAYOUT_BASE,
        "height": 200,
        "margin": dict(l=20, r=20, t=40, b=10),
        "font": dict(
            family="DM Mono, monospace",
            color=TEXT_2,
        ),
    }
    fig.update_layout(**gauge_layout)

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(
            size   = 9,
            color  = TEXT_3,
            family = "DM Mono, monospace",
        )

    return fig


# ─────────────────────────────────────────────────────────────
# 3. STANCE DISTRIBUTION (Enhanced Donut)
# ─────────────────────────────────────────────────────────────

def build_stance_distribution_chart(
    articles: list,
) -> go.Figure:
    """Enhanced donut chart with pull effect on hover."""
    counts = {"Supportive": 0, "Critical": 0, "Neutral": 0}
    for a in articles:
        s = a.get("stance", "Neutral")
        if s in counts:
            counts[s] += 1

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [STANCE_COLOR[l] for l in labels]
    total  = sum(values)

    fig = go.Figure(go.Pie(
        labels        = labels,
        values        = values,
        hole          = 0.65,
        pull          = [0.03, 0.03, 0.03],
        marker        = dict(
            colors = colors,
            line   = dict(color=NAVY_CARD, width=3),
        ),
        textinfo      = "label+percent",
        textfont      = dict(
            size   = 10,
            family = "DM Mono, monospace",
        ),
        hovertemplate = (
            "%{label}<br>"
            "<b>%{value} articles</b><br>"
            "%{percent}"
            "<extra></extra>"
        ),
        rotation = -90,
    ))

    # Center text
    fig.add_annotation(
        text = f"<b>{total}</b>",
        x=0.5, y=0.56,
        font = dict(
            size   = 24,
            color  = TEXT_1,
            family = "Playfair Display, serif",
        ),
        showarrow=False,
    )
    fig.add_annotation(
        text = "articles",
        x=0.5, y=0.40,
        font = dict(
            size   = 8,
            color  = TEXT_3,
            family = "DM Mono, monospace",
        ),
        showarrow=False,
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = "Stance Mix",
            font    = dict(
                size   = 10,
                family = "DM Mono, monospace",
                color  = TEXT_3,
            ),
            x=0.5, xanchor="center",
        ),
        height     = 240,
        margin     = dict(l=10, r=10, t=38, b=10),
        showlegend = False,
        hoverlabel = dict(
            bgcolor    = NAVY_2,
            bordercolor= GOLD,
            font_size  = 11,
            font_family= "DM Mono, monospace",
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────
# 4. WORD COUNT CHART (Enhanced)
# ─────────────────────────────────────────────────────────────

def build_word_count_chart(articles: list) -> go.Figure:
    """Animated horizontal bar with gradient colors."""
    if not articles:
        return go.Figure()

    titles = [
        f"{a.get('title', 'Unknown')[:28]}…"
        if len(a.get("title", "")) > 28
        else a.get("title", "Unknown")
        for a in articles
    ]
    words  = [
        len((a.get("full_text") or "").split())
        if (a.get("full_text") or "").strip()
        else int(a.get("word_count", 0) or 0)
        for a in articles
    ]
    colors = [
        STANCE_COLOR.get(a.get("stance", "Neutral"), TEXT_3)
        for a in articles
    ]

    # Normalize for opacity; guard against all-zero word counts.
    max_w = max(words) if words else 0
    denom = max(max_w, 1)
    opacities = [0.5 + 0.45 * (w / denom) for w in words]

    fig = go.Figure()

    for i, (title, word, color, opacity) in enumerate(
        zip(titles, words, colors, opacities)
    ):
        fig.add_trace(go.Bar(
            x           = [word],
            y           = [title],
            orientation = "h",
            marker      = dict(
                color   = color,
                opacity = opacity,
                line    = dict(color=NAVY_CARD, width=0.5),
            ),
            text         = [f"{word:,}"],
            textposition = "outside",
            textfont     = dict(
                size   = 8,
                color  = TEXT_3,
                family = "DM Mono, monospace",
            ),
            hovertemplate= (
                f"<b>{title}</b><br>"
                f"Words: <b>{word:,}</b>"
                f"<extra></extra>"
            ),
            showlegend = False,
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = "Article Lengths",
            font    = dict(
                size   = 10,
                family = "DM Mono, monospace",
                color  = TEXT_3,
            ),
            x=0, xanchor="left",
        ),
        xaxis = dict(
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            tickfont  = dict(
                size   = 8,
                color  = TEXT_3,
                family = "DM Mono, monospace",
            ),
            zeroline = False,
        ),
        yaxis = dict(
            showgrid  = False,
            autorange = "reversed",
            tickfont  = dict(
                size   = 8,
                color  = TEXT_2,
                family = "DM Mono, monospace",
            ),
        ),
        barmode    = "overlay",
        height     = max(160, len(articles) * 30 + 55),
        margin     = dict(l=16, r=55, t=38, b=16),
        showlegend = False,
        hoverlabel = dict(
            bgcolor    = NAVY_2,
            bordercolor= GOLD,
            font_size  = 11,
            font_family= "DM Mono, monospace",
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────
# 5. BIAS RADAR CHART (Enhanced)
# ─────────────────────────────────────────────────────────────

def build_bias_radar_chart(bias_vector: dict) -> go.Figure:
    """Animated radar with filled area and reference ring."""
    ideological = abs(bias_vector.get("ideological_bias", 0.0))
    emotional   =     bias_vector.get("emotional_bias",   0.0)
    informational=    bias_vector.get("informational_bias",0.0)
    diversity   = 1.0 - bias_vector.get("source_diversity", 0.5)

    categories  = [
        "Ideological",
        "Emotional",
        "Informational",
        "Source\nConcentration",
    ]
    values = [
        round(ideological,   3),
        round(emotional,     3),
        round(informational, 3),
        round(diversity,     3),
    ]

    categories_c = categories + [categories[0]]
    values_c     = values     + [values[0]]

    fig = go.Figure()

    # Reference ring at 0.5
    ref = [0.5] * (len(categories) + 1)
    fig.add_trace(go.Scatterpolar(
        r         = ref,
        theta     = categories_c,
        line      = dict(
            color = "rgba(255,255,255,0.08)",
            width = 1,
            dash  = "dot",
        ),
        name      = "Reference",
        fill      = "none",
        hoverinfo = "skip",
    ))

    # Main bias polygon
    fig.add_trace(go.Scatterpolar(
        r     = values_c,
        theta = categories_c,
        fill  = "toself",
        fillcolor = "rgba(201,168,76,0.10)",
        line  = dict(
            color = GOLD,
            width = 2,
        ),
        name  = "Bias Profile",
        hovertemplate = (
            "<b>%{theta}</b><br>"
            "Score: <b>%{r:.3f}</b>"
            "<extra></extra>"
        ),
        marker = dict(
            color = GOLD,
            size  = 6,
        ),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        polar = dict(
            radialaxis = dict(
                visible   = True,
                range     = [0, 1],
                tickfont  = dict(
                    size   = 8,
                    color  = TEXT_3,
                    family = "DM Mono, monospace",
                ),
                gridcolor = "rgba(255,255,255,0.06)",
                linecolor = "rgba(255,255,255,0.06)",
                tickcolor = TEXT_3,
            ),
            angularaxis = dict(
                tickfont  = dict(
                    size   = 9,
                    color  = TEXT_2,
                    family = "DM Mono, monospace",
                ),
                gridcolor = "rgba(255,255,255,0.06)",
                linecolor = "rgba(255,255,255,0.06)",
            ),
            bgcolor = NAVY_CARD,
        ),
        title = dict(
            text    = "Bias Profile",
            font    = dict(
                size   = 10,
                family = "DM Mono, monospace",
                color  = TEXT_3,
            ),
            x=0.5, xanchor="center",
        ),
        legend = dict(
            font    = dict(
                size   = 8,
                color  = TEXT_3,
                family = "DM Mono, monospace",
            ),
            bgcolor     = "rgba(20,28,46,0.8)",
            bordercolor = BORDER_DIM,
            borderwidth = 1,
        ),
        height = 320,
        margin = dict(l=40, r=40, t=50, b=20),
        hoverlabel = dict(
            bgcolor    = NAVY_2,
            bordercolor= GOLD,
            font_size  = 11,
            font_family= "DM Mono, monospace",
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────
# 6. BIAS BREAKDOWN BARS (Enhanced)
# ─────────────────────────────────────────────────────────────

def build_bias_breakdown_bars(bias_vector: dict) -> go.Figure:
    """Horizontal bars with gradient and animation."""
    dimensions = [
        ("Ideological",   bias_vector.get("ideological_bias",   0.0), True),
        ("Emotional",     bias_vector.get("emotional_bias",      0.0), False),
        ("Informational", bias_vector.get("informational_bias",  0.0), False),
        ("Diversity",     bias_vector.get("source_diversity",    0.0), False),
        ("Entropy",       bias_vector.get("stance_entropy",      0.0), False),
    ]

    labels = [d[0] for d in dimensions]
    values = [d[1] for d in dimensions]
    signed = [d[2] for d in dimensions]

    colors = []
    for val, is_signed in zip(values, signed):
        if is_signed:
            colors.append(
                GREEN if val < 0
                else RED if val > 0
                else BLUE
            )
        else:
            colors.append(GOLD)

    fig = go.Figure(go.Bar(
        x           = [abs(v) for v in values],
        y           = labels,
        orientation = "h",
        marker      = dict(
            color        = colors,
            opacity      = 0.82,
            line         = dict(color=NAVY_CARD, width=0.5),
            cornerradius = 4,
        ),
        text         = [
            f"{v:+.3f}" if s else f"{abs(v):.3f}"
            for v, s in zip(values, signed)
        ],
        textposition = "outside",
        textfont     = dict(
            size   = 9,
            color  = TEXT_2,
            family = "DM Mono, monospace",
        ),
        hovertemplate= (
            "<b>%{y}</b><br>"
            "Score: <b>%{text}</b>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = "Bias Dimensions",
            font    = dict(
                size   = 10,
                family = "DM Mono, monospace",
                color  = TEXT_3,
            ),
            x=0, xanchor="left",
        ),
        xaxis = dict(
            range     = [0, 1.2],
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            tickfont  = dict(
                size   = 8,
                color  = TEXT_3,
                family = "DM Mono, monospace",
            ),
            zeroline = False,
        ),
        yaxis = dict(
            showgrid  = False,
            tickfont  = dict(
                size   = 9,
                color  = TEXT_2,
                family = "DM Mono, monospace",
            ),
            autorange = "reversed",
        ),
        height     = 260,
        margin     = dict(l=16, r=70, t=38, b=16),
        showlegend = False,
        hoverlabel = dict(
            bgcolor    = NAVY_2,
            bordercolor= GOLD,
            font_size  = 11,
            font_family= "DM Mono, monospace",
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────
# 7. SENTIMENT BAR CHART (Enhanced)
# ─────────────────────────────────────────────────────────────

def build_sentiment_bar_chart(
    per_article_sentiment: list,
    topic: str = "",
) -> go.Figure:
    """
    Enhanced sentiment bars with waterfall-style coloring
    and animated threshold lines.
    """
    if not per_article_sentiment:
        return _empty_chart("No sentiment data")

    sources   = []
    compounds = []
    labels    = []
    colors    = []
    stances   = []
    titles_h  = []

    for item in per_article_sentiment:
        source   = (item.get("source", "?") or "?")[:16]
        compound = float(item.get("compound", 0.0))
        label    = item.get("label",  "neutral")
        stance   = item.get("stance", "Neutral")
        title    = (item.get("title", "") or "")[:40]

        sources.append(source)
        compounds.append(round(compound, 4))
        labels.append(label)
        stances.append(stance)
        titles_h.append(title)

        if label == "positive":
            colors.append(GREEN)
        elif label == "negative":
            colors.append(RED)
        else:
            colors.append(BLUE)

    fig = go.Figure()

    # Bars
    fig.add_trace(go.Bar(
        x    = sources,
        y    = compounds,
        name = "VADER Score",
        marker = dict(
            color        = colors,
            opacity      = 0.85,
            line         = dict(color=NAVY_CARD, width=1.5),
            cornerradius = 4,
        ),
        text          = [f"{c:+.3f}" for c in compounds],
        textposition  = "outside",
        textfont      = dict(
            size   = 8,
            color  = TEXT_2,
            family = "DM Mono, monospace",
        ),
        customdata    = list(zip(titles_h, stances, labels)),
        hovertemplate = (
            "<b>%{x}</b><br>"
            "VADER: <b>%{y:+.4f}</b><br>"
            "Sentiment: %{customdata[2]}<br>"
            "Stance: %{customdata[1]}<br>"
            "<i>%{customdata[0]}</i>"
            "<extra></extra>"
        ),
    ))

    # Neutral baseline
    fig.add_hline(
        y=0,
        line_color = TEXT_3,
        line_width = 1,
        line_dash  = "solid",
    )

    # Threshold annotations
    for y_val, color, line_color, text in [
        ( 0.05, GREEN, GREEN_DIM, "+0.05 positive"),
        (-0.05, RED,   RED_DIM,   "-0.05 negative"),
    ]:
        fig.add_hline(
            y          = y_val,
            line_color = line_color,
            line_width = 1,
            line_dash  = "dash",
            annotation_text     = text,
            annotation_position = "right",
            annotation_font     = dict(
                size  = 7,
                color = color,
            ),
        )

    # Average line
    avg = float(np.mean(compounds)) if compounds else 0.0
    fig.add_hline(
        y          = avg,
        line_color = GOLD,
        line_width = 1.5,
        line_dash  = "dot",
        annotation_text     = f"avg {avg:+.3f}",
        annotation_position = "left",
        annotation_font     = dict(
            size  = 8,
            color = GOLD,
        ),
    )

    title_str = (
        f"VADER Sentiment — {topic[:38]}"
        if topic else "VADER Sentiment Scores"
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = title_str,
            font    = dict(
                size   = 11,
                color  = TEXT_1,
                family = "DM Mono, monospace",
            ),
            x=0.5, xanchor="center",
        ),
        xaxis = dict(
            tickfont  = dict(
                size   = 8,
                color  = TEXT_2,
                family = "DM Mono, monospace",
            ),
            showgrid  = False,
            tickangle = -30,
        ),
        yaxis = dict(
            title    = dict(
                text   = "Compound Score",
                font   = dict(
                    size   = 9,
                    color  = TEXT_3,
                    family = "DM Mono, monospace",
                ),
            ),
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            zeroline  = False,
            range     = [
                min(min(compounds) - 0.2, -0.4),
                max(max(compounds) + 0.2,  0.4),
            ],
        ),
        height     = 360,
        margin     = dict(l=55, r=80, t=55, b=75),
        showlegend = False,
        bargap     = 0.30,
        hoverlabel = dict(
            bgcolor    = NAVY_2,
            bordercolor= GOLD,
            font_size  = 11,
            font_family= "DM Mono, monospace",
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────
# 8. WORD CLOUD (Plotly scatter version)
# ─────────────────────────────────────────────────────────────

def build_word_cloud_chart(
    word_frequencies: dict,
    topic: str = "",
) -> go.Figure:
    """Interactive Plotly word cloud with gold gradient."""
    if not word_frequencies:
        return _empty_chart("No keyword data")

    sorted_words = sorted(
        word_frequencies.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:50]

    if not sorted_words:
        return _empty_chart("No keywords found")

    words  = [item[0] for item in sorted_words]
    scores = [item[1] for item in sorted_words]
    max_s  = max(scores) if scores else 1
    min_s  = min(scores) if scores else 0
    rng    = max_s - min_s if max_s != min_s else 1

    normalized = [(s - min_s) / rng for s in scores]
    font_sizes = [int(10 + n * 32) for n in normalized]

    random.seed(99)
    cols = 8

    x_pos = [
        (i % cols) + random.uniform(-0.3, 0.3)
        for i in range(len(words))
    ]
    y_pos = [
        (i // cols) + random.uniform(-0.25, 0.25)
        for i in range(len(words))
    ]

    colors = []
    for n in normalized:
        if n > 0.75:
            colors.append(GOLD_L)
        elif n > 0.50:
            colors.append(GOLD)
        elif n > 0.25:
            colors.append(TEXT_2)
        else:
            colors.append(TEXT_3)

    fig = go.Figure()

    for i, (word, score, x, y, size, color) in enumerate(
        zip(words, scores, x_pos, y_pos, font_sizes, colors)
    ):
        fig.add_trace(go.Scatter(
            x    = [x],
            y    = [y],
            mode = "text",
            text = [word],
            textfont = dict(
                size   = size,
                color  = color,
                family = "DM Sans, sans-serif",
            ),
            hovertemplate = (
                f"<b>{word}</b><br>"
                f"TF-IDF: {score:.4f}<br>"
                f"Rank #{i+1}"
                f"<extra></extra>"
            ),
            showlegend = False,
        ))

    title_str = (
        f"Keywords — {topic[:38]}"
        if topic else "Top TF-IDF Keywords"
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = title_str,
            font    = dict(
                size   = 11,
                color  = TEXT_1,
                family = "DM Mono, monospace",
            ),
            x=0.5, xanchor="center",
        ),
        xaxis = dict(
            showgrid       = False,
            showticklabels = False,
            zeroline       = False,
            showline       = False,
        ),
        yaxis = dict(
            showgrid       = False,
            showticklabels = False,
            zeroline       = False,
            showline       = False,
            autorange      = "reversed",
        ),
        height       = 360,
        margin       = dict(l=10, r=10, t=50, b=10),
        plot_bgcolor = NAVY_CARD,
        hoverlabel   = dict(
            bgcolor    = NAVY_2,
            bordercolor= GOLD,
            font_size  = 11,
            font_family= "DM Mono, monospace",
        ),
    )

    return fig


def build_word_cloud_matplotlib(
    word_frequencies: dict,
    topic: str = "",
):
    """Styled matplotlib word cloud."""
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not word_frequencies:
        return None

    try:
        wc = WordCloud(
            width            = 900,
            height           = 420,
            background_color = NAVY,
            colormap         = "YlOrBr",
            max_words        = 80,
            min_font_size    = 9,
            max_font_size    = 75,
            prefer_horizontal= 0.7,
            collocations     = False,
            margin           = 8,
        ).generate_from_frequencies(word_frequencies)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_facecolor(NAVY)
        ax.set_facecolor(NAVY)

        if topic:
            fig.suptitle(
                f"Word Cloud — {topic[:45]}",
                color      = TEXT_1,
                fontsize   = 10,
                fontfamily = "monospace",
                y          = 1.0,
            )

        plt.tight_layout(pad=0.5)
        return fig

    except Exception as e:
        print(f"[charts] Word cloud error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 9. ENTITY FREQUENCY CHART (Enhanced)
# ─────────────────────────────────────────────────────────────

def build_entity_frequency_chart(
    entity_analysis: dict,
    topic: str = "",
) -> go.Figure:
    """Grouped horizontal bars with color coding by entity type."""
    if not entity_analysis:
        return _empty_chart("No entity data")

    by_type = entity_analysis.get("by_type", {})
    if not by_type:
        return _empty_chart("No entities found")

    MAX_PER  = 6
    type_cfg = {
        "PERSON": {"label": "People",        "color": GREEN},
        "ORG":    {"label": "Organizations", "color": GOLD},
        "GPE":    {"label": "Places",        "color": BLUE},
    }

    fig = go.Figure()

    for etype, cfg in type_cfg.items():
        items = sorted(
            by_type.get(etype, []),
            key=lambda x: x.get("count", 0),
            reverse=True,
        )[:MAX_PER]

        if not items:
            continue

        names  = [item["text"]  for item in items]
        counts = [item["count"] for item in items]

        fig.add_trace(go.Bar(
            name        = cfg["label"],
            x           = counts,
            y           = names,
            orientation = "h",
            marker      = dict(
                color        = cfg["color"],
                opacity      = 0.82,
                line         = dict(color=NAVY_CARD, width=0.5),
                cornerradius = 4,
            ),
            text         = counts,
            textposition = "outside",
            textfont     = dict(
                size   = 8,
                color  = TEXT_2,
                family = "DM Mono, monospace",
            ),
            hovertemplate = (
                f"<b>%{{y}}</b><br>"
                f"Type: {cfg['label']}<br>"
                f"Mentions: <b>%{{x}}</b>"
                f"<extra></extra>"
            ),
        ))

    title_str = (
        f"Entities — {topic[:35]}"
        if topic else "Named Entity Frequency"
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = title_str,
            font    = dict(
                size   = 11,
                color  = TEXT_1,
                family = "DM Mono, monospace",
            ),
            x=0.5, xanchor="center",
        ),
        barmode = "group",
        xaxis   = dict(
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            tickfont  = dict(
                size   = 8,
                color  = TEXT_3,
                family = "DM Mono, monospace",
            ),
            zeroline = False,
        ),
        yaxis   = dict(
            showgrid  = False,
            tickfont  = dict(
                size   = 8,
                color  = TEXT_2,
                family = "DM Mono, monospace",
            ),
            autorange = "reversed",
        ),
        legend = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = -0.28,
            xanchor     = "center",
            x           = 0.5,
            font        = dict(
                size   = 9,
                color  = TEXT_2,
                family = "DM Mono, monospace",
            ),
            bgcolor     = "rgba(20,28,46,0.8)",
            bordercolor = BORDER,
            borderwidth = 1,
        ),
        height     = 380,
        margin     = dict(l=110, r=55, t=50, b=55),
        hoverlabel = dict(
            bgcolor    = NAVY_2,
            bordercolor= GOLD,
            font_size  = 11,
            font_family= "DM Mono, monospace",
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────
# 10. SENTIMENT TIMELINE (Enhanced)
# ─────────────────────────────────────────────────────────────

def build_sentiment_timeline_chart(
    per_article_sentiment: list,
    topic: str = "",
) -> go.Figure:
    """Scatter with connecting lines and stance-colored markers."""
    if not per_article_sentiment:
        return _empty_chart("No sentiment data")

    x_vals       = []
    y_vals       = []
    colors_list  = []
    sources      = []
    stances      = []
    titles_h     = []
    sizes        = []

    for i, item in enumerate(per_article_sentiment):
        compound = float(item.get("compound", 0.0))
        stance   = item.get("stance",  "Neutral")
        source   = (item.get("source", "?") or "?")[:18]
        title    = (item.get("title",  "") or "")[:35]

        x_vals.append(i + 1)
        y_vals.append(round(compound, 4))
        colors_list.append(
            STANCE_COLOR.get(stance, BLUE)
        )
        sources.append(source)
        stances.append(stance)
        titles_h.append(title)
        sizes.append(max(8, int(abs(compound) * 28) + 8))

    fig = go.Figure()

    # Connecting line
    fig.add_trace(go.Scatter(
        x    = x_vals,
        y    = y_vals,
        mode = "lines",
        name = "Trend",
        line = dict(
            color = "rgba(201,168,76,0.25)",
            width = 1.5,
            dash  = "dot",
        ),
        hoverinfo  = "skip",
        showlegend = False,
    ))

    # Zero line area fill
    fig.add_hrect(
        y0          = -0.05,
        y1          = 0.05,
        fillcolor   = "rgba(91,156,246,0.05)",
        line_width  = 0,
        annotation_text     = "neutral zone",
        annotation_position = "right",
        annotation_font     = dict(
            size  = 7,
            color = BLUE,
        ),
    )

    # Points per stance
    for stance_type in ["Supportive", "Critical", "Neutral"]:
        color = STANCE_COLOR[stance_type]
        mask  = [s == stance_type for s in stances]

        if not any(mask):
            continue

        fig.add_trace(go.Scatter(
            x    = [x for x, m in zip(x_vals,    mask) if m],
            y    = [y for y, m in zip(y_vals,    mask) if m],
            mode = "markers+text",
            name = stance_type,
            text = [
                s[:8]
                for s, m in zip(sources, mask) if m
            ],
            textposition = "top center",
            textfont     = dict(
                size   = 7,
                color  = TEXT_3,
                family = "DM Mono, monospace",
            ),
            marker = dict(
                color   = color,
                size    = [d for d, m in zip(sizes, mask) if m],
                opacity = 0.88,
                line    = dict(color=NAVY_CARD, width=1.5),
                symbol  = "circle",
            ),
            customdata = [
                [t, sc]
                for t, sc, m in zip(titles_h, stances, mask)
                if m
            ],
            hovertemplate = (
                "<b>%{text}</b><br>"
                "VADER: <b>%{y:+.4f}</b><br>"
                "Stance: %{customdata[1]}<br>"
                "<i>%{customdata[0]}</i>"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(
        y=0,
        line_color = TEXT_3,
        line_width = 1,
        line_dash  = "dot",
    )

    title_str = (
        f"Sentiment vs Stance — {topic[:30]}"
        if topic else "Sentiment vs Stance"
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = title_str,
            font    = dict(
                size   = 11,
                color  = TEXT_1,
                family = "DM Mono, monospace",
            ),
            x=0.5, xanchor="center",
        ),
        xaxis = dict(
            title    = dict(
                text   = "Article",
                font   = dict(
                    size   = 9,
                    color  = TEXT_3,
                    family = "DM Mono, monospace",
                ),
            ),
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            zeroline  = False,
        ),
        yaxis = dict(
            title    = dict(
                text   = "VADER Compound",
                font   = dict(
                    size   = 9,
                    color  = TEXT_3,
                    family = "DM Mono, monospace",
                ),
            ),
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            zeroline  = False,
        ),
        legend = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = -0.28,
            xanchor     = "center",
            x           = 0.5,
            font        = dict(
                size   = 9,
                color  = TEXT_2,
                family = "DM Mono, monospace",
            ),
            bgcolor     = "rgba(20,28,46,0.8)",
            bordercolor = BORDER,
            borderwidth = 1,
        ),
        height = 340,
        margin = dict(l=55, r=70, t=50, b=65),
        hoverlabel = dict(
            bgcolor    = NAVY_2,
            bordercolor= GOLD,
            font_size  = 11,
            font_family= "DM Mono, monospace",
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def _empty_chart(message: str = "No data") -> go.Figure:
    """Placeholder chart with message."""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text=message,
        showarrow = False,
        font      = dict(
            size   = 12,
            color  = TEXT_3,
            family = "DM Mono, monospace",
        ),
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        height = 180,
        margin = dict(l=20, r=20, t=20, b=20),
        xaxis  = dict(
            showgrid=False, showticklabels=False, zeroline=False
        ),
        yaxis  = dict(
            showgrid=False, showticklabels=False, zeroline=False
        ),
    )
    return fig