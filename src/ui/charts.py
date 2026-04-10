# src/ui/charts.py — Dark editorial theme matching ORBITA's navy/gold palette

import plotly.graph_objects as go
import random

# ── Palette (matches CSS variables) ──────────────────────────────────────────
NAVY      = "#0a0f1e"
NAVY_2    = "#111827"
NAVY_CARD = "#141c2e"
GOLD      = "#c9a84c"
GOLD_L    = "#e8c96a"
RED       = "#e05252"
GREEN     = "#3ec97e"
BLUE      = "#5b9cf6"
TEXT_1    = "#f0ebe0"
TEXT_2    = "#9ba8bb"
TEXT_3    = "#5c6b82"
BORDER    = "rgba(201,168,76,0.18)"

STANCE_COLOR = {
    "Supportive": GREEN,
    "Critical":   RED,
    "Neutral":    BLUE,
}
STANCE_BG = {
    "Supportive": "rgba(62,201,126,0.07)",
    "Critical":   "rgba(224,82,82,0.07)",
    "Neutral":    "rgba(91,156,246,0.07)",
}

_LAYOUT_BASE = dict(
    paper_bgcolor = NAVY_CARD,
    plot_bgcolor  = NAVY_CARD,
    font          = dict(family="DM Mono, monospace", color=TEXT_2),
    # margin removed to avoid conflict with specific margin in update_layout
)


def build_bias_spectrum_graph(
    articles:   list[dict],
    bias_score: float,
    topic:      str,
) -> go.Figure:
    """
    Dark-themed bias spectrum. Articles as coloured dots,
    overall score as a gold diamond.
    """
    # Input validation
    if not isinstance(articles, list):
        articles = []
    if not isinstance(bias_score, (int, float)):
        bias_score = 0.0
    if not isinstance(topic, str):
        topic = str(topic)

    random.seed(42)
    stance_base = {"Supportive": -0.62, "Neutral": 0.0, "Critical": 0.62}

    fig = go.Figure()

    # Background zones
    for x0, x1, color in [
        (-1.0, -0.15, STANCE_BG["Supportive"]),
        (-0.15, 0.15, STANCE_BG["Neutral"]),
        (0.15,  1.0,  STANCE_BG["Critical"]),
    ]:
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=0, y1=1,
            fillcolor=color, line_width=0, layer="below",
        )

    # Zone labels
    for x, lbl, col in [
        (-0.57, "SUPPORTIVE", GREEN),
        (0.0,   "NEUTRAL",    BLUE),
        (0.57,  "CRITICAL",   RED),
    ]:
        fig.add_annotation(
            x=x, y=0.94, text=lbl, showarrow=False,
            font=dict(size=8, color=col,
                      family="DM Mono, monospace"),
            xanchor="center",
        )

    # Centre line
    fig.add_vline(
        x=0, line_dash="dot",
        line_color="rgba(255,255,255,0.1)", line_width=1,
    )

    # Article dots by stance
    for stance in ["Supportive", "Critical", "Neutral"]:
        group = [a for a in articles if a.get("stance") == stance]
        if not group:
            continue

        xs, ys, titles, sources = [], [], [], []
        for a in group:
            base = stance_base.get(stance, 0.0)
            xs.append(max(-0.92, min(0.92,
                       base + random.uniform(-0.11, 0.11))))
            ys.append(random.uniform(0.3, 0.7))
            titles.append(a.get("title", "")[:55])
            sources.append(a.get("source", "Unknown"))

        fig.add_trace(go.Scatter(
            x    = xs, y = ys,
            mode = "markers",
            name = stance,
            marker = dict(
                color   = STANCE_COLOR[stance],
                size    = 13,
                opacity = 0.85,
                line    = dict(color=NAVY, width=2),
            ),
            customdata    = list(zip(titles, sources)),
            hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                f"<i>{stance}</i><extra></extra>"
            ),
        ))

    # Overall bias score diamond
    bias_color = (
        GREEN if bias_score < -0.2
        else RED if bias_score > 0.2
        else BLUE
    )
    text_side = "right" if bias_score <= 0.4 else "left"
    text_str  = (
        f"  {bias_score:+.2f}" if text_side == "right"
        else f"{bias_score:+.2f}  "
    )

    fig.add_trace(go.Scatter(
        x    = [bias_score], y = [0.1],
        mode = "markers+text",
        name = "Overall Score",
        marker = dict(
            color  = bias_color, size=28,
            symbol = "diamond",
            line   = dict(color=NAVY_CARD, width=3),
        ),
        text     = [text_str],
        textfont = dict(size=11, color=bias_color,
                        family="DM Mono, monospace"),
        textposition = f"middle {text_side}",
        hovertemplate = (
            f"<b>Overall Bias Score</b><br>{bias_score:+.2f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = f"<b style='color:{TEXT_1}'>{topic[:52]}</b>",
            font    = dict(size=13, family="DM Mono, monospace",
                           color=TEXT_1),
            x=0.5, xanchor="center", y=0.97,
        ),
        xaxis = dict(
            range     = [-1.05, 1.05],
            tickvals  = [-1, -0.5, 0, 0.5, 1],
            ticktext  = ["−1.0", "−0.5", "0", "+0.5", "+1.0"],
            tickfont  = dict(size=10, family="DM Mono, monospace",
                             color=TEXT_3),
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.04)",
            zeroline  = False,
            color     = TEXT_3,
        ),
        yaxis = dict(
            showticklabels=False, showgrid=False,
            zeroline=False, range=[0, 1],
        ),
        legend = dict(
            orientation="h", yanchor="bottom", y=-0.28,
            xanchor="center", x=0.5,
            font=dict(size=10, family="DM Mono, monospace",
                      color=TEXT_2),
            bgcolor="rgba(20,28,46,0.8)",
            bordercolor=BORDER, borderwidth=1,
        ),
        height    = 360,
        margin    = dict(l=16, r=16, t=46, b=70),
        hovermode = "closest",
    )
    return fig


def build_confidence_gauge(a_score: float, b_score: float) -> go.Figure:
    """Minimal dark bar chart for agent confidence scores."""
    fig = go.Figure()

    for label, score, color in [
        ("Agent A", a_score, GREEN),
        ("Agent B", b_score, RED),
    ]:
        fig.add_trace(go.Bar(
            x            = [label],
            y            = [score],
            marker_color = color,
            marker_opacity = 0.85,
            marker_line  = dict(color=NAVY_CARD, width=1),
            text         = [f"{score:.2f}"],
            textposition = "outside",
            textfont     = dict(size=11, color=color,
                                family="DM Mono, monospace"),
            width        = 0.4,
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = "Agent Confidence",
            font    = dict(size=11, family="DM Mono, monospace",
                           color=TEXT_3),
            x=0.5, xanchor="center",
        ),
        yaxis = dict(
            range=[0, 1.3],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            tickformat=".1f",
            tickfont=dict(size=9, family="DM Mono, monospace",
                          color=TEXT_3),
            zeroline=False,
        ),
        xaxis = dict(
            showgrid=False,
            tickfont=dict(size=10, family="DM Mono, monospace",
                          color=TEXT_2),
        ),
        height     = 220,
        margin     = dict(l=16, r=16, t=38, b=16),
        showlegend = False,
        bargap     = 0.5,
    )
    return fig


def build_stance_distribution_chart(articles: list[dict]) -> go.Figure:
    """Dark donut chart of article stance distribution."""
    counts = {"Supportive": 0, "Critical": 0, "Neutral": 0}
    for a in articles:
        s = a.get("stance", "Neutral")
        if s in counts:
            counts[s] += 1

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [STANCE_COLOR[l] for l in labels]

    fig = go.Figure(go.Pie(
        labels        = labels,
        values        = values,
        hole          = 0.62,
        marker        = dict(
            colors = colors,
            line   = dict(color=NAVY_CARD, width=3),
        ),
        textinfo      = "label+value",
        textfont      = dict(size=10, family="DM Mono, monospace"),
        hovertemplate = (
            "%{label}: %{value} articles (%{percent})<extra></extra>"
        ),
    ))

    total = sum(values)
    fig.add_annotation(
        text      = f"<b>{total}</b>",
        x=0.5, y=0.52,
        font      = dict(size=20, family="DM Mono, monospace",
                         color=TEXT_1),
        showarrow = False,
    )
    fig.add_annotation(
        text      = "articles",
        x=0.5, y=0.38,
        font      = dict(size=9, family="DM Mono, monospace",
                         color=TEXT_3),
        showarrow = False,
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = "Stance Mix",
            font    = dict(size=11, family="DM Mono, monospace",
                           color=TEXT_3),
            x=0.5, xanchor="center",
        ),
        height     = 220,
        margin     = dict(l=16, r=16, t=38, b=16),
        showlegend = False,
    )
    return fig


def build_word_count_chart(articles: list[dict]) -> go.Figure:
    """Horizontal bar — word counts per article coloured by stance."""
    if not articles:
        return go.Figure()

    titles = [
        f"{a.get('title', 'Unknown')[:32]}…"
        if len(a.get("title", "")) > 32
        else a.get("title", "Unknown")
        for a in articles
    ]
    words  = [len((a.get("full_text") or "").split()) for a in articles]
    colors = [STANCE_COLOR.get(a.get("stance", "Neutral"), TEXT_3)
              for a in articles]

    fig = go.Figure(go.Bar(
        x            = words,
        y            = titles,
        orientation  = "h",
        marker_color = colors,
        marker_opacity = 0.75,
        marker_line  = dict(color=NAVY_CARD, width=0.5),
        text         = [f"{w:,}" for w in words],
        textposition = "outside",
        textfont     = dict(size=9, family="DM Mono, monospace",
                            color=TEXT_3),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(
            text    = "Word Counts",
            font    = dict(size=11, family="DM Mono, monospace",
                           color=TEXT_3),
            x=0, xanchor="left",
        ),
        xaxis = dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=9, family="DM Mono, monospace",
                          color=TEXT_3),
            zeroline=False,
        ),
        yaxis = dict(
            showgrid=False,
            autorange="reversed",
            tickfont=dict(size=9, family="DM Mono, monospace",
                          color=TEXT_2),
        ),
        height     = max(180, len(articles) * 32 + 55),
        margin     = dict(l=16, r=55, t=38, b=16),
        showlegend = False,
    )
    return fig