# src/ui/components.py

import streamlit as st


def render_header():
    """Render the ORBITA header."""
    st.markdown("""
        <div class="orbita-header">
            <div class="orbita-title">ORBITA</div>
            <div class="orbita-divider"></div>
            <div class="orbita-tagline">
                Objective Reasoning And Bias Interpretation Tool for Analysis
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_bias_score_display(bias_score: float):
    """Render the bias score number with colour and label."""
    if bias_score < -0.3:
        color = "#10b981"
        label = "Leans Supportive"
    elif bias_score > 0.3:
        color = "#ef4444"
        label = "Leans Critical"
    else:
        color = "#3b82f6"
        label = "Balanced"

    st.markdown(
        f'<div class="bias-display">'
        f'<div class="bias-number" style="color:{color}">'
        f'{bias_score:+.2f}</div>'
        f'<div class="bias-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_metric_cards(
    n_articles:   int,
    n_chunks:     int,
    n_arguments:  int,
    n_counters:   int,
):
    """Render the four metric summary cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Articles Analysed", n_articles)
    with col2:
        st.metric("Text Chunks", n_chunks)
    with col3:
        st.metric("Supporting Args", n_arguments)
    with col4:
        st.metric("Counter Args", n_counters)


def render_agent_a_panel(agent_a: dict):
    """Render Agent A's output — Supporting Arguments."""
    arguments = agent_a.get("arguments", [])
    evidence  = agent_a.get("evidence",  [])
    score     = agent_a.get("confidence_score", 0.0)

    st.markdown(f"**Confidence Score:** `{score:.2f}`")

    if not arguments:
        st.info("No supporting arguments found for this topic.")
        return

    st.markdown("**Supporting Arguments:**")
    for arg in arguments:
        st.markdown(f"""
            <div class="stance-card stance-supportive">
                ✓ {arg}
            </div>
        """, unsafe_allow_html=True)

    if evidence:
        with st.expander("View Evidence & Data Points"):
            for ev in evidence:
                st.markdown(f"• {ev}")


def render_argument_trace(arg):
    st.markdown(
        f"**Argument:** {arg['argument']}"
    )

    with st.expander(
        "View Evidence"
    ):
        st.markdown(
            arg["evidence"]
        )

        st.link_button(
            "Open Source",
            arg["url"]
        )
def render_agent_b_panel(agent_b: dict):
    """Render Agent B's output — Counter Arguments."""
    counters = agent_b.get("counter_arguments", [])
    evidence = agent_b.get("evidence", [])
    score    = agent_b.get("confidence_score", 0.0)

    st.markdown(f"**Confidence Score:** `{score:.2f}`")

    if not counters:
        st.info("No counter-arguments found for this topic.")
        return

    st.markdown("**Counter-Arguments:**")
    for arg in counters:
        st.markdown(f"""
            <div class="stance-card stance-critical">
                ✗ {arg}
            </div>
        """, unsafe_allow_html=True)

    if evidence:
        with st.expander("View Critical Evidence"):
            for ev in evidence:
                st.markdown(f"• {ev}")


def render_synthesis(report: dict):
    """Render Agent C's synthesis report."""
    synthesis = report.get("synthesis_report", "")

    if not synthesis or len(synthesis) < 20:
        st.warning("Synthesis could not be generated.")
        return

    st.markdown(f"""
        <div class="synthesis-box">
            {synthesis.replace(chr(10), '<br>')}
        </div>
    """, unsafe_allow_html=True)

    # Key agreements
    agreements = report.get("key_agreements", [])
    if agreements:
        st.markdown("**Points Both Sides Agree On:**")
        for a in agreements:
            st.markdown(f"""
                <div class="stance-card stance-neutral">
                    ~ {a}
                </div>
            """, unsafe_allow_html=True)

    # Key disagreements
    disagreements = report.get("key_disagreements", [])
    if disagreements:
        st.markdown("**Core Disagreements:**")
        for d in disagreements:
            st.markdown(f"&nbsp;&nbsp;&nbsp;✗ {d}", unsafe_allow_html=True)


def render_loaded_language(report: dict):
    """Render the loaded language removal panel."""
    removed = report.get("loaded_language_removed", [])

    if not removed:
        st.success("No significantly loaded language detected in source articles.")
        return

    st.markdown(f"**{len(removed)} loaded phrase(s) neutralised:**")
    for phrase in removed:
        st.markdown(f"""
            <div class="lang-item">
                {phrase}
            </div>
        """, unsafe_allow_html=True)


def render_hallucination_report(report: dict):
    """Render the hallucination check results."""
    flags = report.get("hallucination_flags", [])

    if not flags:
        st.success(
            "All factual claims by agents were verified "
            "against the source article excerpts."
        )
        return

    st.warning(f"{len(flags)} claim(s) could not be verified against sources:")
    for flag in flags:
        st.markdown(f"""
            <div class="flag-item">
                ⚠ {flag}
            </div>
        """, unsafe_allow_html=True)


def render_source_transparency(articles: list[dict]):
    """Render the source transparency table."""
    if not articles:
        st.info("No source articles available.")
        return

    stance_icons = {
        "Supportive": "🟢",
        "Critical":   "🔴",
        "Neutral":    "🔵",
    }

    for i, article in enumerate(articles, 1):
        title  = article.get("title",  "Unknown Title")[:80]
        source = article.get("source", "Unknown Source")
        stance = article.get("stance", "Neutral")
        url    = article.get("url",    "#")
        words  = len((article.get("full_text") or "").split())
        icon   = stance_icons.get(stance, "⚪")

        col1, col2, col3 = st.columns([6, 2, 2])
        with col1:
            st.markdown(f"**{i}. {title}**")
            st.caption(f"{source}")
        with col2:
            st.markdown(f"{icon} {stance}")
            if words > 0:
                st.caption(f"{words} words")
        with col3:
            st.link_button("Read Article", url, use_container_width=True)

        st.divider()