# src/ui/debate_viz.py
"""
ORBITA Agent Debate Visualization
===================================
Renders the multi-agent debate as a visual animated board.

Shows:
    Left  panel: Agent A (Analyst) — green supporting arguments
    Center panel: Agent C (Arbitrator) — gold verdict
    Right panel:  Agent B (Critic)   — red counter-arguments

Each argument card shows:
    - The argument text
    - Evidence supporting it
    - Source citation
    - Confidence indicator

This makes the multi-agent architecture visible and
understandable to non-technical examiners.

"Sir, you can literally see the three agents debating
 in real time — Agent A argues, Agent B counters,
 and Agent C synthesizes a neutral verdict."
"""

import streamlit as st


# ── Colors ────────────────────────────────────────────────────
NAVY_CARD  = "#141c2e"
NAVY_2     = "#111827"
NAVY_3     = "#1a2236"
GOLD       = "#c9a84c"
GOLD_L     = "#e8c96a"
RED        = "#e05252"
RED_DIM    = "rgba(224,82,82,0.10)"
GREEN      = "#3ec97e"
GREEN_DIM  = "rgba(62,201,126,0.10)"
BLUE       = "#5b9cf6"
BLUE_DIM   = "rgba(91,156,246,0.10)"
TEXT_1     = "#f0ebe0"
TEXT_2     = "#9ba8bb"
TEXT_3     = "#5c6b82"
BORDER     = "rgba(201,168,76,0.18)"
BORDER_DIM = "rgba(255,255,255,0.06)"


def _argument_card(
    text:       str,
    evidence:   str,
    source:     str,
    card_type:  str,   # "pro" | "con" | "verdict"
    index:      int,
    confidence: float = 0.0,
) -> str:
    """Build HTML for a single argument card."""
    if card_type == "pro":
        bg_color    = GREEN_DIM
        border_color= GREEN
        icon        = "+"
        icon_bg     = "rgba(62,201,126,0.2)"
        icon_color  = GREEN
        text_color  = "#b8f0d2"
    elif card_type == "con":
        bg_color    = RED_DIM
        border_color= RED
        icon        = "−"
        icon_bg     = "rgba(224,82,82,0.2)"
        icon_color  = RED
        text_color  = "#f5c0c0"
    else:  # verdict
        bg_color    = "rgba(201,168,76,0.08)"
        border_color= GOLD
        icon        = "◈"
        icon_bg     = "rgba(201,168,76,0.2)"
        icon_color  = GOLD
        text_color  = TEXT_1

    # Confidence bar width
    conf_w = int(confidence * 100) if confidence else 75

    src_short = (source or "")[:20]
    ev_text   = evidence or ""
    tx_text   = text or ""

    evidence_html = ""
    if ev_text:
        evidence_html = (
            f'<div style="font-family:DM Mono,monospace;font-size:0.65rem;'
            f'color:{TEXT_3};border-left:2px solid {border_color}44;'
            f'padding-left:0.4rem;margin-bottom:0.4rem;line-height:1.5">'
            f'{ev_text}'
            f'</div>'
        )

    return (
        f'<div style="'
        f'background:{bg_color};'
        f'border:1px solid {border_color}33;'
        f'border-left:3px solid {border_color};'
        f'border-radius:10px;'
        f'padding:0.85rem 1rem;'
        f'margin-bottom:0.55rem;'
        f'animation:fadeInLeft 0.4s ease {index * 0.08:.2f}s both;'
        f'transition:transform 0.15s,box-shadow 0.15s">'

        # Icon + argument text
        f'<div style="display:flex;gap:0.6rem;align-items:flex-start">'
        f'<div style="'
        f'width:20px;height:20px;border-radius:50%;'
        f'background:{icon_bg};'
        f'color:{icon_color};'
        f'font-weight:700;font-size:0.75rem;'
        f'display:flex;align-items:center;justify-content:center;'
        f'flex-shrink:0;margin-top:0.1rem">{icon}</div>'
        f'<div style="flex:1;min-width:0">'
        f'<div style="font-family:\'DM Sans\',sans-serif;'
        f'font-size:0.83rem;color:{text_color};'
        f'line-height:1.55;margin-bottom:0.4rem">'
        f'{tx_text}'
        f'</div>'

        # Evidence
        f'{evidence_html}'

        # Source + confidence bar
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center">'

        f'<span style="font-family:\'DM Mono\',monospace;'
        f'font-size:0.6rem;color:{TEXT_3};'
        f'background:{NAVY_3};border-radius:3px;'
        f'padding:1px 5px">{src_short}</span>'

        f'<div style="display:flex;align-items:center;gap:0.3rem">'
        f'<div style="width:40px;height:3px;'
        f'background:rgba(255,255,255,0.06);border-radius:2px">'
        f'<div style="width:{conf_w}%;height:3px;'
        f'background:{border_color};border-radius:2px"></div>'
        f'</div>'
        f'<span style="font-family:\'DM Mono\',monospace;'
        f'font-size:0.58rem;color:{TEXT_3}">'
        f'{confidence:.0%}</span>'
        f'</div>'

        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def _verdict_card(
    text:             str,
    bias_score:       float,
    key_agreements:   list,
    key_disagreements:list,
    loaded_removed:   list,
) -> str:
    """Build the central arbitrator verdict card."""
    bias_color = (
        GREEN if bias_score < -0.2
        else RED if bias_score > 0.2
        else BLUE
    )
    direction = (
        "Leans Supportive" if bias_score < -0.2
        else "Leans Critical" if bias_score > 0.2
        else "Balanced"
    )

    agreements_html = "".join(
        f'<div style="display:flex;gap:0.4rem;padding:0.25rem 0;'
        f'border-bottom:1px solid {BORDER_DIM};'
        f'font-size:0.72rem;color:{TEXT_2};line-height:1.4">'
        f'<span style="color:{GOLD};flex-shrink:0">~</span>'
        f'{ag[:80]}</div>'
        for ag in key_agreements[:3]
    )

    disagree_html = "".join(
        f'<div style="display:flex;gap:0.4rem;padding:0.25rem 0;'
        f'border-bottom:1px solid {BORDER_DIM};'
        f'font-size:0.72rem;color:{TEXT_2};line-height:1.4">'
        f'<span style="color:{RED};flex-shrink:0">×</span>'
        f'{dg[:80]}</div>'
        for dg in key_disagreements[:3]
    )

    agreements_section_html = ""
    if agreements_html:
        agreements_section_html = (
            f'<div style="font-family:DM Mono,monospace;font-size:0.58rem;'
            f'letter-spacing:1.5px;color:{GOLD};text-transform:uppercase;'
            f'margin-bottom:0.3rem">Agreements</div>'
            f'{agreements_html}'
        )

    disagreements_section_html = ""
    if disagree_html:
        disagreements_section_html = (
            f'<div style="font-family:DM Mono,monospace;font-size:0.58rem;'
            f'letter-spacing:1.5px;color:{RED};text-transform:uppercase;'
            f'margin:0.5rem 0 0.3rem">Disagreements</div>'
            f'{disagree_html}'
        )

    loaded_removed_html = ""
    if loaded_removed:
        loaded_removed_html = (
            f'<div style="font-family:DM Mono,monospace;font-size:0.58rem;'
            f'color:{TEXT_3};margin-top:0.6rem">'
            f'{len(loaded_removed)} loaded phrase(s) neutralised'
            f'</div>'
        )

    return (
        f'<div style="'
        f'background:rgba(201,168,76,0.05);'
        f'border:1px solid {GOLD}44;'
        f'border-top:3px solid {GOLD};'
        f'border-radius:12px;'
        f'padding:1.1rem 1rem;'
        f'height:100%;'
        f'animation:fadeInUp 0.5s ease both">'

        # Header
        f'<div style="font-family:\'DM Mono\',monospace;'
        f'font-size:0.58rem;letter-spacing:2px;color:{GOLD};'
        f'text-transform:uppercase;margin-bottom:0.5rem">Verdict</div>'

        # Bias score
        f'<div style="display:flex;align-items:baseline;'
        f'gap:0.5rem;margin-bottom:0.8rem">'
        f'<div style="font-family:\'Playfair Display\',serif;'
        f'font-size:2rem;font-weight:700;color:{bias_color}">'
        f'{bias_score:+.3f}</div>'
        f'<div style="font-family:\'DM Mono\',monospace;'
        f'font-size:0.6rem;color:{TEXT_3};'
        f'text-transform:uppercase;letter-spacing:1px">'
        f'{direction}</div>'
        f'</div>'

        # Synthesis preview
        f'<div style="font-family:\'DM Sans\',sans-serif;'
        f'font-size:0.75rem;color:{TEXT_2};'
        f'line-height:1.6;margin-bottom:0.8rem;'
        f'border-bottom:1px solid {BORDER_DIM};'
        f'padding-bottom:0.6rem">'
        f'{text}'
        f'</div>'

        # Agreements
        f'{agreements_section_html}'

        # Disagreements
        f'{disagreements_section_html}'

        # Loaded language removed
        f'{loaded_removed_html}'

        f'</div>'
    )


def render_debate_board(report: dict) -> None:
    """
    Render the complete agent debate visualization.

    This is the main function called from app.py.
    Shows three columns: Agent A | Agent C | Agent B

    Args:
        report: the full report dict from pipeline
    """
    agent_a  = report.get("agent_a", {})
    agent_b  = report.get("agent_b", {})
    agent_c  = report.get("agent_c", {}) or report

    arguments  = agent_a.get("arguments",         [])
    evidences_a= agent_a.get("evidence",           [])
    sources_a  = agent_a.get("key_sources",        [])
    conf_a     = float(agent_a.get("confidence_score", 0.75))

    counters   = agent_b.get("counter_arguments",  [])
    evidences_b= agent_b.get("evidence",           [])
    sources_b  = agent_b.get("key_sources",        [])
    conf_b     = float(agent_b.get("confidence_score", 0.75))

    synthesis     = report.get("synthesis_report",       "")
    bias_score    = float(report.get("bias_score",       0.0))
    agreements    = report.get("key_agreements",         [])
    disagreements = report.get("key_disagreements",      [])
    loaded        = report.get("loaded_language_removed",[])

    # ── Column headers ────────────────────────────────────────
    h_col_a, h_col_c, h_col_b = st.columns([5, 4, 5])

    with h_col_a:
        st.markdown(
            f'<div style="text-align:center;padding:0.6rem;'
            f'background:{GREEN_DIM};'
            f'border:1px solid {GREEN}33;'
            f'border-radius:8px;margin-bottom:0.8rem">'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.6rem;letter-spacing:2px;'
            f'color:{GREEN};text-transform:uppercase">'
            f'⊕ Agent A — Analyst</div>'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.65rem;color:{TEXT_3};margin-top:0.2rem">'
            f'{len(arguments)} supporting arguments · '
            f'conf {conf_a:.0%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with h_col_c:
        st.markdown(
            f'<div style="text-align:center;padding:0.6rem;'
            f'background:rgba(201,168,76,0.08);'
            f'border:1px solid {GOLD}33;'
            f'border-radius:8px;margin-bottom:0.8rem">'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.6rem;letter-spacing:2px;'
            f'color:{GOLD};text-transform:uppercase">'
            f'⚖️ Agent C — Arbitrator</div>'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.65rem;color:{TEXT_3};margin-top:0.2rem">'
            f'Neutral synthesis · hallucination check</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with h_col_b:
        st.markdown(
            f'<div style="text-align:center;padding:0.6rem;'
            f'background:{RED_DIM};'
            f'border:1px solid {RED}33;'
            f'border-radius:8px;margin-bottom:0.8rem">'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.6rem;letter-spacing:2px;'
            f'color:{RED};text-transform:uppercase">'
            f'⊖ Agent B — Critic</div>'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.65rem;color:{TEXT_3};margin-top:0.2rem">'
            f'{len(counters)} counter-arguments · '
            f'conf {conf_b:.0%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Three-column debate board ─────────────────────────────
    col_a, col_c, col_b = st.columns([5, 4, 5])

    # Agent A arguments
    with col_a:
        if not arguments:
            st.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;'
                f'font-size:0.75rem;color:{TEXT_3};'
                f'text-align:center;padding:1rem">No arguments found</div>',
                unsafe_allow_html=True,
            )
        else:
            for i, arg in enumerate(arguments[:6]):
                ev  = evidences_a[i] if i < len(evidences_a) else ""
                src = sources_a[0] if sources_a else ""
                st.markdown(
                    _argument_card(
                        text       = arg,
                        evidence   = ev,
                        source     = src,
                        card_type  = "pro",
                        index      = i,
                        confidence = conf_a,
                    ),
                    unsafe_allow_html=True,
                )

    # Agent C verdict (center)
    with col_c:
        st.markdown(
            _verdict_card(
                text              = synthesis,
                bias_score        = bias_score,
                key_agreements    = agreements,
                key_disagreements = disagreements,
                loaded_removed    = loaded,
            ),
            unsafe_allow_html=True,
        )

    # Agent B counter-arguments
    with col_b:
        if not counters:
            st.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;'
                f'font-size:0.75rem;color:{TEXT_3};'
                f'text-align:center;padding:1rem">'
                f'No counter-arguments found</div>',
                unsafe_allow_html=True,
            )
        else:
            for i, arg in enumerate(counters[:6]):
                ev  = evidences_b[i] if i < len(evidences_b) else ""
                src = sources_b[0] if sources_b else ""
                st.markdown(
                    _argument_card(
                        text       = arg,
                        evidence   = ev,
                        source     = src,
                        card_type  = "con",
                        index      = i,
                        confidence = conf_b,
                    ),
                    unsafe_allow_html=True,
                )

    # ── Arrow indicators ──────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;justify-content:center;'
        f'gap:2rem;margin-top:0.8rem;'
        f'font-family:\'DM Mono\',monospace;font-size:0.65rem;'
        f'color:{TEXT_3}">'
        f'<span style="color:{GREEN}">Agent A argues →</span>'
        f'<span style="color:{GOLD}">Agent C arbitrates</span>'
        f'<span style="color:{RED}">← Agent B counters</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Hallucination flags (below debate) ────────────────────
    flags = report.get("hallucination_flags", [])
    if flags:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(
            f"⚑ {len(flags)} Hallucination Flags — "
            f"Claims Agent C could not verify",
            expanded=False,
        ):
            for flag in flags[:8]:
                st.markdown(
                    f'<div style="background:rgba(201,168,76,0.06);'
                    f'border-left:3px solid {GOLD};'
                    f'padding:0.4rem 0.7rem;border-radius:0 6px 6px 0;'
                    f'margin-bottom:0.3rem;'
                    f'font-family:\'DM Mono\',monospace;'
                    f'font-size:0.72rem;color:{TEXT_2};'
                    f'line-height:1.5">'
                    f'⚑ &nbsp;{flag[:120]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )