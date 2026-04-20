# src/live_meter.py
"""
ORBITA Live Bias Meter
======================
Real-time progress display during pipeline execution.

Why This Exists:
    The ORBITA pipeline takes 2-5 minutes on live runs.
    A blank screen with a spinner looks unprofessional.
    This module shows a live status log, progress bar,
    and animated meter that updates as each phase completes.

Features:
    - Step-by-step log with icons and colors
    - Animated progress bar (0 → 100%)
    - Live bias score preview that refines over time
    - Phase timing display
    - Error handling with recovery messages

Usage:
    from src.live_meter import LiveBiasMeter
    
    meter = LiveBiasMeter()
    meter.start("India Elections 2024")
    meter.phase_complete("fetching",    n_articles=12)
    meter.phase_complete("nlp",         vader_score=-0.23)
    meter.phase_complete("embedding",   n_chunks=87)
    meter.phase_complete("agents",      bias_score=0.31)
    meter.finish(bias_score=0.31)
"""

import time
import streamlit as st
from typing import Optional


# ── Color constants ────────────────────────────────────────────
GOLD       = "#c9a84c"
GREEN      = "#3ec97e"
RED        = "#e05252"
BLUE       = "#5b9cf6"
NAVY_2     = "#111827"
NAVY_CARD  = "#141c2e"
TEXT_1     = "#f0ebe0"
TEXT_2     = "#9ba8bb"
TEXT_3     = "#5c6b82"
BORDER     = "rgba(201,168,76,0.18)"
BORDER_DIM = "rgba(255,255,255,0.06)"


# ── Pipeline phases definition ────────────────────────────────
PHASES = [
    {
        "key":     "start",
        "label":   "Initializing ORBITA",
        "icon":    "◈",
        "color":   GOLD,
        "pct":     2,
        "detail":  "Setting up pipeline and loading models...",
    },
    {
        "key":     "intent",
        "label":   "Decoding Intent (spaCy NER)",
        "icon":    "🔬",
        "color":   BLUE,
        "pct":     8,
        "detail":  "Extracting topic entities and building search queries...",
    },
    {
        "key":     "fetching",
        "label":   "Fetching News Articles",
        "icon":    "📡",
        "color":   BLUE,
        "pct":     18,
        "detail":  "Querying NewsAPI across multiple search angles...",
    },
    {
        "key":     "stance",
        "label":   "Zero-Shot Stance Classification",
        "icon":    "🏷️",
        "color":   GOLD,
        "pct":     28,
        "detail":  "Labeling articles as Supportive / Critical / Neutral...",
    },
    {
        "key":     "scraping",
        "label":   "Scraping Full Article Text",
        "icon":    "📄",
        "color":   BLUE,
        "pct":     38,
        "detail":  "newspaper4k extracting full content from URLs...",
    },
    {
        "key":     "nlp",
        "label":   "NLP Analysis (VADER + spaCy + TF-IDF)",
        "icon":    "🧠",
        "color":   GREEN,
        "pct":     48,
        "detail":  "Computing sentiment scores and extracting entities...",
    },
    {
        "key":     "cnn",
        "label":   "CNN Image Analysis (ResNet-50)",
        "icon":    "🖼️",
        "color":   GOLD,
        "pct":     55,
        "detail":  "Analyzing visual sentiment from article images...",
    },
    {
        "key":     "embedding",
        "label":   "Generating Gemini Embeddings",
        "icon":    "⚛️",
        "color":   BLUE,
        "pct":     65,
        "detail":  "Vectorizing text chunks for semantic retrieval...",
    },
    {
        "key":     "chromadb",
        "label":   "Storing in ChromaDB",
        "icon":    "🗄️",
        "color":   BLUE,
        "pct":     70,
        "detail":  "Persisting vector store for RAG retrieval...",
    },
    {
        "key":     "agent_a",
        "label":   "Agent A — Extracting Supporting Arguments",
        "icon":    "⊕",
        "color":   GREEN,
        "pct":     78,
        "detail":  "RAG retrieval + argument mining from supportive sources...",
    },
    {
        "key":     "agent_b",
        "label":   "Agent B — Extracting Counter-Arguments",
        "icon":    "⊖",
        "color":   RED,
        "pct":     86,
        "detail":  "RAG retrieval + critical argument extraction...",
    },
    {
        "key":     "agent_c",
        "label":   "Agent C — Synthesizing & Hallucination Check",
        "icon":    "⚖️",
        "color":   GOLD,
        "pct":     94,
        "detail":  "Cross-referencing claims + bias vector computation...",
    },
    {
        "key":     "saving",
        "label":   "Saving Report",
        "icon":    "💾",
        "color":   GOLD,
        "pct":     98,
        "detail":  "Writing JSON report and updating cache...",
    },
]

# Map phase key → index for quick lookup
PHASE_INDEX = {p["key"]: i for i, p in enumerate(PHASES)}


class LiveBiasMeter:
    """
    Real-time pipeline progress display for ORBITA.

    Usage pattern:
        meter = LiveBiasMeter()
        meter.start(topic)
        # ... run phase 1 ...
        meter.phase_complete("intent",    queries=4)
        # ... run phase 2 ...
        meter.phase_complete("fetching",  n_articles=12)
        # ... continue ...
        meter.finish(bias_score=0.23)
    """

    def __init__(self):
        # Streamlit placeholder elements
        self._header_box  = None
        self._progress_bar= None
        self._log_box     = None
        self._meter_box   = None
        self._timer_box   = None

        # State
        self._topic       = ""
        self._start_time  = 0.0
        self._done_phases = []  # list of rendered log lines
        self._current_pct = 0
        self._preview_score = None

    # ── Public API ────────────────────────────────────────────

    def start(self, topic: str) -> None:
        """
        Initialize the meter UI and show start state.
        Call this BEFORE running any pipeline phases.
        """
        self._topic      = topic
        self._start_time = time.time()
        self._done_phases= []
        self._current_pct= 0

        # Create all placeholder elements in order
        self._header_box   = st.empty()
        self._meter_box    = st.empty()
        self._progress_bar = st.empty()
        self._timer_box    = st.empty()
        self._log_box      = st.empty()

        # Render initial state
        self._render_header()
        self._render_meter(0, None)
        self._render_progress(2)
        self._render_log()

    def phase_complete(
        self,
        phase_key:   str,
        # Optional detail kwargs for richer messages
        n_articles:  Optional[int]   = None,
        n_chunks:    Optional[int]   = None,
        n_queries:   Optional[int]   = None,
        vader_score: Optional[float] = None,
        bias_score:  Optional[float] = None,
        n_args:      Optional[int]   = None,
        n_counters:  Optional[int]   = None,
        n_images:    Optional[int]   = None,
        extra:       str             = "",
    ) -> None:
        """
        Mark a pipeline phase as complete and update UI.
        Call this immediately after each phase finishes.
        """
        phase = self._get_phase(phase_key)
        if not phase:
            return

        elapsed = round(time.time() - self._start_time, 1)
        pct     = phase["pct"]

        # Build detail message
        detail = self._build_detail(
            phase_key, n_articles, n_chunks, n_queries,
            vader_score, bias_score, n_args, n_counters,
            n_images, extra,
        )

        # Add to log
        self._done_phases.append({
            "icon":    phase["icon"],
            "label":   phase["label"],
            "color":   phase["color"],
            "detail":  detail,
            "elapsed": elapsed,
            "pct":     pct,
        })

        self._current_pct = pct

        # Update preview bias if available
        if bias_score is not None:
            self._preview_score = bias_score
        if vader_score is not None and self._preview_score is None:
            self._preview_score = vader_score * 0.5  # rough preview

        # Re-render all UI elements
        self._render_meter(pct, self._preview_score)
        self._render_progress(pct)
        self._render_timer(elapsed)
        self._render_log()

    def finish(self, bias_score: float) -> None:
        """
        Mark pipeline as complete with final bias score.
        Clears the meter and shows success state.
        """
        elapsed = round(time.time() - self._start_time, 1)

        # Clear all placeholder elements
        self._progress_bar.empty()
        self._timer_box.empty()

        # Show final success state in header
        bias_color = (
            GREEN if bias_score < -0.2
            else RED if bias_score > 0.2
            else BLUE
        )
        direction = (
            "Leans Supportive"  if bias_score < -0.2
            else "Leans Critical" if bias_score > 0.2
            else "Balanced"
        )

        self._header_box.markdown(
            f'<div style="'
            f'background:{NAVY_CARD};'
            f'border:1px solid {bias_color}44;'
            f'border-left:4px solid {bias_color};'
            f'border-radius:12px;'
            f'padding:1rem 1.4rem;'
            f'margin-bottom:0.5rem;'
            f'animation:fadeInUp 0.5s ease">'

            # Top row
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:0.4rem">'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.65rem;letter-spacing:2.5px;'
            f'color:{bias_color};text-transform:uppercase">'
            f'✓ Analysis Complete</div>'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.65rem;color:{TEXT_3}">'
            f'⏱ {elapsed}s total</div>'
            f'</div>'

            # Topic
            f'<div style="font-family:\'Playfair Display\',serif;'
            f'font-size:1.1rem;color:{TEXT_1};margin-bottom:0.5rem">'
            f'{self._topic[:60]}</div>'

            # Bias score + direction
            f'<div style="display:flex;align-items:baseline;gap:0.8rem">'
            f'<div style="font-family:\'Playfair Display\',serif;'
            f'font-size:2rem;font-weight:700;color:{bias_color}">'
            f'{bias_score:+.3f}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.7rem;color:{TEXT_3};letter-spacing:1px;'
            f'text-transform:uppercase">{direction}</div>'
            f'</div>'

            f'</div>',
            unsafe_allow_html=True,
        )

        # Clear meter and log
        self._meter_box.empty()
        self._log_box.empty()

    def error(self, phase_key: str, message: str) -> None:
        """Show an error state for a phase."""
        phase   = self._get_phase(phase_key)
        elapsed = round(time.time() - self._start_time, 1)
        label   = phase["label"] if phase else phase_key

        self._done_phases.append({
            "icon":    "✗",
            "label":   f"{label} — FAILED",
            "color":   RED,
            "detail":  message[:80],
            "elapsed": elapsed,
            "pct":     self._current_pct,
            "is_error": True,
        })
        self._render_log()

    # ── Private Render Methods ────────────────────────────────

    def _render_header(self) -> None:
        """Render the animated 'Analysing...' header card."""
        self._header_box.markdown(
            f'<div style="'
            f'background:{NAVY_CARD};'
            f'border:1px solid {BORDER};'
            f'border-left:4px solid {GOLD};'
            f'border-radius:12px;'
            f'padding:1rem 1.4rem;'
            f'margin-bottom:0.5rem">'

            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.65rem;letter-spacing:2.5px;'
            f'color:{GOLD};text-transform:uppercase;'
            f'margin-bottom:0.4rem">◈ Analysing</div>'

            f'<div style="font-family:\'Playfair Display\',serif;'
            f'font-size:1.15rem;color:{TEXT_1}">'
            f'{self._topic[:65]}</div>'

            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.68rem;color:{TEXT_3};margin-top:0.3rem">'
            f'Do not close this tab — pipeline is running...</div>'

            f'</div>',
            unsafe_allow_html=True,
        )

    def _render_meter(
        self,
        pct:         int,
        bias_preview: Optional[float],
    ) -> None:
        """
        Render the animated bias meter that fills as pipeline runs.
        Shows a live preview of the emerging bias score.
        """
        # Build meter fill color
        if bias_preview is not None:
            if bias_preview < -0.2:
                fill_color = GREEN
                fill_label = f"emerging: {bias_preview:+.2f} supportive"
            elif bias_preview > 0.2:
                fill_color = RED
                fill_label = f"emerging: {bias_preview:+.2f} critical"
            else:
                fill_color = BLUE
                fill_label = f"emerging: {bias_preview:+.2f} balanced"
        else:
            fill_color = GOLD
            fill_label = "computing..."

        # Find current phase label
        current_label = "Initializing..."
        for phase in PHASES:
            if phase["pct"] <= pct:
                current_label = phase["label"]

        self._meter_box.markdown(
            f'<div style="'
            f'background:{NAVY_CARD};'
            f'border:1px solid {BORDER_DIM};'
            f'border-radius:10px;'
            f'padding:1rem 1.2rem;'
            f'margin-bottom:0.4rem">'

            # Phase label
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:0.6rem">'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.72rem;color:{GOLD}">'
            f'▶ {current_label}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.68rem;color:{TEXT_3}">'
            f'{pct}%</div>'
            f'</div>'

            # Wide progress bar
            f'<div style="'
            f'height:6px;'
            f'background:rgba(255,255,255,0.06);'
            f'border-radius:3px;'
            f'margin-bottom:0.5rem;'
            f'overflow:hidden">'
            f'<div style="'
            f'height:6px;'
            f'width:{pct}%;'
            f'background:linear-gradient(90deg,{GOLD},{fill_color});'
            f'border-radius:3px;'
            f'transition:width 0.5s ease"></div>'
            f'</div>'

            # Bias preview label
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.62rem;color:{TEXT_3};text-align:right">'
            f'bias score {fill_label}</div>'

            f'</div>',
            unsafe_allow_html=True,
        )

    def _render_progress(self, pct: int) -> None:
        """Render Streamlit native progress bar."""
        self._progress_bar.progress(
            pct / 100,
            text=None,
        )

    def _render_timer(self, elapsed: float) -> None:
        """Render elapsed time."""
        self._timer_box.markdown(
            f'<div style="font-family:\'DM Mono\',monospace;'
            f'font-size:0.65rem;color:{TEXT_3};'
            f'text-align:right;margin-bottom:0.3rem">'
            f'⏱ {elapsed}s elapsed</div>',
            unsafe_allow_html=True,
        )

    def _render_log(self) -> None:
        """Render the scrollable step log."""
        if not self._done_phases:
            self._log_box.markdown(
                f'<div style="'
                f'background:#07090f;'
                f'border:1px solid {BORDER_DIM};'
                f'border-radius:10px;'
                f'padding:0.8rem 1.2rem;'
                f'font-family:\'DM Mono\',monospace;'
                f'font-size:0.75rem;'
                f'color:{TEXT_3};'
                f'min-height:40px">'
                f'Starting pipeline...'
                f'</div>',
                unsafe_allow_html=True,
            )
            return

        # Build log HTML — most recent at bottom
        lines_html = []
        for i, step in enumerate(self._done_phases):
            color    = step.get("color", GOLD)
            is_error = step.get("is_error", False)
            icon     = "✗" if is_error else "✓"
            icon_color = RED if is_error else color

            line = (
                f'<div style="'
                f'display:flex;gap:0.6rem;align-items:flex-start;'
                f'padding:0.2rem 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.03);'
                f'animation:fadeInUp 0.3s ease">'

                # Icon
                f'<span style="color:{icon_color};flex-shrink:0;'
                f'font-size:0.7rem;margin-top:0.05rem">{icon}</span>'

                # Content
                f'<div style="flex:1;min-width:0">'
                f'<span style="color:{color};font-weight:500">'
                f'{step["label"]}</span>'
                f'<span style="color:{TEXT_3};font-size:0.65rem;'
                f'margin-left:0.5rem"> {step["elapsed"]}s</span>'

                # Detail on new line
                f'<div style="color:{TEXT_3};font-size:0.65rem;'
                f'margin-top:0.1rem;word-break:break-word">'
                f'{step["detail"]}</div>'

                f'</div></div>'
            )
            lines_html.append(line)

        log_html = "".join(lines_html)

        self._log_box.markdown(
            f'<div style="'
            f'background:#07090f;'
            f'border:1px solid {BORDER_DIM};'
            f'border-radius:10px;'
            f'padding:0.8rem 1.2rem;'
            f'font-family:\'DM Mono\',monospace;'
            f'font-size:0.75rem;'
            f'max-height:280px;'
            f'overflow-y:auto;'
            f'line-height:1.7">'
            f'{log_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Private Helpers ───────────────────────────────────────

    def _get_phase(self, key: str) -> Optional[dict]:
        """Look up phase config by key."""
        idx = PHASE_INDEX.get(key)
        if idx is None:
            return None
        return PHASES[idx]

    def _build_detail(
        self,
        phase_key:   str,
        n_articles:  Optional[int],
        n_chunks:    Optional[int],
        n_queries:   Optional[int],
        vader_score: Optional[float],
        bias_score:  Optional[float],
        n_args:      Optional[int],
        n_counters:  Optional[int],
        n_images:    Optional[int],
        extra:       str,
    ) -> str:
        """Build human-readable detail string for a phase."""
        parts = []

        if n_queries  is not None: parts.append(f"{n_queries} queries")
        if n_articles is not None: parts.append(f"{n_articles} articles fetched")
        if n_chunks   is not None: parts.append(f"{n_chunks} chunks embedded")
        if n_images   is not None: parts.append(f"{n_images} images analyzed")
        if n_args     is not None: parts.append(f"{n_args} arguments extracted")
        if n_counters is not None: parts.append(f"{n_counters} counter-args extracted")

        if vader_score is not None:
            direction = (
                "positive" if vader_score > 0.05
                else "negative" if vader_score < -0.05
                else "neutral"
            )
            parts.append(
                f"VADER: {vader_score:+.3f} ({direction})"
            )

        if bias_score is not None:
            direction = (
                "supportive" if bias_score < -0.2
                else "critical" if bias_score > 0.2
                else "balanced"
            )
            parts.append(
                f"bias: {bias_score:+.3f} ({direction})"
            )

        if extra:
            parts.append(extra)

        phase = self._get_phase(phase_key)
        base  = phase["detail"] if phase else ""

        if parts:
            return f"{' · '.join(parts)}"
        return base