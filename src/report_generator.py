# src/report_generator.py
"""
ORBITA PDF Report Generator

Generates a formatted, human-readable PDF report from
the pipeline analysis results.

Why PDF and not HTML or DOCX?
    PDF is universally readable without extra software.
    It is the standard format for research deliverables.
    fpdf2 is lightweight and has no external dependencies.

What the PDF contains:
    Page 1: Cover page with topic, date, bias score
    Page 2: Executive Summary (synthesis report)
    Page 3: Bias Analysis (scores, validation)
    Page 4: Supporting Arguments (Agent A)
    Page 5: Counter Arguments (Agent B)
    Page 6: NLP Analysis (VADER, entities, keywords)
    Page 7: Source Transparency (all articles)
    Page 8: Methodology note

Author: [Your Name]
Project: ORBITA — B.Tech 6th Sem, AIML 2026
"""

import os
import io
import re
from datetime import datetime
from typing import Optional

try:
    from fpdf import FPDF, XPos, YPos
    _fpdf_available = True
except ImportError:
    _fpdf_available = False
    print(
        "[report_generator] WARNING: fpdf2 not installed.\n"
        "  Run: pip install fpdf2"
    )


# ─────────────────────────────────────────────────────────────────────────────
# COLOR CONSTANTS
# ORBITA navy + gold theme mapped to RGB for PDF
# ─────────────────────────────────────────────────────────────────────────────

# Dark navy background equivalent (used for headers)
COLOR_NAVY       = (10,  15,  30)

# Gold accent color
COLOR_GOLD       = (201, 168, 76)

# Light gold (for sub-headings)
COLOR_GOLD_LIGHT = (232, 201, 106)

# Text colors
COLOR_TEXT_DARK  = (30,  30,  40)
COLOR_TEXT_MID   = (80,  90, 110)
COLOR_TEXT_LIGHT = (140, 155, 175)

# Stance colors
COLOR_SUPPORTIVE = (62,  201, 126)
COLOR_CRITICAL   = (224,  82,  82)
COLOR_NEUTRAL    = (91,  156, 246)

# Backgrounds for sections
COLOR_BG_LIGHT   = (248, 248, 252)
COLOR_BG_HEADER  = (20,  28,  46)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Clean text for PDF rendering.

    fpdf2 can handle UTF-8 but some special characters
    need to be normalized to avoid encoding errors.
    """
    if not text:
        return ""

    # Replace common problematic Unicode characters
    replacements = {
        "\u2013": "-",    # en dash
        "\u2014": "--",   # em dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2022": "-",    # bullet
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",    # non-breaking space
        "\u20b9": "Rs.",  # Indian Rupee sign
        "\u2264": "<=",   # less than or equal
        "\u2265": ">=",   # greater than or equal
        "\u00b1": "+/-",  # plus-minus
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Remove any remaining non-ASCII characters that might cause issues
    text = text.encode("ascii", errors="replace").decode("ascii")
    text = text.replace("?", " ")

    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text with ellipsis if too long."""
    if not text:
        return ""
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


# ─────────────────────────────────────────────────────────────────────────────
# ORBITA PDF CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ORBITAReport(FPDF):
    """
    Custom FPDF subclass with ORBITA branding.

    Overrides header() and footer() for consistent
    page layout across all pages.
    """

    def __init__(self, topic: str = "", collection_name: str = ""):
        super().__init__()
        self.topic           = _clean_text(topic)[:60]
        self.collection_name = _clean_text(collection_name)[:40]
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        """
        Page header: ORBITA wordmark + topic on every page.
        Skipped on cover page (page 1).
        """
        if self.page_no() == 1:
            return

        # Header background bar
        self.set_fill_color(*COLOR_BG_HEADER)
        self.rect(0, 0, 210, 12, style="F")

        # ORBITA wordmark
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*COLOR_GOLD)
        self.set_xy(10, 3)
        self.cell(30, 6, "ORBITA", align="L")

        # Topic
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*COLOR_TEXT_LIGHT)
        self.set_xy(45, 3)
        self.cell(120, 6, f"Analysis: {self.topic[:55]}", align="L")

        # Page number
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*COLOR_TEXT_LIGHT)
        self.set_xy(160, 3)
        self.cell(40, 6, f"Page {self.page_no()}", align="R")

        # Reset position after header
        self.set_xy(10, 16)

    def footer(self):
        """Page footer with generation timestamp."""
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*COLOR_TEXT_LIGHT)
        self.cell(
            0, 6,
            f"Generated by ORBITA | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"B.Tech 6th Sem AIML Project",
            align="C"
        )

    def section_title(
        self,
        title:    str,
        icon:     str = "",
        color_bg: tuple = COLOR_BG_HEADER,
        color_fg: tuple = COLOR_GOLD,
    ):
        """
        Draw a styled section title bar.

        Args:
            title:    section title text
            icon:     optional prefix symbol
            color_bg: background color (RGB tuple)
            color_fg: foreground/text color (RGB tuple)
        """
        # Small space before section
        self.ln(3)

        # Background rectangle
        self.set_fill_color(*color_bg)
        self.rect(10, self.get_y(), 190, 8, style="F")

        # Title text
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color_fg)
        display_title = f"  {icon} {title}" if icon else f"  {title}"
        self.cell(190, 8, _clean_text(display_title), align="L")
        self.ln(10)

        # Reset text color
        self.set_text_color(*COLOR_TEXT_DARK)

    def body_text(
        self,
        text:     str,
        indent:   int   = 0,
        size:     int   = 9,
        bold:     bool  = False,
        color:    tuple = COLOR_TEXT_DARK,
    ):
        """
        Write body text with word wrap.

        Args:
            text:   text content
            indent: left indent in mm
            size:   font size
            bold:   whether to use bold
            color:  text color RGB tuple
        """
        if not text:
            return

        style = "B" if bold else ""
        self.set_font("Helvetica", style, size)
        self.set_text_color(*color)
        self.set_x(10 + indent)
        self.multi_cell(
            w       = 190 - indent,
            h       = 5,
            text    = _clean_text(text),
            align   = "J",
        )
        self.ln(1)

    def metric_row(
        self,
        label: str,
        value: str,
        color: tuple = COLOR_GOLD,
    ):
        """
        Draw a label-value pair in a horizontal row.

        Used for bias scores, sentiment values, etc.
        """
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*COLOR_TEXT_MID)
        self.set_x(12)
        self.cell(60, 5, _clean_text(label), align="L")

        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.cell(80, 5, _clean_text(str(value)), align="L")
        self.ln(6)

    def bullet_item(
        self,
        text:        str,
        bullet:      str   = "-",
        indent:      int   = 8,
        size:        int   = 8,
        color:       tuple = COLOR_TEXT_DARK,
        bullet_color: tuple = COLOR_GOLD,
    ):
        """
        Draw a bullet-point list item.

        Args:
            text:         item text
            bullet:       bullet character
            indent:       left indent
            size:         font size
            color:        text color
            bullet_color: bullet symbol color
        """
        if not text:
            return

        cleaned = _clean_text(text)
        if not cleaned:
            return

        # Bullet symbol
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*bullet_color)
        self.set_x(10 + indent)
        self.cell(6, 5, bullet, align="L")

        # Item text
        self.set_font("Helvetica", "", size)
        self.set_text_color(*color)
        self.multi_cell(
            w    = 184 - indent,
            h    = 5,
            text = cleaned[:300],
            align= "L",
        )
        self.ln(1)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_cover_page(
    pdf:             ORBITAReport,
    topic:           str,
    bias_score:      float,
    interpretation:  str,
    collection_name: str,
    n_articles:      int,
    elapsed_seconds: float,
) -> None:
    """
    Build the cover page of the PDF report.

    Shows: ORBITA branding, topic, date, key metrics
    """
    pdf.add_page()

    # ── Full-page dark background ─────────────────────────────────
    pdf.set_fill_color(*COLOR_BG_HEADER)
    pdf.rect(0, 0, 210, 80, style="F")

    # ── ORBITA wordmark ───────────────────────────────────────────
    pdf.set_xy(0, 18)
    pdf.set_font("Helvetica", "B", 42)
    pdf.set_text_color(*COLOR_GOLD)
    pdf.cell(210, 18, "ORBITA", align="C")

    # ── Tagline ───────────────────────────────────────────────────
    pdf.set_xy(0, 38)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*COLOR_TEXT_LIGHT)
    pdf.cell(
        210, 6,
        "Objective Reasoning And Bias Interpretation Tool for Analysis",
        align="C"
    )

    # ── Decorative line ───────────────────────────────────────────
    pdf.set_draw_color(*COLOR_GOLD)
    pdf.set_line_width(0.5)
    pdf.line(75, 48, 135, 48)

    # ── B.Tech info ───────────────────────────────────────────────
    pdf.set_xy(0, 51)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*COLOR_TEXT_LIGHT)
    pdf.cell(210, 5, "B.Tech 6th Sem | AIML 2026 | Research Project", align="C")

    # ── Reset background ─────────────────────────────────────────
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 82, 210, 215, style="F")

    # ── Topic heading ─────────────────────────────────────────────
    pdf.set_xy(10, 88)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*COLOR_TEXT_MID)
    pdf.cell(190, 6, "ANALYSIS TOPIC", align="C")

    pdf.set_xy(10, 95)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*COLOR_TEXT_DARK)
    topic_display = _truncate(topic, 70)
    pdf.multi_cell(190, 8, topic_display, align="C")

    # ── Collection name ───────────────────────────────────────────
    if collection_name:
        pdf.set_xy(10, pdf.get_y() + 3)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*COLOR_TEXT_LIGHT)
        pdf.cell(
            190, 6,
            f"Session: {_clean_text(collection_name)}",
            align="C"
        )

    # ── Date ──────────────────────────────────────────────────────
    pdf.set_xy(10, pdf.get_y() + 4)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*COLOR_TEXT_MID)
    pdf.cell(
        190, 6,
        datetime.now().strftime("%B %d, %Y at %H:%M"),
        align="C"
    )

    # ── Bias score box ────────────────────────────────────────────
    box_y = pdf.get_y() + 10

    # Determine box color based on bias direction
    if bias_score < -0.2:
        box_color   = COLOR_SUPPORTIVE
        label_color = COLOR_SUPPORTIVE
    elif bias_score > 0.2:
        box_color   = COLOR_CRITICAL
        label_color = COLOR_CRITICAL
    else:
        box_color   = COLOR_NEUTRAL
        label_color = COLOR_NEUTRAL

    # Outer box border
    pdf.set_draw_color(*box_color)
    pdf.set_line_width(1.0)
    pdf.rect(55, box_y, 100, 35)

    # Bias score value
    pdf.set_xy(55, box_y + 4)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*label_color)
    pdf.cell(100, 14, f"{bias_score:+.3f}", align="C")

    # Interpretation label
    pdf.set_xy(55, box_y + 18)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*label_color)
    pdf.cell(100, 6, _clean_text(interpretation), align="C")

    # Scale label
    pdf.set_xy(55, box_y + 26)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*COLOR_TEXT_LIGHT)
    pdf.cell(100, 6, "-1.0 (Supportive) <- 0 -> +1.0 (Critical)", align="C")

    # ── Quick stats row ───────────────────────────────────────────
    stats_y = box_y + 44

    stats = [
        ("Articles", str(n_articles)),
        ("Time", f"{elapsed_seconds:.0f}s"),
        ("Method", "Hybrid NLP+AI"),
    ]

    col_w = 60
    for i, (label, value) in enumerate(stats):
        x = 15 + i * col_w

        pdf.set_xy(x, stats_y)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*COLOR_GOLD)
        pdf.cell(col_w, 8, _clean_text(value), align="C")

        pdf.set_xy(x, stats_y + 9)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*COLOR_TEXT_MID)
        pdf.cell(col_w, 5, _clean_text(label), align="C")

    # ── Divider line ──────────────────────────────────────────────
    pdf.set_draw_color(*COLOR_GOLD)
    pdf.set_line_width(0.3)
    pdf.line(30, stats_y + 22, 180, stats_y + 22)

    # ── Disclaimer ────────────────────────────────────────────────
    pdf.set_xy(20, stats_y + 26)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*COLOR_TEXT_LIGHT)
    pdf.multi_cell(
        170, 4,
        "This report was generated automatically by ORBITA using "
        "a hybrid approach combining VADER sentiment analysis, "
        "spaCy NER, TF-IDF keyword extraction, and Google Gemini AI. "
        "The bias score represents the ideological leaning of "
        "retrieved news coverage, not an objective truth.",
        align="C"
    )


def _build_synthesis_page(
    pdf:      ORBITAReport,
    report:   dict,
) -> None:
    """Build the synthesis/executive summary page."""
    pdf.add_page()
    pdf.section_title("Executive Summary — Unbiased 360 Synthesis", icon="*")

    synthesis = report.get("synthesis_report", "")
    if synthesis:
        pdf.body_text(synthesis, size=9)
    else:
        pdf.body_text("Synthesis not available.", color=COLOR_TEXT_LIGHT)

    pdf.ln(4)

    # Key agreements
    agreements = report.get("key_agreements", [])
    if agreements:
        pdf.section_title(
            "Points of Agreement",
            icon        = "~",
            color_bg    = (20, 40, 25),
            color_fg    = COLOR_SUPPORTIVE,
        )
        for item in agreements[:6]:
            pdf.bullet_item(
                item,
                bullet       = "~",
                bullet_color = COLOR_SUPPORTIVE,
                color        = COLOR_TEXT_DARK,
            )
        pdf.ln(2)

    # Key disagreements
    disagreements = report.get("key_disagreements", [])
    if disagreements:
        pdf.section_title(
            "Core Disagreements",
            icon        = "x",
            color_bg    = (40, 15, 15),
            color_fg    = COLOR_CRITICAL,
        )
        for item in disagreements[:6]:
            pdf.bullet_item(
                item,
                bullet       = "x",
                bullet_color = COLOR_CRITICAL,
                color        = COLOR_TEXT_DARK,
            )

    # Loaded language removed
    removed = report.get("loaded_language_removed", [])
    if removed:
        pdf.ln(4)
        pdf.section_title(
            "Loaded Language Neutralised",
            icon        = "~",
            color_bg    = (35, 30, 10),
            color_fg    = COLOR_GOLD,
        )
        phrases_text = " | ".join(
            f'"{_clean_text(p)}"' for p in removed[:8]
        )
        pdf.body_text(phrases_text, indent=4, size=8,
                      color=COLOR_TEXT_MID)


def _build_bias_analysis_page(
    pdf:         ORBITAReport,
    report:      dict,
    nlp_results: dict,
) -> None:
    """Build the bias analysis page with all scores."""
    pdf.add_page()
    pdf.section_title("Bias Analysis", icon="@")

    bias_vector  = report.get("bias_vector", {})
    nlp_summary  = report.get("nlp_summary", {})

    # ── Multi-dimensional bias scores ─────────────────────────────
    pdf.body_text("Multi-Dimensional Bias Vector:", bold=True, size=9)
    pdf.ln(2)

    scores = [
        ("Composite Score",    f"{report.get('bias_score', 0):+.4f}",  COLOR_GOLD),
        ("Ideological Bias",   f"{bias_vector.get('ideological_bias', 0):+.4f}", COLOR_GOLD_LIGHT),
        ("Emotional Bias",     f"{bias_vector.get('emotional_bias', 0):.4f}",    COLOR_TEXT_MID),
        ("Informational Bias", f"{bias_vector.get('informational_bias', 0):.4f}", COLOR_TEXT_MID),
        ("Source Diversity",   f"{bias_vector.get('source_diversity', 0):.4f}",  COLOR_SUPPORTIVE),
        ("Stance Entropy",     f"{bias_vector.get('stance_entropy', 0):.4f}",    COLOR_SUPPORTIVE),
        ("Interpretation",     bias_vector.get("interpretation", "N/A"),         COLOR_GOLD),
    ]

    for label, value, color in scores:
        pdf.metric_row(label, value, color)

    pdf.ln(4)

    # ── NLP Validation ────────────────────────────────────────────
    pdf.section_title(
        "Manual NLP Validation (Independent of AI)",
        icon     = "v",
        color_bg = (15, 25, 40),
        color_fg = COLOR_GOLD_LIGHT,
    )

    manual_score  = nlp_summary.get("manual_bias_score", None)
    gemini_val    = nlp_summary.get("gemini_validation", {})
    vader_avg     = nlp_summary.get("avg_vader_compound", None)
    nlp_val_note  = report.get("nlp_validation_note", "")

    if manual_score is not None:
        pdf.metric_row("Manual NLP Bias Score", f"{manual_score:+.4f}", COLOR_GOLD)
    if vader_avg is not None:
        pdf.metric_row("Avg VADER Compound",    f"{vader_avg:+.4f}", COLOR_TEXT_MID)

    if gemini_val:
        pdf.metric_row(
            "Agreement Level",
            gemini_val.get("agreement_level", "N/A"),
            COLOR_SUPPORTIVE,
        )
        diff = gemini_val.get("absolute_diff")
        if diff is not None:
            pdf.metric_row("Absolute Difference", f"{diff:.4f}", COLOR_TEXT_MID)

        dir_agrees = gemini_val.get("direction_agrees")
        if dir_agrees is not None:
            pdf.metric_row(
                "Direction Agrees",
                "Yes" if dir_agrees else "No",
                COLOR_SUPPORTIVE if dir_agrees else COLOR_CRITICAL,
            )

    if nlp_val_note:
        pdf.ln(2)
        pdf.body_text(
            f"Validation Note: {nlp_val_note}",
            indent = 4,
            size   = 8,
            color  = COLOR_TEXT_MID,
        )

    # ── Sentiment distribution ────────────────────────────────────
    sent_dist = nlp_summary.get("sentiment_distribution", {})
    if sent_dist:
        pdf.ln(4)
        pdf.body_text("VADER Sentiment Distribution:", bold=True, size=9)
        pdf.ln(2)

        pos = sent_dist.get("positive", 0)
        neg = sent_dist.get("negative", 0)
        neu = sent_dist.get("neutral",  0)

        pdf.metric_row("Positive Articles", str(pos), COLOR_SUPPORTIVE)
        pdf.metric_row("Negative Articles", str(neg), COLOR_CRITICAL)
        pdf.metric_row("Neutral Articles",  str(neu), COLOR_NEUTRAL)


def _build_arguments_page(
    pdf:         ORBITAReport,
    agent_a:     dict,
    agent_b:     dict,
) -> None:
    """Build supporting and counter arguments pages."""

    # ── Agent A: Supporting Arguments ─────────────────────────────
    pdf.add_page()
    pdf.section_title(
        "Supporting Arguments — Agent A (Analyst)",
        icon     = "+",
        color_bg = (15, 35, 20),
        color_fg = COLOR_SUPPORTIVE,
    )

    pdf.metric_row(
        "Confidence Score",
        f"{agent_a.get('confidence_score', 0):.2f}",
        COLOR_SUPPORTIVE,
    )

    nlp_used_a = agent_a.get("nlp_context_used", False)
    pdf.metric_row(
        "NLP Context Used",
        "Yes (VADER + spaCy)" if nlp_used_a else "No",
        COLOR_SUPPORTIVE if nlp_used_a else COLOR_TEXT_LIGHT,
    )
    pdf.ln(3)

    arguments = agent_a.get("arguments", [])
    if arguments:
        for i, arg in enumerate(arguments[:10], 1):
            pdf.bullet_item(
                f"{i}. {arg}",
                bullet       = "+",
                indent       = 5,
                bullet_color = COLOR_SUPPORTIVE,
            )
    else:
        pdf.body_text(
            "No supporting arguments found.",
            color = COLOR_TEXT_LIGHT,
        )

    # NLP-validated arguments
    nlp_validated = agent_a.get("nlp_validated_arguments", [])
    if nlp_validated:
        pdf.ln(3)
        pdf.body_text(
            "NLP-Validated Arguments (backed by positive VADER scores):",
            bold  = True,
            size  = 8,
            color = COLOR_SUPPORTIVE,
        )
        for arg in nlp_validated[:4]:
            pdf.bullet_item(
                arg,
                bullet       = "*",
                indent       = 8,
                size         = 8,
                bullet_color = COLOR_SUPPORTIVE,
            )

    # ── Agent B: Counter Arguments ────────────────────────────────
    pdf.add_page()
    pdf.section_title(
        "Counter-Arguments — Agent B (Critic)",
        icon     = "-",
        color_bg = (40, 15, 15),
        color_fg = COLOR_CRITICAL,
    )

    pdf.metric_row(
        "Confidence Score",
        f"{agent_b.get('confidence_score', 0):.2f}",
        COLOR_CRITICAL,
    )

    nlp_used_b = agent_b.get("nlp_context_used", False)
    pdf.metric_row(
        "NLP Context Used",
        "Yes (VADER + spaCy)" if nlp_used_b else "No",
        COLOR_SUPPORTIVE if nlp_used_b else COLOR_TEXT_LIGHT,
    )
    pdf.ln(3)

    counters = agent_b.get("counter_arguments", [])
    if counters:
        for i, arg in enumerate(counters[:10], 1):
            pdf.bullet_item(
                f"{i}. {arg}",
                bullet       = "-",
                indent       = 5,
                bullet_color = COLOR_CRITICAL,
            )
    else:
        pdf.body_text(
            "No counter-arguments found.",
            color = COLOR_TEXT_LIGHT,
        )

    # NLP-validated counter arguments
    nlp_validated_b = agent_b.get(
        "nlp_validated_counter_arguments", []
    )
    if nlp_validated_b:
        pdf.ln(3)
        pdf.body_text(
            "NLP-Validated Counters (backed by negative VADER scores):",
            bold  = True,
            size  = 8,
            color = COLOR_CRITICAL,
        )
        for arg in nlp_validated_b[:4]:
            pdf.bullet_item(
                arg,
                bullet       = "*",
                indent       = 8,
                size         = 8,
                bullet_color = COLOR_CRITICAL,
            )


def _build_nlp_page(
    pdf:         ORBITAReport,
    nlp_results: dict,
    report:      dict,
) -> None:
    """Build the NLP analysis page."""
    if not nlp_results:
        return

    pdf.add_page()
    pdf.section_title(
        "Manual NLP Analysis (VADER + spaCy + TF-IDF)",
        icon     = "#",
        color_bg = (15, 25, 40),
        color_fg = COLOR_GOLD_LIGHT,
    )

    pdf.body_text(
        "The following analysis was computed entirely using rule-based "
        "NLP methods, independent of any AI/LLM system. "
        "This provides an independent validation of the AI-generated bias scores.",
        size  = 8,
        color = COLOR_TEXT_MID,
    )
    pdf.ln(3)

    # Per-article sentiment table header
    per_article = nlp_results.get("per_article_sentiment", [])
    if per_article:
        pdf.body_text("VADER Sentiment Per Article:", bold=True, size=9)
        pdf.ln(2)

        # Table header
        pdf.set_fill_color(*COLOR_BG_HEADER)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*COLOR_GOLD)
        pdf.set_x(10)
        pdf.cell(70, 5, "Source",  fill=True)
        pdf.cell(30, 5, "Stance",  fill=True)
        pdf.cell(30, 5, "VADER",   fill=True)
        pdf.cell(30, 5, "Label",   fill=True)
        pdf.ln(6)

        # Table rows
        for item in per_article[:8]:
            source   = _truncate(item.get("source",   ""), 28)
            stance   = _clean_text(item.get("stance",   "Neutral"))
            compound = item.get("compound", 0.0)
            label    = _clean_text(item.get("label",   "neutral"))

            # Row color based on sentiment
            if label == "positive":
                row_color = COLOR_SUPPORTIVE
            elif label == "negative":
                row_color = COLOR_CRITICAL
            else:
                row_color = COLOR_TEXT_MID

            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*COLOR_TEXT_DARK)
            pdf.set_x(10)
            pdf.cell(70, 5, source)
            pdf.cell(30, 5, stance)

            pdf.set_text_color(*row_color)
            pdf.cell(30, 5, f"{compound:+.3f}")
            pdf.cell(30, 5, label.capitalize())
            pdf.ln(6)

        pdf.set_text_color(*COLOR_TEXT_DARK)
        pdf.ln(3)

    # Top keywords
    keyword_analysis = nlp_results.get("keyword_analysis", {})
    top_keywords     = keyword_analysis.get("top_keywords", [])

    if top_keywords:
        pdf.body_text("Top TF-IDF Keywords:", bold=True, size=9)
        pdf.ln(1)

        kw_text = " | ".join(
            kw.get("word", "") for kw in top_keywords[:15]
        )
        pdf.body_text(
            _clean_text(kw_text),
            indent = 4,
            size   = 8,
            color  = COLOR_TEXT_MID,
        )
        pdf.ln(3)

    # Top entities
    entity_analysis = nlp_results.get("entity_analysis", {})
    top_entities    = entity_analysis.get("top_entities", [])

    if top_entities:
        pdf.body_text("Top Named Entities (spaCy NER):", bold=True, size=9)
        pdf.ln(2)

        for ent in top_entities[:10]:
            text       = _clean_text(ent.get("text",       ""))
            label_name = _clean_text(ent.get("label_name", ""))
            count      = ent.get("count", 0)

            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*COLOR_TEXT_DARK)
            pdf.set_x(14)
            pdf.cell(80, 5, text)

            pdf.set_text_color(*COLOR_TEXT_MID)
            pdf.cell(50, 5, label_name)

            pdf.set_text_color(*COLOR_GOLD)
            pdf.cell(30, 5, f"{count}x")
            pdf.ln(6)


def _build_sources_page(
    pdf:      ORBITAReport,
    articles: list,
) -> None:
    """Build the source transparency page."""
    pdf.add_page()
    pdf.section_title(
        f"Source Transparency — {len(articles)} Articles",
        icon = "S",
    )

    stance_colors = {
        "Supportive": COLOR_SUPPORTIVE,
        "Critical":   COLOR_CRITICAL,
        "Neutral":    COLOR_NEUTRAL,
    }

    for i, article in enumerate(articles, 1):
        title  = _truncate(
            article.get("title",  "Unknown Title"), 75
        )
        source = _clean_text(
            article.get("source", "Unknown")
        )[:25]
        stance = _clean_text(
            article.get("stance", "Neutral")
        )
        url    = _clean_text(
            article.get("url",    "")
        )[:60]
        words  = len(
            (article.get("full_text") or "").split()
        )

        stance_color = stance_colors.get(stance, COLOR_TEXT_MID)

        # Article number + title
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*COLOR_TEXT_DARK)
        pdf.set_x(10)
        pdf.cell(10, 5, f"{i}.")
        pdf.multi_cell(175, 5, title)

        # Source + stance + words
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*COLOR_TEXT_MID)
        pdf.set_x(20)
        pdf.cell(60, 4, source)

        pdf.set_text_color(*stance_color)
        pdf.cell(40, 4, stance)

        pdf.set_text_color(*COLOR_TEXT_LIGHT)
        pdf.cell(40, 4, f"{words:,} words" if words > 0 else "")
        pdf.ln(5)

        # URL (smaller, muted)
        if url:
            pdf.set_font("Helvetica", "I", 6)
            pdf.set_text_color(*COLOR_TEXT_LIGHT)
            pdf.set_x(20)
            pdf.cell(180, 4, url[:70])
            pdf.ln(5)

        # Small separator
        pdf.set_draw_color(220, 220, 230)
        pdf.set_line_width(0.1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        # Check if we're near page bottom
        if pdf.get_y() > 265:
            pdf.add_page()
            pdf.section_title(
                "Source Transparency (continued)",
                icon = "S",
            )


def _build_methodology_page(pdf: ORBITAReport) -> None:
    """Build the methodology note page."""
    pdf.add_page()
    pdf.section_title("Methodology", icon="M")

    methodology_text = (
        "ORBITA employs a hybrid approach combining rule-based NLP "
        "with large language model (LLM) reasoning for comprehensive "
        "news bias analysis. "

        "Phase 2 (Data Engineering): Articles are retrieved from NewsAPI "
        "and classified using TF-IDF cosine similarity stance detection. "
        "Full article text is extracted using newspaper4k. "

        "Phase 2.5 (Manual NLP): VADER sentiment analysis computes "
        "compound scores per article. spaCy Named Entity Recognition "
        "extracts people, organizations, and locations. TF-IDF keyword "
        "extraction identifies the most distinctive terms. These provide "
        "an independent, rule-based bias estimate. "

        "Phase 3 (Embeddings): Article chunks are embedded using "
        "Google Gemini embedding API and stored in ChromaDB "
        "for semantic retrieval. "

        "Phase 4 (Multi-Agent RAG): Three specialized agents perform "
        "retrieval-augmented analysis. Agent A (Analyst) extracts "
        "supporting arguments. Agent B (Critic) extracts "
        "counter-arguments. Agent C (Arbitrator) synthesizes both "
        "perspectives with hallucination checking against source chunks. "
        "All three agents receive NLP context from Phase 2.5, "
        "enabling hybrid validation. "

        "Bias Score: A multi-dimensional bias vector is computed "
        "combining ideological stance, emotional content, "
        "informational density, and source diversity into a "
        "composite score in the range [-1.0, +1.0]. "
        "Independent VADER-based validation is compared with the "
        "AI-generated score to assess reliability."
    )

    pdf.body_text(methodology_text, size=8, color=COLOR_TEXT_DARK)

    pdf.ln(8)
    pdf.body_text(
        "Libraries Used: vaderSentiment, spaCy (en_core_web_sm), "
        "scikit-learn (TF-IDF), LangChain, ChromaDB, "
        "Google Gemini API, newspaper4k, Streamlit, Plotly",
        size  = 7,
        color = COLOR_TEXT_MID,
    )

    pdf.ln(4)
    pdf.body_text(
        "Note: The bias score represents the ideological leaning "
        "of retrieved news coverage, not an objective measure of truth. "
        "ORBITA does not make editorial judgments.",
        size  = 7,
        color = COLOR_TEXT_LIGHT,
        bold  = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    pipeline_result:  dict,
    collection_name:  str = "",
) -> Optional[bytes]:
    """
    Generate a complete PDF report from pipeline results.

    This is the main function called from app.py.

    Args:
        pipeline_result:  the dict returned by run_pipeline()
        collection_name:  optional session name from UI input

    Returns:
        PDF as bytes (ready for st.download_button),
        or None if fpdf2 is not installed
    """
    if not _fpdf_available:
        print(
            "[report_generator] Cannot generate PDF: fpdf2 not installed.\n"
            "  Run: pip install fpdf2"
        )
        return None

    report      = pipeline_result.get("report",      {})
    articles    = pipeline_result.get("articles",    [])
    topic       = pipeline_result.get("topic",       "Unknown Topic")
    nlp_results = pipeline_result.get("nlp_analysis", {})
    elapsed     = pipeline_result.get("elapsed_seconds", 0)

    agent_a     = report.get("agent_a", {})
    agent_b     = report.get("agent_b", {})
    bias_score  = report.get("bias_score", 0.0)
    bias_vector = report.get("bias_vector", {})
    interpretation = bias_vector.get("interpretation", "Unknown")

    print(f"[report_generator] Generating PDF for '{topic}'...")

    try:
        # Create PDF instance
        pdf = ORBITAReport(
            topic           = topic,
            collection_name = collection_name,
        )

        # Build all pages
        _build_cover_page(
            pdf             = pdf,
            topic           = topic,
            bias_score      = bias_score,
            interpretation  = interpretation,
            collection_name = collection_name,
            n_articles      = len(articles),
            elapsed_seconds = elapsed,
        )

        _build_synthesis_page(pdf, report)

        _build_bias_analysis_page(pdf, report, nlp_results)

        _build_arguments_page(pdf, agent_a, agent_b)

        if nlp_results:
            _build_nlp_page(pdf, nlp_results, report)

        if articles:
            _build_sources_page(pdf, articles)

        _build_methodology_page(pdf)

        # Output as bytes
        pdf_bytes = pdf.output()

        print(
            f"[report_generator] PDF generated: "
            f"{len(pdf_bytes):,} bytes, "
            f"{pdf.page_no()} pages"
        )

        return bytes(pdf_bytes)

    except Exception as e:
        print(f"[report_generator] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_pdf_filename(topic: str, collection_name: str = "") -> str:
    """
    Generate a clean filename for the PDF download.

    Args:
        topic:           analysis topic
        collection_name: optional session name

    Returns:
        filename string like "ORBITA_Farm_Laws_India_20260417.pdf"
    """
    safe_topic = re.sub(r"[^\w\s]", "", topic)
    safe_topic = re.sub(r"\s+", "_", safe_topic.strip())[:30]

    date_str = datetime.now().strftime("%Y%m%d")

    if collection_name:
        safe_coll = re.sub(r"[^\w\s]", "", collection_name)
        safe_coll = re.sub(r"\s+", "_", safe_coll.strip())[:20]
        return f"ORBITA_{safe_topic}_{safe_coll}_{date_str}.pdf"

    return f"ORBITA_{safe_topic}_{date_str}.pdf"