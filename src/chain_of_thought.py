# src/chain_of_thought.py
"""
ORBITA Chain of Thought (CoT) Recorder

Records the complete reasoning trail of the ORBITA pipeline
so every decision is explainable and traceable.

Research Justification:
    Explainability is a core requirement for trustworthy AI.
    (Arrieta et al. 2020 - "Explainable AI: Concepts and Challenges")
    
    Without CoT: ORBITA is a black box.
    With CoT:    ORBITA is an Explainable AI (XAI) system.
    
    Every bias score can now be traced back to:
    - Which articles were retrieved
    - What the NLP analysis showed
    - How the agents reasoned
    - Why the final score was assigned

Step Types:
    PIPELINE  : Overall pipeline phase transitions
    RETRIEVAL : ChromaDB semantic search steps  
    NLP       : Manual NLP analysis steps (VADER, spaCy, TF-IDF)
    SENTIMENT : VADER sentiment per article
    ENTITY    : spaCy NER extraction
    KEYWORD   : TF-IDF keyword results
    ARGUMENT  : Agent argument extraction
    VALIDATION: Cross-validation between methods
    DECISION  : Final scoring decisions
    SYNTHESIS : Agent C synthesis reasoning
    IMAGE     : Visual analysis steps

Author: [Your Name]
"""

import time
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# STEP TYPES
# ─────────────────────────────────────────────────────────────────────────────

class CoTStepType(Enum):
    """
    Types of reasoning steps in the ORBITA pipeline.
    Each type gets a different icon and color in the UI.
    """
    PIPELINE   = "pipeline"    # Phase transitions
    RETRIEVAL  = "retrieval"   # ChromaDB search
    NLP        = "nlp"         # Manual NLP analysis
    SENTIMENT  = "sentiment"   # VADER scores
    ENTITY     = "entity"      # spaCy NER
    KEYWORD    = "keyword"     # TF-IDF keywords
    ARGUMENT   = "argument"    # Agent argument extraction
    VALIDATION = "validation"  # Cross-method validation
    DECISION   = "decision"    # Scoring decisions
    SYNTHESIS  = "synthesis"   # Agent C synthesis
    IMAGE      = "image"       # Visual analysis
    ERROR      = "error"       # Errors (non-fatal)


# Icons for each step type (used in UI)
STEP_ICONS = {
    CoTStepType.PIPELINE:   "🔄",
    CoTStepType.RETRIEVAL:  "🔍",
    CoTStepType.NLP:        "📊",
    CoTStepType.SENTIMENT:  "💭",
    CoTStepType.ENTITY:     "👤",
    CoTStepType.KEYWORD:    "🔑",
    CoTStepType.ARGUMENT:   "⚖️",
    CoTStepType.VALIDATION: "✅",
    CoTStepType.DECISION:   "🎯",
    CoTStepType.SYNTHESIS:  "📝",
    CoTStepType.IMAGE:      "🖼️",
    CoTStepType.ERROR:      "⚠️",
}

# Colors for each step type (CSS hex colors)
STEP_COLORS = {
    CoTStepType.PIPELINE:   "#c9a84c",   # gold
    CoTStepType.RETRIEVAL:  "#5b9cf6",   # blue
    CoTStepType.NLP:        "#9b59b6",   # purple
    CoTStepType.SENTIMENT:  "#3ec97e",   # green
    CoTStepType.ENTITY:     "#e67e22",   # orange
    CoTStepType.KEYWORD:    "#1abc9c",   # teal
    CoTStepType.ARGUMENT:   "#c9a84c",   # gold
    CoTStepType.VALIDATION: "#3ec97e",   # green
    CoTStepType.DECISION:   "#e05252",   # red
    CoTStepType.SYNTHESIS:  "#5b9cf6",   # blue
    CoTStepType.IMAGE:      "#fd79a8",   # pink
    CoTStepType.ERROR:      "#e05252",   # red
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoTStep:
    """
    A single reasoning step in the chain of thought.
    
    Each step records WHAT was done, WHY, and WHAT was found.
    """
    step_type:   str                    # CoTStepType value
    phase:       str                    # Pipeline phase name
    title:       str                    # Short title (shown in UI)
    detail:      str                    # Detailed explanation
    evidence:    list = field(default_factory=list)   # Supporting evidence
    confidence:  float = 0.0            # How confident (0-1)
    score:       Optional[float] = None # Numeric output if applicable
    timestamp:   str = ""               # When this step ran
    elapsed_ms:  int = 0                # Time taken in milliseconds
    agent:       str = ""               # Which agent (A/B/C/pipeline)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def icon(self) -> str:
        try:
            return STEP_ICONS[CoTStepType(self.step_type)]
        except (ValueError, KeyError):
            return "•"

    @property
    def color(self) -> str:
        try:
            return STEP_COLORS[CoTStepType(self.step_type)]
        except (ValueError, KeyError):
            return "#9ba8bb"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ORBITACoT:
    """
    Chain of Thought recorder for the entire ORBITA pipeline.
    
    Usage:
        cot = ORBITACoT(topic="Farm Laws India")
        
        # Record steps as pipeline runs
        cot.add_step(
            step_type  = CoTStepType.RETRIEVAL,
            phase      = "Phase 3",
            title      = "ChromaDB Semantic Search",
            detail     = "Searched for supporting arguments",
            evidence   = ["Retrieved 12 chunks", "8 Supportive"],
            confidence = 0.85,
            agent      = "Agent A",
        )
        
        # Get formatted chain
        chain = cot.get_chain()
        
        # Save to JSON
        cot.save("cot_farm_laws.json")
    """

    def __init__(self, topic: str = ""):
        self.topic      = topic
        self.steps: list[CoTStep] = []
        self.start_time = time.time()
        self._step_start: float = time.time()

    def start_step_timer(self):
        """Call before each step to measure elapsed time."""
        self._step_start = time.time()

    def add_step(
        self,
        step_type:  CoTStepType,
        phase:      str,
        title:      str,
        detail:     str,
        evidence:   list        = None,
        confidence: float       = 0.0,
        score:      Optional[float] = None,
        agent:      str         = "",
    ) -> CoTStep:
        """
        Add a reasoning step to the chain.
        
        Args:
            step_type:  type of step (from CoTStepType enum)
            phase:      pipeline phase name
            title:      short title shown in UI timeline
            detail:     full explanation of this step
            evidence:   list of supporting facts/numbers
            confidence: confidence in this step (0-1)
            score:      numeric output (bias score, etc.)
            agent:      which agent/component added this step
        
        Returns:
            The created CoTStep object
        """
        elapsed_ms = int(
            (time.time() - self._step_start) * 1000
        )
        self._step_start = time.time()

        step = CoTStep(
            step_type  = step_type.value,
            phase      = phase,
            title      = title,
            detail     = detail,
            evidence   = evidence or [],
            confidence = round(confidence, 4),
            score      = round(score, 4) if score is not None else None,
            elapsed_ms = elapsed_ms,
            agent      = agent,
        )
        self.steps.append(step)
        return step

    def add_pipeline_step(
        self,
        phase:   str,
        title:   str,
        detail:  str,
        evidence: list = None,
    ):
        """Convenience: add a pipeline transition step."""
        return self.add_step(
            CoTStepType.PIPELINE, phase, title,
            detail, evidence, confidence=1.0,
        )

    def add_retrieval_step(
        self,
        agent:      str,
        query:      str,
        n_results:  int,
        stance_filter: str = "",
        top_sources: list  = None,
    ):
        """Convenience: add a ChromaDB retrieval step."""
        evidence = [f"Retrieved {n_results} chunks"]
        if stance_filter:
            evidence.append(f"Stance filter: {stance_filter}")
        if top_sources:
            evidence.append(
                f"Sources: {', '.join(top_sources[:3])}"
            )

        return self.add_step(
            step_type  = CoTStepType.RETRIEVAL,
            phase      = "Phase 3",
            title      = f"Semantic Search — {agent}",
            detail     = (
                f"Query: \"{query[:80]}\"\n"
                f"Retrieved {n_results} relevant chunks from "
                f"ChromaDB vector store."
                + (f" Filtered by stance: {stance_filter}."
                   if stance_filter else "")
            ),
            evidence   = evidence,
            confidence = min(1.0, n_results / 10),
            agent      = agent,
        )

    def add_sentiment_step(
        self,
        per_article: list,
        avg_compound: float,
    ):
        """Convenience: add VADER sentiment analysis step."""
        evidence = [
            f"Avg VADER compound: {avg_compound:+.3f}",
        ]
        for item in per_article[:4]:
            evidence.append(
                f"{item.get('source','?')[:20]}: "
                f"{item.get('compound', 0):+.3f} "
                f"({item.get('label', 'neutral')})"
            )

        label = (
            "negative tone" if avg_compound < -0.05
            else "positive tone" if avg_compound > 0.05
            else "neutral tone"
        )

        return self.add_step(
            step_type  = CoTStepType.SENTIMENT,
            phase      = "Phase 2.5",
            title      = f"VADER Sentiment → {label}",
            detail     = (
                f"VADER sentiment analysis on {len(per_article)} articles.\n"
                f"Average compound score: {avg_compound:+.4f}\n"
                f"This provides INDEPENDENT validation of bias direction."
            ),
            evidence   = evidence,
            confidence = min(1.0, len(per_article) / 6),
            score      = avg_compound,
            agent      = "NLP Analyzer",
        )

    def add_entity_step(
        self,
        top_entities: list,
        n_total:      int,
    ):
        """Convenience: add spaCy NER step."""
        evidence = [f"Total unique entities: {n_total}"]
        for ent in top_entities[:5]:
            evidence.append(
                f"{ent.get('text', '?')} "
                f"({ent.get('label_name', '?')}): "
                f"{ent.get('count', 0)}x"
            )

        return self.add_step(
            step_type  = CoTStepType.ENTITY,
            phase      = "Phase 2.5",
            title      = f"spaCy NER → {n_total} entities found",
            detail     = (
                f"Named Entity Recognition extracted {n_total} unique "
                f"entities from all articles.\n"
                f"Top entities reveal WHO and WHAT the coverage focuses on, "
                f"which can indicate coverage bias."
            ),
            evidence   = evidence,
            confidence = 0.9,
            agent      = "NLP Analyzer",
        )

    def add_keyword_step(
        self,
        top_keywords: list,
    ):
        """Convenience: add TF-IDF keyword step."""
        kw_str = ", ".join(
            kw.get("word", "") for kw in top_keywords[:8]
        )
        evidence = [
            f"Top {len(top_keywords)} keywords extracted",
            f"Keywords: {kw_str}",
        ]

        return self.add_step(
            step_type  = CoTStepType.KEYWORD,
            phase      = "Phase 2.5",
            title      = f"TF-IDF → Top keywords: {kw_str[:40]}",
            detail     = (
                f"TF-IDF keyword extraction identified the most "
                f"distinctive terms across all articles.\n"
                f"These reveal the TOPICAL FOCUS of coverage."
            ),
            evidence   = evidence,
            confidence = 0.9,
            agent      = "NLP Analyzer",
        )

    def add_argument_step(
        self,
        agent:      str,
        n_args:     int,
        confidence: float,
        top_args:   list,
        nlp_used:   bool = False,
    ):
        """Convenience: add argument extraction step."""
        evidence = [
            f"Extracted {n_args} arguments",
            f"Agent confidence: {confidence:.2f}",
        ]
        if nlp_used:
            evidence.append("NLP context (VADER+spaCy) used")
        for arg in top_args[:2]:
            evidence.append(f"• {arg[:80]}")

        role = "supporting" if agent == "Agent A" else "counter"

        return self.add_step(
            step_type  = CoTStepType.ARGUMENT,
            phase      = "Phase 4",
            title      = (
                f"{agent} → {n_args} {role} arguments "
                f"(conf: {confidence:.2f})"
            ),
            detail     = (
                f"{agent} extracted {n_args} {role} arguments "
                f"using RAG retrieval from ChromaDB.\n"
                + ("NLP context provided: VADER sentiment scores "
                   "and named entities helped identify strongest evidence.\n"
                   if nlp_used else "")
                + f"Confidence score: {confidence:.2f}"
            ),
            evidence   = evidence,
            confidence = confidence,
            agent      = agent,
        )

    def add_validation_step(
        self,
        manual_score:  float,
        gemini_score:  float,
        agreement:     str,
        diff:          float,
    ):
        """Convenience: add NLP vs AI validation step."""
        direction_ok = (
            (manual_score > 0 and gemini_score > 0) or
            (manual_score < 0 and gemini_score < 0) or
            (abs(manual_score) < 0.1 and abs(gemini_score) < 0.1)
        )

        evidence = [
            f"Manual NLP score: {manual_score:+.4f}",
            f"Gemini AI score:  {gemini_score:+.4f}",
            f"Absolute diff:    {diff:.4f}",
            f"Direction: {'AGREES' if direction_ok else 'DISAGREES'}",
        ]

        return self.add_step(
            step_type  = CoTStepType.VALIDATION,
            phase      = "Post-Phase 4",
            title      = (
                f"Cross-Validation → {agreement} "
                f"(diff={diff:.3f})"
            ),
            detail     = (
                f"HYBRID VALIDATION: Manual NLP (VADER) vs Gemini AI\n"
                f"This is a key research contribution — two independent "
                f"methods agree on bias direction.\n"
                f"Agreement level: {agreement}\n"
                f"{'✓ Direction agrees' if direction_ok else '✗ Direction disagrees'}"
            ),
            evidence   = evidence,
            confidence = max(0.0, 1.0 - diff),
            score      = gemini_score,
            agent      = "Validator",
        )

    def add_decision_step(
        self,
        final_score:    float,
        interpretation: str,
        reasoning:      str,
        dimensions:     dict = None,
    ):
        """Convenience: add final decision step."""
        evidence = [
            f"Final composite score: {final_score:+.4f}",
            f"Interpretation: {interpretation}",
        ]
        if dimensions:
            for dim, val in list(dimensions.items())[:4]:
                evidence.append(f"{dim}: {val}")

        return self.add_step(
            step_type  = CoTStepType.DECISION,
            phase      = "Final",
            title      = (
                f"Final Decision → {final_score:+.3f} "
                f"({interpretation})"
            ),
            detail     = reasoning,
            evidence   = evidence,
            confidence = 0.9,
            score      = final_score,
            agent      = "Agent C",
        )

    def add_synthesis_step(
        self,
        n_words:         int,
        n_hallucinations: int,
        validation_note: str,
    ):
        """Convenience: add synthesis step."""
        evidence = [
            f"Synthesis length: {n_words} words",
            f"Hallucination flags: {n_hallucinations}",
        ]
        if validation_note:
            evidence.append(f"NLP note: {validation_note[:80]}")

        return self.add_step(
            step_type  = CoTStepType.SYNTHESIS,
            phase      = "Phase 4",
            title      = (
                f"Agent C Synthesis → {n_words} words, "
                f"{n_hallucinations} flags"
            ),
            detail     = (
                f"Agent C synthesized arguments from both sides "
                f"into a {n_words}-word neutral report.\n"
                f"Hallucination check: {n_hallucinations} claims "
                f"flagged as potentially ungrounded.\n"
                + (f"NLP validation: {validation_note}"
                   if validation_note else "")
            ),
            evidence   = evidence,
            confidence = max(0.5, 1.0 - n_hallucinations * 0.05),
            agent      = "Agent C",
        )

    def get_chain(self) -> list:
        """Return all steps as list of dicts."""
        return [s.to_dict() for s in self.steps]

    def get_steps_by_type(
        self, step_type: CoTStepType
    ) -> list:
        """Return steps filtered by type."""
        return [
            s for s in self.steps
            if s.step_type == step_type.value
        ]

    def get_summary(self) -> dict:
        """Return a summary of the chain."""
        total_elapsed = round(time.time() - self.start_time, 2)

        return {
            "topic":          self.topic,
            "total_steps":    len(self.steps),
            "total_elapsed":  total_elapsed,
            "step_breakdown": {
                st.value: len(self.get_steps_by_type(st))
                for st in CoTStepType
                if self.get_steps_by_type(st)
            },
            "phases": list(dict.fromkeys(
                s.phase for s in self.steps
            )),
        }

    def to_json(self) -> str:
        """Serialize the full chain to JSON string."""
        return json.dumps({
            "topic":    self.topic,
            "created":  datetime.now().isoformat(),
            "summary":  self.get_summary(),
            "steps":    self.get_chain(),
        }, indent=2, ensure_ascii=False)

    def save(self, filepath: str) -> None:
        """Save chain to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        print(f"[CoT] Saved chain to {filepath}")

    def print_chain(self) -> None:
        """Print the chain to terminal for debugging."""
        print(f"\n{'='*60}")
        print(f"  ORBITA Chain of Thought — {self.topic}")
        print(f"{'='*60}")
        for i, step in enumerate(self.steps, 1):
            icon = step.icon
            print(f"\n  Step {i:02d} | {icon} [{step.step_type.upper()}]")
            print(f"  Phase:  {step.phase}")
            print(f"  Title:  {step.title}")
            print(f"  Detail: {step.detail[:100]}...")
            if step.evidence:
                for ev in step.evidence[:3]:
                    print(f"    • {ev}")
            if step.score is not None:
                print(f"  Score:  {step.score:+.4f}")
            print(f"  Conf:   {step.confidence:.2f} | "
                  f"Time: {step.elapsed_ms}ms")
        print(f"\n{'='*60}")
        summary = self.get_summary()
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Total time:  {summary['total_elapsed']}s")
        print(f"{'='*60}\n")