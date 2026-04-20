# src/agent_b.py
"""
ORBITA Agent B — The Critic

Role: Extract counter-arguments and critical evidence.

Enhanced with hybrid context:
    - RAG text chunks from ChromaDB
    - Visual context from Gemini Vision
    - NLP context from VADER + spaCy + TF-IDF

The NLP context helps Agent B identify which articles have
genuinely negative framing (low VADER score) vs which are
just labeled "Critical" by the stance classifier.
"""

import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.genai as genai
try:
    from .chain_of_thought import ORBITACoT, CoTStepType
except ImportError:
    from chain_of_thought import ORBITACoT, CoTStepType
try:
    from .config import (
        GEMINI_API_KEY, GEMINI_MODEL,
        AGENT_TEMPERATURE, AGENT_MAX_TOKENS, AGENT_B_TOP_K
    )
    from .vector_store import retrieve_chunks
except ImportError:
    from config import (
        GEMINI_API_KEY, GEMINI_MODEL,
        AGENT_TEMPERATURE, AGENT_MAX_TOKENS, AGENT_B_TOP_K
    )
    from vector_store import retrieve_chunks

try:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    genai_client = None


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

AGENT_B_SYSTEM_PROMPT = """You are Agent B — The Critic for ORBITA.

TASK: Extract counter-arguments and criticisms from the provided
article excerpts and NLP analysis data.

OUTPUT: Return ONLY a raw JSON object. No markdown. No backticks.
No text outside the JSON. Start with { and end with }.

RULES:
- Extract ONLY what is explicitly stated in the excerpts
- Each counter-argument must be a complete, specific sentence
- Use NLP context (VADER scores, entities) to identify which
  sources have strongest critical framing
- Do not add opinions or external knowledge
- Minimum 3 counter-arguments if the text supports it
- confidence_score: 0.0 (no criticism) to 1.0 (strong criticism)

JSON STRUCTURE:
{
  "counter_arguments": ["specific counter-argument 1"],
  "evidence": ["critical data point or quote 1"],
  "key_sources": ["Source Name 1"],
  "confidence_score": 0.75,
  "nlp_validated_counter_arguments": ["arg backed by negative VADER"],
  "top_critical_entities": ["entity1", "entity2"]
}"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _build_nlp_section_for_prompt(nlp_context: str) -> str:
    """
    Format NLP context for injection into Agent B's prompt.

    Agent B is the CRITIC, so we highlight:
    - Articles with NEGATIVE VADER scores
    - Entities mentioned in critical coverage
    - Keywords associated with critical framing
    - Emotional language detected
    """
    if not nlp_context or len(nlp_context.strip()) < 20:
        return ""

    lines    = nlp_context.strip().split("\n")
    relevant = []

    for line in lines:
        line_lower = line.lower()

        # Keep lines relevant to critical/negative analysis
        if any(keyword in line_lower for keyword in [
            "negative", "critical", "compound", "sentiment",
            "entities", "keywords", "vader", "nlp", "manual bias",
            "emotional", "loaded", "per-article"
        ]):
            relevant.append(line)

    if not relevant:
        relevant = lines[:8]

    context_text = "\n".join(relevant[:12])

    return (
        f"\n\nINDEPENDENT NLP ANALYSIS (VADER + spaCy + TF-IDF):\n"
        f"{context_text}\n"
        f"Use articles with negative VADER compound scores as "
        f"stronger evidence for counter-arguments. "
        f"Note any emotionally loaded language as additional evidence."
    )


def _parse_json_response(raw: str) -> dict:
    """Robustly parse JSON from Gemini's response."""
    if not raw or not raw.strip():
        return {
            "counter_arguments": [], "evidence": [],
            "key_sources": [], "confidence_score": 0.0,
            "nlp_validated_counter_arguments": [],
            "top_critical_entities": [],
        }

    # Strategy 1: direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    cleaned = re.sub(
        r"```(?:json)?\s*(.*?)\s*```", r"\1",
        raw, flags=re.DOTALL
    ).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find outermost braces
    first = raw.find("{")
    last  = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(raw[first:last + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 4: fix trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    first = fixed.find("{")
    last  = fixed.rfind("}")
    if first != -1 and last != -1:
        try:
            return json.loads(fixed[first:last + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 5: field-by-field extraction
    result = {}
    for field in [
        "counter_arguments", "evidence", "key_sources",
        "nlp_validated_counter_arguments", "top_critical_entities"
    ]:
        m = re.search(
            rf'"{field}"\s*:\s*\[(.*?)\]', raw, re.DOTALL
        )
        if m:
            result[field] = re.findall(r'"(.*?)"', m.group(1))
        else:
            result[field] = []

    m = re.search(
        r'"confidence_score"\s*:\s*(-?\d+\.?\d*)', raw
    )
    result["confidence_score"] = float(m.group(1)) if m else 0.5

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_b(
    topic:          str,
    visual_context: str = "",
    nlp_context:    str = "",
    cot:            "ORBITACoT" = None,
) -> dict:

    print("\n[Agent B — Critic] Starting...")
    if cot:
        cot.start_step_timer()

    query  = f"criticism problems risks opposition concerns about {topic}"
    chunks = retrieve_chunks(
        query         = query,
        n_results     = AGENT_B_TOP_K,
        stance_filter = "Critical",
    )

    if not chunks:
        print("  No Critical chunks — using unfiltered retrieval")
        chunks = retrieve_chunks(query=query, n_results=AGENT_B_TOP_K)

    print(f"  Retrieved {len(chunks)} chunks")

    if cot:
        top_sources = list(dict.fromkeys(
            c.get("source", "") for c in chunks[:5]
            if c.get("source")
        ))
        cot.add_retrieval_step(
            agent         = "Agent B",
            query         = query,
            n_results     = len(chunks),
            stance_filter = "Critical",
            top_sources   = top_sources,
        )

    context = "\n\n".join([
        f"[{i+1}] {c.get('source', '?')}: {c.get('text', '')[:400]}"
        for i, c in enumerate(chunks)
    ])

    visual_section = ""
    if visual_context and len(visual_context.strip()) > 20:
        visual_section = (
            f"\n\nVISUAL EVIDENCE:\n{visual_context[:400]}\n"
        )

    nlp_section = _build_nlp_section_for_prompt(nlp_context)

    prompt = (
        f"TOPIC: {topic}\n\n"
        f"TEXT EXCERPTS:\n{context}"
        f"{visual_section}"
        f"{nlp_section}\n\n"
        f"Extract ALL critical counter-arguments. "
        f"Return ONLY raw JSON starting with {{."
    )

    if cot:
        cot.start_step_timer()

    try:
        response = genai_client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = [prompt],
            config   = {
                "temperature":       AGENT_TEMPERATURE,
                "max_output_tokens": AGENT_MAX_TOKENS,
                "system_instruction": AGENT_B_SYSTEM_PROMPT,
            },
        )
        raw = response.text
    except Exception as e:
        if cot:
            cot.add_step(
                CoTStepType.ERROR, "Phase 4",
                f"Agent B — Gemini Error: {str(e)[:50]}",
                str(e), agent="Agent B",
            )
        return {
            "counter_arguments": [], "evidence": [],
            "key_sources": [], "confidence_score": 0.0,
            "retrieved_chunks": chunks,
            "nlp_validated_counter_arguments": [],
            "top_critical_entities": [],
            "visual_context_used": bool(visual_context),
            "nlp_context_used": bool(nlp_context),
        }

    result     = _parse_json_response(raw)
    confidence = result.get("confidence_score", 0.5)

    result["retrieved_chunks"]    = chunks
    result["visual_context_used"] = bool(visual_context)
    result["nlp_context_used"]    = bool(nlp_context)

    if "nlp_validated_counter_arguments" not in result:
        result["nlp_validated_counter_arguments"] = []
    if "top_critical_entities" not in result:
        result["top_critical_entities"] = []

    if not result.get("counter_arguments"):
        result["counter_arguments"] = [
            f"Sources indicate concerns about {topic}."
        ]
        result["confidence_score"] = 0.3
        confidence = 0.3

    if cot:
        cot.add_argument_step(
            agent      = "Agent B",
            n_args     = len(result.get("counter_arguments", [])),
            confidence = confidence,
            top_args   = result.get("counter_arguments", [])[:2],
            nlp_used   = bool(nlp_context),
        )

    print(f"  Counter-arguments: {len(result.get('counter_arguments', []))}")
    print(f"  Confidence: {confidence:.2f}")
    print("  [Agent B] Done.")

    return result