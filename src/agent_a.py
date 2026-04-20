# src/agent_a.py
"""
ORBITA Agent A — The Analyst

Role: Extract supporting arguments from retrieved article chunks.

Enhanced with hybrid context:
    - RAG text chunks from ChromaDB (semantic search)
    - Visual context from Gemini Vision image analysis
    - NLP context from VADER + spaCy + TF-IDF (manual NLP)

The NLP context helps Agent A identify which articles have
genuinely positive framing (high VADER score) vs which are
just labeled "Supportive" by the stance classifier.

Research justification:
    "Agent A receives independent NLP validation data,
     allowing it to distinguish emotionally positive
     framing from factually supportive content."
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
        AGENT_TEMPERATURE, AGENT_MAX_TOKENS, AGENT_A_TOP_K
    )
    from .vector_store import retrieve_chunks
except ImportError:
    from config import (
        GEMINI_API_KEY, GEMINI_MODEL,
        AGENT_TEMPERATURE, AGENT_MAX_TOKENS, AGENT_A_TOP_K
    )
    from vector_store import retrieve_chunks

try:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    genai_client = None


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

AGENT_A_SYSTEM_PROMPT = """You are Agent A — The Analyst for ORBITA.

TASK: Extract supporting arguments and evidence from the provided
article excerpts and NLP analysis data.

OUTPUT: Return ONLY a raw JSON object. No markdown. No backticks.
No text outside the JSON. Start with { and end with }.

RULES:
- Extract ONLY what is explicitly stated in the excerpts
- Each argument must be a complete, specific sentence
- Use NLP context (VADER scores, entities, keywords) to
  identify which sources have strongest supportive framing
- Do not add opinions or external knowledge
- Minimum 3 arguments if the text supports it
- confidence_score: 0.0 (no support) to 1.0 (strong support)

JSON STRUCTURE:
{
  "arguments": ["specific argument 1", "specific argument 2"],
  "evidence": ["data point or quote 1", "data point 2"],
  "key_sources": ["Source Name 1"],
  "confidence_score": 0.75,
  "nlp_validated_arguments": ["argument backed by positive VADER score"],
  "top_entities_referenced": ["entity1", "entity2"]
}"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _build_nlp_section_for_prompt(nlp_context: str) -> str:
    """
    Format NLP context for injection into Agent A's prompt.

    Agent A is the ANALYST (supportive side), so we highlight:
    - Articles with POSITIVE VADER scores
    - Entities mentioned in supportive coverage
    - Keywords associated with supportive framing

    Args:
        nlp_context: raw context string from nlp_analyzer

    Returns:
        formatted string for prompt injection,
        or empty string if no context available
    """
    if not nlp_context or len(nlp_context.strip()) < 20:
        return ""

    # Extract just the most relevant lines for Agent A
    lines      = nlp_context.strip().split("\n")
    relevant   = []

    for line in lines:
        line_lower = line.lower()

        # Keep lines relevant to supporting/positive analysis
        if any(keyword in line_lower for keyword in [
            "positive", "supportive", "compound", "sentiment",
            "entities", "keywords", "vader", "nlp", "manual bias",
            "validation", "per-article"
        ]):
            relevant.append(line)

    if not relevant:
        # If no specific lines matched, include first 8 lines
        relevant = lines[:8]

    context_text = "\n".join(relevant[:12])  # Cap at 12 lines

    return (
        f"\n\nINDEPENDENT NLP ANALYSIS (VADER + spaCy + TF-IDF):\n"
        f"{context_text}\n"
        f"Use articles with positive VADER compound scores as "
        f"stronger evidence for supporting arguments. "
        f"Reference entities and keywords where relevant."
    )


def _parse_json_response(raw: str) -> dict:
    """
    Robustly parse JSON from Gemini's response.
    Tries multiple strategies before falling back.
    """
    if not raw or not raw.strip():
        return {
            "arguments": [], "evidence": [],
            "key_sources": [], "confidence_score": 0.0,
            "nlp_validated_arguments": [],
            "top_entities_referenced": [],
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
    for field in ["arguments", "evidence", "key_sources",
                  "nlp_validated_arguments", "top_entities_referenced"]:
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


def _build_argument_trace(
    arguments: list,
    evidence:  list,
    chunks:    list,
) -> list:
    """
    Map arguments to source evidence using semantic similarity.

    Uses TF-IDF cosine similarity to match each argument
    to its most relevant source chunk.

    This enables full traceability:
    Argument → Evidence → Source → URL
    """
    if not arguments:
        return []

    traced = []

    for i, arg in enumerate(arguments):
        best_ev    = evidence[i] if i < len(evidence) else ""
        best_chunk = None

        if evidence and chunks:
            try:
                corpus = [arg] + evidence
                vec    = TfidfVectorizer(
                    stop_words="english", max_features=500
                )
                mat  = vec.fit_transform(corpus)
                sims = cosine_similarity(mat[0:1], mat[1:])[0]
                best_ev_idx = int(np.argmax(sims))
                best_ev     = evidence[best_ev_idx]

                # Find matching chunk for the best evidence
                chunk_texts = [c.get("text", "") for c in chunks]
                corpus2     = [best_ev] + chunk_texts
                vec2        = TfidfVectorizer(
                    stop_words="english", max_features=500
                )
                mat2  = vec2.fit_transform(corpus2)
                sims2 = cosine_similarity(mat2[0:1], mat2[1:])[0]
                best_chunk = chunks[int(np.argmax(sims2))]

            except Exception:
                pass

        traced.append({
            "argument": arg,
            "evidence": best_ev,
            "source":   (
                best_chunk.get("source", "Unknown")
                if best_chunk else "Unknown"
            ),
            "url": (
                best_chunk.get("url", "#")
                if best_chunk else "#"
            ),
        })

    return traced


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_a(
    topic:          str,
    visual_context: str = "",
    nlp_context:    str = "",
    cot:            "ORBITACoT" = None,
) -> dict:

    print("\n[Agent A — Analyst] Starting...")
    if cot:
        cot.start_step_timer()

    # RAG Retrieval
    query  = f"benefits advantages support arguments in favour of {topic}"
    chunks = retrieve_chunks(
        query         = query,
        n_results     = AGENT_A_TOP_K,
        stance_filter = "Supportive",
    )

    if not chunks:
        print("  No Supportive chunks — using unfiltered retrieval")
        chunks = retrieve_chunks(query=query, n_results=AGENT_A_TOP_K)

    print(f"  Retrieved {len(chunks)} chunks")

    # CoT: record retrieval
    if cot:
        top_sources = list(dict.fromkeys(
            c.get("source", "") for c in chunks[:5]
            if c.get("source")
        ))
        cot.add_retrieval_step(
            agent         = "Agent A",
            query         = query,
            n_results     = len(chunks),
            stance_filter = "Supportive",
            top_sources   = top_sources,
        )

    # Build context
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
        f"Extract ALL supporting arguments. "
        f"Return ONLY raw JSON starting with {{."
    )

    # CoT: record prompt construction
    if cot:
        cot.add_step(
            step_type  = CoTStepType.ARGUMENT,
            phase      = "Phase 4",
            title      = "Agent A — Building Prompt",
            detail     = (
                f"Constructed prompt with:\n"
                f"• {len(chunks)} RAG chunks\n"
                f"• Visual context: {'yes' if visual_context else 'no'}\n"
                f"• NLP context: {'yes' if nlp_context else 'no'}"
            ),
            evidence   = [
                f"RAG chunks: {len(chunks)}",
                f"Prompt length: {len(prompt)} chars",
            ],
            confidence = 0.9,
            agent      = "Agent A",
        )
        cot.start_step_timer()

    # Call Gemini
    try:
        response = genai_client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = [prompt],
            config   = {
                "temperature":       AGENT_TEMPERATURE,
                "max_output_tokens": AGENT_MAX_TOKENS,
                "system_instruction": AGENT_A_SYSTEM_PROMPT,
            },
        )
        raw = response.text
    except Exception as e:
        if cot:
            cot.add_step(
                CoTStepType.ERROR, "Phase 4",
                f"Agent A — Gemini Error: {str(e)[:50]}",
                str(e), agent="Agent A",
            )
        return {
            "arguments": [], "evidence": [],
            "key_sources": [], "confidence_score": 0.0,
            "argument_traces": [], "retrieved_chunks": chunks,
            "nlp_validated_arguments": [],
            "top_entities_referenced": [],
            "visual_context_used": bool(visual_context),
            "nlp_context_used": bool(nlp_context),
        }

    result    = _parse_json_response(raw)
    arguments = result.get("arguments", [])
    evidence  = result.get("evidence",  [])
    confidence = result.get("confidence_score", 0.5)

    # Build argument traces
    argument_traces = _build_argument_trace(
        arguments, evidence, chunks
    )

    result["argument_traces"]     = argument_traces
    result["retrieved_chunks"]    = chunks
    result["visual_context_used"] = bool(visual_context)
    result["nlp_context_used"]    = bool(nlp_context)

    if "nlp_validated_arguments"  not in result:
        result["nlp_validated_arguments"]  = []
    if "top_entities_referenced"  not in result:
        result["top_entities_referenced"]  = []

    if not result.get("arguments"):
        result["arguments"]        = [
            f"Sources indicate support for {topic}."
        ]
        result["confidence_score"] = 0.3
        confidence                  = 0.3

    # CoT: record extraction results
    if cot:
        cot.add_argument_step(
            agent      = "Agent A",
            n_args     = len(result.get("arguments", [])),
            confidence = confidence,
            top_args   = result.get("arguments", [])[:2],
            nlp_used   = bool(nlp_context),
        )

    print(f"  Arguments: {len(result.get('arguments', []))}")
    print(f"  Confidence: {confidence:.2f}")
    print("  [Agent A] Done.")

    return result