# src/agent_b.py

import json
import re
import google.genai as genai

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


AGENT_B_SYSTEM_PROMPT = """You are Agent B — The Critic for ORBITA.

TASK: Extract counter-arguments and criticisms from the provided article excerpts.

OUTPUT: Return ONLY a raw JSON object. No markdown. No backticks. No text 
outside the JSON. Start with { and end with }.

RULES:
- Extract ONLY what is explicitly stated in the excerpts
- Each counter-argument must be a complete, specific sentence
- Do not add opinions or external knowledge
- Minimum 3 counter-arguments if the text supports it
- confidence_score: 0.0 (no criticism) to 1.0 (strong criticism)

JSON STRUCTURE:
{
  "counter_arguments": ["specific counter-argument 1", "specific counter-argument 2"],
  "evidence": ["critical data point or quote 1", "critical data point 2"],
  "key_sources": ["Source Name 1"],
  "confidence_score": 0.75
}"""


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        title  = chunk.get("title",  "Unknown")[:60]
        text   = chunk.get("text",   "").strip()
        lines.append(
            f"[EXCERPT {i}]\n"
            f"Source: {source} | Article: {title}\n"
            f"{text}\n"
        )
    return "\n---\n".join(lines)


def _parse_json_response(raw: str) -> dict:
    """
    Robustly parse JSON from Gemini's response.
    """
    if not raw or not raw.strip():
        return {
            "counter_arguments": [], "evidence": [],
            "key_sources": [], "confidence_score": 0.0,
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
    for field in ["counter_arguments", "evidence", "key_sources"]:
        m = re.search(rf'"{field}"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
        if m:
            result[field] = re.findall(r'"(.*?)"', m.group(1))
        else:
            result[field] = []

    m = re.search(r'"confidence_score"\s*:\s*(-?\d+\.?\d*)', raw)
    result["confidence_score"] = float(m.group(1)) if m else 0.5

    return result


def _fallback_agent_b(chunks: list[dict]) -> dict:
    """Fallback logic when Gemini API is unavailable."""
    counter_arguments = []
    evidence = []
    key_sources = set()

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if text:
            if len(counter_arguments) < 2 and text not in counter_arguments:
                counter_arguments.append(text[:160])
            if len(evidence) < 2 and text not in evidence:
                evidence.append(text[:160])
        key_sources.add(chunk.get("source", "Unknown"))

    if not counter_arguments:
        counter_arguments = ["No counter-arguments found in chunks."]
    if not evidence:
        evidence = ["No counter-evidence found in chunks."]

    return {
        "counter_arguments": counter_arguments,
        "evidence":          evidence,
        "key_sources":       list(key_sources),
        "confidence_score":  0.5,
        "retrieved_chunks":  chunks,
    }


def run_agent_b(topic: str) -> dict:
    """Agent B — extracts counter-arguments via RAG."""
    print("\n[Agent B — Critic] Starting...")

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

    context = "\n\n".join([
        f"[{i+1}] {c.get('source','?')}: {c.get('text','')[:400]}"
        for i, c in enumerate(chunks)
    ])

    prompt = (
        f"TOPIC: {topic}\n\n"
        f"EXCERPTS:\n{context}\n\n"
        f"Extract all critical counter-arguments. "
        f"Return ONLY raw JSON starting with {{."
    )

    try:
        response = genai_client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = [prompt],
            config   = {
                "temperature": AGENT_TEMPERATURE,
                "max_output_tokens": AGENT_MAX_TOKENS,
                "system_instruction": AGENT_B_SYSTEM_PROMPT,
            },
        )
        raw = response.text
    except Exception as e:
        print(f"  Gemini error: {e}")
        return {
            "counter_arguments": [], "evidence": [],
            "key_sources": [], "confidence_score": 0.0,
            "retrieved_chunks": chunks,
        }

    result = _parse_json_response(raw)
    result["retrieved_chunks"] = chunks

    if not result.get("counter_arguments"):
        result["counter_arguments"] = [
            f"Sources indicate concerns about {topic} "
            f"based on retrieved articles."
        ]
        result["confidence_score"] = 0.3

    print(f"  Counter-arguments: {len(result.get('counter_arguments', []))}")
    print(f"  Confidence: {result.get('confidence_score', 0):.2f}")
    print("  [Agent B] Done.")
    return result