# src/agent_a.py

import json
import re
import google.genai as genai

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

# New google.genai client interface (no configure API)
try:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    genai_client = None


AGENT_A_SYSTEM_PROMPT = """You are Agent A — The Analyst for ORBITA.

TASK: Extract supporting arguments from the provided article excerpts.

OUTPUT: Return ONLY a raw JSON object. No markdown. No backticks. No text 
outside the JSON. Start with { and end with }.

RULES:
- Extract ONLY what is explicitly stated in the excerpts
- Each argument must be a complete, specific sentence
- Do not add opinions or external knowledge
- Minimum 3 arguments if the text supports it
- confidence_score: 0.0 (no support) to 1.0 (strong support)

JSON STRUCTURE:
{
  "arguments": ["specific argument 1", "specific argument 2"],
  "evidence": ["data point or quote 1", "data point or quote 2"],
  "key_sources": ["Source Name 1"],
  "confidence_score": 0.75
}"""


def _build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context block
    for the Gemini prompt.
    """
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
    Tries multiple strategies before falling back.
    """
    if not raw or not raw.strip():
        return {
            "arguments": [], "evidence": [],
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

    # Strategy 5: field-by-field regex extraction
    result = {}
    for field in ["arguments", "evidence", "key_sources"]:
        m = re.search(rf'"{field}"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
        if m:
            result[field] = re.findall(r'"(.*?)"', m.group(1))
        else:
            result[field] = []

    m = re.search(r'"confidence_score"\s*:\s*(-?\d+\.?\d*)', raw)
    result["confidence_score"] = float(m.group(1)) if m else 0.5

    return result

def _fallback_agent_a(chunks: list[dict]) -> dict:
    """Fallback logic when Gemini API is unavailable."""
    arguments = []
    evidence   = []
    key_sources = set()

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if text:
            if len(arguments) < 2 and text not in arguments:
                arguments.append(text[:160])
            if len(evidence) < 2 and text not in evidence:
                evidence.append(text[:160])
        key_sources.add(chunk.get("source", "Unknown"))

    if not arguments:
        arguments = ["No supporting arguments found in chunks."]
    if not evidence:
        evidence = ["No supporting evidence found in chunks."]

    traces = _build_argument_trace(
        arguments,
        evidence,
        chunks
    )

    return {
        "arguments": arguments,
        "evidence": evidence,
        "key_sources": list(key_sources),
        "confidence_score": 0.5,
        "argument_traces": traces,
        "retrieved_chunks": chunks,
        }

def _build_argument_trace(
    arguments: list,
    evidence: list,
    chunks: list
) -> list[dict]:
    """
    Map arguments to source evidence.
    Enables traceability and transparency.
    """

    traced_arguments = []

    for i, arg in enumerate(arguments):
        ev = evidence[i] if i < len(evidence) else ""

        matched_chunk = None

        # Try matching evidence to chunk text
        for chunk in chunks:
            if ev and ev[:40] in chunk.get("text", ""):
                matched_chunk = chunk
                break

        traced_arguments.append({
            "argument": arg,
            "evidence": ev,
            "source": (
                matched_chunk.get("source", "Unknown")
                if matched_chunk else "Unknown"
            ),
            "url": (
                matched_chunk.get("url", "#")
                if matched_chunk else "#"
            )
        })

    return traced_arguments

def run_agent_a(topic: str) -> dict:
    """Agent A — extracts supporting arguments via RAG."""
    print("\n[Agent A — Analyst] Starting...")

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

    # Build a concise prompt — avoid token limit issues
    context = "\n\n".join([
        f"[{i+1}] {c.get('source','?')}: {c.get('text','')[:400]}"
        for i, c in enumerate(chunks)
    ])

    prompt = (
        f"TOPIC: {topic}\n\n"
        f"EXCERPTS:\n{context}\n\n"
        f"Extract all supporting arguments. "
        f"Return ONLY raw JSON starting with {{."
    )

    try:
        response = genai_client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = [prompt],
            config   = {
                "temperature": AGENT_TEMPERATURE,
                "max_output_tokens": AGENT_MAX_TOKENS,
                "system_instruction": AGENT_A_SYSTEM_PROMPT,
            },
        )
        raw = response.text
    except Exception as e:
        print(f"  Gemini error: {e}")
        return {
            "arguments": [], "evidence": [],
            "key_sources": [], "confidence_score": 0.0,
            "retrieved_chunks": chunks,
        }

    result = _parse_json_response(raw)

    arguments = result.get("arguments", [])
    evidence  = result.get("evidence", [])

    # Build traceable structure
    argument_traces = _build_argument_trace(
        arguments,
        evidence,
        chunks
    )

    result["argument_traces"] = argument_traces
    result["retrieved_chunks"] = chunks

    # Validate — ensure we have actual content
    if not result.get("arguments"):
        result["arguments"] = [
            f"Sources indicate support for {topic} "
            f"based on retrieved articles."
        ]
        result["confidence_score"] = 0.3

    print(f"  Arguments: {len(result.get('arguments', []))}")
    print(f"  Confidence: {result.get('confidence_score', 0):.2f}")
    print("  [Agent A] Done.")
    return result