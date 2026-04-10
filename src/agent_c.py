# src/agent_c.py

import json
import re
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import google.genai as genai
from src.source_bias import compute_weighted_bias

from src.config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    AGENT_TEMPERATURE, AGENT_MAX_TOKENS,
    AGENT_C_TOP_K, HALLUCINATION_THRESH
)
from src.vector_store import retrieve_chunks

genai_client = genai.Client(api_key=GEMINI_API_KEY)


AGENT_C_SYSTEM_PROMPT = """You are Agent C — The Arbitrator for ORBITA.

Your job: Write a neutral 360-degree synthesis of a news topic using the 
arguments from Agent A and Agent B, grounded in the source excerpts.

CRITICAL RULES:
1. You MUST return ONLY a valid JSON object. 
2. Do NOT include any text before or after the JSON.
3. Do NOT wrap the JSON in markdown code blocks.
4. Do NOT use triple backticks anywhere.
5. Start your response directly with { and end with }
6. The bias_score field MUST be a number between -1.0 and 1.0
7. synthesis_report MUST be at least 200 words.

EXACT OUTPUT FORMAT — copy this structure precisely:
{
  "synthesis_report": "your 300-500 word neutral report here",
  "bias_score": 0.0,
  "loaded_language_removed": ["phrase1", "phrase2"],
  "key_agreements": ["agreement1", "agreement2"],
  "key_disagreements": ["disagreement1", "disagreement2"],
  "source_citations": ["Source1", "Source2"],
  "hallucination_flags": []
}

bias_score guide:
-1.0 = overwhelmingly supportive coverage found
 0.0 = perfectly balanced
+1.0 = overwhelmingly critical coverage found"""


def _parse_json_robust(raw: str) -> dict | None:
    """
    Aggressively extract JSON from Gemini's response.
    Tries multiple strategies in order of reliability.
    Returns None only if all strategies fail.
    """
    if not raw or not raw.strip():
        return None

    # ── Strategy 1: Direct parse ──────────────────────────────────
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # ── Strategy 2: Strip markdown fences ────────────────────────
    # Handles ```json ... ``` and ``` ... ```
    cleaned = re.sub(
        r"```(?:json)?\s*(.*?)\s*```",
        r"\1",
        raw,
        flags=re.DOTALL
    ).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ── Strategy 3: Find the outermost { } block ──────────────────
    # Works when Gemini adds explanation before/after the JSON
    first_brace = raw.find("{")
    last_brace  = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = raw[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # ── Strategy 4: Fix common Gemini JSON mistakes ───────────────
    # Trailing commas before } or ]
    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    # Unescaped newlines inside strings
    fixed = re.sub(r'(?<!\\)\n(?=(?:[^"]*"[^"]*")*[^"]*"[^"]*$)', r"\\n", fixed)
    first_brace = fixed.find("{")
    last_brace  = fixed.rfind("}")
    if first_brace != -1 and last_brace != -1:
        candidate = fixed[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # ── Strategy 5: Field-by-field extraction ────────────────────
    # Last resort: extract each field using regex when JSON is malformed
    result = {}

    # Extract synthesis_report
    m = re.search(
        r'"synthesis_report"\s*:\s*"(.*?)(?:"\s*,\s*"[a-z_]+"|\s*})',
        raw, re.DOTALL
    )
    if m:
        result["synthesis_report"] = m.group(1).replace('\\"', '"').strip()

    # Extract bias_score — look for any number after the key
    m = re.search(
        r'"bias_score"\s*:\s*(-?\d+\.?\d*)',
        raw
    )
    if m:
        try:
            result["bias_score"] = float(m.group(1))
        except ValueError:
            result["bias_score"] = 0.0

    # Extract arrays
    for field in ["loaded_language_removed", "key_agreements",
                  "key_disagreements", "source_citations",
                  "hallucination_flags"]:
        m = re.search(
            rf'"{field}"\s*:\s*\[(.*?)\]',
            raw, re.DOTALL
        )
        if m:
            items_raw = m.group(1)
            items = re.findall(r'"(.*?)"', items_raw)
            result[field] = items

    if "synthesis_report" in result or "bias_score" in result:
        return result

    return None


def _normalise_bias_score(result: dict) -> float:
    """
    Extract bias_score from result dict, trying multiple key variants
    that Gemini sometimes uses.
    """
    # Try all common key variants Gemini returns
    for key in ["bias_score", "biasScore", "bias score",
                "bias_rating", "biasRating", "score"]:
        val = result.get(key)
        if val is not None:
            try:
                score = float(val)
                # Clamp to valid range
                return max(-1.0, min(1.0, score))
            except (TypeError, ValueError):
                continue

    # If no numeric score found, try to infer from synthesis text
    synthesis = result.get("synthesis_report", "").lower()
    supportive_words = ["supportive", "favourable", "positive",
                        "beneficial", "advantage"]
    critical_words   = ["critical", "opposed", "negative",
                        "harmful", "problem", "risk"]

    s_count = sum(1 for w in supportive_words if w in synthesis)
    c_count = sum(1 for w in critical_words   if w in synthesis)

    if s_count > c_count * 2:
        return -0.3
    if c_count > s_count * 2:
        return 0.3
    return 0.0


def _hallucination_check(
    claims: list[str],
    source_chunks: list[dict],
) -> tuple[list[str], list[str]]:
    """Check claims against source chunks. Returns (supported, flagged)."""
    if not claims or not source_chunks:
        return claims, []

    source_texts = [c.get("text", "") for c in source_chunks]
    all_texts    = claims + source_texts

    try:
        vec   = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vec.fit_transform(all_texts)
    except ValueError:
        return claims, []

    claim_vecs  = tfidf[:len(claims)]
    source_vecs = tfidf[len(claims):]
    sim_matrix  = cosine_similarity(claim_vecs, source_vecs)

    supported, flagged = [], []
    for i, claim in enumerate(claims):
        if float(np.max(sim_matrix[i])) >= HALLUCINATION_THRESH:
            supported.append(claim)
        else:
            flagged.append(claim)

    return supported, flagged


def _call_gemini_with_retry(prompt: str, max_retries: int = 3) -> str:
    """
    Call Gemini with automatic retry on failure.
    Returns the raw response text.
    """

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Gemini call attempt {attempt}/{max_retries}...")
            response = genai_client.models.generate_content(
                model    = GEMINI_MODEL,
                contents = [prompt],
                config   = {
                    "temperature": 0.1,   # very low for consistent JSON
                    "max_output_tokens": AGENT_MAX_TOKENS,
                    "system_instruction": AGENT_C_SYSTEM_PROMPT,
                },
            )
            raw = response.text.strip()

            if not raw:
                print(f"  Attempt {attempt}: empty response, retrying...")
                time.sleep(3)
                continue

            print(f"  Attempt {attempt}: got {len(raw)} chars")
            return raw

        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = attempt * 5
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)

    raise RuntimeError(
        f"Gemini failed after {max_retries} attempts."
    )


def _build_compact_prompt(
    topic: str,
    agent_a: dict,
    agent_b: dict,
    chunks: list[dict],
) -> str:
    """
    Build a concise prompt that stays well within Gemini's token limit.
    We cap the number of chunks and truncate long texts to avoid
    the response being cut off mid-JSON.
    """
    # Cap to 8 chunks max and 300 chars each to stay under token limit
    MAX_CHUNKS    = 8
    MAX_CHUNK_LEN = 300

    chunk_lines = []
    for i, c in enumerate(chunks[:MAX_CHUNKS], 1):
        text   = c.get("text", "")[:MAX_CHUNK_LEN]
        source = c.get("source", "Unknown")
        stance = c.get("stance", "Unknown")
        chunk_lines.append(f"[{i}] ({source}/{stance}): {text}")

    chunks_block = "\n".join(chunk_lines)

    # Cap agent outputs too
    a_args = agent_a.get("arguments", [])[:4]
    a_evid = agent_a.get("evidence",  [])[:2]
    b_args = agent_b.get("counter_arguments", [])[:4]
    b_evid = agent_b.get("evidence",          [])[:2]

    a_score = agent_a.get("confidence_score", 0.0)
    b_score = agent_b.get("confidence_score", 0.0)

    prompt = f"""TOPIC: {topic}

AGENT A SUPPORTING ARGUMENTS (confidence {a_score:.2f}):
{chr(10).join(f'- {a}' for a in a_args)}
EVIDENCE: {chr(10).join(f'- {e}' for e in a_evid)}

AGENT B COUNTER-ARGUMENTS (confidence {b_score:.2f}):
{chr(10).join(f'- {b}' for b in b_args)}
EVIDENCE: {chr(10).join(f'- {e}' for e in b_evid)}

SOURCE EXCERPTS:
{chunks_block}

Write a neutral 360-degree synthesis. Return ONLY a JSON object starting \
with {{ and ending with }}. No markdown. No backticks. No text outside the JSON."""

    return prompt

def stance_to_numeric(stance: str) -> float:
    """
    Convert stance label to numeric score.
    Supportive → -1
    Neutral    → 0
    Critical   → +1
    """

    if stance == "Supportive":
        return -1.0

    elif stance == "Critical":
        return 1.0

    return 0.0
def run_agent_c(
    topic: str,
    agent_a_output: dict,
    agent_b_output: dict,
) -> dict:
    """Main Agent C function."""
    print("\n[Agent C — Arbitrator] Starting...")

    # RAG retrieval
    chunks = retrieve_chunks(
        query     = f"facts analysis evidence about {topic}",
        n_results = AGENT_C_TOP_K,
    )
    print(f"  Retrieved {len(chunks)} chunks")

    # Combine with agent chunks and deduplicate
    all_chunks = chunks + \
                 agent_a_output.get("retrieved_chunks", []) + \
                 agent_b_output.get("retrieved_chunks", [])
    seen, unique = set(), []
    for c in all_chunks:
        t = c.get("text", "")[:80]
        if t not in seen:
            seen.add(t)
            unique.append(c)

    print(f"  Unique source chunks for grounding: {len(unique)}")

    # Hallucination check
    a_claims = (agent_a_output.get("arguments", []) +
                agent_a_output.get("evidence",  []))
    b_claims = (agent_b_output.get("counter_arguments", []) +
                agent_b_output.get("evidence",          []))
    _, a_flagged = _hallucination_check(a_claims, unique)
    _, b_flagged = _hallucination_check(b_claims, unique)
    all_flagged  = a_flagged + b_flagged

    if all_flagged:
        print(f"  Hallucination flags: {len(all_flagged)}")
    else:
        print("  Hallucination check: all claims grounded")

    # Build compact prompt
    prompt = _build_compact_prompt(topic, agent_a_output,
                                   agent_b_output, unique)

    # Call Gemini with retry
    print("  Calling Gemini API...")
    raw_text = _call_gemini_with_retry(prompt)

    print(f"\n  --- RAW GEMINI RESPONSE (first 300 chars) ---")
    print(f"  {raw_text[:300]}")
    print(f"  --- END RAW RESPONSE ---\n")

    # Parse with robust parser
    parsed = _parse_json_robust(raw_text)

    if parsed is None:
        print("  WARNING: All JSON parsing strategies failed.")
        print("  Building result from raw text...")

        # Last resort: use the raw text as synthesis
        parsed = {
            "synthesis_report":        raw_text[:1500],
            "bias_score":              0.0,
            "loaded_language_removed": [],
            "key_agreements":          [],
            "key_disagreements":       [],
            "source_citations":        [],
            "hallucination_flags":     [],
        }
    else:
        print("  JSON parsed successfully.")

    # ── Compute weighted bias using sources ─────────────────

    bias_scores = []

    for chunk in unique:

        stance = chunk.get("stance", "Neutral")

        stance_score = stance_to_numeric(
            stance
        )

        source_name = chunk.get(
            "source",
            "Unknown"
        )

        weighted = compute_weighted_bias(
            stance_score,
            source_name
        )

        bias_scores.append(weighted)

    # Final bias calculation
    if bias_scores:
        final_bias = sum(bias_scores) / len(bias_scores)
    else:
        final_bias = 0.0

    # Override Gemini bias
    parsed["bias_score"] = final_bias

    # Ensure synthesis is not empty
    synthesis = parsed.get("synthesis_report", "").strip()
    if not synthesis or len(synthesis) < 50:
        # Try to use raw text directly
        parsed["synthesis_report"] = raw_text.strip()[:2000]
        print("  Used raw response as synthesis fallback.")

    # Add hallucination flags
    existing = parsed.get("hallucination_flags", [])
    parsed["hallucination_flags"] = list(set(existing + all_flagged))

    # Validate and print results
    bias = parsed.get("bias_score", 0.0)
    synth_words = len(parsed.get("synthesis_report", "").split())
    print(f"  Bias score:       {bias:+.2f}")
    print(f"  Synthesis:        {synth_words} words")
    print(f"  Hallucination flags: {len(parsed.get('hallucination_flags', []))}")
    print("  [Agent C] Done.")

    return parsed