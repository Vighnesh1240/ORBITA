# src/agent_c.py
"""
ORBITA Agent C — The Arbitrator

Role: Synthesize Agent A and B outputs into an unbiased
360-degree report with hallucination checking.

Enhanced with hybrid context:
    - RAG text chunks from ChromaDB
    - Visual context from Gemini Vision
    - NLP context from VADER + spaCy + TF-IDF

Agent C benefits MOST from NLP context because:
1. It can reference independent VADER scores as validation
2. It knows which claims are emotionally loaded (from NLP)
3. It can compare manual bias estimate with its own
4. Entity data helps ground the synthesis factually
"""

import json
import re
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import google.genai as genai
try:
    from .chain_of_thought import ORBITACoT, CoTStepType
except ImportError:
    from chain_of_thought import ORBITACoT, CoTStepType
try:
    from .bias_model  import compute_bias_vector
    from .source_bias import compute_weighted_bias
    from .config import (
        GEMINI_API_KEY, GEMINI_MODEL,
        AGENT_TEMPERATURE, AGENT_MAX_TOKENS,
        AGENT_C_TOP_K, HALLUCINATION_THRESH
    )
    from .vector_store import retrieve_chunks
except ImportError:
    from bias_model  import compute_bias_vector
    from source_bias import compute_weighted_bias
    from config import (
        GEMINI_API_KEY, GEMINI_MODEL,
        AGENT_TEMPERATURE, AGENT_MAX_TOKENS,
        AGENT_C_TOP_K, HALLUCINATION_THRESH
    )
    from vector_store import retrieve_chunks

genai_client = genai.Client(api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# CITATION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_real_citations(agent_c_response: dict, articles: list) -> list:
    """
    Extract and validate source citations from Agent C response.
    
    Agent C's JSON includes source_citations field. This function
    validates and deduplicates the citations against the actual
    articles list and ensures we only include real sources.
    
    Args:
        agent_c_response: dict from Agent C with 'source_citations' field
        articles:        list of article dicts with 'title', 'url', etc.
    
    Returns:
        list of validated citation strings (article titles/URLs)
    """
    if not agent_c_response:
        return []
    
    citations = agent_c_response.get("source_citations", [])
    if not citations:
        return []
    
    if not articles:
        # No articles to validate against, return as-is (deduplicated)
        return list(dict.fromkeys(citations))
    
    # Build lookup: article titles and URLs that are "real"
    article_titles = {
        a.get("title", "").lower() for a in articles if a.get("title")
    }
    article_urls = {a.get("url", "").lower() for a in articles if a.get("url")}
    article_sources = {
        a.get("source", "").lower() for a in articles if a.get("source")
    }
    
    validated = []
    seen = set()
    
    for citation in citations:
        if not citation or not isinstance(citation, str):
            continue
        
        citation_lower = citation.lower().strip()
        if citation_lower in seen:
            continue  # Skip duplicates
        
        # Check if this citation matches any real article
        is_real = (
            citation_lower in article_titles or
            citation_lower in article_urls or
            citation_lower in article_sources or
            any(
                article_titles_word in citation_lower
                for article_titles_word in article_titles
                if len(article_titles_word) > 10
            )
        )
        
        if is_real:
            validated.append(citation)
            seen.add(citation_lower)
    
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

AGENT_C_SYSTEM_PROMPT = """You are Agent C — The Arbitrator for ORBITA.

Your job: Write a neutral 360-degree synthesis using arguments
from Agent A and Agent B, grounded in source excerpts and
validated against independent NLP analysis data.

CRITICAL RULES:
1. Return ONLY a valid JSON object
2. Do NOT include any text before or after the JSON
3. Do NOT wrap in markdown code blocks
4. Start your response directly with { and end with }
5. bias_score MUST be a number between -1.0 and 1.0
6. synthesis_report MUST be at least 200 words
7. If NLP validation data is provided, reference it in synthesis

EXACT OUTPUT FORMAT:
{
  "synthesis_report": "your 300-500 word neutral report here",
  "bias_score": 0.0,
  "loaded_language_removed": ["phrase1"],
  "key_agreements": ["agreement1"],
  "key_disagreements": ["disagreement1"],
  "source_citations": ["Source1"],
  "hallucination_flags": [],
  "nlp_validation_note": "one sentence noting agreement/disagreement with manual NLP"
}

bias_score:
  -1.0 = overwhelmingly supportive coverage
   0.0 = perfectly balanced
  +1.0 = overwhelmingly critical coverage"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _build_nlp_section_for_prompt(nlp_context: str) -> str:
    """
    Format NLP context for injection into Agent C's prompt.

    Agent C gets the FULL NLP context because it needs:
    - Overall sentiment distribution
    - Manual bias estimate for cross-validation
    - Entity data for factual grounding
    - Keyword data for topic relevance
    - Agreement/disagreement with Gemini

    Agent C should reference this in its synthesis to
    demonstrate hybrid validation.
    """
    if not nlp_context or len(nlp_context.strip()) < 20:
        return ""

    # Agent C gets more context than A or B
    # because it writes the final synthesis
    context_text = nlp_context.strip()[:700]

    return (
        f"\n\nINDEPENDENT NLP VALIDATION DATA:\n"
        f"{context_text}\n"
        f"IMPORTANT: In your synthesis_report, briefly note "
        f"whether the manual NLP analysis (VADER sentiment, "
        f"named entities) supports or contradicts the arguments "
        f"from Agent A and Agent B. This cross-validation "
        f"strengthens the credibility of your synthesis. "
        f"Also populate nlp_validation_note with a one-sentence "
        f"summary of this cross-validation."
    )


def _parse_json_robust(raw: str) -> dict:
    """
    Aggressively extract JSON from Gemini's response.
    Tries multiple strategies in order of reliability.
    """
    if not raw or not raw.strip():
        return None

    # Strategy 1: Direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown fences
    cleaned = re.sub(
        r"```(?:json)?\s*(.*?)\s*```",
        r"\1", raw, flags=re.DOTALL
    ).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find outermost { }
    first_brace = raw.find("{")
    last_brace  = raw.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(raw[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 4: Fix trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    f = fixed.find("{")
    l = fixed.rfind("}")
    if f != -1 and l > f:
        try:
            return json.loads(fixed[f:l + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 5: Field-by-field extraction
    result = {}

    m = re.search(
        r'"synthesis_report"\s*:\s*"(.*?)(?:"\s*,\s*"[a-z_]+"|\s*})',
        raw, re.DOTALL
    )
    if m:
        result["synthesis_report"] = (
            m.group(1).replace('\\"', '"').strip()
        )

    m = re.search(
        r'"bias_score"\s*:\s*(-?\d+\.?\d*)', raw
    )
    if m:
        try:
            result["bias_score"] = float(m.group(1))
        except ValueError:
            result["bias_score"] = 0.0

    for field in [
        "loaded_language_removed", "key_agreements",
        "key_disagreements", "source_citations", "hallucination_flags"
    ]:
        m = re.search(
            rf'"{field}"\s*:\s*\[(.*?)\]', raw, re.DOTALL
        )
        if m:
            result[field] = re.findall(r'"(.*?)"', m.group(1))
        else:
            result[field] = []

    m = re.search(
        r'"nlp_validation_note"\s*:\s*"([^"]+)"', raw
    )
    if m:
        result["nlp_validation_note"] = m.group(1)

    if "synthesis_report" in result or "bias_score" in result:
        return result

    return None


def _hallucination_check(
    claims:        list,
    source_chunks: list,
) -> tuple:
    """
    Check claims against source chunks.

    Uses TF-IDF cosine similarity — a claim is flagged as
    potentially hallucinated if it has low similarity to
    all source chunks (it wasn't grounded in the sources).

    Args:
        claims:        list of argument/claim strings
        source_chunks: list of chunk dicts from ChromaDB

    Returns:
        tuple (supported_claims, flagged_claims)
    """
    # Filter short claims — can't meaningfully check them
    claims = [c for c in claims if len(c.split()) >= 6]

    if not claims or not source_chunks:
        return claims, []

    source_texts = [c.get("text", "") for c in source_chunks]
    all_texts    = claims + source_texts

    try:
        vec   = TfidfVectorizer(
            stop_words   = "english",
            max_features = 5000,
            ngram_range  = (1, 2),
        )
        tfidf = vec.fit_transform(all_texts)
    except ValueError:
        return claims, []

    claim_vecs  = tfidf[:len(claims)]
    source_vecs = tfidf[len(claims):]
    sim_matrix  = cosine_similarity(claim_vecs, source_vecs)

    supported = []
    flagged   = []

    for i, claim in enumerate(claims):
        max_sim = float(np.max(sim_matrix[i]))
        if max_sim >= HALLUCINATION_THRESH:
            supported.append(claim)
        else:
            flagged.append(claim)

    return supported, flagged


def _call_gemini_with_retry(
    prompt:      str,
    max_retries: int = 3,
) -> str:
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
                    "temperature":       0.1,
                    "max_output_tokens": AGENT_MAX_TOKENS,
                    "system_instruction": AGENT_C_SYSTEM_PROMPT,
                },
            )
            raw = response.text.strip()

            if not raw:
                print(f"  Attempt {attempt}: empty response")
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

    raise RuntimeError(f"Gemini failed after {max_retries} attempts.")


def _build_compact_prompt(
    topic:         str,
    agent_a:       dict,
    agent_b:       dict,
    chunks:        list,
    visual_section: str = "",
    nlp_section:    str = "",
) -> str:
    """
    Build a concise prompt for Agent C.

    Includes agent outputs + source chunks + visual + NLP context.
    Capped to stay within Gemini's token limit.
    """
    MAX_CHUNKS    = 8
    MAX_CHUNK_LEN = 300

    chunk_lines = []
    for i, c in enumerate(chunks[:MAX_CHUNKS], 1):
        text   = c.get("text",   "")[:MAX_CHUNK_LEN]
        source = c.get("source", "Unknown")
        stance = c.get("stance", "Unknown")
        chunk_lines.append(
            f"[{i}] ({source}/{stance}): {text}"
        )

    chunks_block = "\n".join(chunk_lines)

    a_args = agent_a.get("arguments",           [])[:4]
    a_evid = agent_a.get("evidence",             [])[:2]
    b_args = agent_b.get("counter_arguments",   [])[:4]
    b_evid = agent_b.get("evidence",             [])[:2]
    a_conf = agent_a.get("confidence_score", 0.0)
    b_conf = agent_b.get("confidence_score", 0.0)

    # Add NLP validation info in a compact way
    nlp_compact = ""
    if nlp_section and len(nlp_section.strip()) > 20:
        # Take first 400 chars of NLP section
        nlp_compact = nlp_section[:400]

    prompt = (
        f"TOPIC: {topic}\n\n"
        f"AGENT A SUPPORTING ARGUMENTS (confidence {a_conf:.2f}):\n"
        + "\n".join(f"- {a}" for a in a_args) +
        f"\nEVIDENCE: " + " | ".join(a_evid) +

        f"\n\nAGENT B COUNTER-ARGUMENTS (confidence {b_conf:.2f}):\n"
        + "\n".join(f"- {b}" for b in b_args) +
        f"\nEVIDENCE: " + " | ".join(b_evid) +

        f"\n\nSOURCE EXCERPTS:\n{chunks_block}"
    )

    if visual_section:
        prompt += visual_section

    if nlp_compact:
        prompt += nlp_compact

    prompt += (
        "\n\nWrite a neutral 360-degree synthesis. "
        "Return ONLY a JSON object starting with { "
        "and ending with }. No markdown. No backticks."
    )

    return prompt


def stance_to_numeric(stance: str) -> float:
    """Convert stance label to numeric score."""
    if stance == "Supportive":
        return -1.0
    elif stance == "Critical":
        return 1.0
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_c(
    topic:          str,
    agent_a_output: dict,
    agent_b_output: dict,
    visual_context: str = "",
    nlp_context:    str = "",
    cot:            "ORBITACoT" = None,
) -> dict:

    print("\n[Agent C — Arbitrator] Starting...")
    if cot:
        cot.start_step_timer()

    chunks = retrieve_chunks(
        query     = f"facts analysis evidence about {topic}",
        n_results = AGENT_C_TOP_K,
    )
    print(f"  Retrieved {len(chunks)} chunks")

    if cot:
        cot.add_retrieval_step(
            agent        = "Agent C",
            query        = f"facts analysis evidence about {topic}",
            n_results    = len(chunks),
            top_sources  = list(dict.fromkeys(
                c.get("source", "") for c in chunks[:5]
                if c.get("source")
            )),
        )

    # Combine chunks
    all_chunks = (
        chunks +
        agent_a_output.get("retrieved_chunks", []) +
        agent_b_output.get("retrieved_chunks", [])
    )
    seen, unique = set(), []
    for c in all_chunks:
        t = c.get("text", "")[:80]
        if t not in seen:
            seen.add(t)
            unique.append(c)

    # Hallucination check
    a_claims = (
        agent_a_output.get("arguments", []) +
        agent_a_output.get("evidence",  [])
    )
    b_claims = (
        agent_b_output.get("counter_arguments", []) +
        agent_b_output.get("evidence",           [])
    )
    _, a_flagged = _hallucination_check(a_claims, unique)
    _, b_flagged = _hallucination_check(b_claims, unique)
    all_flagged  = a_flagged + b_flagged

    if cot:
        cot.add_step(
            step_type  = CoTStepType.VALIDATION,
            phase      = "Phase 4",
            title      = (
                f"Hallucination Check — "
                f"{len(all_flagged)} flags"
            ),
            detail     = (
                f"Checked {len(a_claims + b_claims)} claims "
                f"against {len(unique)} source chunks.\n"
                f"Flagged {len(all_flagged)} potentially "
                f"ungrounded claims.\n"
                f"Method: TF-IDF cosine similarity "
                f"(threshold={HALLUCINATION_THRESH})"
            ),
            evidence   = [
                f"Total claims checked: {len(a_claims + b_claims)}",
                f"Source chunks used: {len(unique)}",
                f"Claims flagged: {len(all_flagged)}",
                f"Claims verified: {len(a_claims+b_claims)-len(all_flagged)}",
            ],
            confidence = max(0.5, 1.0 - len(all_flagged) * 0.05),
            agent      = "Agent C",
        )
        cot.start_step_timer()

    # Build prompt
    visual_section = ""
    if visual_context and len(visual_context.strip()) > 20:
        visual_section = (
            f"\n\nVISUAL ANALYSIS:\n{visual_context[:500]}\n"
        )

    nlp_section = _build_nlp_section_for_prompt(nlp_context)

    prompt = _build_compact_prompt(
        topic          = topic,
        agent_a        = agent_a_output,
        agent_b        = agent_b_output,
        chunks         = unique,
        visual_section = visual_section,
        nlp_section    = nlp_section,
    )

    print("  Calling Gemini API...")
    raw_text = _call_gemini_with_retry(prompt)

    if cot:
        cot.add_step(
            step_type  = CoTStepType.SYNTHESIS,
            phase      = "Phase 4",
            title      = "Gemini Synthesis Complete",
            detail     = (
                f"Agent C called Gemini API with combined context:\n"
                f"Agent A args + Agent B counters + RAG chunks + "
                f"NLP validation data.\n"
                f"Response length: {len(raw_text)} chars"
            ),
            evidence   = [
                f"Prompt length: {len(prompt)} chars",
                f"Response length: {len(raw_text)} chars",
            ],
            confidence = 0.85,
            agent      = "Agent C",
        )

    parsed = _parse_json_robust(raw_text)
    if parsed is None:
        parsed = {
            "synthesis_report":        raw_text[:1500],
            "bias_score":              0.0,
            "loaded_language_removed": [],
            "key_agreements":          [],
            "key_disagreements":       [],
            "source_citations":        [],
            "hallucination_flags":     [],
            "nlp_validation_note":     "",
        }

    # Compute bias vector
    seen_urls      = set()
    chunk_articles = []
    for chunk in unique:
        url = chunk.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            chunk_articles.append({
                "url":         url,
                "source":      chunk.get("source",  "Unknown"),
                "stance":      chunk.get("stance",  "Neutral"),
                "title":       chunk.get("title",   ""),
                "full_text":   chunk.get("text",    ""),
                "description": chunk.get("text",    "")[:200],
            })

    bias_vector = compute_bias_vector(
        articles       = chunk_articles,
        agent_a_output = agent_a_output,
        agent_b_output = agent_b_output,
        agent_c_output = parsed,
    )

    final_bias            = bias_vector["composite_score"]
    parsed["bias_score"]  = final_bias
    parsed["bias_vector"] = bias_vector

    if cot:
        cot.add_step(
            step_type  = CoTStepType.DECISION,
            phase      = "Phase 4",
            title      = (
                f"Bias Vector Computed → "
                f"{final_bias:+.4f} "
                f"({bias_vector.get('interpretation', 'N/A')})"
            ),
            detail     = (
                f"Multi-dimensional bias vector computed:\n"
                f"Ideological: {bias_vector.get('ideological_bias', 0):+.4f}\n"
                f"Emotional:   {bias_vector.get('emotional_bias', 0):.4f}\n"
                f"Diversity:   {bias_vector.get('source_diversity', 0):.4f}\n"
                f"Composite:   {final_bias:+.4f}"
            ),
            evidence   = [
                f"Ideological: {bias_vector.get('ideological_bias',0):+.4f}",
                f"Emotional:   {bias_vector.get('emotional_bias',0):.4f}",
                f"Info:        {bias_vector.get('informational_bias',0):.4f}",
                f"Diversity:   {bias_vector.get('source_diversity',0):.4f}",
                f"Composite:   {final_bias:+.4f}",
            ],
            confidence = bias_vector.get("confidence", 0.8),
            score      = final_bias,
            agent      = "Agent C",
        )

    synthesis = parsed.get("synthesis_report", "").strip()
    if not synthesis or len(synthesis) < 50:
        parsed["synthesis_report"] = raw_text.strip()[:2000]

    existing = parsed.get("hallucination_flags", [])
    parsed["hallucination_flags"] = list(set(existing + all_flagged))
    parsed["visual_context_used"] = bool(visual_context)
    parsed["nlp_context_used"]    = bool(nlp_context)

    if "nlp_validation_note" not in parsed:
        parsed["nlp_validation_note"] = ""

    print(f"  Bias: {final_bias:+.4f}")
    print(f"  Synthesis: {len(parsed.get('synthesis_report','').split())} words")
    print("  [Agent C] Done.")

    return parsed