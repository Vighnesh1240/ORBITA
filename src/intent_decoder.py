# src/intent_decoder.py

import re
import spacy
from urllib.parse import urlparse

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

# Words that add no search value on their own
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "up", "down", "then", "than", "so",
    "if", "when", "where", "how", "what", "which", "who", "that",
    "this", "these", "those", "it", "its", "i", "you", "he", "she",
    "we", "they", "news", "latest", "recent", "new", "top",
}


def is_url(text: str) -> bool:
    return text.strip().startswith(("http://", "https://", "www."))


def extract_topic_from_url(url: str) -> str:
    parsed = urlparse(url)
    path   = parsed.path.strip("/")
    topic  = re.sub(r"[-_/]", " ", path)
    topic  = re.sub(r"\.\w+$", "", topic)
    segments = [s for s in topic.split() if len(s) > 2]
    return " ".join(segments[:6])


def _keep_full_input(text: str) -> str:
    """
    Clean the raw user input minimally — remove punctuation
    and extra spaces, but keep EVERY meaningful word.
    This is the primary topic used for search queries.
    """
    # Remove punctuation except hyphens between words
    cleaned = re.sub(r"[^\w\s-]", " ", text)
    # Collapse whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _extract_ner_entities(text: str) -> list[str]:
    """
    Use spaCy NER to find named entities.
    Returns entity strings — used as supplementary signals,
    NOT as the primary topic anymore.
    """
    doc      = nlp(text)
    priority = {"ORG", "GPE", "PERSON", "LAW", "EVENT",
                "NORP", "PRODUCT", "MONEY", "DATE", "FAC"}

    priority_ents = []
    other_ents    = []

    for ent in doc.ents:
        if ent.label_ in priority:
            priority_ents.append(ent.text)
        else:
            other_ents.append(ent.text)

    all_ents = priority_ents + other_ents

    # Deduplicate preserving order
    seen, unique = set(), []
    for e in all_ents:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


def _meaningful_words(text: str) -> list[str]:
    """
    Extract all words from user input that are not stop words.
    This preserves domain-specific terms like 'budget', 'lpg',
    'subsidy', 'regulation', etc. that NER misses.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def build_search_queries(
    full_topic: str,
    entities:   list[str],
    keywords:   list[str],
) -> list[str]:
    """
    Build 3–4 diverse search queries.

    Strategy:
    1. The full cleaned input — always the first query
    2. Top 2 NER entities joined (if different from full input)
    3. Full input + "debate analysis" for opinion coverage
    4. Full input + "India" if India not already present
    """
    queries = []

    # Query 1: full user input as-is (most important)
    queries.append(full_topic)

    # Query 2: NER entities joined (supplementary diversity)
    # Avoid generic one-word entity queries like "india".
    generic_single_entities = {
        "india", "indian", "government", "ministry", "state", "country"
    }

    if len(entities) >= 2:
        ent_q = " ".join(entities[:2])
        if ent_q.lower() != full_topic.lower():
            queries.append(ent_q)
    elif len(entities) == 1 and entities[0].lower() != full_topic.lower():
        ent = entities[0].strip()
        if ent.lower() not in generic_single_entities:
            queries.append(ent)

    # Query 3: full input + "debate analysis"
    queries.append(f"{full_topic} debate analysis")

    # Domain expansion: 'crypto' often appears as 'cryptocurrency' in headlines.
    if "crypto" in full_topic.lower() and "cryptocurrency" not in full_topic.lower():
        queries.append(full_topic.lower().replace("crypto", "cryptocurrency"))

    # Query 4: add India context if missing
    if "india" not in full_topic.lower():
        queries.append(f"{full_topic} India")

    # Deduplicate
    seen, unique = set(), []
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)

    return unique[:4]


def decode_intent(user_input: str) -> dict:
    """
    Main function. Takes raw user input (topic or URL).

    KEY CHANGE from previous version:
    - The 'topic' is now the FULL cleaned user input, not just NER entities
    - NER entities are kept as supplementary metadata
    - All meaningful words are preserved regardless of NER classification

    This fixes issues like:
    - "Budget 2026"  → topic = "Budget 2026"  (not just "2026")
    - "India LPG"   → topic = "India LPG"    (not just "India")
    - "UPI payments" → topic = "UPI payments" (not just "UPI")

    Returns:
        {
          original_input:  raw user text
          topic:           full cleaned topic string
          entities:        NER entities found
          keywords:        meaningful words extracted
          search_queries:  list of queries for NewsAPI
        }
    """
    user_input = user_input.strip()

    if not user_input:
        raise ValueError("Input cannot be empty.")

    if is_url(user_input):
        # URL input — extract topic from path
        topic    = extract_topic_from_url(user_input)
        entities = _extract_ner_entities(topic)
        keywords = _meaningful_words(topic)
    else:
        # Text input — keep full input as topic
        topic    = _keep_full_input(user_input)
        entities = _extract_ner_entities(user_input)
        keywords = _meaningful_words(user_input)

    # Build search queries using full topic (not just entities)
    search_queries = build_search_queries(topic, entities, keywords)

    return {
        "original_input":  user_input,
        "topic":           topic,
        "entities":        entities,
        "keywords":        keywords,
        "search_queries":  search_queries,
    }