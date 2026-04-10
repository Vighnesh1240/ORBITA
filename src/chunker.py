# src/chunker.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from .config import CHUNK_SIZE, CHUNK_OVERLAP
except ImportError:
    from config import CHUNK_SIZE, CHUNK_OVERLAP


def _make_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a LangChain text splitter.

    CHUNK_SIZE and CHUNK_OVERLAP are in tokens.
    We use chars = tokens * 4 as a practical approximation
    (average English word is ~4 chars, ~1.3 tokens, so 4 chars/token is safe).
    """
    return RecursiveCharacterTextSplitter(
        chunk_size        = CHUNK_SIZE * 4,      # ~500 tokens in characters
        chunk_overlap     = CHUNK_OVERLAP * 4,   # ~50 token overlap
        length_function   = len,
        separators        = ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )


def chunk_article(article: dict) -> list[dict]:
    """
    Split a single article into overlapping text chunks.
    Each chunk is a dict containing the text and metadata.

    Args:
        article: article dict with 'full_text', 'url', 'title',
                 'source', 'stance' fields

    Returns:
        list of chunk dicts, each with:
          - chunk_id:   unique string ID
          - text:       the chunk text
          - url:        source article URL
          - title:      source article title
          - source:     news source name
          - stance:     Supportive / Critical / Neutral
          - chunk_index: position of this chunk within the article
    """
    full_text = article.get("full_text", "").strip()

    if not full_text:
        return []

    splitter = _make_splitter()
    raw_chunks = splitter.split_text(full_text)

    # Build chunk dicts with metadata
    chunks = []
    url    = article.get("url", "")
    title  = article.get("title", "Unknown")
    source = article.get("source", "Unknown")
    stance = article.get("stance", "Neutral")

    # Create a short safe ID from the URL
    url_id = "".join(c if c.isalnum() else "_" for c in url)[-40:]

    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if len(chunk_text) < 50:
            # Skip tiny fragments (usually leftover punctuation)
            continue

        chunks.append({
            "chunk_id":    f"{url_id}_chunk_{i}",
            "text":        chunk_text,
            "url":         url,
            "title":       title,
            "source":      source,
            "stance":      stance,
            "chunk_index": i,
        })

    return chunks


def chunk_all_articles(articles: list[dict]) -> list[dict]:
    """
    Main function. Chunks every article in the list.

    Args:
        articles: list of article dicts from Step 2

    Returns:
        flat list of all chunk dicts across all articles
    """
    print(f"\n[chunker] Splitting {len(articles)} articles into chunks...")

    all_chunks = []

    for i, article in enumerate(articles):
        title  = (article.get("title") or "Untitled")[:50]
        chunks = chunk_article(article)
        all_chunks.extend(chunks)
        print(f"  [{i+1}/{len(articles)}] '{title}' → {len(chunks)} chunks")

    # Print summary by stance
    stance_counts = {}
    for chunk in all_chunks:
        s = chunk.get("stance", "Unknown")
        stance_counts[s] = stance_counts.get(s, 0) + 1

    print(f"\n[chunker] Total chunks: {len(all_chunks)}")
    print(f"  By stance: {stance_counts}")

    return all_chunks