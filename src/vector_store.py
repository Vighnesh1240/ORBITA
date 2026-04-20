# src/vector_store.py

import os
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress all warnings from pydantic to avoid ArbitraryTypeWarning
warnings.filterwarnings("ignore", module="pydantic.*")

import chromadb
from chromadb.config import Settings

try:
    from .config import (
        CHROMA_DB_DIR,
        CHROMA_COLLECTION,
        TOP_K_RETRIEVAL,
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
        GEMINI_API_KEY,
    )
except ImportError:
    from config import (
        CHROMA_DB_DIR,
        CHROMA_COLLECTION,
        TOP_K_RETRIEVAL,
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
        GEMINI_API_KEY,
    )

import google.genai as genai


# Create client for embeddings
client = genai.Client(api_key=GEMINI_API_KEY)


def _normalize_embedding_dim(vector: list[float], target_dim: int = EMBEDDING_DIM) -> list[float] | None:
    """Pad/trim one embedding vector to a consistent size for Chroma."""
    if vector is None:
        return None

    try:
        normalized = [float(v) for v in vector]
    except Exception:
        return None

    if len(normalized) < target_dim:
        normalized.extend([0.0] * (target_dim - len(normalized)))
    elif len(normalized) > target_dim:
        normalized = normalized[:target_dim]

    return normalized


def _get_client() -> chromadb.PersistentClient:
    """
    Get a persistent ChromaDB client.
    Creates the chroma_db/ directory if it does not exist.
    """
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)


def _get_collection(client: chromadb.PersistentClient,
                    reset: bool = False) -> chromadb.Collection:
    """
    Get or create the ORBITA ChromaDB collection.

    Args:
        client: ChromaDB persistent client
        reset:  if True, delete and recreate the collection
                (use this when rerunning the pipeline on a new topic)
    """
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            print(f"  [vector_store] Existing collection deleted (reset=True)")
        except Exception:
            pass  # Collection didn't exist — that's fine

    collection = client.get_or_create_collection(
        name     = CHROMA_COLLECTION,
        metadata = {"hnsw:space": "cosine"},  # use cosine similarity
    )

    return collection


def store_chunks(chunks: list[dict], reset: bool = True) -> chromadb.Collection:
    """
    Store all embedded chunks into ChromaDB.

    Args:
        chunks: list of chunk dicts with 'embedding' field
        reset:  whether to wipe and recreate the collection first

    Returns:
        the ChromaDB collection (for immediate use or inspection)
    """
    print(f"\n[vector_store] Storing {len(chunks)} chunks in ChromaDB...")
    print(f"  Path: {CHROMA_DB_DIR}")

    client     = _get_client()
    collection = _get_collection(client, reset=reset)

    # ChromaDB requires:
    # - ids:        list of unique strings
    # - embeddings: list of vectors (list of floats)
    # - documents:  list of text strings
    # - metadatas:  list of dicts (must contain only str/int/float/bool values)

    ids         = []
    embeddings  = []
    documents   = []
    metadatas   = []

    skipped_invalid = 0

    for chunk in chunks:
        if "embedding" not in chunk:
            continue  # skip chunks that failed embedding

        normalized_embedding = _normalize_embedding_dim(chunk.get("embedding"))
        if normalized_embedding is None:
            skipped_invalid += 1
            continue

        ids.append(chunk["chunk_id"])
        embeddings.append(normalized_embedding)
        documents.append(chunk["text"])
        metadatas.append({
            "url":         chunk.get("url", ""),
            "title":       chunk.get("title", "")[:200],  # ChromaDB has metadata size limits
            "source":      chunk.get("source", ""),
            "stance":      chunk.get("stance", "Neutral"),
            "chunk_index": chunk.get("chunk_index", 0),
        })

    if not ids:
        raise RuntimeError("No valid chunks to store — all failed embedding.")

    if skipped_invalid:
        print(f"  [vector_store] Skipped {skipped_invalid} chunk(s) with invalid embeddings")

    # ChromaDB can handle large upserts but we batch for safety
    BATCH_SIZE = 50
    total_stored = 0

    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids        = ids[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]
        batch_documents  = documents[i:i + BATCH_SIZE]
        batch_metadatas  = metadatas[i:i + BATCH_SIZE]

        collection.upsert(
            ids        = batch_ids,
            embeddings = batch_embeddings,
            documents  = batch_documents,
            metadatas  = batch_metadatas,
        )
        total_stored += len(batch_ids)

    print(f"  Stored {total_stored} chunks successfully.")
    print(f"  Collection total: {collection.count()} chunks")

    # Print breakdown by stance
    print(f"\n  Breakdown by stance:")
    for stance in ["Supportive", "Critical", "Neutral"]:
        results = collection.get(where={"stance": stance})
        count   = len(results["ids"])
        print(f"    {stance}: {count} chunks")

    return collection


def _fallback_query_embedding(query_text: str) -> list[float]:
    """Create a repeatable local embedding fallback for a query."""
    vect = TfidfVectorizer(stop_words="english", max_features=min(EMBEDDING_DIM, 1024))
    try:
        tfidf = vect.fit_transform([query_text])
    except ValueError:
        return [0.0] * EMBEDDING_DIM

    arr = tfidf.toarray().astype(float)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    if norm[0] == 0:
        return [0.0] * EMBEDDING_DIM

    row = (arr / norm).tolist()[0]

    if len(row) < EMBEDDING_DIM:
        row.extend([0.0] * (EMBEDDING_DIM - len(row)))
    else:
        row = row[:EMBEDDING_DIM]

    return row

def embed_query(query_text: str) -> list[float]:
    """
    Embed a query string using the same Gemini model.
    Use task_type RETRIEVAL_QUERY (different from RETRIEVAL_DOCUMENT).
    """
    if not query_text:
        return [0.0] * 512

    try:
        result = client.models.embed_content(
            model    = EMBEDDING_MODEL,
            contents = [query_text],
            config   = {"task_type": "RETRIEVAL_QUERY"},
        )

        normalized = _normalize_embedding_dim(result.embeddings[0].values)
        if normalized is None:
            return _fallback_query_embedding(query_text)
        return normalized

    except Exception as e:
        print(f"[vector_store] Warning: embedding API error: {e}")
        return _fallback_query_embedding(query_text)


def retrieve_chunks(
    query: str,
    n_results: int = TOP_K_RETRIEVAL,
    stance_filter: str = None,
) -> list[dict]:
    """
    Semantic search against the ChromaDB collection.
    Returns the most relevant chunks for the given query.

    Args:
        query:        natural language query string
        n_results:    how many chunks to return
        stance_filter: if set (e.g. "Supportive"), only return chunks
                       with that stance label. If None, return all stances.

    Returns:
        list of dicts, each with 'text', 'url', 'title',
        'source', 'stance', 'distance' fields
    """
    client     = _get_client()
    collection = _get_collection(client, reset=False)

    if collection.count() == 0:
        raise RuntimeError(
            "ChromaDB collection is empty. Run the pipeline first."
        )

    # Embed the query
    query_embedding = embed_query(query)

    # Build optional where filter
    where = None
    if stance_filter and stance_filter in ["Supportive", "Critical", "Neutral"]:
        where = {"stance": stance_filter}

    # Query ChromaDB
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = min(n_results, collection.count()),
        where            = where,
        include          = ["documents", "metadatas", "distances"],
    )

    # Format output
    output = []
    docs       = results["documents"][0]
    metas      = results["metadatas"][0]
    distances  = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        output.append({
            "text":     doc,
            "url":      meta.get("url", ""),
            "title":    meta.get("title", ""),
            "source":   meta.get("source", ""),
            "stance":   meta.get("stance", ""),
            "distance": round(dist, 4),
        })

    return output


def get_collection_stats() -> dict:
    """
    Return basic stats about the stored collection.
    Useful for verifying Step 3 completed correctly.
    """
    client     = _get_client()
    collection = _get_collection(client, reset=False)
    total      = collection.count()

    stance_counts = {}
    for stance in ["Supportive", "Critical", "Neutral"]:
        results = collection.get(where={"stance": stance})
        stance_counts[stance] = len(results["ids"])

    return {
        "total_chunks": total,
        "by_stance":    stance_counts,
        "db_path":      CHROMA_DB_DIR,
    }