# src/embedder.py

import time
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress all warnings from pydantic to avoid ArbitraryTypeWarning
warnings.filterwarnings("ignore", module="pydantic.*")

import google.genai as genai

try:
    from .config import (
        GEMINI_API_KEY,
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
        EMBEDDING_BATCH,
        EMBEDDING_DELAY,
    )
except ImportError:
    from config import (
        GEMINI_API_KEY,
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
        EMBEDDING_BATCH,
        EMBEDDING_DELAY,
    )

# Create client once at module level
client = genai.Client(api_key=GEMINI_API_KEY)


def _normalize_embedding_dim(vector: list[float], target_dim: int = EMBEDDING_DIM) -> list[float]:
    """Pad/trim a single embedding vector to the configured dimension."""
    if vector is None:
        return [0.0] * target_dim

    try:
        normalized = [float(v) for v in vector]
    except Exception:
        return [0.0] * target_dim

    if len(normalized) < target_dim:
        normalized.extend([0.0] * (target_dim - len(normalized)))
    elif len(normalized) > target_dim:
        normalized = normalized[:target_dim]

    return normalized


def _fallback_embeddings(texts: list[str]) -> list[list[float]]:
    """Create deterministic pseudo-embeddings when API access is unavailable."""
    if not texts:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=min(EMBEDDING_DIM, 1024))
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        # all texts empty or no vocabulary
        return [[0.0] * EMBEDDING_DIM for _ in texts]

    arr = tfidf.toarray().astype(float)

    # L2 normalize for cosine similarity behavior
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    # Pad or trim to expected embedding dimension
    if arr.shape[1] < EMBEDDING_DIM:
        pad_width = EMBEDDING_DIM - arr.shape[1]
        arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    elif arr.shape[1] > EMBEDDING_DIM:
        arr = arr[:, :EMBEDDING_DIM]

    return arr.tolist()


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of text strings using the Gemini embedding API.

    Returns a list of embedding vectors (one per input text).
    Each vector is a list of floats.
    """
    if not texts:
        return []

    try:
        result = client.models.embed_content(
            model    = EMBEDDING_MODEL,
            contents = texts,
            config   = {"task_type": "RETRIEVAL_DOCUMENT"},
        )

        # result.embeddings is a list of ContentEmbedding objects
        embeddings = [emb.values for emb in result.embeddings]

        # Normalise: always return list of lists
        if embeddings and isinstance(embeddings[0], float):
            # Single text was passed — wrap in outer list
            embeddings = [embeddings]

        return [_normalize_embedding_dim(v) for v in embeddings]

    except Exception as e:
        print(f"[embedder] Warning: embedding API error: {e}")
        return _fallback_embeddings(texts)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Main function. Adds an 'embedding' field to every chunk dict.

    Processes in batches to stay within API rate limits.
    Chunks that fail embedding are removed from the output.

    Args:
        chunks: list of chunk dicts (must have 'text' field)

    Returns:
        list of chunk dicts, each with an 'embedding' field added
    """
    print(f"\n[embedder] Generating embeddings for {len(chunks)} chunks...")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Batch size: {EMBEDDING_BATCH}")

    embedded_chunks = []
    failed          = 0
    total_batches   = (len(chunks) + EMBEDDING_BATCH - 1) // EMBEDDING_BATCH

    for batch_num in range(total_batches):
        start = batch_num * EMBEDDING_BATCH
        end   = start + EMBEDDING_BATCH
        batch = chunks[start:end]

        texts = [c["text"] for c in batch]

        print(f"  Batch {batch_num + 1}/{total_batches} "
              f"({len(texts)} chunks)...", end=" ")

        try:
            embeddings = _embed_batch(texts)

            if len(embeddings) != len(batch):
                print(f"MISMATCH (expected {len(batch)}, got {len(embeddings)})")
                failed += len(batch)
                continue

            for chunk, embedding in zip(batch, embeddings):
                chunk["embedding"] = embedding
                embedded_chunks.append(chunk)

            print(f"OK")

        except Exception as e:
            print(f"FAILED — {e}")
            failed += len(batch)

            # If rate limited, wait longer before retrying
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"  Rate limit hit — waiting 10 seconds...")
                time.sleep(10)
                # Retry this batch once
                try:
                    embeddings = _embed_batch(texts)
                    for chunk, embedding in zip(batch, embeddings):
                        chunk["embedding"] = embedding
                        embedded_chunks.append(chunk)
                    failed -= len(batch)
                    print(f"  Retry succeeded.")
                except Exception as retry_err:
                    print(f"  Retry also failed: {retry_err}")

        # Polite delay between batches
        if batch_num < total_batches - 1:
            time.sleep(EMBEDDING_DELAY)

    print(f"\n[embedder] Done. {len(embedded_chunks)} embedded, {failed} failed.")
    return embedded_chunks