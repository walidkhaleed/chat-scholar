
"""
Chat Scholar V12 - Lightweight ONNX embeddings.

Uses FastEmbed instead of PyTorch.
Designed for a memory-constrained portfolio demo.
"""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# A distinct identifier prevents confusion with
# existing PyTorch-generated embedding indexes.
BACKEND_ID = "fastembed-onnx-minilm-l6-v2-v1"

EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def get_model():
    """Load the ONNX embedding model only once."""

    from fastembed import TextEmbedding

    cache_dir = Path(
        os.getenv(
            "CHAT_SCHOLAR_MODEL_CACHE",
            str(
                Path.home()
                / ".cache"
                / "chat-scholar-v12"
            )
        )
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(cache_dir),
        threads=1,
    )


def normalize_vectors(vectors):
    """Return normalized float32 vectors."""

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    if vectors.ndim not in (1, 2):
        raise ValueError(
            "Expected a one- or two-dimensional vector array."
        )

    norms = np.linalg.norm(
        vectors,
        axis=-1,
        keepdims=True
    )

    return (
        vectors / np.maximum(norms, 1e-12)
    ).astype(np.float32)


def embed_passages(texts: list[str]) -> np.ndarray:
    """
    Embed document chunks.

    Returns shape:
        (number_of_chunks, 384)
    """

    if not texts or any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "Document chunks must contain non-empty text."
        )

    model = get_model()

    vectors = np.stack(
        list(model.passage_embed(texts))
    )

    vectors = normalize_vectors(vectors)

    if vectors.shape != (
        len(texts),
        EMBEDDING_DIMENSION
    ):
        raise RuntimeError(
            "Unexpected embedding dimensions."
        )

    return vectors


def embed_question(question: str) -> np.ndarray:
    """
    Embed a user question.

    Returns shape:
        (384,)
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    model = get_model()

    vector = next(
        model.query_embed(question)
    )

    vector = normalize_vectors(vector)

    if vector.shape != (EMBEDDING_DIMENSION,):
        raise RuntimeError(
            "Unexpected query dimensions."
        )

    return vector

@lru_cache(maxsize=1)
def get_untruncated_tokenizer():
    """
    Create a tokenizer copy with truncation disabled.

    Returns the tokenizer and the effective model
    input limit. The actual embedding tokenizer
    remains unchanged.
    """
    from tokenizers import Tokenizer

    model = get_model()
    embedding_tokenizer = model.model.tokenizer

    truncation = embedding_tokenizer.truncation

    if not truncation or not truncation.get("max_length"):
        raise RuntimeError(
            "ONNX tokenizer has no configured token limit."
        )

    token_limit = int(truncation["max_length"])

    # Clone the tokenizer without modifying the one
    # that FastEmbed uses for actual inference.
    counter = Tokenizer.from_str(
        embedding_tokenizer.to_str()
    )

    counter.no_truncation()

    return counter, token_limit


def inspect_token_lengths(texts: list[str]) -> dict:
    """
    Count input tokens before ONNX truncation.

    Returns the effective token limit, individual
    token counts, and the number of oversized texts.
    """
    if not texts or any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "Texts must contain non-empty strings."
        )

    tokenizer, token_limit = get_untruncated_tokenizer()

    token_counts = [
        len(
            tokenizer.encode(
                text,
                add_special_tokens=True
            ).ids
        )
        for text in texts
    ]

    return {
        "token_limit": token_limit,
        "token_counts": token_counts,
        "truncated_count": sum(
            count > token_limit
            for count in token_counts
        ),
    }
