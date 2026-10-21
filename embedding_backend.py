
"""
Chat Scholar V12 - Embedding backend selector.

local:
    Original Sentence Transformers + PyTorch.

onnx:
    Lightweight FastEmbed + ONNX Runtime.

The backend is selected through:
CHAT_SCHOLAR_EMBEDDING_BACKEND
"""

import os

import numpy as np


LOCAL_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def get_backend_name() -> str:
    """Read and validate the selected backend."""

    backend = os.getenv(
        "CHAT_SCHOLAR_EMBEDDING_BACKEND",
        "local"
    ).strip().lower()

    if backend not in {"local", "onnx"}:
        raise ValueError(
            "Invalid embedding backend. "
            "Expected 'local' or 'onnx'."
        )

    return backend


def get_backend_model_id() -> str:
    """Return the identifier stored with embeddings."""

    if get_backend_name() == "onnx":
        from render_embeddings import BACKEND_ID
        return BACKEND_ID

    # Preserve existing local embedding metadata.
    return LOCAL_MODEL_ID


def embed_documents(texts: list[str]) -> np.ndarray:
    """Generate normalized float32 document vectors."""

    if not texts or any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "Documents must contain non-empty text."
        )

    if get_backend_name() == "onnx":
        from render_embeddings import embed_passages
        return embed_passages(texts)

    # Import PyTorch-dependent code only when needed.
    from embedding_processor import get_embedding_model

    model = get_embedding_model()

    vectors = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(vectors, dtype=np.float32)


def embed_question(question: str) -> np.ndarray:
    """Generate one normalized float32 query vector."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if get_backend_name() == "onnx":
        from render_embeddings import embed_question as onnx_embed
        return onnx_embed(question)

    from embedding_processor import get_embedding_model

    model = get_embedding_model()

    vector = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(vector, dtype=np.float32)
