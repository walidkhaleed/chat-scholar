
import json
import logging
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import numpy as np

from chunk_processor import load_chunks
from embedding_backend import (
    embed_question as backend_embed_question,
    embed_documents as backend_embed_documents,
    get_backend_name,
    get_backend_model_id,
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_DIR = BASE_DIR / "embeddings"

EMBEDDING_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Small batch size for 8 GB RAM
BATCH_SIZE = 8


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the embedding model once and reuse it.

    The model runs on CPU.
    """

    logger.info("Loading embedding model: %s", MODEL_NAME)
    @lru_cache(maxsize=1)
    def get_embedding_model():
        """
        Load the original local embedding model only when needed.
        """
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", MODEL_NAME)

        model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

        return model
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    return model


def generate_embeddings(document_id: str) -> dict:
    """
    Generate embeddings for all chunks of a document.

    Requires V3 chunking to have been completed.
    """

    # Validate document identifier
    document_id = str(UUID(document_id))

    # Load previously generated chunks
    chunk_data = load_chunks(document_id)

    if chunk_data is None:
        raise FileNotFoundError(
            "Chunks not found. Run /chunk-document first."
        )

    chunks = chunk_data["chunks"]

    if not chunks:
        raise ValueError(
            "No text chunks available for embedding."
        )

    texts = [chunk["text"] for chunk in chunks]

    # Load cached model

    # V12: Preserve the original local token-length
    # analysis, but avoid loading PyTorch on ONNX.

    # V12: Inspect token lengths with the selected backend.
    if get_backend_name() == "local":

        model = get_embedding_model()
        token_limit = model.max_seq_length

        truncated_count = 0

        for text in texts:
            token_count = len(
                model.tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False
                )["input_ids"]
            )

            if token_count > token_limit:
                truncated_count += 1

    else:
        # ONNX: count tokens before truncation.
        from render_embeddings import inspect_token_lengths

        token_info = inspect_token_lengths(texts)

        truncated_count = token_info["truncated_count"]

        logger.info(
            "ONNX embedding: %s chunks, %s exceed %s tokens",
            len(texts),
            truncated_count,
            token_info["token_limit"],
        )


    # Generate embeddings with the selected backend.
    vectors = backend_embed_documents(texts)


    # Store as float32 for FAISS compatibility
    vectors = np.asarray(vectors, dtype=np.float32)

    # Verify output shape
    if vectors.ndim != 2:
        raise RuntimeError("Invalid embedding dimensions.")

    if vectors.shape[0] != len(chunks):
        raise RuntimeError("Embedding count mismatch.")

    # Save vectors to a NumPy binary file
    vector_path = (
        EMBEDDING_DIR / f"{document_id}.npy"
    )

    np.save(vector_path, vectors)

    # Store metadata separately
    metadata = {
        "document_id": document_id,
        "filename": chunk_data["filename"],
        "model": get_backend_model_id(),
        "embedding_dimension": int(vectors.shape[1]),
        "total_embeddings": int(vectors.shape[0]),
        "normalized": True,
        "chunk_size": chunk_data["chunk_size"],
        "chunk_overlap": chunk_data["chunk_overlap"],
        "truncated_chunks": truncated_count,
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"]
            }
            for chunk in chunks
        ]
    }

    metadata_path = (
        EMBEDDING_DIR / f"{document_id}.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    return {
        "document_id": document_id,
        "filename": chunk_data["filename"],
        "model": get_backend_model_id(),
        "total_embeddings": int(vectors.shape[0]),
        "embedding_dimension": int(vectors.shape[1]),
        "truncated_chunks": truncated_count,
        "vector_file": vector_path.name,
        "metadata_file": metadata_path.name,
        "first_vector_preview": vectors[0][:8].tolist()
    }


def load_embeddings(document_id: str) -> dict | None:
    """
    Load stored embedding metadata and vector information.
    """

    document_id = str(UUID(document_id))

    vector_path = (
        EMBEDDING_DIR / f"{document_id}.npy"
    )

    metadata_path = (
        EMBEDDING_DIR / f"{document_id}.json"
    )

    if not vector_path.exists() or not metadata_path.exists():
        return None

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    vectors = np.load(
        vector_path,
        allow_pickle=False
    )

    if vectors.ndim != 2:
        raise ValueError("Stored embedding array is invalid.")

    if vectors.shape[0] != metadata["total_embeddings"]:
        raise ValueError("Stored embedding count mismatch.")

    if vectors.shape[1] != metadata["embedding_dimension"]:
        raise ValueError("Stored embedding dimension mismatch.")

    return {
        "document_id": document_id,
        "filename": metadata["filename"],
        "model": metadata["model"],
        "total_embeddings": metadata["total_embeddings"],
        "embedding_dimension": metadata["embedding_dimension"],
        "normalized": metadata["normalized"],
        "truncated_chunks": metadata["truncated_chunks"],
        "vector_shape": list(vectors.shape),
        "first_vector_preview": vectors[0][:8].tolist()
    }


def embed_query(question: str) -> np.ndarray:
    """
    Generate a normalized question embedding using
    the configured embedding backend.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    vector = backend_embed_question(question)

    vector = np.asarray(vector, dtype=np.float32)

    if vector.shape != (384,):
        raise RuntimeError(
            "Unexpected query embedding dimensions."
        )

    return vector



def check_embedding_lengths(document_id: str) -> dict:
    """
    Check PDF chunk token lengths using the active
    embedding backend without generating embeddings.

    Supports:
    - local: Sentence Transformers / PyTorch
    - onnx: FastEmbed / ONNX Runtime
    """
    document_id = str(UUID(document_id))

    chunk_data = load_chunks(document_id)

    if chunk_data is None:
        raise FileNotFoundError(
            "Chunks not found. Run /chunk-document first."
        )

    chunks = chunk_data["chunks"]

    if not chunks:
        raise ValueError(
            "Document has no extractable text chunks."
        )

    texts = [chunk["text"] for chunk in chunks]

    if get_backend_name() == "onnx":
        from render_embeddings import inspect_token_lengths

        token_info = inspect_token_lengths(texts)

        limit = token_info["token_limit"]
        token_counts = token_info["token_counts"]

    else:
        # Preserve original local MiniLM implementation.
        model = get_embedding_model()
        limit = model.max_seq_length

        token_counts = [
            len(
                model.tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False
                )["input_ids"]
            )
            for text in texts
        ]

    oversized = []

    for chunk, token_count in zip(chunks, token_counts):
        if token_count > limit:
            oversized.append({
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "token_count": token_count
            })

    return {
        "document_id": document_id,
        "total_chunks": len(chunks),
        "token_limit": limit,
        "oversized_chunks": oversized,
        "valid": len(oversized) == 0
    }
