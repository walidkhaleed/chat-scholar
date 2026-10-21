
import hashlib
import json
from pathlib import Path
from uuid import UUID

import faiss
import numpy as np
from embedding_backend import get_backend_model_id

from embedding_processor import (
    EMBEDDING_DIR,
    MODEL_NAME,
    embed_query,
)
from chunk_processor import load_chunks


BASE_DIR = Path(__file__).resolve().parent
FAISS_DIR = BASE_DIR / "faiss_indexes"
FAISS_DIR.mkdir(exist_ok=True)


def validate_id(document_id: str) -> str:
    """Validate document ID before accessing files."""
    return str(UUID(document_id))


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 fingerprint."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def get_paths(document_id: str) -> dict:
    document_id = validate_id(document_id)

    return {
        "vectors": EMBEDDING_DIR / f"{document_id}.npy",
        "metadata": EMBEDDING_DIR / f"{document_id}.json",
        "chunks": BASE_DIR / "chunks" / f"{document_id}.json",
        "index": FAISS_DIR / f"{document_id}.faiss",
        "index_metadata": FAISS_DIR / f"{document_id}.json",
    }


def load_source_data(document_id: str):
    """
    Load V4 vectors and metadata.
    Verify that they match the current V3 chunks.
    """
    paths = get_paths(document_id)

    if not paths["vectors"].exists():
        raise FileNotFoundError(
            "Embeddings not found. Run /generate-embeddings first."
        )

    if not paths["metadata"].exists():
        raise FileNotFoundError("Embedding metadata not found.")

    if not paths["chunks"].exists():
        raise FileNotFoundError("Chunks not found.")

    vectors = np.load(
        paths["vectors"],
        allow_pickle=False
    )

    with open(paths["metadata"], encoding="utf-8") as file:
        metadata = json.load(file)

    chunk_data = load_chunks(document_id)

    if chunk_data is None:
        raise FileNotFoundError("Chunks not found.")

    chunks = chunk_data["chunks"]

    if vectors.ndim != 2 or vectors.shape[1] != 384:
        raise ValueError("Expected 384-dimensional vectors.")

    if len(vectors) == 0:
        raise ValueError("No embeddings available.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain invalid values.")

    if metadata.get("model") != get_backend_model_id():
        raise ValueError(
            "Embedding backend mismatch. "
            "Regenerate embeddings with the active backend."
        )
    if metadata.get("total_embeddings") != len(vectors):
        raise ValueError("Embedding count mismatch.")

    if len(chunks) != len(vectors):
        raise ValueError(
            "Chunk count changed. Regenerate embeddings."
        )

    # Check whether chunk IDs and content still match.
    saved_chunks = metadata["chunks"]

    for current, saved in zip(chunks, saved_chunks):
        if (
            current["chunk_id"] != saved["chunk_id"]
            or current["text"] != saved["text"]
            or current["page_number"] != saved["page_number"]
        ):
            raise ValueError(
                "Chunks changed. Regenerate embeddings."
            )

    # Inner product equals cosine similarity for unit vectors.
    norms = np.linalg.norm(vectors, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            "Vectors are not normalized. Regenerate embeddings."
        )

    return (
        np.ascontiguousarray(vectors, dtype=np.float32),
        metadata,
        paths
    )


def build_faiss_index(document_id: str) -> dict:
    """
    Build and save an exact FAISS inner-product index.
    """
    document_id = validate_id(document_id)

    vectors, metadata, paths = load_source_data(document_id)

    dimension = vectors.shape[1]

    # Exact nearest-neighbor index
    index = faiss.IndexFlatIP(dimension)

    # Add all document embeddings
    index.add(vectors)

    # Save the FAISS index
    faiss.write_index(index, str(paths["index"]))

    # Save fingerprints for stale-index detection
    index_metadata = {
        "document_id": document_id,
        "filename": metadata["filename"],
        "model": metadata["model"],
        "index_type": "IndexFlatIP",
        "dimension": dimension,
        "total_vectors": index.ntotal,
        "normalized": True,
        "vectors_sha256": sha256_file(paths["vectors"]),
        "metadata_sha256": sha256_file(paths["metadata"]),
        "chunks_sha256": sha256_file(paths["chunks"]),
    }

    with open(
        paths["index_metadata"],
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(index_metadata, file, indent=2)

    return {
        "message": "FAISS index created successfully.",
        "document_id": document_id,
        "filename": metadata["filename"],
        "index_type": "IndexFlatIP",
        "dimension": dimension,
        "total_vectors": index.ntotal,
        "index_file": paths["index"].name,
    }


def load_faiss_index(document_id: str):
    """
    Load an index and reject outdated source files.

    Only load indexes created locally by this application.
    FAISS index files should not be accepted from users.
    """
    document_id = validate_id(document_id)
    paths = get_paths(document_id)

    if not paths["index"].exists():
        raise FileNotFoundError(
            "FAISS index not found. Run /build-faiss first."
        )

    if not paths["index_metadata"].exists():
        raise FileNotFoundError("FAISS metadata not found.")

    with open(
        paths["index_metadata"],
        encoding="utf-8"
    ) as file:
        info = json.load(file)

    # V12: Reject indexes created by a different
# embedding backend, even if dimensions match.
    if info.get("model") != get_backend_model_id():
        raise ValueError(
            "FAISS index embedding backend mismatch. "
            "Rebuild the index using the active backend."
        )
    fingerprints = {
        "vectors_sha256": "vectors",
        "metadata_sha256": "metadata",
        "chunks_sha256": "chunks",
    }

    for field, path_name in fingerprints.items():
        path = paths[path_name]

        if not path.exists():
            raise ValueError(
                "Source file missing. Rebuild the pipeline."
            )

        if sha256_file(path) != info[field]:
            raise ValueError(
                "Source data changed. Rebuild the FAISS index."
            )

    index = faiss.read_index(str(paths["index"]))

    if index.d != info["dimension"]:
        raise ValueError("FAISS dimension mismatch.")

    if index.ntotal != info["total_vectors"]:
        raise ValueError("FAISS vector count mismatch.")

    return index, info


def get_faiss_info(document_id: str) -> dict:
    """Return information about a saved FAISS index."""
    index, info = load_faiss_index(document_id)

    return {
        "document_id": info["document_id"],
        "filename": info["filename"],
        "model": info["model"],
        "index_type": info["index_type"],
        "dimension": index.d,
        "total_vectors": index.ntotal,
        "status": "ready",
    }


def semantic_search(
    document_id: str,
    question: str,
    top_k: int = 3
) -> dict:
    """
    Search a PDF using cosine similarity.

    Returns the most relevant chunks and page numbers.
    """
    document_id = validate_id(document_id)

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20.")

    # Verify source data and load saved index
    index, info = load_faiss_index(document_id)

    paths = get_paths(document_id)

    with open(paths["metadata"], encoding="utf-8") as file:
        metadata = json.load(file)

    # Convert question to a vector using the V4 model
    query_vector = embed_query(question)

    query_vector = np.ascontiguousarray(
        query_vector.reshape(1, -1),
        dtype=np.float32
    )

    if query_vector.shape[1] != index.d:
        raise ValueError("Query embedding dimension mismatch.")

    # Find top-k nearest neighbors
    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_vector, k)

    results = []

    for rank, (score, position) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):
        if position < 0:
            continue

        chunk = metadata["chunks"][int(position)]

        results.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "similarity_score": round(float(score), 4),
            "text": chunk["text"],
        })

    return {
        "document_id": document_id,
        "filename": info["filename"],
        "question": question,
        "model": info["model"],
        "total_indexed_chunks": index.ntotal,
        "results_count": len(results),
        "results": results,
    }
