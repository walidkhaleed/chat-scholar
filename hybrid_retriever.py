
import re
from uuid import UUID

import numpy as np
from rank_bm25 import BM25Okapi

from chunk_processor import load_chunks
from faiss_processor import (
    load_faiss_index,
    load_source_data
)
from embedding_processor import embed_query


RRF_K = 60


def tokenize(text: str) -> list[str]:
    """
    Lightweight tokenizer for English academic text.
    """
    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE
    )


def hybrid_search(
    document_id: str,
    question: str,
    top_k: int = 5,
    candidate_k: int = 15
) -> dict:
    """
    Combine FAISS and BM25 retrieval with RRF.

    Both methods search the same current V3 chunks.
    """

    document_id = str(UUID(document_id))

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    if not 1 <= candidate_k <= 50:
        raise ValueError(
            "candidate_k must be between 1 and 50."
        )

    candidate_k = max(candidate_k, top_k)

    # Load and validate the existing FAISS index.
    index, index_info = load_faiss_index(document_id)

    # Verify the vectors still match the source chunks.
    vectors, metadata, _ = load_source_data(document_id)

    chunk_data = load_chunks(document_id)

    if chunk_data is None:
        raise FileNotFoundError("Document chunks not found.")

    chunks = chunk_data["chunks"]

    if len(chunks) != index.ntotal:
        raise ValueError("FAISS and chunk count mismatch.")

    # -------------------------------------------
    # 1. Semantic retrieval using FAISS
    # -------------------------------------------

    query_vector = embed_query(question)

    query_vector = np.ascontiguousarray(
        query_vector.reshape(1, -1),
        dtype=np.float32
    )

    search_k = min(candidate_k, index.ntotal)

    semantic_scores, semantic_indices = index.search(
        query_vector,
        search_k
    )

    semantic_ranks = {}

    for rank, position in enumerate(
        semantic_indices[0],
        start=1
    ):
        if position >= 0:
            semantic_ranks[int(position)] = rank

    # -------------------------------------------
    # 2. Lexical retrieval using BM25
    # -------------------------------------------

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    if not any(tokenized_corpus):
        raise ValueError("Document has no searchable text.")

    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenize(question)

    if not query_tokens:
        raise ValueError("Question has no searchable terms.")

    lexical_scores = bm25.get_scores(query_tokens)

    lexical_indices = np.argsort(
        -lexical_scores,
        kind="stable"
    )[:search_k]

    # Only include BM25 candidates with positive
    # lexical relevance.
    lexical_ranks = {}

    for position in lexical_indices:
        position = int(position)

        if lexical_scores[position] > 0:
            lexical_ranks[position] = (
                len(lexical_ranks) + 1
            )

    # -------------------------------------------
    # 3. Reciprocal Rank Fusion
    # -------------------------------------------

    candidate_positions = (
        set(semantic_ranks) | set(lexical_ranks)
    )

    fused_results = []

    for position in candidate_positions:
        semantic_rank = semantic_ranks.get(position)
        lexical_rank = lexical_ranks.get(position)

        score = 0.0

        if semantic_rank is not None:
            score += 1.0 / (RRF_K + semantic_rank)

        if lexical_rank is not None:
            score += 1.0 / (RRF_K + lexical_rank)

        chunk = chunks[position]

        # Exact cosine score for each candidate.
        # The V4 vectors are normalized.
        cosine = float(
            np.dot(vectors[position], query_vector[0])
        )

        fused_results.append({
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "rrf_score": round(score, 6),
            "semantic_score": round(cosine, 4),
            "semantic_rank": semantic_rank,
            "lexical_rank": lexical_rank,
        })

    fused_results.sort(
        key=lambda item: (
            -item["rrf_score"],
            -item["semantic_score"],
            item["chunk_id"]
        )
    )

    results = fused_results[:top_k]

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return {
        "document_id": document_id,
        "filename": chunk_data["filename"],
        "question": question,
        "retrieval_method": "FAISS + BM25 + RRF",
        "total_chunks": len(chunks),
        "results_count": len(results),
        "results": results
    }
