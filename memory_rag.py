
import asyncio
from uuid import UUID

from starlette.concurrency import run_in_threadpool

from pdf_memory import (
    get_session,
    get_messages,
    save_turn
)
from question_rewriter import rewrite_question
from advanced_rag import advanced_pdf_answer


MAX_HISTORY_TURNS = 5

# In-process locks serialize requests for each session.
# Use a single Uvicorn worker for this local version.
_session_locks = {}


def get_session_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()

    return _session_locks[session_id]


async def chat_with_pdf_memory(
    session_id: str,
    question: str,
    top_k: int = 5
) -> dict:
    """
    Answer PDF questions with persistent conversation
    memory and V7 document-grounded retrieval.
    """

    session_id = str(UUID(session_id))
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    lock = get_session_lock(session_id)

    async with lock:

        # 1. Load session
        session = await run_in_threadpool(
            get_session,
            session_id
        )

        if session is None:
            raise FileNotFoundError(
                "PDF session not found."
            )

        document_id = session["document_id"]

        # 2. Load recent conversation history
        history = await run_in_threadpool(
            get_messages,
            session_id,
            MAX_HISTORY_TURNS
        )

        # 3. Rewrite follow-up question
        
        # Only checked answers are useful for resolving
        # references to previously discussed findings.
        rewrite_history = [
            item
            for item in history
            if item["status"] == "checks_passed"
        ]

        standalone_question = await rewrite_question(
            question,
            rewrite_history
        )


        # 4. Use V7 advanced RAG
        rag_result = await advanced_pdf_answer(
            document_id=document_id,
            question=standalone_question,
            top_k=top_k
        )

        status = rag_result["status"]

        # 5. Store the complete result.
        # Save abstentions and verification failures too,
        # so the visible conversation can be resumed.
        message_id = await run_in_threadpool(
            save_turn,
            session_id,
            question,
            standalone_question,
            rag_result["answer"],
            status,
            rag_result["sources"],
            rag_result["verification"]
        )

        # 6. Return answer and evidence
        return {
            "session_id": session_id,
            "document_id": document_id,
            "message_id": message_id,
            "question": question,
            "standalone_question": standalone_question,
            "answer": rag_result["answer"],
            "status": status,
            "model": rag_result["model"],
            "history_turns_used": len(history),
            "sources": rag_result["sources"],
            "verification": rag_result["verification"]
        }
