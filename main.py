
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File
)


from pydantic import BaseModel, Field

from groq import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    RateLimitError
)

from groq_client import ask_groq, MODEL

from memory import (
    get_history,
    save_message,
    clear_history
)

from pdf_processor import (
    UPLOAD_DIR,
    extract_pdf_text,
    save_document,
    load_document,
    list_documents
)


from uuid import UUID

from chunk_processor import (
    create_chunks,
    load_chunks
)


from embedding_processor import (
    generate_embeddings,
    load_embeddings
)


from starlette.concurrency import run_in_threadpool

from faiss_processor import (
    build_faiss_index,
    semantic_search,
    get_faiss_info,
)


from rag_processor import (
    prepare_document,
    answer_pdf_question,
    get_rag_status
)


from hybrid_retriever import hybrid_search
from advanced_rag import advanced_pdf_answer


from pdf_memory import (
    create_session,
    get_session,
    list_sessions,
    get_messages,
    delete_session
)

from memory_rag import (
    chat_with_pdf_memory,
    get_session_lock
)

from rag_processor import get_rag_status


from fastapi import Form
from pydantic import ValidationError

from grading_models import (
    RubricCreate,
    EssayTextRequest,
    GradeRequest
)

from grading_store import (
    save_rubric,
    get_rubric,
    list_rubrics,
    save_essay,
    get_essay,
    get_grade,
    extract_essay_text
)

from essay_grader import grade_essay

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from auth_routes import router as auth_router
from auth_guard import authentication_guard
from csrf_guard import csrf_guard
# --------------------------------------------------
# Configuration
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB








app = FastAPI(
    title="Chat Scholar API",
    description="Academic AI Assistant with RAG and Essay Grading",
    version="9.0.0"
    
)

app.include_router(auth_router)
app.middleware("http")(authentication_guard)
app.middleware("http")(csrf_guard)

# --------------------------------------------------
# Pydantic models
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    model: str


# --------------------------------------------------
# Existing V1 endpoints
# --------------------------------------------------

@app.get("/")
async def home():
    return {
        "application": "Chat Scholar",
        "version": "V2",
        "status": "running",
        "features": [
            "AI Chatbot",
            "Conversation Memory",
            "PDF Upload",
            "PDF Text Extraction",
            "Document Storage"
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    session_id = request.session_id or str(uuid4())

    history = get_history(session_id)

    try:
        answer = await ask_groq(
            question=request.question,
            history=history
        )

        save_message(
            session_id,
            "user",
            request.question
        )

        save_message(
            session_id,
            "assistant",
            answer
        )

        return ChatResponse(
            session_id=session_id,
            question=request.question,
            answer=answer,
            model=MODEL
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Groq rate limit reached."
        )

    except (APIConnectionError, APITimeoutError):
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Groq."
        )

    except APIStatusError as e:
        logger.error(
            "Groq API error %s: %s",
            e.status_code,
            e.response.text
        )

        raise HTTPException(
            status_code=502,
            detail=f"Groq API returned HTTP {e.status_code}."
        )

    except Exception:
        logger.exception("Unexpected chat error")

        raise HTTPException(
            status_code=500,
            detail="Internal server error."
        )


@app.get("/history/{session_id}")
async def history(session_id: str):
    return {
        "session_id": session_id,
        "messages": get_history(session_id)
    }


@app.delete("/history/{session_id}")
async def delete_history(session_id: str):

    clear_history(session_id)

    return {
        "session_id": session_id,
        "message": "Conversation history cleared."
    }


# --------------------------------------------------
# New V2 endpoints
# --------------------------------------------------

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required."
        )

    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # Generate a unique document ID
    document_id = str(uuid4())

    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"

    try:
        # Read up to the maximum size plus one byte
        content = await file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the 20 MB limit."
            )

        # Check the PDF file signature
        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file."
            )

        # Save the original PDF
        pdf_path.write_bytes(content)

        # Extract text
        extracted = extract_pdf_text(pdf_path)

        # Prepare document metadata
        document_data = {
            "document_id": document_id,
            "filename": Path(file.filename).name,
            "page_count": extracted["page_count"],
            "character_count": len(extracted["full_text"]),
            "pages": extracted["pages"],
            "full_text": extracted["full_text"]
        }

        # Save extracted content
        save_document(document_id, document_data)

        return {
            "message": "PDF uploaded and processed successfully.",
            "document_id": document_id,
            "filename": document_data["filename"],
            "page_count": document_data["page_count"],
            "character_count": document_data["character_count"],
            "text_preview": extracted["full_text"][:500]
        }

    except HTTPException:
        pdf_path.unlink(missing_ok=True)
        raise

    except (ValueError, pymupdf.FileDataError) as e:
        pdf_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        pdf_path.unlink(missing_ok=True)
        logger.exception("PDF processing failed")

        raise HTTPException(
            status_code=500,
            detail="PDF processing failed."
        )

    finally:
        await file.close()


@app.get("/documents")
async def get_documents():
    documents = list_documents()

    return {
        "total_documents": len(documents),
        "documents": documents
    }


@app.get("/documents/{document_id}")
async def get_document(document_id: str):

    # Validate the ID to avoid unsafe file paths
    try:
        document_id = str(
            __import__("uuid").UUID(document_id)
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID."
        )

    document = load_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return document

# --------------------------------------------------
# V3: Text Chunking Endpoints
# --------------------------------------------------

class ChunkRequest(BaseModel):
    document_id: str
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=5000)


def validate_document_id(document_id: str) -> str:
    """
    Validate and normalize a document UUID.
    """
    try:
        return str(UUID(document_id))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID."
        )


@app.post("/chunk-document")
async def chunk_document(request: ChunkRequest):

    document_id = validate_document_id(request.document_id)

    try:
        result = create_chunks(
            document_id=document_id,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        return {
            "message": "Document chunked successfully.",
            "document_id": result["document_id"],
            "filename": result["filename"],
            "chunk_size": result["chunk_size"],
            "chunk_overlap": result["chunk_overlap"],
            "total_chunks": result["total_chunks"],
            "preview": result["chunks"][:3]
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        logger.exception("Text chunking failed")

        raise HTTPException(
            status_code=500,
            detail="Text chunking failed."
        )


@app.get("/chunks/{document_id}")
async def get_document_chunks(document_id: str):

    document_id = validate_document_id(document_id)

    result = load_chunks(document_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Chunks not found. Process the document first."
        )

    return result


# --------------------------------------------------
# V4: Local Embedding Endpoints
# --------------------------------------------------

class EmbeddingRequest(BaseModel):
    document_id: str


@app.post("/generate-embeddings")
async def create_document_embeddings(
    request: EmbeddingRequest
):
    """
    Generate and store local embeddings for a document.
    """

    document_id = validate_document_id(
        request.document_id
    )

    try:
        # Run CPU-heavy embedding work in a thread
        from starlette.concurrency import run_in_threadpool

        result = await run_in_threadpool(
            generate_embeddings,
            document_id
        )

        return {
            "message": "Embeddings generated successfully.",
            **result
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "Document chunks not found. "
                "Run /chunk-document first."
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        logger.exception("Embedding generation failed")

        raise HTTPException(
            status_code=500,
            detail="Embedding generation failed."
        )


@app.get("/embeddings/{document_id}")
async def get_document_embeddings(document_id: str):
    """
    Return stored embedding metadata and a vector preview.
    """

    document_id = validate_document_id(document_id)

    try:
        result = load_embeddings(document_id)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Embeddings not found. "
                    "Run /generate-embeddings first."
                )
            )

        return result

    except HTTPException:
        raise

    except Exception:
        logger.exception("Failed to load embeddings")

        raise HTTPException(
            status_code=500,
            detail="Failed to load embeddings."
        )


# --------------------------------------------------
# V5: FAISS Vector Search
# --------------------------------------------------

class FAISSBuildRequest(BaseModel):
    document_id: str


class SemanticSearchRequest(BaseModel):
    document_id: str
    question: str = Field(
        min_length=1,
        max_length=5000
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20
    )


@app.post("/build-faiss")
async def create_faiss_index(request: FAISSBuildRequest):
    """
    Build a FAISS index using existing V4 embeddings.
    """
    document_id = validate_document_id(
        request.document_id
    )

    try:
        result = await run_in_threadpool(
            build_faiss_index,
            document_id
        )

        return result

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        logger.exception("FAISS indexing failed")

        raise HTTPException(
            status_code=500,
            detail="FAISS indexing failed."
        )


@app.post("/semantic-search")
async def search_document(request: SemanticSearchRequest):
    """
    Retrieve semantically relevant PDF chunks.
    """
    document_id = validate_document_id(
        request.document_id
    )

    try:
        result = await run_in_threadpool(
            semantic_search,
            document_id,
            request.question,
            request.top_k
        )

        return result

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        logger.exception("Semantic search failed")

        raise HTTPException(
            status_code=500,
            detail="Semantic search failed."
        )


@app.get("/faiss/{document_id}")
async def inspect_faiss_index(document_id: str):
    """
    Inspect the stored FAISS index.
    """
    document_id = validate_document_id(document_id)

    try:
        return await run_in_threadpool(
            get_faiss_info,
            document_id
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        logger.exception("Failed to inspect FAISS index")

        raise HTTPException(
            status_code=500,
            detail="Failed to inspect FAISS index."
        )

# --------------------------------------------------
# V6: Complete RAG Pipeline
# --------------------------------------------------

class PrepareDocumentRequest(BaseModel):
    document_id: str
    chunk_size: int = Field(
        default=700,
        ge=100,
        le=10000
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=5000
    )


class PDFChatRequest(BaseModel):
    document_id: str
    question: str = Field(
        min_length=1,
        max_length=5000
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10
    )


@app.post("/prepare-document")
async def prepare_pdf_for_rag(
    request: PrepareDocumentRequest
):
    """
    Run chunking, embedding generation,
    and FAISS indexing for an uploaded PDF.
    """

    document_id = validate_document_id(
        request.document_id
    )

    try:
        result = await run_in_threadpool(
            prepare_document,
            document_id,
            request.chunk_size,
            request.chunk_overlap
        )

        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception:
        logger.exception("RAG document preparation failed")

        raise HTTPException(
            status_code=500,
            detail="RAG document preparation failed."
        )


@app.post("/pdf-chat")
async def chat_with_pdf(request: PDFChatRequest):
    """
    Answer a question using retrieved PDF evidence.
    """

    document_id = validate_document_id(
        request.document_id
    )

    try:
        result = await answer_pdf_question(
            document_id=document_id,
            question=request.question,
            top_k=request.top_k
        )

        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Groq rate limit reached. Try again later."
        )

    except (APIConnectionError, APITimeoutError):
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Groq."
        )

    except APIStatusError as exc:
        logger.error(
            "Groq API error %s: %s",
            exc.status_code,
            exc.response.text
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"Groq API returned HTTP {exc.status_code}."
            )
        )

    except Exception:
        logger.exception("PDF question answering failed")

        raise HTTPException(
            status_code=500,
            detail="PDF question answering failed."
        )


@app.get("/rag-status/{document_id}")
async def rag_document_status(document_id: str):
    """
    Check whether a document has a valid
    FAISS index and is ready for PDF Q&A.
    """

    document_id = validate_document_id(document_id)

    try:
        return await run_in_threadpool(
            get_rag_status,
            document_id
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except Exception:
        logger.exception("RAG status check failed")

        raise HTTPException(
            status_code=500,
            detail="RAG status check failed."
        )

# --------------------------------------------------
# V7: Advanced RAG and Citation Verification
# --------------------------------------------------

class HybridSearchRequest(BaseModel):
    document_id: str
    question: str = Field(
        min_length=1,
        max_length=5000
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10
    )


class AdvancedPDFChatRequest(BaseModel):
    document_id: str
    question: str = Field(
        min_length=1,
        max_length=5000
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10
    )


@app.post("/hybrid-search")
async def search_pdf_hybrid(
    request: HybridSearchRequest
):
    """
    Search PDF chunks using FAISS + BM25 + RRF.
    """

    document_id = validate_document_id(
        request.document_id
    )

    try:
        return await run_in_threadpool(
            hybrid_search,
            document_id,
            request.question,
            request.top_k
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception:
        logger.exception("Hybrid search failed")

        raise HTTPException(
            status_code=500,
            detail="Hybrid search failed."
        )


@app.post("/advanced-pdf-chat")
async def chat_with_pdf_advanced(
    request: AdvancedPDFChatRequest
):
    """
    Advanced RAG with validated source references.
    """

    document_id = validate_document_id(
        request.document_id
    )

    try:
        return await advanced_pdf_answer(
            document_id=document_id,
            question=request.question,
            top_k=request.top_k
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Groq rate limit reached."
        )

    except (APIConnectionError, APITimeoutError):
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Groq."
        )

    except APIStatusError as exc:
        logger.error(
            "Groq API error %s: %s",
            exc.status_code,
            exc.response.text
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"Groq API returned HTTP {exc.status_code}."
            )
        )

    except Exception:
        logger.exception("Advanced RAG failed")

        raise HTTPException(
            status_code=500,
            detail="Advanced RAG failed."
        )

# --------------------------------------------------
# V8: PDF Conversation Memory
# --------------------------------------------------

class CreatePDFSessionRequest(BaseModel):
    document_id: str
    title: str = Field(
        default="PDF Conversation",
        max_length=200
    )


class PDFMemoryChatRequest(BaseModel):
    session_id: str
    question: str = Field(
        min_length=1,
        max_length=5000
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10
    )


@app.post("/pdf-sessions")
async def create_pdf_session(
    request: CreatePDFSessionRequest
):
    """
    Create a persistent conversation session
    associated with one prepared PDF.
    """

    document_id = validate_document_id(
        request.document_id
    )

    try:
        status = await run_in_threadpool(
            get_rag_status,
            document_id
        )

        if not status["ready"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Document is not ready for RAG. "
                    "Run /prepare-document first."
                )
            )

        return await run_in_threadpool(
            create_session,
            document_id,
            request.title
        )

    except HTTPException:
        raise

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception:
        logger.exception("PDF session creation failed")
        raise HTTPException(
            status_code=500,
            detail="PDF session creation failed."
        )


@app.get("/pdf-sessions")
async def get_pdf_sessions():
    """
    List persistent PDF conversation sessions.
    """

    sessions = await run_in_threadpool(
        list_sessions
    )

    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }


@app.get("/pdf-sessions/{session_id}")
async def get_pdf_session_details(session_id: str):
    """
    Retrieve session information and conversation history.
    """

    session_id = validate_document_id(session_id)

    session = await run_in_threadpool(
        get_session,
        session_id
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="PDF session not found."
        )

    messages = await run_in_threadpool(
        get_messages,
        session_id
    )

    return {
        **session,
        "total_turns": len(messages),
        "messages": messages
    }


@app.delete("/pdf-sessions/{session_id}")
async def remove_pdf_session(session_id: str):
    """
    Delete a session and its stored conversation history.
    """

    session_id = validate_document_id(session_id)

    # Wait for any in-progress chat in this session.
    lock = get_session_lock(session_id)

    async with lock:
        deleted = await run_in_threadpool(
            delete_session,
            session_id
        )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="PDF session not found."
        )

    return {
        "session_id": session_id,
        "message": "PDF session and history deleted."
    }


@app.post("/pdf-chat-memory")
async def pdf_chat_with_memory(
    request: PDFMemoryChatRequest
):
    """
    PDF question answering with persistent
    conversation memory and V7 evidence checks.
    """

    session_id = validate_document_id(
        request.session_id
    )

    try:
        return await chat_with_pdf_memory(
            session_id=session_id,
            question=request.question,
            top_k=request.top_k
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Groq rate limit reached."
        )

    except (APIConnectionError, APITimeoutError):
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Groq."
        )

    except APIStatusError as exc:
        logger.error(
            "Groq API error %s: %s",
            exc.status_code,
            exc.response.text
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"Groq API returned HTTP {exc.status_code}."
            )
        )

    except Exception:
        logger.exception("PDF memory chat failed")
        raise HTTPException(
            status_code=500,
            detail="PDF memory chat failed."
        )


# --------------------------------------------------
# V9: Academic Essay Grading
# --------------------------------------------------

MAX_ESSAY_UPLOAD_BYTES = 5 * 1024 * 1024


@app.post("/rubrics")
async def create_grading_rubric(
    rubric: RubricCreate
):
    return await run_in_threadpool(
        save_rubric,
        rubric
    )


@app.get("/rubrics")
async def get_all_rubrics():
    rubrics = await run_in_threadpool(
        list_rubrics
    )

    return {
        "total_rubrics": len(rubrics),
        "rubrics": rubrics
    }


@app.get("/rubrics/{rubric_id}")
async def get_grading_rubric(rubric_id: str):
    rubric_id = validate_document_id(rubric_id)

    rubric = await run_in_threadpool(
        get_rubric,
        rubric_id
    )

    if rubric is None:
        raise HTTPException(
            status_code=404,
            detail="Rubric not found."
        )

    return rubric


@app.post("/essays/text")
async def create_text_essay(
    request: EssayTextRequest
):
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Essay is empty."
        )

    try:
        return await run_in_threadpool(
            save_essay,
            request.title,
            request.text
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


@app.post("/essays/upload")
async def upload_essay(
    file: UploadFile = File(...),
    title: str = Form(default="")
):
    """
    Upload an essay as TXT, PDF, or DOCX.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename required."
        )

    try:
        content = await file.read(
            MAX_ESSAY_UPLOAD_BYTES + 1
        )

        if len(content) > MAX_ESSAY_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Essay exceeds 5 MB."
            )

        text = await run_in_threadpool(
            extract_essay_text,
            file.filename,
            content
        )

        essay_title = (
            title.strip()
            or Path(file.filename).stem
        )

        if not essay_title or len(essay_title) > 200:
            raise HTTPException(
                status_code=400,
                detail="Invalid essay title."
            )

        return await run_in_threadpool(
            save_essay,
            essay_title,
            text,
            Path(file.filename).name
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception:
        logger.exception("Essay upload failed")

        raise HTTPException(
            status_code=500,
            detail="Essay upload failed."
        )

    finally:
        await file.close()


@app.get("/essays/{essay_id}")
async def retrieve_essay(essay_id: str):
    essay_id = validate_document_id(essay_id)

    essay = await run_in_threadpool(
        get_essay,
        essay_id
    )

    if essay is None:
        raise HTTPException(
            status_code=404,
            detail="Essay not found."
        )

    return essay


@app.post("/grade-essay")
async def grade_submitted_essay(
    request: GradeRequest
):
    """
    Generate and validate a rubric-based grade.
    """

    essay_id = validate_document_id(
        request.essay_id
    )

    rubric_id = validate_document_id(
        request.rubric_id
    )

    try:
        return await grade_essay(
            essay_id,
            rubric_id
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except (
        ValueError,
        ValidationError
    ) as exc:
        logger.warning(
            "Grading validation failed: %s",
            exc
        )

        raise HTTPException(
            status_code=422,
            detail=(
                "The generated assessment failed "
                "validation. No grade was saved."
            )
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Groq rate limit reached."
        )

    except (APIConnectionError, APITimeoutError):
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Groq."
        )

    except APIStatusError as exc:
        logger.error(
            "Groq API error %s: %s",
            exc.status_code,
            exc.response.text
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"Groq API returned HTTP "
                f"{exc.status_code}."
            )
        )

    except Exception:
        logger.exception("Essay grading failed")

        raise HTTPException(
            status_code=500,
            detail="Essay grading failed."
        )


@app.get("/grades/{grade_id}")
async def retrieve_grade(grade_id: str):
    grade_id = validate_document_id(grade_id)

    report = await run_in_threadpool(
        get_grade,
        grade_id
    )

    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Grading report not found."
        )

    return report

BASE_PATH = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_PATH / "static"),
    name="static"
)


@app.get("/app", include_in_schema=False)
async def web_interface():
    return FileResponse(
        BASE_PATH / "templates" / "index.html",
        media_type="text/html"
    )