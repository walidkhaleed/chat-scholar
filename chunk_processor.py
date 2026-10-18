
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_processor import load_document


BASE_DIR = Path(__file__).resolve().parent
CHUNK_DIR = BASE_DIR / "chunks"

CHUNK_DIR.mkdir(exist_ok=True)


def create_chunks(
    document_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> dict:
    """
    Split extracted PDF text into overlapping chunks.

    Each chunk preserves:
    - Document ID
    - Source filename
    - Page number
    - Chunk ID
    - Chunk text
    """

    # Validate chunk configuration
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be between 0 and chunk_size - 1."
        )

    # Load extracted document
    document = load_document(document_id)

    if document is None:
        raise FileNotFoundError("Document not found.")

    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    # Process each PDF page separately
    for page in document["pages"]:

        page_number = page["page_number"]
        page_text = page["text"]

        if not page_text.strip():
            continue

        text_chunks = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(
            text_chunks,
            start=1
        ):

            chunk_id = (
                f"{document_id}_p{page_number}_c{chunk_index}"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "filename": document["filename"],
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "character_count": len(chunk_text)
            })

    result = {
        "document_id": document_id,
        "filename": document["filename"],
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    # Save chunks as JSON
    output_path = CHUNK_DIR / f"{document_id}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )

    return result


def load_chunks(document_id: str) -> dict | None:
    """
    Load previously generated chunks.
    """

    file_path = CHUNK_DIR / f"{document_id}.json"

    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
