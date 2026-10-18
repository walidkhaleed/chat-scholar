
import json
from pathlib import Path

import pymupdf


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
TEXT_DIR = BASE_DIR / "extracted_text"

UPLOAD_DIR.mkdir(exist_ok=True)
TEXT_DIR.mkdir(exist_ok=True)


def extract_pdf_text(pdf_path: Path) -> dict:
    """
    Extract text from every page of a PDF.

    Returns:
        page_count: Number of pages
        pages: List of page numbers and extracted text
        full_text: Combined document text
    """

    pages = []

    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)

            pages.append({
                "page_number": page_number,
                "text": text
            })

    full_text = "\n\n".join(
        page["text"] for page in pages
    )

    return {
        "page_count": len(pages),
        "pages": pages,
        "full_text": full_text
    }


def save_document(document_id: str, data: dict) -> Path:
    """
    Save extracted PDF data as a JSON file.
    """

    output_path = TEXT_DIR / f"{document_id}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return output_path


def load_document(document_id: str) -> dict | None:
    """
    Load a previously processed document.
    """

    document_path = TEXT_DIR / f"{document_id}.json"

    if not document_path.exists():
        return None

    with open(document_path, "r", encoding="utf-8") as file:
        return json.load(file)


def list_documents() -> list:
    """
    Return metadata for all processed documents.
    """

    documents = []

    for file_path in sorted(TEXT_DIR.glob("*.json")):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        documents.append({
            "document_id": data["document_id"],
            "filename": data["filename"],
            "page_count": data["page_count"],
            "character_count": data["character_count"]
        })

    return documents
