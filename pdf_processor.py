
import json
from pathlib import Path

import pymupdf


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
TEXT_DIR = BASE_DIR / "extracted_text"

UPLOAD_DIR.mkdir(exist_ok=True)
TEXT_DIR.mkdir(exist_ok=True)

class PDFResourceLimitError(ValueError):
    """Raised when a PDF exceeds configured resource limits."""
    pass

def extract_pdf_text(
    pdf_path: Path,
    max_pages: int | None = None,
    max_characters: int | None = None,
) -> dict:
    """
    Extract PDF text while optionally enforcing
    resource limits.

    Defaults preserve existing local behavior.
    """

    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive.")

    if max_characters is not None and max_characters <= 0:
        raise ValueError("max_characters must be positive.")

    pages = []
    total_characters = 0

    with pymupdf.open(pdf_path) as document:

        if document.needs_pass:
            raise ValueError(
                "Password-protected PDFs are not supported."
            )

        # Reject oversized documents before extracting text.
        if (
            max_pages is not None
            and len(document) > max_pages
        ):
            raise PDFResourceLimitError(
                f"PDF exceeds the {max_pages}-page limit."
            )

        for page_number, page in enumerate(
            document, start=1
        ):
            text = page.get_text("text", sort=True)

            # Include separators added between pages.
            separator_length = 2 if pages else 0

            new_total = (
                total_characters
                + separator_length
                + len(text)
            )

            if (
                max_characters is not None
                and new_total > max_characters
            ):
                raise PDFResourceLimitError(
                    "PDF exceeds the extracted-text limit "
                    f"of {max_characters:,} characters."
                )

            pages.append({
                "page_number": page_number,
                "text": text
            })

            total_characters = new_total

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
