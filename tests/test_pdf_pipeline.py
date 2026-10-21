
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import psutil
import pymupdf

import pdf_processor
import chunk_processor
import embedding_processor
import faiss_processor

from embedding_backend import get_backend_model_id
from hybrid_retriever import hybrid_search


class RealPDFPipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Temporary data directory: never touch
        # the application's existing saved PDFs.
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)

        cls.root = Path(cls.temp.name)

        for folder in (
            "extracted_text",
            "chunks",
            "embeddings",
            "faiss_indexes",
        ):
            (cls.root / folder).mkdir()

        patches = [
            patch.dict(
                os.environ,
                {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "onnx"}
            ),
            patch.object(
                pdf_processor,
                "TEXT_DIR",
                cls.root / "extracted_text",
            ),
            patch.object(
                chunk_processor,
                "CHUNK_DIR",
                cls.root / "chunks",
            ),
            patch.object(
                embedding_processor,
                "EMBEDDING_DIR",
                cls.root / "embeddings",
            ),
            patch.object(
                faiss_processor,
                "EMBEDDING_DIR",
                cls.root / "embeddings",
            ),
            patch.object(
                faiss_processor,
                "FAISS_DIR",
                cls.root / "faiss_indexes",
            ),
            patch.object(
                faiss_processor,
                "BASE_DIR",
                cls.root,
            ),
        ]

        for item in patches:
            item.start()
            cls.addClassCleanup(item.stop)

        cls.document_id = str(uuid4())
        cls.pdf_path = cls.root / "portfolio_test.pdf"

        # Generate an actual three-page PDF.
        page_texts = [
            (
                "Company payroll policy. "
                "Employees receive their salaries on Friday. "
                "The human resources department handles payroll."
            ),
            (
                "Geography lesson. Warsaw is the capital "
                "city of Poland. Poland is located in Europe."
            ),
            (
                "Computer science lesson. Machine learning "
                "algorithms identify patterns in data."
            ),
        ]

        with pymupdf.open() as document:
            for text in page_texts:
                page = document.new_page()
                page.insert_text(
                    (72, 72),
                    text,
                    fontsize=11,
                )

            document.save(str(cls.pdf_path))

        # 1. Extract real PDF text.
        cls.extracted = pdf_processor.extract_pdf_text(
            cls.pdf_path
        )

        pdf_processor.save_document(
            cls.document_id,
            {
                "document_id": cls.document_id,
                "filename": cls.pdf_path.name,
                "page_count": cls.extracted["page_count"],
                "character_count": len(
                    cls.extracted["full_text"]
                ),
                "pages": cls.extracted["pages"],
                "full_text": cls.extracted["full_text"],
            },
        )

        # 2. Create real document chunks.
        cls.chunks = chunk_processor.create_chunks(
            cls.document_id,
            chunk_size=400,
            chunk_overlap=40,
        )

        # 3. Validate actual ONNX token lengths.
        cls.validation = (
            embedding_processor.check_embedding_lengths(
                cls.document_id
            )
        )

        if not cls.validation["valid"]:
            raise AssertionError(
                "Test PDF unexpectedly exceeds token limit."
            )

        # 4. Generate real ONNX embeddings.
        cls.embeddings = (
            embedding_processor.generate_embeddings(
                cls.document_id
            )
        )

        # 5. Build the actual application FAISS index.
        cls.faiss_result = (
            faiss_processor.build_faiss_index(
                cls.document_id
            )
        )

        cls.question = (
            "When do employees receive their salaries?"
        )

        # 6. Run real FAISS semantic retrieval.
        cls.semantic = faiss_processor.semantic_search(
            cls.document_id,
            cls.question,
            top_k=3,
        )

        # 7. Run real FAISS + BM25 + RRF retrieval.
        cls.hybrid = hybrid_search(
            cls.document_id,
            cls.question,
            top_k=3,
            candidate_k=3,
        )

    def test_real_pdf_extraction(self):
        self.assertEqual(
            self.extracted["page_count"], 3
        )
        self.assertIn(
            "salaries",
            self.extracted["full_text"]
        )

    def test_page_aware_chunking(self):
        self.assertEqual(
            self.chunks["total_chunks"], 3
        )

        pages = {
            chunk["page_number"]
            for chunk in self.chunks["chunks"]
        }

        self.assertEqual(pages, {1, 2, 3})

    def test_token_validation(self):
        self.assertTrue(self.validation["valid"])
        self.assertEqual(
            self.validation["token_limit"], 256
        )

    def test_real_onnx_embeddings(self):
        self.assertEqual(
            self.embeddings["embedding_dimension"], 384
        )

        self.assertEqual(
            self.embeddings["model"],
            get_backend_model_id()
        )

        self.assertEqual(
            self.embeddings["truncated_chunks"], 0
        )

    def test_application_faiss_index(self):
        self.assertEqual(
            self.faiss_result["total_vectors"], 3
        )

        self.assertEqual(
            self.faiss_result["dimension"], 384
        )

    def test_semantic_retrieval(self):
        best = self.semantic["results"][0]

        print(
            "\nBest FAISS result:",
            best["text"],
        )

        self.assertEqual(best["page_number"], 1)
        self.assertIn(
            "salaries",
            best["text"].lower()
        )

    def test_hybrid_retrieval(self):
        print(
            "\nHybrid retrieval:",
            json.dumps(
                self.hybrid,
                indent=2,
                default=str,
            )[:2000],
        )

        self.assertIsInstance(self.hybrid, dict)

        self.assertIn(
            "salaries",
            json.dumps(self.hybrid).lower()
        )

    def test_process_memory(self):
        process = psutil.Process(os.getpid())
        memory_mb = (
            process.memory_info().rss / 1024**2
        )

        print(
            f"\nPDF + ONNX + FAISS + BM25 RAM: "
            f"{memory_mb:.1f} MB"
        )

        self.assertGreater(memory_mb, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
