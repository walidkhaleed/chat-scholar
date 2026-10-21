
import os
import subprocess
import sys
import tempfile
import unittest

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pymupdf

from fastapi import HTTPException
from starlette.datastructures import UploadFile

import main
from pdf_processor import (
    extract_pdf_text,
    PDFResourceLimitError,
)


def make_pdf(page_texts):
    """Create a real PDF entirely in memory."""

    document = pymupdf.open()

    try:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text(
                    (72, 72),
                    text,
                    fontsize=10,
                )

        return document.tobytes()

    finally:
        document.close()


class PDFResourceLimitTests(
    unittest.IsolatedAsyncioTestCase
):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.upload_dir = Path(self.temp.name)

        patcher = patch.object(
            main,
            "UPLOAD_DIR",
            self.upload_dir,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        # Never write test document metadata to the
        # application's real extracted_text directory.
        metadata_patcher = patch.object(
            main,
            "save_document",
        )
        self.mock_save = metadata_patcher.start()
        self.addCleanup(metadata_patcher.stop)

        self.patch_limits = [
            patch.object(
                main, "PDF_UPLOAD_LIMIT",
                5 * 1024 * 1024
            ),
            patch.object(
                main, "PDF_PAGE_LIMIT", 30
            ),
            patch.object(
                main, "PDF_TEXT_LIMIT", 200_000
            ),
        ]

        for item in self.patch_limits:
            item.start()
            self.addCleanup(item.stop)

    async def upload(self, contents, filename="test.pdf"):
        file = UploadFile(
            file=BytesIO(contents),
            filename=filename,
        )

        try:
            return await main.upload_pdf(file)
        finally:
            await file.close()

    def assert_upload_dir_empty(self):
        self.assertEqual(
            list(self.upload_dir.iterdir()),
            []
        )

    def test_01_public_demo_environment(self):
        env = os.environ.copy()
        env["CHAT_SCHOLAR_PUBLIC_DEMO_LIMITS"] = "true"

        code = (
            "import main; "
            "assert main.PUBLIC_DEMO_LIMITS is True; "
            "assert main.PDF_UPLOAD_LIMIT == 5*1024*1024; "
            "assert main.PDF_PAGE_LIMIT == 30; "
            "assert main.PDF_TEXT_LIMIT == 200000"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(
            result.returncode,
            0,
            result.stderr,
        )

    async def test_02_pdf_over_5mb_rejected(self):
        oversized = (
            b"%PDF-" +
            b"x" * (5 * 1024 * 1024)
        )

        with self.assertRaises(HTTPException) as error:
            await self.upload(oversized)

        self.assertEqual(
            error.exception.status_code, 413
        )
        self.assert_upload_dir_empty()

    async def test_03_pdf_over_30_pages_rejected(self):
        pdf = make_pdf(
            ["Page content"] * 31
        )

        with self.assertRaises(HTTPException) as error:
            await self.upload(pdf)

        self.assertEqual(
            error.exception.status_code, 413
        )
        self.assertIn(
            "30-page",
            error.exception.detail,
        )
        self.assert_upload_dir_empty()

    async def test_04_excessive_text_rejected(self):
        # Use a lower threshold in this particular
        # test so a small real PDF can exercise
        # the same enforcement logic.
        text = (
            "Academic research methods and analysis.\n"
            * 5
        )
        pdf = make_pdf([text])

        with patch.object(
            main,
            "PDF_TEXT_LIMIT",
            80,
        ):
            with self.assertRaises(
                HTTPException
            ) as error:
                await self.upload(pdf)

        self.assertEqual(
            error.exception.status_code, 413
        )
        self.assertIn(
            "extracted-text",
            error.exception.detail,
        )
        self.assert_upload_dir_empty()

    async def test_05_valid_pdf_succeeds(self):
        pdf = make_pdf([
            "Employees receive salaries on Friday.",
            "Warsaw is the capital of Poland.",
        ])

        result = await self.upload(pdf)

        self.assertEqual(
            result["page_count"], 2
        )
        self.assertIn(
            "document_id", result
        )
        self.assertIn(
            "successfully",
            result["message"],
        )

        self.mock_save.assert_called_once()

    async def test_06_invalid_signature_rejected(self):
        with self.assertRaises(HTTPException) as error:
            await self.upload(
                b"Not actually a PDF"
            )

        self.assertEqual(
            error.exception.status_code, 400
        )
        self.assert_upload_dir_empty()

    def test_07_original_extraction_preserved(self):
        pdf_path = (
            self.upload_dir / "local_test.pdf"
        )

        pdf_path.write_bytes(
            make_pdf(["Normal page"] * 31)
        )

        # Without explicit limits, the original
        # extraction behavior remains available.
        extracted = extract_pdf_text(pdf_path)

        self.assertEqual(
            extracted["page_count"], 31
        )

    def test_08_direct_text_limit_enforced(self):
        pdf_path = (
            self.upload_dir / "text_test.pdf"
        )

        pdf_path.write_bytes(
            make_pdf([
                "Academic research methods and analysis.\n"
                * 5
            ])
        )

        with self.assertRaises(
            PDFResourceLimitError
        ):
            extract_pdf_text(
                pdf_path,
                max_characters=80,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
