
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from faiss_processor import load_faiss_index


class FAISSBackendGuardTests(unittest.TestCase):

    def check_mismatch(self, active_backend, stored_model):
        document_id = str(uuid4())

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            index_path = root / "test.faiss"
            metadata_path = root / "test.json"

            index_path.write_bytes(b"placeholder")

            metadata_path.write_text(
                json.dumps({
                    "document_id": document_id,
                    "model": stored_model,
                    "dimension": 384,
                    "total_vectors": 1,
                }),
                encoding="utf-8",
            )

            paths = {
                "index": index_path,
                "index_metadata": metadata_path,
            }

            with patch.dict(
                os.environ,
                {
                    "CHAT_SCHOLAR_EMBEDDING_BACKEND":
                        active_backend
                },
            ):
                with patch(
                    "faiss_processor.get_paths",
                    return_value=paths,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "backend mismatch",
                    ):
                        load_faiss_index(document_id)

    def test_local_rejects_onnx_index(self):
        self.check_mismatch(
            active_backend="local",
            stored_model="fastembed-onnx-minilm-l6-v2-v1",
        )

    def test_onnx_rejects_local_index(self):
        self.check_mismatch(
            active_backend="onnx",
            stored_model="sentence-transformers/all-MiniLM-L6-v2",
        )

    def test_onnx_rejects_missing_backend_metadata(self):
        self.check_mismatch(
            active_backend="onnx",
            stored_model=None,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
