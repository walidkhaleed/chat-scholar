
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np

import embedding_processor


class DocumentRoutingTests(unittest.TestCase):

    def test_onnx_document_generation(self):
        document_id = str(uuid4())

        fake_chunks = {
            "filename": "example.pdf",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "page_number": 1,
                    "chunk_index": 0,
                    "text": "Machine learning uses data."
                },
                {
                    "chunk_id": "chunk-2",
                    "page_number": 2,
                    "chunk_index": 1,
                    "text": "Warsaw is in Poland."
                }
            ]
        }

        vectors = np.zeros((2, 384), dtype=np.float32)
        vectors[:, 0] = 1.0

        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(
                os.environ,
                {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "onnx"}
            ):
                with patch.object(
                    embedding_processor,
                    "EMBEDDING_DIR",
                    Path(folder)
                ):
                    with patch.object(
                        embedding_processor,
                        "load_chunks",
                        return_value=fake_chunks
                    ):
                        with patch(
                            "render_embeddings.inspect_token_lengths",
                            return_value={
                                "token_limit": 256,
                                "token_counts": [7, 8],
                                "truncated_count": 0,
                            }
                        ) as mock_tokens:
                            with patch.object(
                                embedding_processor,
                                "backend_embed_documents",
                                return_value=vectors
                            ) as mock_embed:
                                with patch.object(
                                    embedding_processor,
                                    "get_embedding_model"
                                ) as mock_local_model:

                                    result = (
                                        embedding_processor
                                        .generate_embeddings(document_id)
                                    )

                                    mock_local_model.assert_not_called()

                            mock_tokens.assert_called_once_with([
                                "Machine learning uses data.",
                                "Warsaw is in Poland."
                            ])

                        metadata_path = (
                            Path(folder) / f"{document_id}.json"
                        )

                        vector_path = (
                            Path(folder) / f"{document_id}.npy"
                        )

                        metadata = json.loads(
                            metadata_path.read_text(
                                encoding="utf-8"
                            )
                        )

                        saved_vectors = np.load(
                            vector_path,
                            allow_pickle=False
                        )

        self.assertEqual(
            metadata["model"],
            "fastembed-onnx-minilm-l6-v2-v1"
        )

        self.assertEqual(
            result["model"],
            metadata["model"]
        )

        self.assertEqual(saved_vectors.shape, (2, 384))
        self.assertEqual(saved_vectors.dtype, np.float32)
        self.assertEqual(result["total_embeddings"], 2)

        # We have not implemented ONNX token
        # length inspection yet.
        self.assertEqual(metadata["truncated_chunks"], 0)
        self.assertEqual(result["truncated_chunks"], 0)

    def test_local_model_identifier_preserved(self):
        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "local"}
        ):
            from embedding_backend import get_backend_model_id

            self.assertEqual(
                get_backend_model_id(),
                "sentence-transformers/all-MiniLM-L6-v2"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
