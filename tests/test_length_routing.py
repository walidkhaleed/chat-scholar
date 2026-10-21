
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

import embedding_processor


class TokenLengthRoutingTests(unittest.TestCase):

    def setUp(self):
        self.document_id = str(uuid4())

        self.chunk_data = {
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "page_number": 1,
                    "text": "Short paragraph."
                },
                {
                    "chunk_id": "chunk-2",
                    "page_number": 2,
                    "text": "Long academic paragraph."
                }
            ]
        }

    def test_onnx_detects_oversized_chunk(self):
        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "onnx"}
        ):
            with patch.object(
                embedding_processor,
                "load_chunks",
                return_value=self.chunk_data
            ):
                with patch(
                    "render_embeddings.inspect_token_lengths",
                    return_value={
                        "token_limit": 256,
                        "token_counts": [8, 340],
                        "truncated_count": 1
                    }
                ) as mock_inspection:
                    with patch.object(
                        embedding_processor,
                        "get_embedding_model"
                    ) as mock_local:
                        result = (
                            embedding_processor
                            .check_embedding_lengths(
                                self.document_id
                            )
                        )

                        mock_local.assert_not_called()

        mock_inspection.assert_called_once_with([
            "Short paragraph.",
            "Long academic paragraph."
        ])

        self.assertEqual(result["total_chunks"], 2)
        self.assertEqual(result["token_limit"], 256)
        self.assertFalse(result["valid"])
        self.assertEqual(len(result["oversized_chunks"]), 1)

        oversized = result["oversized_chunks"][0]

        self.assertEqual(oversized["chunk_id"], "chunk-2")
        self.assertEqual(oversized["page_number"], 2)
        self.assertEqual(oversized["token_count"], 340)

    def test_onnx_accepts_short_chunks(self):
        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "onnx"}
        ):
            with patch.object(
                embedding_processor,
                "load_chunks",
                return_value=self.chunk_data
            ):
                with patch(
                    "render_embeddings.inspect_token_lengths",
                    return_value={
                        "token_limit": 256,
                        "token_counts": [8, 10],
                        "truncated_count": 0
                    }
                ):
                    result = (
                        embedding_processor
                        .check_embedding_lengths(
                            self.document_id
                        )
                    )

        self.assertTrue(result["valid"])
        self.assertEqual(result["oversized_chunks"], [])

    def test_missing_chunks_rejected(self):
        with patch.object(
            embedding_processor,
            "load_chunks",
            return_value=None
        ):
            with self.assertRaises(FileNotFoundError):
                embedding_processor.check_embedding_lengths(
                    self.document_id
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
