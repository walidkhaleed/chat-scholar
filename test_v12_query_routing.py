
import os
import unittest
from unittest.mock import patch

import numpy as np

from embedding_processor import embed_query


class QueryRoutingTests(unittest.TestCase):

    def test_onnx_query_routing(self):
        fake_vector = np.ones(384, dtype=np.float32)

        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "onnx"}
        ):
            with patch(
                "embedding_processor.backend_embed_question",
                return_value=fake_vector
            ) as mock_embed:
                result = embed_query("What is machine learning?")

        mock_embed.assert_called_once_with(
            "What is machine learning?"
        )
        self.assertEqual(result.shape, (384,))
        self.assertEqual(result.dtype, np.float32)

    def test_local_query_routing(self):
        fake_vector = np.ones(384, dtype=np.float32)

        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "local"}
        ):
            with patch(
                "embedding_processor.backend_embed_question",
                return_value=fake_vector
            ) as mock_embed:
                result = embed_query("Explain neural networks")

        mock_embed.assert_called_once()
        self.assertEqual(result.shape, (384,))

    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            embed_query("   ")

    def test_invalid_vector_dimension_rejected(self):
        with patch(
            "embedding_processor.backend_embed_question",
            return_value=np.ones(128, dtype=np.float32)
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "dimensions"
            ):
                embed_query("Test question")


if __name__ == "__main__":
    unittest.main(verbosity=2)
