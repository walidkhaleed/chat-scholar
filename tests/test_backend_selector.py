
import os
import unittest
from unittest.mock import patch

import numpy as np

from embedding_backend import (
    get_backend_name,
    get_backend_model_id,
    embed_documents,
    embed_question,
)


class BackendSelectorTests(unittest.TestCase):

    def test_local_is_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                get_backend_name(),
                "local"
            )

    def test_onnx_selection(self):
        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "onnx"}
        ):
            self.assertEqual(
                get_backend_name(),
                "onnx"
            )

    def test_invalid_backend_rejected(self):
        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "unknown"}
        ):
            with self.assertRaises(ValueError):
                get_backend_name()

    def test_distinct_model_identifiers(self):
        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "local"}
        ):
            local_id = get_backend_model_id()

        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "onnx"}
        ):
            onnx_id = get_backend_model_id()

        self.assertNotEqual(local_id, onnx_id)

    def test_onnx_document_and_query_vectors(self):
        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_EMBEDDING_BACKEND": "onnx"}
        ):
            documents = [
                "Employees receive their salaries on Friday.",
                "Warsaw is the capital of Poland.",
            ]

            vectors = embed_documents(documents)
            query = embed_question(
                "When do employees get paid?"
            )

        self.assertEqual(vectors.shape, (2, 384))
        self.assertEqual(query.shape, (384,))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(query.dtype, np.float32)

        best_match = int(np.argmax(vectors @ query))

        self.assertEqual(best_match, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
