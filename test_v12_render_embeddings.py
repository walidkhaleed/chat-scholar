
import unittest

import numpy as np

from render_embeddings import (
    BACKEND_ID,
    EMBEDDING_DIMENSION,
    embed_passages,
    embed_question,
    get_model,
)


class RenderEmbeddingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.documents = [
            "Employees receive their salaries on Friday.",
            "Machine learning identifies patterns in data.",
            "Warsaw is the capital city of Poland.",
        ]

        cls.vectors = embed_passages(cls.documents)
        cls.query = embed_question(
            "When do workers get paid?"
        )

    def test_document_dimensions(self):
        self.assertEqual(
            self.vectors.shape,
            (3, EMBEDDING_DIMENSION)
        )

    def test_query_dimensions(self):
        self.assertEqual(
            self.query.shape,
            (EMBEDDING_DIMENSION,)
        )

    def test_float32(self):
        self.assertEqual(
            self.vectors.dtype,
            np.float32
        )
        self.assertEqual(
            self.query.dtype,
            np.float32
        )

    def test_normalized_vectors(self):
        np.testing.assert_allclose(
            np.linalg.norm(
                self.vectors,
                axis=1
            ),
            np.ones(3),
            atol=1e-5
        )

    def test_semantic_retrieval(self):
        scores = self.vectors @ self.query
        best = int(np.argmax(scores))

        self.assertEqual(best, 0)

    def test_model_cached(self):
        self.assertIs(
            get_model(),
            get_model()
        )

    def test_backend_identified(self):
        self.assertIn(
            "onnx",
            BACKEND_ID
        )

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            embed_question("")

    def test_empty_chunks_rejected(self):
        with self.assertRaises(ValueError):
            embed_passages([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
