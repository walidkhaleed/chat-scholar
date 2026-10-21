
import unittest

import faiss
import numpy as np
import psutil
import os

from render_embeddings import (
    embed_passages,
    embed_question,
    EMBEDDING_DIMENSION,
)


class RealONNXFAISSTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.documents = [
            "Employees receive their salaries on Friday.",
            "Warsaw is the capital of Poland.",
            "Machine learning identifies patterns in data.",
            "The university library opens at eight in the morning.",
        ]

        cls.question = "When do workers get paid?"

        # Generate real ONNX embeddings.
        cls.vectors = embed_passages(cls.documents)
        cls.query_vector = embed_question(cls.question)

        # Build a real FAISS index.
        cls.index = faiss.IndexFlatIP(
            EMBEDDING_DIMENSION
        )

        cls.index.add(
            np.ascontiguousarray(
                cls.vectors,
                dtype=np.float32
            )
        )

    def test_index_dimensions(self):
        self.assertEqual(self.index.d, 384)

    def test_index_contains_all_documents(self):
        self.assertEqual(self.index.ntotal, 4)

    def test_real_semantic_search(self):
        query = np.ascontiguousarray(
            self.query_vector.reshape(1, -1),
            dtype=np.float32
        )

        scores, indices = self.index.search(
            query,
            3
        )

        best_index = int(indices[0][0])
        best_score = float(scores[0][0])

        print("\n=== REAL ONNX + FAISS RESULTS ===")

        for rank, (score, index) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):
            if index >= 0:
                print(
                    f"{rank}. "
                    f"Score: {float(score):.4f} | "
                    f"{self.documents[int(index)]}"
                )

        print("Best similarity:", round(best_score, 4))

        self.assertEqual(best_index, 0)
        self.assertTrue(np.isfinite(best_score))

    def test_normalized_vectors(self):
        norms = np.linalg.norm(
            self.vectors,
            axis=1
        )

        np.testing.assert_allclose(
            norms,
            np.ones(len(self.documents)),
            atol=1e-5
        )

    def test_memory_usage(self):
        process = psutil.Process(os.getpid())

        memory_mb = (
            process.memory_info().rss / 1024**2
        )

        print(
            f"\nONNX + FAISS process RAM: "
            f"{memory_mb:.1f} MB"
        )

        self.assertGreater(memory_mb, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
