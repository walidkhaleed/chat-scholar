
import unittest

from render_embeddings import inspect_token_lengths


class ONNXTokenLengthTests(unittest.TestCase):

    def test_short_text(self):
        result = inspect_token_lengths([
            "Machine learning uses data."
        ])

        self.assertEqual(result["token_limit"], 256)
        self.assertEqual(result["truncated_count"], 0)
        self.assertEqual(len(result["token_counts"]), 1)

    def test_oversized_text_detected(self):
        long_text = "Machine learning is useful. " * 300

        result = inspect_token_lengths([long_text])

        self.assertGreater(
            result["token_counts"][0],
            result["token_limit"]
        )
        self.assertEqual(result["truncated_count"], 1)

    def test_mixed_chunks(self):
        texts = [
            "This is a short academic paragraph.",
            "Artificial intelligence and research. " * 300
        ]

        result = inspect_token_lengths(texts)

        self.assertEqual(len(result["token_counts"]), 2)
        self.assertEqual(result["truncated_count"], 1)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            inspect_token_lengths([])

    def test_blank_text_rejected(self):
        with self.assertRaises(ValueError):
            inspect_token_lengths(["   "])


if __name__ == "__main__":
    unittest.main(verbosity=2)
