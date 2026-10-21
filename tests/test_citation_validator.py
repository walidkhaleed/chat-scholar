
import unittest

from citation_validator import (
    extract_citations,
    validate_citations,
)


class CitationValidatorTests(unittest.TestCase):

    def setUp(self):
        self.sources = [
            {
                "source_id": "S1",
                "page_number": 1,
                "text": (
                    "Employees receive salaries on Friday. "
                    "The study enrolled 12 volunteers."
                ),
            },
            {
                "source_id": "S2",
                "page_number": 2,
                "text": (
                    "The research concludes that "
                    "training improved employee performance."
                ),
            },
        ]

    def test_01_valid_citation(self):
        answer = (
            "Employees receive salaries on Friday. [S1]"
        )

        result = validate_citations(
            answer, self.sources
        )

        self.assertTrue(
            result["evidence_checks_passed"]
        )

    def test_02_multiple_valid_citations(self):
        answer = (
            "Employees receive salaries on Friday [S1]. "
            "Training improved performance [S2]."
        )

        result = validate_citations(
            answer, self.sources
        )

        self.assertTrue(
            result["evidence_checks_passed"]
        )

    def test_03_unknown_source_rejected(self):
        answer = (
            "Employees receive salaries on Friday [S99]."
        )

        result = validate_citations(
            answer, self.sources
        )

        self.assertFalse(
            result["evidence_checks_passed"]
        )

        self.assertIn(
            "S99", result["invalid_source_ids"]
        )

    def test_04_uncited_sentence_rejected(self):
        answer = (
            "Employees receive salaries on Friday. "
            "This is the company's standard policy."
        )

        result = validate_citations(
            answer, self.sources
        )

        self.assertFalse(
            result["evidence_checks_passed"]
        )

        self.assertTrue(
            result["uncited_sentences"]
        )

    def test_05_unsupported_number_rejected(self):
        answer = (
            "The study enrolled 42 volunteers [S1]."
        )

        result = validate_citations(
            answer, self.sources
        )

        self.assertFalse(
            result["evidence_checks_passed"]
        )

        self.assertIn(
            "42", result["unsupported_numbers"]
        )

    def test_06_supported_number_accepted(self):
        answer = (
            "The study enrolled 12 volunteers [S1]."
        )

        result = validate_citations(
            answer, self.sources
        )

        self.assertTrue(
            result["evidence_checks_passed"]
        )

    def test_07_unicode_citation_spacing(self):
        # U+202F = narrow no-break space,
        # previously observed in a real Groq response.
        answer = (
            "Employees receive salaries on Friday "
            "[S\u202f1]."
        )

        result = validate_citations(
            answer, self.sources
        )

        self.assertTrue(
            result["evidence_checks_passed"],
            msg=f"Verification details: {result}"
        )
    def test_08_groq_corner_bracket_citations(self):
        answer = (
            "Employees receive salaries on Friday【S1】. "
            "Training improved employee performance【S2】."
        )

        result = validate_citations(
            answer,
            self.sources
        )

        self.assertTrue(
            result["evidence_checks_passed"],
            msg=str(result)
        )

        self.assertEqual(
            result["cited_source_ids"],
            ["S1", "S2"]
        )

        self.assertEqual(
            result["uncited_sentences"],
            []
        )

        self.assertEqual(
            result["unsupported_numbers"],
            []
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
