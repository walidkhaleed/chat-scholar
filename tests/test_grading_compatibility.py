
import unittest

from pydantic import ValidationError

from grading_models import (
    RubricCreate,
    RubricCriterion,
    CriterionAssessment,
)

from essay_grader import verify_quotes


class GradingCompatibilityTests(unittest.TestCase):

    def create_criterion(self, criterion_id, weight):
        return RubricCriterion(
            criterion_id=criterion_id,
            name="Academic Assessment",
            description="Evaluate the quality of academic writing.",
            weight=weight,
            max_score=10
        )

    def create_assessment(self, quotes):
        return CriterionAssessment(
            criterion_id="argument",
            score=8,
            justification="The argument is clearly presented.",
            evidence_quotes=quotes,
            improvement="Provide more supporting evidence."
        )

    def test_valid_rubric(self):
        rubric = RubricCreate(
            title="Academic Essay Rubric",
            criteria=[
                self.create_criterion("argument", 60),
                self.create_criterion("evidence", 40)
            ]
        )

        self.assertEqual(len(rubric.criteria), 2)
        self.assertEqual(
            sum(c.weight for c in rubric.criteria),
            100
        )

    def test_invalid_rubric_weights(self):
        with self.assertRaises(ValidationError):
            RubricCreate(
                title="Invalid Rubric",
                criteria=[
                    self.create_criterion("argument", 60),
                    self.create_criterion("evidence", 30)
                ]
            )

    def test_duplicate_criterion_ids(self):
        with self.assertRaises(ValidationError):
            RubricCreate(
                title="Duplicate IDs",
                criteria=[
                    self.create_criterion("argument", 50),
                    self.create_criterion("argument", 50)
                ]
            )

    def test_valid_quotation(self):
        essay = "AI supports learning in modern education."

        assessment = self.create_assessment([
            "AI supports learning"
        ])

        result = verify_quotes(
            essay,
            [assessment]
        )

        self.assertEqual(result["verified_quotes"], 1)
        self.assertEqual(result["removed_quotes"], 0)

        self.assertEqual(
            assessment.evidence_quotes,
            ["AI supports learning"]
        )

    def test_fabricated_quotation(self):
        essay = "AI supports learning in modern education."

        assessment = self.create_assessment([
            "The moon is made of green cheese."
        ])

        result = verify_quotes(
            essay,
            [assessment]
        )

        self.assertEqual(result["verified_quotes"], 0)
        self.assertEqual(result["removed_quotes"], 1)
        self.assertEqual(assessment.evidence_quotes, [])

    def test_mixed_quotations(self):
        essay = "AI supports learning in modern education."

        assessment = self.create_assessment([
            "AI supports learning",
            "This quotation was fabricated."
        ])

        result = verify_quotes(
            essay,
            [assessment]
        )

        self.assertEqual(result["verified_quotes"], 1)
        self.assertEqual(result["removed_quotes"], 1)

        self.assertEqual(
            assessment.evidence_quotes,
            ["AI supports learning"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
