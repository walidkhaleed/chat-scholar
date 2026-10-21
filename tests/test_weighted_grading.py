
import unittest

from grading_models import (
    RubricCreate,
    RubricCriterion,
    CriterionAssessment,
)

from essay_grader import calculate_scores


class WeightedGradingTests(unittest.TestCase):

    def criterion(self, criterion_id, weight, max_score=10):
        return RubricCriterion(
            criterion_id=criterion_id,
            name=f"Criterion {criterion_id}",
            description="Evaluate the quality of the essay.",
            weight=weight,
            max_score=max_score
        )

    def assessment(self, criterion_id, score):
        return CriterionAssessment(
            criterion_id=criterion_id,
            score=score,
            justification="The essay demonstrates this quality.",
            evidence_quotes=[],
            improvement="Provide additional relevant examples."
        )

    def rubric(self):
        return RubricCreate(
            title="Academic Essay Assessment",
            criteria=[
                self.criterion("argument", 60),
                self.criterion("evidence", 40),
            ]
        )

    # TEST 1: Standard weighted calculation
    def test_standard_weighted_score(self):
        total, details = calculate_scores(
            self.rubric(),
            [
                self.assessment("argument", 8),
                self.assessment("evidence", 5),
            ]
        )

        self.assertEqual(total, 68.0)
        self.assertEqual(details[0]["weighted_points"], 48.0)
        self.assertEqual(details[1]["weighted_points"], 20.0)

    # TEST 2: Perfect essay
    def test_perfect_score(self):
        total, details = calculate_scores(
            self.rubric(),
            [
                self.assessment("argument", 10),
                self.assessment("evidence", 10),
            ]
        )

        self.assertEqual(total, 100.0)
        self.assertEqual(len(details), 2)

    # TEST 3: Zero scores
    def test_zero_score(self):
        total, _ = calculate_scores(
            self.rubric(),
            [
                self.assessment("argument", 0),
                self.assessment("evidence", 0),
            ]
        )

        self.assertEqual(total, 0.0)

    # TEST 4: Assessments may arrive out of order
    def test_assessment_order_independence(self):
        total, details = calculate_scores(
            self.rubric(),
            [
                self.assessment("evidence", 5),
                self.assessment("argument", 8),
            ]
        )

        self.assertEqual(total, 68.0)

        self.assertEqual(
            [d["criterion_id"] for d in details],
            ["argument", "evidence"]
        )

    # TEST 5: Different maximum scores and weights
    def test_different_maximum_scores(self):
        rubric = RubricCreate(
            title="Mixed Scoring Scale",
            criteria=[
                self.criterion("structure", 30, 5),
                self.criterion("research", 70, 20),
            ]
        )

        total, details = calculate_scores(
            rubric,
            [
                self.assessment("structure", 4),
                self.assessment("research", 13),
            ]
        )

        # (4/5)*30 + (13/20)*70 = 69.5
        self.assertEqual(total, 69.5)
        self.assertEqual(details[0]["weighted_points"], 24.0)
        self.assertEqual(details[1]["weighted_points"], 45.5)

    # TEST 6: Decimal rounding uses ROUND_HALF_UP
    def test_decimal_rounding(self):
        rubric = RubricCreate(
            title="Decimal Rounding",
            criteria=[
                self.criterion("precision", 100, 10)
            ]
        )

        total, details = calculate_scores(
            rubric,
            [self.assessment("precision", 0.1005)]
        )

        # 0.1005 / 10 * 100 = 1.005
        # ROUND_HALF_UP to two decimal places = 1.01
        self.assertEqual(total, 1.01)
        self.assertEqual(details[0]["weighted_points"], 1.01)

    # TEST 7: Score greater than maximum
    def test_score_above_maximum_rejected(self):
        with self.assertRaises(ValueError):
            calculate_scores(
                self.rubric(),
                [
                    self.assessment("argument", 11),
                    self.assessment("evidence", 5),
                ]
            )

    # TEST 8: Negative score
    def test_negative_score_rejected(self):
        with self.assertRaises(ValueError):
            calculate_scores(
                self.rubric(),
                [
                    self.assessment("argument", -1),
                    self.assessment("evidence", 5),
                ]
            )

    # TEST 9: Missing assessment
    def test_missing_criterion_rejected(self):
        with self.assertRaises(ValueError):
            calculate_scores(
                self.rubric(),
                [self.assessment("argument", 8)]
            )

    # TEST 10: Unexpected assessment
    def test_unexpected_criterion_rejected(self):
        with self.assertRaises(ValueError):
            calculate_scores(
                self.rubric(),
                [
                    self.assessment("argument", 8),
                    self.assessment("evidence", 5),
                    self.assessment("extra", 7),
                ]
            )

    # TEST 11: Duplicate assessment
    def test_duplicate_criterion_rejected(self):
        with self.assertRaises(ValueError):
            calculate_scores(
                self.rubric(),
                [
                    self.assessment("argument", 8),
                    self.assessment("argument", 7),
                    self.assessment("evidence", 5),
                ]
            )

    # TEST 12: Grading details are preserved
    def test_grading_details(self):
        total, details = calculate_scores(
            self.rubric(),
            [
                self.assessment("argument", 8),
                self.assessment("evidence", 5),
            ]
        )

        self.assertEqual(total, 68.0)
        self.assertEqual(len(details), 2)

        for detail in details:
            self.assertIn("criterion_id", detail)
            self.assertIn("name", detail)
            self.assertIn("weight", detail)
            self.assertIn("score", detail)
            self.assertIn("max_score", detail)
            self.assertIn("weighted_points", detail)
            self.assertIn("justification", detail)
            self.assertIn("evidence_quotes", detail)
            self.assertIn("improvement", detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
