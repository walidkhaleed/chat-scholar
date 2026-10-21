
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

import grading_store
from main import app

from auth_security import (
    SESSION_COOKIE_NAME,
    create_session_token,
)
from session_revocation import SessionRevocationStore


class GradePersistenceTests(unittest.TestCase):

    def setUp(self):
        # Create isolated storage for every test.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.test_path = Path(self.temp_dir.name)
        self.grade_dir = self.test_path / "grades"
        self.grade_dir.mkdir()

        # Redirect grading_store to the test directory.
        patcher = patch.object(
            grading_store,
            "GRADE_DIR",
            self.grade_dir
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def sample_report(self):
        return {
            "essay_id": str(uuid4()),
            "essay_title": "Artificial Intelligence in Education",
            "rubric_id": str(uuid4()),
            "rubric_title": "Academic Essay Assessment",
            "model": "test-model",
            "status": "completed",
            "percentage": 68.0,
            "criteria": [
                {
                    "criterion_id": "argument",
                    "name": "Argument",
                    "weight": 60,
                    "score": 8.0,
                    "max_score": 10,
                    "weighted_points": 48.0,
                    "justification": "A clear central argument.",
                    "evidence_quotes": [
                        "AI supports learning"
                    ],
                    "improvement": "Expand the counterargument."
                },
                {
                    "criterion_id": "evidence",
                    "name": "Evidence",
                    "weight": 40,
                    "score": 5.0,
                    "max_score": 10,
                    "weighted_points": 20.0,
                    "justification": "Some evidence is relevant.",
                    "evidence_quotes": [],
                    "improvement": "Add more academic sources."
                }
            ],
            "overall_feedback": (
                "The essay has a clear argument but "
                "requires stronger supporting evidence."
            ),
            "strengths": [
                "Clear thesis",
                "Good organization"
            ],
            "improvements": [
                "Use additional supporting sources"
            ],
            "grading_notice": (
                "AI-generated formative assessment. "
                "Scores require human review before "
                "use in consequential grading."
            ),
            "evidence_verification": {
                "verified_quotes": 1,
                "removed_quotes": 0
            }
        }

    # TEST 1: Save a report successfully.
    def test_save_report(self):
        report = self.sample_report()

        saved = grading_store.save_grade(report)

        self.assertIn("grade_id", saved)
        self.assertIn("created_at", saved)

        file_path = (
            self.grade_dir / f"{saved['grade_id']}.json"
        )

        self.assertTrue(file_path.exists())
        print("1. Report saved to JSON: PASS")

    # TEST 2: Retrieve a saved report.
    def test_retrieve_report(self):
        saved = grading_store.save_grade(
            self.sample_report()
        )

        loaded = grading_store.get_grade(
            saved["grade_id"]
        )

        self.assertEqual(loaded, saved)
        print("2. Saved report retrieved: PASS")

    # TEST 3: Verify score and grading details.
    def test_report_score_consistency(self):
        saved = grading_store.save_grade(
            self.sample_report()
        )

        loaded = grading_store.get_grade(
            saved["grade_id"]
        )

        self.assertEqual(loaded["percentage"], 68.0)

        points = sum(
            item["weighted_points"]
            for item in loaded["criteria"]
        )

        self.assertAlmostEqual(points, 68.0)
        self.assertEqual(len(loaded["criteria"]), 2)

        print("3. Scores and criteria preserved: PASS")

    # TEST 4: Verify feedback and quotations.
    def test_report_feedback_preserved(self):
        saved = grading_store.save_grade(
            self.sample_report()
        )

        loaded = grading_store.get_grade(
            saved["grade_id"]
        )

        self.assertEqual(
            loaded["criteria"][0]["evidence_quotes"],
            ["AI supports learning"]
        )

        self.assertEqual(
            loaded["evidence_verification"]["verified_quotes"],
            1
        )

        self.assertEqual(
            len(loaded["strengths"]),
            2
        )

        self.assertIn(
            "human review",
            loaded["grading_notice"]
        )

        print("4. Feedback and quotations preserved: PASS")

    # TEST 5: Verify unique report IDs.
    def test_unique_report_ids(self):
        first = grading_store.save_grade(
            self.sample_report()
        )

        second = grading_store.save_grade(
            self.sample_report()
        )

        self.assertNotEqual(
            first["grade_id"],
            second["grade_id"]
        )

        self.assertEqual(
            len(list(self.grade_dir.glob("*.json"))),
            2
        )

        print("5. Unique report IDs: PASS")

    # TEST 6: Verify persistence on a fresh read.
    def test_report_survives_reopening(self):
        saved = grading_store.save_grade(
            self.sample_report()
        )

        # Read the JSON directly from disk.
        file_path = (
            self.grade_dir / f"{saved['grade_id']}.json"
        )

        persisted = grading_store.read_json(file_path)

        self.assertEqual(
            persisted["grade_id"],
            saved["grade_id"]
        )

        self.assertEqual(
            persisted["percentage"],
            68.0
        )

        print("6. Report persists on disk: PASS")

    # TEST 7: Confirm missing reports return None.
    def test_missing_report(self):
        nonexistent_id = str(uuid4())

        result = grading_store.get_grade(
            nonexistent_id
        )

        self.assertIsNone(result)
        print("7. Missing report handled: PASS")

    # TEST 8: Verify secure API retrieval.
    def test_secure_report_endpoint(self):
        saved = grading_store.save_grade(
            self.sample_report()
        )

        url = f"/grades/{saved['grade_id']}"

        test_sessions = SessionRevocationStore(
            self.test_path / "sessions.sqlite3"
        )

        with patch(
            "auth_guard.session_store",
            test_sessions
        ):
            with TestClient(app) as client:

                # Anonymous access must be rejected.
                response = client.get(url)

                self.assertEqual(
                    response.status_code,
                    401
                )

                # Authenticate using a test session.
                client.cookies.set(
                    SESSION_COOKIE_NAME,
                    create_session_token()
                )

                # Existing report must be returned.
                response = client.get(url)

                self.assertEqual(
                    response.status_code,
                    200
                )

                self.assertEqual(
                    response.json()["percentage"],
                    68.0
                )

                # A missing report must return 404.
                response = client.get(
                    f"/grades/{uuid4()}"
                )

                self.assertEqual(
                    response.status_code,
                    404
                )

        print("8. Secure report API retrieval: PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
