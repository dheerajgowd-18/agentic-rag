import unittest
from app.agents.nodes.responder import validate_citations


class TestCitationValidation(unittest.TestCase):
    def test_valid_citations_in_range(self):
        answer = "Kubernetes pods run in nodes [1], and networking is managed by CNI [2]."
        sanitized, meta = validate_citations(answer, num_documents=2)
        self.assertEqual(sanitized, answer)
        self.assertTrue(meta["all_citations_valid"])
        self.assertEqual(meta["valid_citations"], [1, 2])
        self.assertEqual(meta["invalid_citations"], [])
        self.assertEqual(meta["total_citations"], 2)

    def test_out_of_range_citation_flagged_and_sanitized(self):
        # 2 documents provided, but answer cites [3] and [9]
        answer = "Redis queue is initialized [1], but celery worker is used [3], and Kafka is deployed [9]."
        sanitized, meta = validate_citations(answer, num_documents=2)
        self.assertFalse(meta["all_citations_valid"])
        self.assertEqual(meta["valid_citations"], [1])
        self.assertEqual(meta["invalid_citations"], [3, 9])
        self.assertIn("[unverified-cite-3]", sanitized)
        self.assertIn("[unverified-cite-9]", sanitized)
        self.assertIn("[1]", sanitized)

    def test_combined_citations(self):
        answer = "Both techniques are complementary [1][2]."
        sanitized, meta = validate_citations(answer, num_documents=2)
        self.assertTrue(meta["all_citations_valid"])
        self.assertEqual(meta["valid_citations"], [1, 2])

    def test_repeated_citations(self):
        answer = "Step one [1]. Step two [1]. Step three [2]."
        sanitized, meta = validate_citations(answer, num_documents=2)
        self.assertTrue(meta["all_citations_valid"])
        self.assertEqual(meta["total_citations"], 3)
        self.assertEqual(meta["valid_citations"], [1, 2])

    def test_non_citation_brackets_preserved(self):
        answer = "Log level is [INFO], status is [ACTIVE], and version is [v1.2] with reference [1]."
        sanitized, meta = validate_citations(answer, num_documents=1)
        self.assertTrue(meta["all_citations_valid"])
        self.assertEqual(meta["valid_citations"], [1])
        self.assertIn("[INFO]", sanitized)
        self.assertIn("[ACTIVE]", sanitized)
        self.assertIn("[v1.2]", sanitized)

    def test_no_citations_present(self):
        answer = "The documentation does not provide instructions for this task."
        sanitized, meta = validate_citations(answer, num_documents=3)
        self.assertEqual(sanitized, answer)
        self.assertTrue(meta["all_citations_valid"])
        self.assertEqual(meta["total_citations"], 0)
        self.assertEqual(meta["valid_citations"], [])
        self.assertEqual(meta["invalid_citations"], [])

    def test_comma_separated_citations(self):
        answer = "Features are documented here [1, 2]."
        sanitized, meta = validate_citations(answer, num_documents=2)
        self.assertEqual(sanitized, answer)
        self.assertTrue(meta["all_citations_valid"])
        self.assertEqual(meta["valid_citations"], [1, 2])
        self.assertEqual(meta["total_citations"], 2)

    def test_comma_separated_with_out_of_range_sanitized(self):
        answer = "Deployment steps [1, 5] apply to nodes."
        sanitized, meta = validate_citations(answer, num_documents=2)
        self.assertFalse(meta["all_citations_valid"])
        self.assertEqual(meta["valid_citations"], [1])
        self.assertEqual(meta["invalid_citations"], [5])
        self.assertIn("[1][unverified-cite-5]", sanitized)

    def test_nonpositive_and_negative_citations(self):
        answer = "Root entry [0] and previous index [-1]."
        sanitized, meta = validate_citations(answer, num_documents=2)
        self.assertFalse(meta["all_citations_valid"])
        self.assertIn("[unverified-cite-0]", sanitized)
        self.assertIn("[unverified-cite--1]", sanitized)
        self.assertEqual(meta["invalid_citations"], [-1, 0])


if __name__ == "__main__":
    unittest.main()
