import unittest
from unittest.mock import patch, MagicMock
from app.services.retrieval.ranking_service import (
    rerank_documents,
    rerank_documents_detailed,
    RerankOutcome,
)


class TestRerankingService(unittest.TestCase):
    def test_empty_documents_handling(self):
        outcome = rerank_documents_detailed("test query", [])
        self.assertEqual(len(outcome.documents), 0)
        self.assertFalse(outcome.reranked)
        self.assertFalse(outcome.fallback_used)

    @patch("app.services.retrieval.ranking_service._get_ranker")
    def test_reranker_success(self, mock_get_ranker):
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": 1, "text": "Doc B", "score": 0.95},
            {"id": 0, "text": "Doc A", "score": 0.65},
        ]
        mock_get_ranker.return_value = mock_ranker

        docs = [
            {"content": "Doc A", "source": "a.txt", "score": 0.5},
            {"content": "Doc B", "source": "b.txt", "score": 0.4},
        ]

        outcome = rerank_documents_detailed("Kubernetes networking", docs, top_n=2)
        self.assertTrue(outcome.reranked)
        self.assertFalse(outcome.fallback_used)
        self.assertEqual(len(outcome.documents), 2)
        # Verify Doc B was reordered to first place with its new cross-encoder score
        self.assertEqual(outcome.documents[0]["source"], "b.txt")
        self.assertEqual(outcome.documents[0]["score"], 0.95)

    @patch("app.services.retrieval.ranking_service._get_ranker")
    def test_reranker_failure_fallback_observability(self, mock_get_ranker):
        mock_ranker = MagicMock()
        mock_ranker.rerank.side_effect = RuntimeError("ONNX runtime memory allocation failed")
        mock_get_ranker.return_value = mock_ranker

        docs = [
            {"content": "Doc 1", "source": "1.txt"},
            {"content": "Doc 2", "source": "2.txt"},
            {"content": "Doc 3", "source": "3.txt"},
        ]

        outcome = rerank_documents_detailed("Failure test query", docs, top_n=2)
        self.assertFalse(outcome.reranked)
        self.assertTrue(outcome.fallback_used)
        self.assertTrue(outcome.original_order_returned)
        self.assertIn("RuntimeError", outcome.error)
        # Preserves original order up to top_n
        self.assertEqual(len(outcome.documents), 2)
        self.assertEqual(outcome.documents[0]["source"], "1.txt")
        self.assertEqual(outcome.documents[1]["source"], "2.txt")

    @patch("app.services.retrieval.ranking_service._get_ranker")
    def test_string_documents_support(self, mock_get_ranker):
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": 1, "text": "Passage B", "score": 0.88},
            {"id": 0, "text": "Passage A", "score": 0.44},
        ]
        mock_get_ranker.return_value = mock_ranker

        docs = ["Passage A", "Passage B"]
        res = rerank_documents("query", docs, top_n=2)
        self.assertEqual(res[0], "Passage B")
        self.assertTrue(hasattr(res, "outcome"))
        self.assertTrue(res.outcome.reranked)


if __name__ == "__main__":
    unittest.main()
