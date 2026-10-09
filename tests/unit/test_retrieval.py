import unittest
from unittest.mock import patch, MagicMock
from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge,
    search_enterprise_knowledge_detailed,
    validate_collection_compatibility,
    RetrievalStatus,
    classify_retrieval_exception,
)


class TestRetrievalService(unittest.TestCase):
    @patch("app.services.retrieval.qdrant_service.embed_query", return_value=[0.1] * 384)
    @patch("app.services.retrieval.qdrant_service.get_qdrant_client")
    def test_successful_retrieval(self, mock_get_client, mock_embed):
        mock_client = MagicMock()
        mock_point = MagicMock()
        mock_point.payload = {"text": "Kubernetes Pod configuration", "source": "k8s.pdf", "source_type": "true"}
        mock_point.score = 0.85
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points.return_value = mock_response
        mock_get_client.return_value = mock_client

        outcome = search_enterprise_knowledge_detailed("Kubernetes Pod", limit=5)
        self.assertEqual(outcome.status, RetrievalStatus.SUCCESS)
        self.assertEqual(len(outcome.documents), 1)
        self.assertEqual(outcome.documents[0]["source"], "k8s.pdf")
        self.assertEqual(outcome.documents[0]["score"], 0.85)

    @patch("app.services.retrieval.qdrant_service.embed_query", return_value=[0.1] * 384)
    @patch("app.services.retrieval.qdrant_service.get_qdrant_client")
    def test_no_results_status(self, mock_get_client, mock_embed):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response
        mock_get_client.return_value = mock_client

        outcome = search_enterprise_knowledge_detailed("Nonexistent term", limit=5)
        self.assertEqual(outcome.status, RetrievalStatus.NO_RESULTS)
        self.assertEqual(len(outcome.documents), 0)
        self.assertTrue(outcome.is_success)  # Genuine no-match is a successful query

    @patch("app.services.retrieval.qdrant_service.embed_query", side_effect=Exception("Model embed error"))
    def test_embedding_failure_outcome(self, mock_embed):
        outcome = search_enterprise_knowledge_detailed("Any query")
        self.assertEqual(outcome.status, RetrievalStatus.EMBEDDING_FAILURE)
        self.assertFalse(outcome.is_success)
        self.assertEqual(len(outcome.documents), 0)

    @patch("app.services.retrieval.qdrant_service.embed_query", return_value=[0.1] * 384)
    @patch("app.services.retrieval.qdrant_service.get_qdrant_client")
    def test_database_connection_unavailable(self, mock_get_client, mock_embed):
        mock_client = MagicMock()
        mock_client.query_points.side_effect = Exception("Connection refused: failed to connect to host")
        mock_get_client.return_value = mock_client

        outcome = search_enterprise_knowledge_detailed("Query with down DB")
        self.assertEqual(outcome.status, RetrievalStatus.VECTOR_DB_UNAVAILABLE)
        self.assertFalse(outcome.is_success)

    @patch("app.services.retrieval.qdrant_service.embed_query", return_value=[0.1] * 384)
    @patch("app.services.retrieval.qdrant_service.get_qdrant_client")
    def test_auth_failure_classified(self, mock_get_client, mock_embed):
        mock_client = MagicMock()
        mock_client.query_points.side_effect = Exception("401 Unauthorized: Invalid API key")
        mock_get_client.return_value = mock_client

        outcome = search_enterprise_knowledge_detailed("Query with bad key")
        self.assertEqual(outcome.status, RetrievalStatus.AUTH_FAILURE)

    def test_exception_classifier(self):
        s1, _ = classify_retrieval_exception(Exception("Connection timed out after 30s"))
        self.assertEqual(s1, RetrievalStatus.VECTOR_DB_UNAVAILABLE)

        s2, _ = classify_retrieval_exception(Exception("Forbidden: 403 Access Denied"))
        self.assertEqual(s2, RetrievalStatus.AUTH_FAILURE)

        s3, _ = classify_retrieval_exception(Exception("Wrong vector dimension: expected 3072, got 384"))
        self.assertEqual(s3, RetrievalStatus.CONFIG_MISMATCH)

    @patch("app.services.retrieval.qdrant_service.get_embedding_dim", return_value=384)
    @patch("app.services.retrieval.qdrant_service.get_qdrant_client")
    def test_validate_collection_compatibility_mismatch(self, mock_get_client, mock_get_dim):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_info = MagicMock()
        mock_info.config.params.vectors.size = 3072  # Collection is 3072, active model is 384
        mock_client.get_collection.return_value = mock_info
        mock_get_client.return_value = mock_client

        report = validate_collection_compatibility("test_col")
        self.assertFalse(report["compatible"])
        self.assertIn("Dimension mismatch", report["reason"])

    @patch("app.services.retrieval.qdrant_service.get_embedding_dim", return_value=384)
    @patch("app.services.retrieval.qdrant_service.get_active_embedding_metadata")
    @patch("app.services.retrieval.qdrant_service.get_qdrant_client")
    def test_validate_collection_model_identity_mismatch(self, mock_get_client, mock_meta, mock_dim):
        mock_meta.return_value = {"provider": "local", "model_name": "all-MiniLM-L6-v2"}
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_info = MagicMock()
        mock_info.config.params.vectors.size = 384
        mock_info.points_count = 100
        mock_client.get_collection.return_value = mock_info

        # Sample point was indexed with different model
        mock_point = MagicMock()
        mock_point.payload = {"embedding_model": "bge-small-en-v1.5", "embedding_provider": "local"}
        mock_client.scroll.return_value = ([mock_point], None)
        mock_get_client.return_value = mock_client

        report = validate_collection_compatibility("test_col")
        self.assertFalse(report["compatible"])
        self.assertIn("Embedding model mismatch", report["reason"])
        self.assertIn("bge-small-en-v1.5", report["reason"])

    @patch("app.services.retrieval.qdrant_service.get_embedding_dim", return_value=384)
    @patch("app.services.retrieval.qdrant_service.get_qdrant_client")
    def test_validate_collection_nonexistent(self, mock_get_client, mock_dim):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_get_client.return_value = mock_client

        report = validate_collection_compatibility("missing_col")
        self.assertFalse(report["compatible"])
        self.assertIn("does not exist", report["reason"])


if __name__ == "__main__":
    unittest.main()
