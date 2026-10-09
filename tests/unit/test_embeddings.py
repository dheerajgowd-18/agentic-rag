import unittest
from unittest.mock import patch, MagicMock
from app.services.retrieval import embedding


class TestEmbeddingService(unittest.TestCase):
    def setUp(self):
        # Reset internal globals between tests
        embedding._active_model = None
        embedding._model_type = None
        embedding._active_model_name = None
        embedding._fallback_used = False

    def tearDown(self):
        embedding._active_model = None
        embedding._model_type = None
        embedding._active_model_name = None
        embedding._fallback_used = False

    def test_unsupported_provider_raises_value_error(self):
        with patch.object(embedding.settings, "EMBEDDING_PROVIDER", "invalid_provider"):
            with self.assertRaises(ValueError) as ctx:
                embedding.get_embedding_dim()
            self.assertIn("Unsupported EMBEDDING_PROVIDER", str(ctx.exception))

    def test_local_provider_dimension(self):
        with patch.object(embedding.settings, "EMBEDDING_PROVIDER", "local"):
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
                dim = embedding.get_embedding_dim()
                self.assertEqual(dim, 384)
                meta = embedding.get_active_embedding_metadata()
                self.assertEqual(meta["provider"], "local")
                self.assertFalse(meta["fallback_used"])

    def test_gemini_failure_raises_when_fallback_disabled(self):
        with patch.object(embedding.settings, "EMBEDDING_PROVIDER", "gemini"), \
             patch.object(embedding.settings, "ALLOW_EMBEDDING_FALLBACK", False), \
             patch.object(embedding.settings, "GEMINI_API_KEY", "dummy_key"):
            with patch("app.services.retrieval.embedding._probe_gemini", side_effect=Exception("API Quota Exceeded")):
                with self.assertRaises(RuntimeError) as ctx:
                    embedding.get_embedding_dim()
                self.assertIn("Refusing to silently switch", str(ctx.exception))

    def test_gemini_failure_falls_back_when_explicitly_allowed(self):
        with patch.object(embedding.settings, "EMBEDDING_PROVIDER", "gemini"), \
             patch.object(embedding.settings, "ALLOW_EMBEDDING_FALLBACK", True), \
             patch.object(embedding.settings, "GEMINI_API_KEY", "dummy_key"):
            mock_local = MagicMock()
            mock_local.get_sentence_embedding_dimension.return_value = 384
            with patch("app.services.retrieval.embedding._probe_gemini", side_effect=Exception("API Quota Exceeded")), \
                 patch("sentence_transformers.SentenceTransformer", return_value=mock_local):
                dim = embedding.get_embedding_dim()
                self.assertEqual(dim, 384)
                meta = embedding.get_active_embedding_metadata()
                self.assertTrue(meta["fallback_used"])
                self.assertEqual(meta["provider"], "local")

    def test_clean_state_reset_on_initialization_error(self):
        with patch.object(embedding.settings, "EMBEDDING_PROVIDER", "nonexistent"):
            try:
                embedding.get_embedding_dim()
            except ValueError:
                pass
            # Verify internal globals reset cleanly rather than holding partial state
            self.assertIsNone(embedding._active_model)
            self.assertIsNone(embedding._model_type)


if __name__ == "__main__":
    unittest.main()
