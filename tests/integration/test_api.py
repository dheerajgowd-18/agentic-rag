import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.guardrails.rails import GuardrailOutcome, GuardrailResult


class TestApiIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("embedding_provider", data)

    @patch("app.main.validate_collection_compatibility")
    @patch("app.main.settings.validate_configuration")
    def test_ready_endpoint_success(self, mock_cfg, mock_compat):
        mock_cfg.return_value = {"valid": True, "issues": [], "warnings": []}
        mock_compat.return_value = {"compatible": True, "actual_dimension": 384}

        resp = self.client.get("/api/ready")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["dimension"], 384)

    @patch("app.main.validate_collection_compatibility")
    def test_ready_endpoint_degraded(self, mock_compat):
        mock_compat.return_value = {"compatible": False, "reason": "Collection not found"}

        resp = self.client.get("/api/ready")
        self.assertEqual(resp.status_code, 503)
        data = resp.json()
        self.assertEqual(data["status"], "not_ready")

    @patch("app.main.guard_detailed")
    def test_query_guardrails_blocked(self, mock_guard):
        mock_guard.return_value = GuardrailResult(
            outcome=GuardrailOutcome.BLOCKED,
            content="I can only assist with technical topics.",
            reason="Off-topic query"
        )
        resp = self.client.post("/query", json={"q": "tell me a joke", "thread_id": "test_t1"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "Blocked by guardrails.")
        self.assertIn("only assist with technical topics", data["answer"])
        self.assertEqual(data["execution_metadata"]["guardrail_decision"], "BLOCKED")

    @patch("app.main.guard_detailed")
    @patch("app.main.rag_agent.invoke")
    def test_query_successful_flow(self, mock_invoke, mock_guard):
        mock_guard.return_value = GuardrailResult(
            outcome=GuardrailOutcome.ALLOWED,
            content=None,
            reason="Allowed"
        )
        mock_invoke.return_value = {
            "final_answer": "Redis is configured using redis-pod.yaml [1].",
            "plan": ["Intent: Technical", "Context Retrieved"],
            "status": "Response generated.",
            "documents": [{"source": "redis.txt", "content": "redis doc", "score": 0.9}],
            "execution_metadata": {"planner_intent": "TECHNICAL", "retrieval_executed": True}
        }

        resp = self.client.post("/query", json={"q": "How to start Redis?", "thread_id": "test_t2"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("redis-pod.yaml", data["answer"])
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["source"], "redis.txt")
        self.assertIn("execution_metadata", data)
        self.assertTrue(data["execution_metadata"]["retrieval_executed"])

    @patch("app.main.guard_detailed", side_effect=Exception("Critical connection crash"))
    def test_query_server_error_sanitization(self, mock_guard):
        resp = self.client.post("/query", json={"q": "Crash test", "thread_id": "crash_t"})
        self.assertEqual(resp.status_code, 500)
        data = resp.json()
        # Verify no raw exception traceback or sensitive details leaked
        self.assertNotIn("Critical connection crash", data.get("answer", ""))
        self.assertEqual(data.get("status"), "error")


if __name__ == "__main__":
    unittest.main()
