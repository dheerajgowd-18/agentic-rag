import json
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.guardrails.rails import GuardrailOutcome, GuardrailResult
from app.agents.nodes.planner import PlannerDecision


class TestStreamingIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("app.main.guard_async_detailed")
    def test_streaming_guardrails_blocked(self, mock_guard):
        mock_guard.return_value = GuardrailResult(
            outcome=GuardrailOutcome.BLOCKED,
            content="Jailbreak detected and blocked.",
            reason="Safety policy"
        )

        resp = self.client.post("/query/stream", json={"q": "jailbreak prompt", "thread_id": "stream_block"})
        self.assertEqual(resp.status_code, 200)

        events = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        event_types = [e["type"] for e in events]
        self.assertIn("thought", event_types)
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)

        token_event = next(e for e in events if e["type"] == "token")
        self.assertIn("Jailbreak detected", token_event["content"])

    @patch("app.main.guard_async_detailed")
    @patch("app.main.evaluate_planner")
    @patch("app.main.execute_shared_retrieval")
    @patch("app.main.portkey_client.chat.completions.create")
    def test_streaming_technical_query_events(self, mock_create, mock_retrieval, mock_eval, mock_guard):
        mock_guard.return_value = GuardrailResult(
            outcome=GuardrailOutcome.ALLOWED,
            content=None,
            reason="Allowed"
        )
        mock_eval.return_value = PlannerDecision(
            intent="TECHNICAL",
            search_query="Kubernetes Redis",
            reasoning="Tech query"
        )
        mock_retrieval.return_value = (
            [{"source": "doc1.txt", "content": "redis doc"}],
            [{"source": "doc1.txt", "content": "redis doc", "score": 0.9}],
            {"candidate_count": 1, "selected_passage_count": 1}
        )

        # Mock token stream chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Redis is deployed "
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "using redis-pod.yaml [1]."
        mock_create.return_value = iter([chunk1, chunk2])

        resp = self.client.post("/query/stream", json={"q": "How to start redis?", "thread_id": "stream_tech"})
        self.assertEqual(resp.status_code, 200)

        events = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        event_types = [e["type"] for e in events]
        self.assertIn("thought", event_types)
        self.assertIn("sources", event_types)
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)

        sources_event = next(e for e in events if e["type"] == "sources")
        self.assertEqual(len(sources_event["sources"]), 1)
        self.assertEqual(sources_event["sources"][0]["source"], "doc1.txt")

    @patch("starlette.requests.Request.is_disconnected")
    @patch("app.main.guard_async_detailed")
    @patch("app.main.evaluate_planner")
    @patch("app.main.execute_shared_retrieval")
    @patch("app.main.portkey_client.chat.completions.create")
    def test_streaming_client_cancellation_discards_incomplete_turn(
        self, mock_create, mock_retrieval, mock_eval, mock_guard, mock_disconnected
    ):
        from unittest.mock import AsyncMock
        mock_disconnected.side_effect = AsyncMock(return_value=True)
        from app.agents.graph import rag_agent
        cancel_thread = "stream_cancelled_thread"
        cfg = {"configurable": {"thread_id": cancel_thread}}
        # Initialize thread
        rag_agent.update_state(cfg, {"messages": [{"role": "system", "content": "__CLEAR__"}]})

        mock_guard.return_value = GuardrailResult(outcome=GuardrailOutcome.ALLOWED, content=None, reason="Allowed")
        mock_eval.return_value = PlannerDecision(intent="TECHNICAL", search_query="redis", reasoning="tech")
        mock_retrieval.return_value = ([], [], {"candidate_count": 0, "selected_passage_count": 0})

        # Simulate stream that produces one token then disconnects
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Partial answer token"
        mock_stream = MagicMock()
        mock_stream.__iter__.return_value = [chunk1]
        mock_create.return_value = mock_stream

        # Client sends request, disconnection is detected during token streaming
        try:
            resp = self.client.post("/query/stream", json={"q": "Question?", "thread_id": cancel_thread})
            for _ in resp.iter_lines():
                pass
        except Exception:
            pass

        # Verify that conversational memory did NOT commit a partial assistant message
        state = rag_agent.get_state(cfg)
        assistant_msgs = [m for m in state.values.get("messages", []) if m.get("role") == "assistant"]
        self.assertEqual(len(assistant_msgs), 0)
        self.assertEqual(state.values.get("status"), "Interrupted by client cancellation.")


if __name__ == "__main__":
    unittest.main()
