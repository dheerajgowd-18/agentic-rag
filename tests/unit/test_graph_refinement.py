import unittest
from app.agents.graph import (
    simplify_search_query,
    context_grader_node,
    query_rewriter_node,
    route_grader,
    route_planner,
    get_active_persistence_mode,
    MAX_RETRIEVAL_RETRIES,
)


class TestGraphRefinement(unittest.TestCase):
    def test_simplify_search_query_stop_word_stripping(self):
        query = "How do you start Redis for a Kubernetes work queue?"
        simplified = simplify_search_query(query)
        self.assertIn("Redis", simplified)
        self.assertIn("Kubernetes", simplified)
        self.assertNotIn("How", simplified.split())
        self.assertNotIn("you", simplified.split())
        self.assertNotIn("for", simplified.split())

    def test_simplify_search_query_empty_and_fallback(self):
        self.assertEqual(simplify_search_query(""), "")
        # Query with only stop words preserves tokens up to limit
        res = simplify_search_query("how do you do")
        self.assertTrue(len(res) > 0)

    def test_context_grader_with_substantive_documents(self):
        state = {
            "documents": [{"content": "Redis pod configuration instructions.", "source": "k8s.txt"}],
            "retry_count": 0,
            "plan": ["Retrieve"],
            "execution_metadata": {},
        }
        res = context_grader_node(state)
        self.assertEqual(res["execution_metadata"]["grader_outcome"], "sufficient")
        self.assertIn("Sufficient evidence", res["plan"][-1])

    def test_context_grader_insufficient_documents_triggers_retry(self):
        state = {
            "documents": [],
            "retry_count": 0,
            "plan": ["Retrieve"],
            "execution_metadata": {},
        }
        res = context_grader_node(state)
        self.assertEqual(res["execution_metadata"]["grader_outcome"], "retry_needed")
        self.assertIn("initiating rewrite", res["plan"][-1])

    def test_context_grader_exhausted_at_max_retries(self):
        state = {
            "documents": [],
            "retry_count": MAX_RETRIEVAL_RETRIES,
            "plan": ["Retrieve"],
            "execution_metadata": {},
        }
        res = context_grader_node(state)
        self.assertEqual(res["execution_metadata"]["grader_outcome"], "exhausted")
        self.assertIn("Max retries reached", res["plan"][-1])

    def test_query_rewriter_increments_retry_count(self):
        state = {
            "current_query": "How do I configure HPA in Kubernetes?",
            "retry_count": 0,
            "plan": ["Retrieve"],
            "execution_metadata": {},
        }
        res = query_rewriter_node(state)
        self.assertEqual(res["retry_count"], 1)
        self.assertTrue(res["execution_metadata"]["retry_executed"])
        self.assertEqual(res["current_query"], res["execution_metadata"]["rewritten_query"])
        self.assertIn("configure HPA Kubernetes", res["current_query"])

    def test_route_grader_routing_decisions(self):
        # Case 1: Retry needed with retry_count=0 -> query_rewriter
        state_retry = {"execution_metadata": {"grader_outcome": "retry_needed"}, "retry_count": 0}
        self.assertEqual(route_grader(state_retry), "query_rewriter")

        # Case 2: Retry needed but retry_count at max -> responder
        state_exhausted = {"execution_metadata": {"grader_outcome": "retry_needed"}, "retry_count": MAX_RETRIEVAL_RETRIES}
        self.assertEqual(route_grader(state_exhausted), "responder")

        # Case 3: Sufficient evidence -> responder
        state_sufficient = {"execution_metadata": {"grader_outcome": "sufficient"}, "retry_count": 0}
        self.assertEqual(route_grader(state_sufficient), "responder")

    def test_route_planner(self):
        self.assertEqual(route_planner({"current_query": "CONVERSATIONAL"}), "responder")
        self.assertEqual(route_planner({"current_query": "Kubernetes Pod"}), "retriever")

    def test_active_persistence_mode_reporting(self):
        mode = get_active_persistence_mode()
        self.assertIn(mode, ("memory", "sqlite", "memory (degraded fallback)"))


if __name__ == "__main__":
    unittest.main()
