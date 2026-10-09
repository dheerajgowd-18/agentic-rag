import unittest
from unittest.mock import patch, MagicMock
from app.agents.nodes.planner import (
    parse_planner_decision,
    evaluate_planner,
    planner_node,
    PlannerDecision,
)


class TestPlanner(unittest.TestCase):
    def test_json_parsing_conversational(self):
        json_output = '{"intent": "CONVERSATIONAL", "search_query": "", "reasoning": "Greeting"}'
        decision = parse_planner_decision(json_output, "hello")
        self.assertEqual(decision.intent, "CONVERSATIONAL")
        self.assertEqual(decision.search_query, "")

    def test_json_parsing_technical(self):
        json_output = '{"intent": "TECHNICAL", "search_query": "Kubernetes worker queue Redis", "reasoning": "Technical query"}'
        decision = parse_planner_decision(json_output, "how do I run redis wq?")
        self.assertEqual(decision.intent, "TECHNICAL")
        self.assertEqual(decision.search_query, "Kubernetes worker queue Redis")

    def test_technical_empty_query_fallback(self):
        # When model says TECHNICAL but provides empty search query, it should fallback to latest user message
        json_output = '{"intent": "TECHNICAL", "search_query": "  ", "reasoning": "Technical"}'
        decision = parse_planner_decision(json_output, "How to configure SR-IOV on Intel NIC?")
        self.assertEqual(decision.intent, "TECHNICAL")
        self.assertEqual(decision.search_query, "How to configure SR-IOV on Intel NIC?")

    def test_plain_text_conversational_fallback(self):
        decision = parse_planner_decision("CONVERSATIONAL", "hi there")
        self.assertEqual(decision.intent, "CONVERSATIONAL")
        self.assertEqual(decision.search_query, "")

    def test_plain_text_query_fallback(self):
        decision = parse_planner_decision("Refined query: BGP routing configuration Linux", "tell me about bgp")
        self.assertEqual(decision.intent, "TECHNICAL")
        self.assertEqual(decision.search_query, "BGP routing configuration Linux")

    @patch("app.agents.nodes.planner.llm")
    def test_evaluate_planner_with_mocked_llm(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = '{"intent": "TECHNICAL", "search_query": "Intel FPGA OpenCL", "reasoning": "Hardware question"}'
        mock_llm.invoke.return_value = mock_response

        messages = [
            {"role": "user", "content": "How do I program Intel FPGAs?"}
        ]
        decision = evaluate_planner(messages)
        self.assertEqual(decision.intent, "TECHNICAL")
        self.assertEqual(decision.search_query, "Intel FPGA OpenCL")

    @patch("app.agents.nodes.planner.evaluate_planner")
    def test_planner_node_state_update(self, mock_eval):
        mock_eval.return_value = PlannerDecision(
            intent="TECHNICAL",
            search_query="Kubernetes HPA scaling",
            reasoning="Autoscaling query"
        )
        state = {"messages": [{"role": "user", "content": "How does HPA scale pods?"}]}
        result = planner_node(state)

        self.assertEqual(result["current_query"], "Kubernetes HPA scaling")
        self.assertEqual(result["execution_metadata"]["planner_intent"], "TECHNICAL")
        self.assertTrue(result["execution_metadata"]["retrieval_executed"])

    @patch("app.agents.nodes.planner.llm")
    def test_multi_turn_technical_follow_up_classified_technical(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = '{"intent": "TECHNICAL", "search_query": "Kubernetes HPA custom metrics CPU memory", "reasoning": "Technical follow up"}'
        mock_llm.invoke.return_value = mock_response

        messages = [
            {"role": "user", "content": "How do I configure Horizontal Pod Autoscaler?"},
            {"role": "assistant", "content": "You configure HPA via the autoscaling/v2 API."},
            {"role": "user", "content": "What metrics does it support?"}
        ]
        decision = evaluate_planner(messages)
        self.assertEqual(decision.intent, "TECHNICAL")
        self.assertIn("metrics", decision.search_query.lower())

    def test_malformed_json_fallback(self):
        # Broken JSON string should not crash; falls back to user query
        broken_json = '{"intent": "TECHNICAL", "search_query": '
        decision = parse_planner_decision(broken_json, "How to configure BGP on Linux?")
        self.assertEqual(decision.intent, "TECHNICAL")
        self.assertEqual(decision.search_query, "How to configure BGP on Linux?")

    def test_null_json_fields(self):
        null_json = '{"intent": "TECHNICAL", "search_query": null, "reasoning": null}'
        decision = parse_planner_decision(null_json, "How to tune Linux sysctl?")
        self.assertEqual(decision.intent, "TECHNICAL")
        self.assertEqual(decision.search_query, "How to tune Linux sysctl?")

    def test_empty_query_and_empty_user_message(self):
        decision = parse_planner_decision("", "")
        self.assertEqual(decision.intent, "CONVERSATIONAL")
        self.assertEqual(decision.search_query, "")


if __name__ == "__main__":
    unittest.main()

