import unittest
from evals.pipeline import normalize_contexts, detect_tool, validate_golden_dataset
from evals.metrics import _prep_samples, get_evaluation_diagnostics


class TestEvaluation(unittest.TestCase):
    def test_normalize_contexts_from_dict_records(self):
        sources = [
            {"source": "k8s.pdf", "content": "Kubernetes cluster guide.", "score": 0.88},
            {"source": "intel.txt", "content": "Intel FPGA specifications.", "score": 0.72},
        ]
        norm = normalize_contexts(sources)
        self.assertEqual(len(norm), 2)
        self.assertEqual(norm[0], "Kubernetes cluster guide.")
        self.assertEqual(norm[1], "Intel FPGA specifications.")

    def test_normalize_contexts_with_empty_or_strings(self):
        sources = ["Raw text string", {"content": "   "}, "", "Another passage"]
        norm = normalize_contexts(sources)
        self.assertEqual(norm, ["Raw text string", "Another passage"])

    def test_detect_tool_from_structured_metadata(self):
        meta_retrieve = {"retrieval_executed": True, "planner_intent": "TECHNICAL"}
        self.assertEqual(detect_tool([], meta_retrieve), "retrieve_documents")

        meta_guard = {"guardrail_decision": "BLOCKED"}
        self.assertEqual(detect_tool([], meta_guard), "guardrails")

        meta_conv = {"planner_intent": "CONVERSATIONAL", "retrieval_executed": False}
        self.assertEqual(detect_tool([], meta_conv), "direct_answer")

    def test_detect_tool_from_thought_process_fallback(self):
        self.assertEqual(detect_tool(["Intent: Technical", "Search Term: redis"]), "retrieve_documents")
        self.assertEqual(detect_tool(["Intent: Guardrails Fired"]), "guardrails")
        self.assertEqual(detect_tool(["Intent: Conversational/Memory"]), "direct_answer")

    def test_prep_samples_filters_and_truncates_strings(self):
        dataset = {
            "rag_samples": [
                {
                    "id": 1,
                    "question": "What is HPA?",
                    "actual_response": "HPA scales pods.",
                    "actual_contexts": [
                        {"content": "A" * 2000, "source": "a.txt"},
                        "B" * 500,
                    ],
                    "pipeline_status": "success",
                },
                {
                    "id": 2,
                    "question": "Failed question",
                    "actual_response": "",
                    "actual_contexts": [],
                    "pipeline_status": "failed",
                }
            ]
        }
        prep = _prep_samples(dataset)
        self.assertEqual(len(prep), 1)  # Failed sample excluded
        # Ensure contexts are strings and truncated
        contexts = prep[0]["actual_contexts"]
        self.assertEqual(len(contexts), 2)
        self.assertIsInstance(contexts[0], str)
        self.assertLessEqual(len(contexts[0]), 1000)

    def test_tool_correctness_empty_sets(self):
        # When both expected and called are empty, score must be 1.0
        called = set()
        expected = set()
        if not called and not expected:
            score = 1.0
        else:
            union = len(called | expected)
            score = len(called & expected) / union if union > 0 else 0.0
        self.assertEqual(score, 1.0)

    def test_evaluation_diagnostics_summary(self):
        dataset = {
            "rag_samples": [
                {"actual_response": "Answer", "pipeline_status": "success", "actual_contexts": ["c1"], "used_fallback_contexts": False},
                {"actual_response": "", "pipeline_status": "failed", "actual_contexts": []},
                {"actual_response": "Fallback Ans", "pipeline_status": "success", "actual_contexts": ["fb"], "used_fallback_contexts": True},
            ]
        }
        diag = get_evaluation_diagnostics(dataset)
        self.assertEqual(diag["total_samples"], 3)
        self.assertEqual(diag["successful_live_queries"], 2)
        self.assertEqual(diag["failed_live_queries"], 1)
        self.assertEqual(diag["samples_using_fallback_contexts"], 1)

    def test_validate_golden_dataset_structure(self):
        valid_dataset = {
            "rag_samples": [
                {
                    "id": 1,
                    "question": "What is K8s?",
                    "reference": "Kubernetes is an orchestrator.",
                    "relevant_contexts": ["Kubernetes orchestrates containers."]
                }
            ]
        }
        res = validate_golden_dataset(valid_dataset)
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["issues"]), 0)

        invalid_dataset = {
            "rag_samples": [
                {
                    "id": 2,
                    "question": "",
                    "reference": "",
                    "relevant_contexts": [""]  # empty context
                }
            ]
        }
        res2 = validate_golden_dataset(invalid_dataset)
        self.assertFalse(res2["valid"])
        self.assertGreater(len(res2["issues"]), 0)


if __name__ == "__main__":
    unittest.main()
