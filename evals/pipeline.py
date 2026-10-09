"""
Phase 1 — Live Pipeline.
Calls the running FastAPI /query endpoint for each golden sample.
Captures: actual_response (truncated to 300 chars), actual_contexts (from sources),
and actual_tools_called (detected from thought_process).
"""


import time
import copy
import json
import os
import requests
import logfire

API_URL = os.getenv("EVAL_API_URL", "http://localhost:8000/query")
RESPONSE_TRUNCATE = int(os.getenv("EVAL_RESPONSE_TRUNCATE", "2000"))
DELAY_BETWEEN_CALLS = int(os.getenv("EVAL_DELAY_BETWEEN_CALLS", "10"))   # seconds — stays within Groq RPM on the main key
REQUEST_TIMEOUT = int(os.getenv("EVAL_REQUEST_TIMEOUT", "120"))      # seconds — guardrails + LangGraph + Groq can take >60s




import uuid

def normalize_contexts(sources: list) -> list[str]:
    """
    Normalizes retrieved source records into plain textual passages expected by RAGAS.
    Handles dicts (extracting 'content' or 'text'), strings, and empty/missing entries.
    """
    if not sources:
        return []
    extracted = []
    for s in sources:
        if isinstance(s, dict):
            text = s.get("content") or s.get("text") or ""
        elif isinstance(s, str):
            text = s
        else:
            text = str(s)
        text = text.strip()
        if text:
            extracted.append(text)
    return extracted


def detect_tool(thought_process: list, execution_metadata: dict = None) -> str:
    """
    Maps execution metadata or thought_process from /query response to a workflow tool name.
    Prioritizes structured execution metadata over free-form string matching.
    """
    if execution_metadata and isinstance(execution_metadata, dict):
        guard_decision = execution_metadata.get("guardrail_decision")
        if guard_decision in ("BLOCKED", "GUARDRAILS_BLOCKED"):
            return "guardrails"
        if execution_metadata.get("retrieval_executed"):
            return "retrieve_documents"
        intent = execution_metadata.get("planner_intent")
        if intent == "CONVERSATIONAL" or execution_metadata.get("retrieval_executed") is False:
            return "direct_answer"

    joined = " ".join(thought_process).lower()
    if "guardrail" in joined or "blocked" in joined:
        return "guardrails"
    if any(k in joined for k in ("intent: technical", "search term:", "context retrieved", "retrieved", "qdrant", "rerank")):
        return "retrieve_documents"
    if any(k in joined for k in ("conversational", "memory", "direct_answer")):
        return "direct_answer"
    return "unknown"


def run_pipeline(golden_dataset: dict, progress_callback=None) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    Returns a deep copy with actual_response, actual_contexts, actual_tools_called filled.
    Uses unique thread IDs per evaluation run to prevent checkpoint pollution.
    Never silently replaces failed live retrieval with reference contexts.
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset.get("rag_samples", [])
    n = len(samples)
    run_id = uuid.uuid4().hex[:8]

    diagnostics = {
        "run_id": run_id,
        "total_samples": n,
        "successful_live_queries": 0,
        "failed_live_queries": 0,
        "samples_with_contexts": 0,
        "samples_empty_contexts": 0,
    }

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n, run_id=run_id):
        for i, sample in enumerate(samples):
            question = sample["question"]
            sample_id = sample.get("id", i + 1)
            isolated_thread_id = f"eval_run_{run_id}_{sample_id}"

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
                thread_id=isolated_thread_id,
            ):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"q": question, "thread_id": isolated_thread_id},
                        timeout=REQUEST_TIMEOUT,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    raw_answer = data.get("answer") or ""
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []
                    exec_meta = data.get("execution_metadata") or {}

                    normalized_ctx = normalize_contexts(sources)

                    sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = normalized_ctx[:5]
                    sample["raw_sources"] = sources[:5]
                    sample["actual_tools_called"] = [detect_tool(thought_process, exec_meta)]
                    sample["execution_metadata"] = exec_meta
                    sample["pipeline_status"] = "success"
                    sample["used_fallback_contexts"] = False

                    diagnostics["successful_live_queries"] += 1
                    if normalized_ctx:
                        diagnostics["samples_with_contexts"] += 1
                    else:
                        diagnostics["samples_empty_contexts"] += 1

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        context_chunks=len(normalized_ctx),
                    )

                except requests.exceptions.ConnectionError as ce:
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = []
                    sample["raw_sources"] = []
                    sample["actual_tools_called"] = ["failed_call"]
                    sample["pipeline_status"] = "failed"
                    sample["error_detail"] = f"ConnectionError: {ce}"
                    sample["used_fallback_contexts"] = False
                    diagnostics["failed_live_queries"] += 1

                except Exception as e:
                    logfire.error(f"❌ Query failed: {e}")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = []
                    sample["raw_sources"] = []
                    sample["actual_tools_called"] = ["failed_call"]
                    sample["pipeline_status"] = "failed"
                    sample["error_detail"] = str(e)
                    sample["used_fallback_contexts"] = False
                    diagnostics["failed_live_queries"] += 1

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    dataset["pipeline_diagnostics"] = diagnostics
    return dataset


def validate_golden_dataset(dataset: dict) -> dict:
    """
    Validates structure of golden dataset, verifying reference contexts,
    questions, references, and expected tools. Reports invalid or malformed samples.
    """
    if not isinstance(dataset, dict) or "rag_samples" not in dataset:
        raise ValueError("Golden dataset must be a dictionary containing 'rag_samples'.")

    samples = dataset.get("rag_samples", [])
    issues = []

    for i, s in enumerate(samples):
        sample_id = s.get("id", i + 1)
        if not s.get("question") or not str(s.get("question")).strip():
            issues.append(f"Sample {sample_id}: missing or empty 'question'")
        if not s.get("reference") or not str(s.get("reference")).strip():
            issues.append(f"Sample {sample_id}: missing or empty 'reference'")

        ref_ctxs = s.get("relevant_contexts") or s.get("reference_contexts") or []
        if not isinstance(ref_ctxs, list):
            issues.append(f"Sample {sample_id}: reference contexts must be a list of strings")
        else:
            for c_idx, c in enumerate(ref_ctxs):
                if not isinstance(c, str) or not c.strip():
                    issues.append(f"Sample {sample_id}: context #{c_idx} is not a valid non-empty string")

    return {
        "valid": len(issues) == 0,
        "total_samples": len(samples),
        "issues": issues,
    }


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)


def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(golden_path, encoding="utf-8") as f:
        data = json.load(f)
    validation = validate_golden_dataset(data)
    if not validation["valid"]:
        logfire.warning(f"Golden dataset validation issues detected: {validation['issues']}")
    return data
