"""
Guardrails binary evaluation.
Sends each test input to the live /query API and checks if the guardrail fired.
Classifies each result as TP / TN / FP / FN and computes precision + recall.
"""


import time
import copy
import requests
import logfire

API_URL = "http://localhost:8000/query"


def _is_blocked(response_json: dict) -> bool:
    """Detects guardrail block decision from structured metadata, status, or thought_process."""
    exec_meta = response_json.get("execution_metadata") or {}
    guard_decision = exec_meta.get("guardrail_decision")
    if guard_decision in ("BLOCKED", "GUARDRAILS_BLOCKED"):
        return True
    if guard_decision == "ALLOWED":
        return False
    status_str = (response_json.get("status") or "").lower()
    if "blocked by guardrails" in status_str:
        return True
    tp = response_json.get("thought_process") or []
    return any("guardrail" in step.lower() and "fired" in step.lower() for step in tp)


def run_guardrails_eval(guardrails_samples: list, progress_callback=None) -> list:
    """
    Runs each guardrails test case against the live API.
    Adds actual_blocked and result (TP/TN/FP/FN/ERROR) to each sample in place.
    Returns the enriched list.
    """
    samples = copy.deepcopy(guardrails_samples)
    n = len(samples)

    with logfire.span("🛡️ Eval — Guardrails Tests", total=n):
        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i, n, sample["input"])

            with logfire.span(
                f"🛡️ Test {sample['id']}",
                input_text=sample["input"][:80],
                expected_blocked=sample["expected_blocked"],
            ):
                blocked = None
                has_error = False

                try:
                    resp = requests.post(
                        API_URL,
                        json={"q": sample["input"], "thread_id": f"guardrail_eval_{i}"},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    blocked = _is_blocked(resp.json())

                except requests.exceptions.ConnectionError as ce:
                    logfire.error(f"❌ Cannot reach FastAPI — is the app running on :8000? ({ce})")
                    has_error = True
                    sample["error_detail"] = f"ConnectionError: {ce}"

                except Exception as e:
                    logfire.error(f"❌ Guardrails test error: {e}")
                    has_error = True
                    sample["error_detail"] = str(e)

                expected = sample["expected_blocked"]
                sample["actual_blocked"] = blocked

                if has_error:
                    sample["result"] = "ERROR"
                elif expected and blocked:
                    sample["result"] = "TP"
                elif expected and not blocked:
                    sample["result"] = "FN"
                elif not expected and not blocked:
                    sample["result"] = "TN"
                else:
                    sample["result"] = "FP"

                logfire.info(
                    f"🛡️ {sample['result']}",
                    expected_blocked=expected,
                    actual_blocked=blocked,
                    input_preview=sample["input"][:60],
                )

            time.sleep(2)

    return samples


def compute_guardrails_metrics(results: list) -> dict:
    valid_results = [r for r in results if r.get("result") in ("TP", "TN", "FP", "FN")]
    errors = sum(1 for r in results if r.get("result") == "ERROR")

    tp = sum(1 for r in valid_results if r["result"] == "TP")
    tn = sum(1 for r in valid_results if r["result"] == "TN")
    fp = sum(1 for r in valid_results if r["result"] == "FP")
    fn = sum(1 for r in valid_results if r["result"] == "FN")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / len(valid_results) if valid_results else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "total": len(results),
        "valid_evaluations": len(valid_results),
        "errors": errors,
        "correct": tp + tn,
    }


