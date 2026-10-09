"""
Phase 2 — RAGAS + Tool Correctness metrics.
Uses JUDGE_GROQ key so production GROQ_API_KEY is never exhausted by eval runs.
All LLM-based metrics run in batches of 5 with 30s cooldowns between sub-batches
and 60s cooldowns between experiments — calibrated for Groq's 6,000 TPM on_demand tier.
Contexts are truncated to 300 chars (2 chunks max) so no single request exceeds the limit.
"""


import os
import asyncio
import logfire
import pandas as pd
from openai import AsyncOpenAI


# Ragas imports are lazily loaded inside _build_judge and run_all_metrics
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or "openai/gpt-oss-20b"
COOLDOWN_STANDARD = int(os.getenv("EVAL_COOLDOWN_STANDARD", "62"))
COOLDOWN_MINI = int(os.getenv("EVAL_COOLDOWN_MINI", "40"))       # between individual samples — lets sliding TPM window recover (~2,800 tok/sample)
GENERAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "1"))  # one sample at a time: abatch_score fires calls concurrently per sample,
                         # so batch>1 stacks multiple samples' async calls inside the same second
CONTEXT_TRUNCATE = int(os.getenv("EVAL_CONTEXT_TRUNCATE", "1000"))  # chars per context chunk — configurable for high-fidelity evaluation
CONTEXT_LIMIT = int(os.getenv("EVAL_CONTEXT_LIMIT", "3"))       # number of context chunks passed to RAGAS per sample


def _build_judge():
    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings

    api_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    llm = llm_factory(JUDGE_MODEL, provider="openai", client=client)
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )
    return llm, embeddings

async def _cooldown(seconds: int, label: str, status_cb=None):
    msg = f"⏳ {seconds}s cooldown after {label} (Groq TPM buffer)..."
    if status_cb:
        status_cb(msg)
    for _ in range(seconds // 10):
        await asyncio.sleep(10)
    if status_cb:
        status_cb(f"✅ Ready — starting next experiment.")
        
        
def get_evaluation_diagnostics(golden_dataset: dict) -> dict:
    """
    Summarizes diagnostics for an evaluation dataset.
    Reports total samples, successes, failures, missing contexts, exclusions, and fallback usage.
    """
    samples = golden_dataset.get("rag_samples", [])
    total = len(samples)
    successful = sum(1 for s in samples if s.get("actual_response", "").strip() and s.get("pipeline_status") != "failed")
    failed = sum(1 for s in samples if s.get("pipeline_status") == "failed" or not s.get("actual_response", "").strip())
    with_contexts = sum(1 for s in samples if s.get("actual_contexts"))
    empty_contexts = sum(1 for s in samples if not s.get("actual_contexts"))
    fallback_contexts = sum(1 for s in samples if s.get("used_fallback_contexts"))
    excluded = total - successful

    return {
        "total_samples": total,
        "successful_live_queries": successful,
        "failed_live_queries": failed,
        "samples_with_contexts": with_contexts,
        "samples_empty_contexts": empty_contexts,
        "samples_using_fallback_contexts": fallback_contexts,
        "excluded_from_metrics": excluded,
    }


def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns only samples with actual_response populated and not failed.
    Normalizes contexts to text strings, truncates to CONTEXT_TRUNCATE chars and limits to CONTEXT_LIMIT chunks
    so a single RAGAS LLM call stays well under the 6,000 TPM ceiling.
    Does not substitute reference contexts for failed live retrieval.
    """
    valid = []
    for s in golden_dataset.get("rag_samples", []):
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        if s.get("pipeline_status") == "failed":
            continue
        raw_contexts = s.get("actual_contexts")
        if raw_contexts is None:
            raw_contexts = []
        contexts = []
        for c in raw_contexts[:CONTEXT_LIMIT]:
            if isinstance(c, dict):
                text = c.get("content") or c.get("text") or ""
            elif isinstance(c, str):
                text = c
            else:
                text = str(c)
            text = text.strip()
            if text:
                contexts.append(text[:CONTEXT_TRUNCATE])
        valid.append({**s, "actual_contexts": contexts})
    return valid


def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    return pd.DataFrame([
        {"question": s["question"][:65], metric_key: round(float(r.value), 3)}
        for s, r in zip(samples, scores)
    ])


async def _batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = "") -> list:
    """
    Runs abatch_score in chunks of GENERAL_BATCH_SIZE with cooldowns between chunks.
    Keeps each burst under 6,000 TPM on Groq's on_demand tier.
    """
    all_scores = []
    batches = [inputs[i : i + GENERAL_BATCH_SIZE] for i in range(0, len(inputs), GENERAL_BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        if b_idx > 0:
            await _cooldown(COOLDOWN_MINI, f"{label} batch {b_idx}", status_cb)
        scores = await metric.abatch_score(batch)
        all_scores.extend(scores)
    return all_scores

async def run_all_metrics(golden_dataset: dict, status_cb=None) -> dict:
    """
    Runs all 6 experiments. Returns dict keyed by metric name → DataFrame.
    status_cb(message: str) is called for live UI updates.
    """
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
    )
    judge_llm, ragas_embeddings = _build_judge()
    samples = _prep_samples(golden_dataset)

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    results = {}

    with logfire.span("🧪 Eval Phase 2 — All Metrics", total_samples=len(samples)):

        # ── Exp 1: Faithfulness ───────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 1/6 — Faithfulness ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 1 — Faithfulness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(Faithfulness(llm=judge_llm), inputs, samples, status_cb, "Faithfulness")
            df = _score_df("faithfulness", samples, scores)
            results["faithfulness"] = df
            logfire.info("🧪 Faithfulness done", avg=round(df["faithfulness"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Faithfulness", status_cb)

        # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 2/6 — Answer Relevancy ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 2 — Answer Relevancy"):
            inputs = [
                {"user_input": s["question"], "response": s["actual_response"]}
                for s in samples
            ]
            scores = await _batched_score(
                AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Relevancy"
            )
            df = _score_df("answer_relevancy", samples, scores)
            results["answer_relevancy"] = df
            logfire.info("🧪 Answer Relevancy done", avg=round(df["answer_relevancy"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Answer Relevancy", status_cb)

        # ── Exp 3: Context Precision ──────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 3/6 — Context Precision ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 3 — Context Precision"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(ContextPrecision(llm=judge_llm), inputs, samples, status_cb, "Context Precision")
            df = _score_df("context_precision", samples, scores)
            results["context_precision"] = df
            logfire.info("🧪 Context Precision done", avg=round(df["context_precision"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Precision", status_cb)

        # ── Exp 4: Context Recall ─────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 4/6 — Context Recall ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 4 — Context Recall"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(ContextRecall(llm=judge_llm), inputs, samples, status_cb, "Context Recall")
            df = _score_df("context_recall", samples, scores)
            results["context_recall"] = df
            logfire.info("🧪 Context Recall done", avg=round(df["context_recall"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Recall", status_cb)

        # ── Exp 5: Answer Correctness (split into batches) ────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 5/6 — Answer Correctness batch 1/2...")
        with logfire.span("🧪 Exp 5 — Answer Correctness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "reference": s["reference"],
                }
                for s in samples
            ]
            all_scores = await _batched_score(
                AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Correctness"
            )
            df = _score_df("answer_correctness", samples, all_scores)
            results["answer_correctness"] = df
            logfire.info("🧪 Answer Correctness done", avg=round(df["answer_correctness"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Answer Correctness", status_cb)

        # ── Exp 6: Workflow Tool / Routing Correctness (Jaccard similarity) ───
        if status_cb:
            status_cb("⚡ Exp 6/6 — Workflow Execution Correctness (zero LLM calls)...")
        with logfire.span("🧪 Exp 6 — Workflow Routing Correctness"):
            tool_rows = []
            for s in samples:
                called = set(s.get("actual_tools_called") or [])
                expected = set(s.get("expected_tools") or [])
                if not called and not expected:
                    score = 1.0  # Agreement: neither requires tools
                else:
                    union = len(called | expected)
                    score = len(called & expected) / union if union > 0 else 0.0
                tool_rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
            df = pd.DataFrame(tool_rows)
            results["tool_correctness"] = df
            logfire.info("🧪 Tool Correctness done", avg=round(df["tool_correctness"].mean(), 3))

        if status_cb:
            status_cb("✅ All 6 experiments complete!")

    return results
