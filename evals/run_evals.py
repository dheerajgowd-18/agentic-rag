import os
import sys
import json
import argparse
import asyncio

from evals import run_pipeline, load_golden_dataset, run_guardrails_eval, compute_guardrails_metrics, run_all_metrics
from evals.pipeline import save_results

def main():
    parser = argparse.ArgumentParser(description="CLI Evaluation Suite for Enterprise Agentic RAG")
    parser.add_argument("--mode", choices=["all", "pipeline", "guardrails", "metrics"], default="guardrails",
                        help="Eval mode: pipeline, guardrails, metrics, or all (default: guardrails)")
    parser.add_argument("--output", default="evals/eval_results.json", help="Path to save evaluation results")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Enterprise Agentic RAG — CLI Evaluation Suite")
    print("=" * 60)

    dataset = load_golden_dataset()
    results = {}

    if args.mode in ("guardrails", "all"):
        print("\n🛡️ Running Guardrails Binary Evaluation...")
        gr_samples = dataset.get("guardrails_samples", [])
        gr_results = run_guardrails_eval(gr_samples, progress_callback=lambda i, n, inp: print(f"  [{i+1}/{n}] Testing: {inp[:60]}..."))
        gr_metrics = compute_guardrails_metrics(gr_results)
        results["guardrails"] = {
            "metrics": gr_metrics,
            "results": gr_results
        }
        print("\n🛡️ Guardrails Summary:")
        print(f"  Accuracy:  {gr_metrics['accuracy'] * 100:.1f}%")
        print(f"  Precision: {gr_metrics['precision'] * 100:.1f}%")
        print(f"  Recall:    {gr_metrics['recall'] * 100:.1f}%")
        print(f"  Total:     {gr_metrics['total']} (Correct: {gr_metrics['correct']})")

    if args.mode in ("pipeline", "all"):
        print("\n⚡ Running Phase 1 — Live Pipeline against /query endpoint...")
        pipeline_dataset = run_pipeline(dataset, progress_callback=lambda i, n, q, s, r='': print(f"  [{i+1}/{n}] {s.upper()}: {q[:60]}"))
        results["pipeline"] = pipeline_dataset
        save_results(pipeline_dataset, args.output)
        print(f"✅ Pipeline run complete. Enriched dataset saved to {args.output}")

    if args.mode in ("metrics", "all"):
        print("\n🧪 Running Phase 2 — RAGAS Metrics...")
        enriched = results.get("pipeline") or dataset
        try:
            metrics_results = asyncio.run(run_all_metrics(enriched, status_cb=lambda msg: print(f"  {msg}")))
            results["metrics"] = {k: v.to_dict() for k, v in metrics_results.items()}
            print("\n🧪 RAGAS Metrics Summary:")
            for m_name, df in metrics_results.items():
                if m_name in df.columns:
                    print(f"  {m_name}: {df[m_name].mean():.3f}")
        except Exception as e:
            print(f"❌ Metrics evaluation error: {e}")

    # Save final aggregated results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📊 Complete evaluation results saved to: {args.output}")

if __name__ == "__main__":
    main()