from evals.pipeline import run_pipeline, load_golden_dataset
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics

def run_all_metrics(*args, **kwargs):
    import sys
    try:
        import langchain_google_vertexai
        sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai
    except Exception:
        pass
    from evals.metrics import run_all_metrics as _ram
    return _ram(*args, **kwargs)
