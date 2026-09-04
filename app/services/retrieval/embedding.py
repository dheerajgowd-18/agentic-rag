import os
import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BATCH_SIZE = 32
_GEMINI_DIM = 3072
_LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
_LOCAL_DIM = 384

_active_model = None
_model_type: str | None = None  # "gemini" or "fallback"


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. Will use sentence-transformers fallback.")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    model_name = getattr(settings, "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    logfire.info(f"Loading local sentence-transformers ({model_name}).")
    model = SentenceTransformer(model_name)
    global _LOCAL_DIM
    try:
        _LOCAL_DIM = model.get_sentence_embedding_dimension()
    except Exception:
        _LOCAL_DIM = 384
    return model


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    provider = (getattr(settings, "EMBEDDING_PROVIDER", None) or os.getenv("EMBEDDING_PROVIDER") or "local").lower().strip()
    if provider in ("local", "sentence-transformers", "sentence_transformers"):
        _active_model = _load_fallback()
        _model_type = "fallback"
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _LOCAL_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        # Backoff: 10s → 20s → 30s → 40s to allow Gemini's 60s sliding quota window to recover
        for attempt in range(5):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < 4:
                    wait = 10 * (attempt + 1)
                    logfire.warning(
                        f"Gemini rate limit hit — waiting {wait}s for quota recovery "
                        f"(attempt {attempt + 1}/5)..."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 5 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
        if _model_type == "gemini" and (i + BATCH_SIZE) < len(texts):
            # Polite pause to stay within Gemini 15 RPM free-tier ceiling
            time.sleep(2)
    return all_embeddings
