import os
import time
import threading
import logfire
from app.config import settings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BATCH_SIZE = 32
_GEMINI_DIM = 3072
_LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
_LOCAL_DIM = 384

_init_lock = threading.Lock()
_active_model = None
_model_type: str | None = None  # "gemini" or "local"
_active_model_name: str | None = None
_fallback_used: bool = False

SUPPORTED_PROVIDERS = {"local", "sentence-transformers", "sentence_transformers", "gemini"}


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or raises."""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as ie:
        raise ImportError(
            "langchain-google-genai package is required when EMBEDDING_PROVIDER is 'gemini'. "
            f"Install it or switch to 'local'. Details: {ie}"
        )

    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured but EMBEDDING_PROVIDER is set to 'gemini'.")

    model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
        google_api_key=settings.GEMINI_API_KEY,
    )
    model.embed_query("probe")
    logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
    return model


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
    return model, model_name


def _init():
    """Initialise embedding model once per process in a thread-safe manner. Called lazily on first use."""
    global _active_model, _model_type, _active_model_name, _fallback_used
    if _active_model is not None:
        return

    with _init_lock:
        if _active_model is not None:
            return

        try:
            raw_provider = getattr(settings, "EMBEDDING_PROVIDER", None) or os.getenv("EMBEDDING_PROVIDER") or "local"
            provider = raw_provider.lower().strip()

            if provider not in SUPPORTED_PROVIDERS:
                raise ValueError(
                    f"Unsupported EMBEDDING_PROVIDER '{provider}'. "
                    f"Supported providers are: {sorted(SUPPORTED_PROVIDERS)}"
                )

            if provider in ("local", "sentence-transformers", "sentence_transformers"):
                model, name = _load_fallback()
                _active_model = model
                _model_type = "local"
                _active_model_name = name
                _fallback_used = False
                return

            if provider == "gemini":
                try:
                    gemini = _probe_gemini()
                    _active_model = gemini
                    _model_type = "gemini"
                    _active_model_name = "models/gemini-embedding-2-preview"
                    _fallback_used = False
                except Exception as e:
                    allow_fallback = getattr(settings, "ALLOW_EMBEDDING_FALLBACK", False) or os.getenv("ALLOW_EMBEDDING_FALLBACK", "false").lower() == "true"
                    if allow_fallback:
                        logfire.warning(f"⚠️ Gemini embedding failed ({e}); falling back to local sentence-transformers because ALLOW_EMBEDDING_FALLBACK is enabled.")
                        model, name = _load_fallback()
                        _active_model = model
                        _model_type = "local"
                        _active_model_name = name
                        _fallback_used = True
                    else:
                        logfire.error(f"❌ Gemini embedding initialization failed: {e}")
                        raise RuntimeError(
                            f"Configured EMBEDDING_PROVIDER 'gemini' failed to initialize: {e}. "
                            "Refusing to silently switch embedding models against vector collection. "
                            "Set ALLOW_EMBEDDING_FALLBACK=true to allow automatic fallback."
                        ) from e
        except Exception:
            _active_model = None
            _model_type = None
            _active_model_name = None
            _fallback_used = False
            raise


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _LOCAL_DIM


def get_active_embedding_metadata() -> dict:
    """Return runtime metadata about the active embedding configuration."""
    _init()
    return {
        "provider": _model_type,
        "model_name": _active_model_name,
        "dimension": get_embedding_dim(),
        "fallback_used": _fallback_used,
    }


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


import asyncio

async def embed_query_async(query: str) -> list[float]:
    """Non-blocking async wrapper around embed_query."""
    return await asyncio.to_thread(embed_query, query)



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
