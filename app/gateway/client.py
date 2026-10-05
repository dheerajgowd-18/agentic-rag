import os
import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# Production gateway config:
#   - Fallback: primary @rag/openai/gpt-oss-120b → @brag/openai/gpt-oss-20b on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/{settings.GROQ_FALLBACK_MODEL}"}},
    ]
}

PORTKEY_CONFIG_ID = getattr(settings, "PORTKEY_CONFIG_ID", None) or os.getenv("PORTKEY_CONFIG_ID")

# Determine active gateway configuration (remote ID takes precedence over local dict)
ACTIVE_GATEWAY_CONFIG = PORTKEY_CONFIG_ID or GATEWAY_CONFIG

try:
    portkey_client = Portkey(
        api_key=settings.PORTKEY_API_KEY,
        config=ACTIVE_GATEWAY_CONFIG
    )
    logfire.info("🌐 Portkey gateway initialized with fallback & retry policies.")
except Exception as e:
    logfire.warning(f"⚠️ Portkey initialization with config failed ({e}), falling back to standard client.")
    portkey_client = Portkey(api_key=settings.PORTKEY_API_KEY)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.
    Inherits retry, caching, and fallback policies via Portkey headers.
    """
    header_kwargs = {
        "api_key": settings.PORTKEY_API_KEY,
        "config": ACTIVE_GATEWAY_CONFIG,
        "metadata": {
            "feature": feature,
            "_user": "rag-system",
            "environment": "production"
        }
    }

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
        temperature=0,
        default_headers=createHeaders(**header_kwargs)
    )


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"