import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    # --- EMBEDDINGS (LOCAL / GEMINI) ---
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    ALLOW_EMBEDDING_FALLBACK: bool = os.getenv("ALLOW_EMBEDDING_FALLBACK", "false").lower() == "true"

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL: str | None = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "enterprise_rag")
    QDRANT_SCORE_THRESHOLD: float = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.30"))

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
    GROQ_FALLBACK_API_KEY: str | None = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_FALLBACK_MODEL: str = os.getenv("GROQ_FALLBACK_MODEL") or "openai/gpt-oss-20b"
    GROQ_GUARD_MODEL: str = os.getenv("GROQ_GUARD_MODEL") or "openai/gpt-oss-20b"

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY: str | None = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG: str = os.getenv("PORTKEY_GROQ_SLUG") or "rag"
    GROQ_SLUG_2: str = os.getenv("PORTKEY_GROQ_SLUG_2") or "brag"
    PORTKEY_CONFIG_ID: str | None = os.getenv("PORTKEY_CONFIG_ID")

    # --- GUARDRAILS ---
    GUARDRAILS_FAIL_CLOSED: bool = os.getenv("GUARDRAILS_FAIL_CLOSED", "true").lower() == "true"

    # --- SESSION PERSISTENCE ---
    CHECKPOINT_PERSISTENCE: str = os.getenv("CHECKPOINT_PERSISTENCE", "memory").lower()
    CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "checkpoints.sqlite")
    CHECKPOINT_PERSISTENCE_REQUIRED: bool = os.getenv("CHECKPOINT_PERSISTENCE_REQUIRED", "false").lower() == "true"

    # --- SECURITY & CORS ---
    # Explicit allowed origins; never wildcard with credentials
    CORS_ALLOWED_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000"
        ).split(",")
        if origin.strip()
    ]

    # --- OBSERVABILITY (OPTIONAL) ---
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    @classmethod
    def mask_secret(cls, secret: str | None) -> str:
        """Helper to mask secret keys without leaking values in logs."""
        if not secret:
            return "(not set)"
        if len(secret) <= 8:
            return "***"
        return f"{secret[:4]}...{secret[-4:]}"

    @classmethod
    def validate_configuration(cls) -> Dict[str, Any]:
        """
        Validates core configuration for startup and readiness checks.
        Differentiates required settings from optional integrations.
        """
        issues = []
        warnings = []

        # Vector DB
        if not cls.QDRANT_URL:
            issues.append("QDRANT_CLUSTER_ENDPOINT is not configured.")
        if not cls.QDRANT_API_KEY:
            issues.append("QDRANT_API_KEY is not configured.")

        # Embedding Provider
        if cls.EMBEDDING_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            issues.append("EMBEDDING_PROVIDER is 'gemini' but GEMINI_API_KEY is missing.")

        # Reasoning & Gateway
        if not cls.PORTKEY_API_KEY and not cls.GROQ_API_KEY:
            issues.append("Neither PORTKEY_API_KEY nor GROQ_API_KEY is configured.")

        # Optional Observability
        if cls.LANGSMITH_TRACING.lower() == "true" and not cls.LANGSMITH_API_KEY:
            warnings.append("LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not provided.")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "embedding_provider": cls.EMBEDDING_PROVIDER,
            "checkpoint_mode": cls.CHECKPOINT_PERSISTENCE,
            "qdrant_configured": bool(cls.QDRANT_URL and cls.QDRANT_API_KEY),
            "portkey_configured": bool(cls.PORTKEY_API_KEY),
        }


# Apply LangChain environment variables for automatic tracing if enabled
if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()
