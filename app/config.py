import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # --- EMBEDDINGS (LOCAL / GEMINI) ---
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL") or "openai/gpt-oss-20b"
    GROQ_GUARD_MODEL = os.getenv("GROQ_GUARD_MODEL") or "openai/gpt-oss-20b"

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = os.getenv("PORTKEY_GROQ_SLUG") or "rag"       # primary slug in Portkey
    GROQ_SLUG_2 = os.getenv("PORTKEY_GROQ_SLUG_2") or "brag"  # fallback slug in Portkey
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")

    # --- SESSION PERSISTENCE ---
    CHECKPOINT_PERSISTENCE = os.getenv("CHECKPOINT_PERSISTENCE", "memory").lower()
    CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "checkpoints.sqlite")

    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()
