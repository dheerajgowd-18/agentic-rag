import os
import logfire
from enum import Enum
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_query, get_embedding_dim, get_active_embedding_metadata


class RetrievalStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_RESULTS = "NO_RESULTS"
    VECTOR_DB_UNAVAILABLE = "VECTOR_DB_UNAVAILABLE"
    AUTH_FAILURE = "AUTH_FAILURE"
    EMBEDDING_FAILURE = "EMBEDDING_FAILURE"
    CONFIG_MISMATCH = "CONFIG_MISMATCH"
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"


class RetrievalOutcome:
    """
    Represents the explicit outcome of a retrieval attempt.
    Distinguishes legitimate zero-match queries from infrastructure,
    authentication, configuration, and embedding failures.
    """
    def __init__(
        self,
        status: RetrievalStatus,
        documents: List[Dict[str, Any]],
        query: str,
        total_candidates: int = 0,
        error_message: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ):
        self.status = status
        self.documents = documents
        self.query = query
        self.total_candidates = total_candidates
        self.error_message = error_message
        self.score_threshold = score_threshold

    @property
    def is_success(self) -> bool:
        return self.status in (RetrievalStatus.SUCCESS, RetrievalStatus.NO_RESULTS)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "documents": self.documents,
            "query": self.query,
            "total_candidates": self.total_candidates,
            "error_message": self.error_message,
            "score_threshold": self.score_threshold,
        }

    def __repr__(self) -> str:
        return f"<RetrievalOutcome status={self.status.value} docs={len(self.documents)} candidates={self.total_candidates}>"


# Client getter with lazy initialization support
_client_instance = None

def get_qdrant_client() -> QdrantClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    return _client_instance

# Module-level client proxy for existing imports
class _ClientProxy:
    def __getattr__(self, name):
        return getattr(get_qdrant_client(), name)

client = _ClientProxy()

DEFAULT_SCORE_THRESHOLD = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.30"))


def classify_retrieval_exception(err: Exception) -> tuple[RetrievalStatus, str]:
    """Classifies exceptions into typed retrieval statuses without leaking sensitive details."""
    err_msg = str(err)
    err_lower = err_msg.lower()

    if any(k in err_lower for k in ("unauthorized", "forbidden", "401", "403", "api key", "authentication")):
        return RetrievalStatus.AUTH_FAILURE, "Vector database authentication failed."
    if any(k in err_lower for k in ("connection refused", "timeout", "timed out", "connect", "endpoint", "network", "failed to connect", "unreachable", "getaddrinfo")):
        return RetrievalStatus.VECTOR_DB_UNAVAILABLE, "Vector database service is unreachable or timed out."
    if any(k in err_lower for k in ("dimension", "vector size", "wrong vector", "does not exist", "not found", "collection")):
        return RetrievalStatus.CONFIG_MISMATCH, "Vector collection configuration or dimension mismatch."
    return RetrievalStatus.UNEXPECTED_ERROR, f"Unexpected retrieval failure: {type(err).__name__}"


def validate_collection_compatibility(collection_name: str = None) -> dict:
    """
    Validates that the Qdrant collection exists and its vector dimension
    and embedding model identity match the active embedding configuration.
    """
    col_name = collection_name or settings.QDRANT_COLLECTION
    q_client = get_qdrant_client()
    expected_dim = get_embedding_dim()
    active_meta = get_active_embedding_metadata()
    active_model = active_meta.get("model_name")
    active_provider = active_meta.get("provider")

    try:
        exists = q_client.collection_exists(col_name)
        if not exists:
            return {
                "compatible": False,
                "reason": f"Collection '{col_name}' does not exist.",
                "collection_name": col_name,
                "expected_dimension": expected_dim,
                "actual_dimension": None,
            }

        info = q_client.get_collection(col_name)
        vectors_config = info.config.params.vectors
        actual_size = None
        if hasattr(vectors_config, "size"):
            actual_size = vectors_config.size
        elif isinstance(vectors_config, dict):
            actual_size = vectors_config.get("size")

        if actual_size is not None and actual_size != expected_dim:
            return {
                "compatible": False,
                "reason": f"Dimension mismatch: collection expects {actual_size}, but embedding model produces {expected_dim}.",
                "collection_name": col_name,
                "expected_dimension": expected_dim,
                "actual_dimension": actual_size,
            }

        # Validate model identity from collection sample points if collection contains vectors
        points_count = getattr(info, "points_count", None)
        if points_count is None and hasattr(info, "vectors_count"):
            points_count = info.vectors_count

        stored_model = None
        stored_provider = None
        if points_count and points_count > 0:
            try:
                scroll_res = q_client.scroll(collection_name=col_name, limit=1, with_payload=True)
                sample_points = scroll_res[0] if isinstance(scroll_res, tuple) else getattr(scroll_res, "points", [])
                if sample_points and len(sample_points) > 0:
                    first_payload = getattr(sample_points[0], "payload", {}) or {}
                    stored_model = first_payload.get("embedding_model")
                    stored_provider = first_payload.get("embedding_provider")

                    if stored_model and active_model and stored_model != active_model:
                        return {
                            "compatible": False,
                            "reason": (
                                f"Embedding model mismatch: collection '{col_name}' was indexed with "
                                f"'{stored_model}' ({stored_provider or 'unknown'}), but active query model is "
                                f"'{active_model}' ({active_provider}). Searching requires reindexing."
                            ),
                            "collection_name": col_name,
                            "expected_dimension": expected_dim,
                            "actual_dimension": actual_size,
                            "stored_model": stored_model,
                            "active_model": active_model,
                        }
            except Exception as scroll_err:
                logfire.warning(f"Could not verify point metadata for collection '{col_name}': {scroll_err}")

        return {
            "compatible": True,
            "collection_name": col_name,
            "expected_dimension": expected_dim,
            "actual_dimension": actual_size,
            "stored_model": stored_model or active_model,
            "active_model": active_model,
        }
    except Exception as e:
        status, safe_msg = classify_retrieval_exception(e)
        return {
            "compatible": False,
            "reason": f"{safe_msg} ({e})",
            "collection_name": col_name,
            "expected_dimension": expected_dim,
            "actual_dimension": None,
        }


def search_enterprise_knowledge_detailed(
    query: str,
    limit: int = 8,
    score_threshold: float = None
) -> RetrievalOutcome:
    """
    Performs vector search in the enterprise knowledge base, returning a
    structured RetrievalOutcome that explicitly distinguishes success,
    no-match, DB outage, auth failure, embedding failure, and config mismatches.
    """
    threshold = score_threshold if score_threshold is not None else DEFAULT_SCORE_THRESHOLD

    # 1. Embed query
    try:
        query_vector = embed_query(query)
    except Exception as e:
        logfire.error(f"❌ Embedding Generation Failed: {e}")
        return RetrievalOutcome(
            status=RetrievalStatus.EMBEDDING_FAILURE,
            documents=[],
            query=query,
            total_candidates=0,
            error_message=f"Embedding generation failed: {type(e).__name__}",
            score_threshold=threshold,
        )

    # 2. Query Qdrant
    try:
        q_client = get_qdrant_client()
        query_kwargs = {
            "collection_name": settings.QDRANT_COLLECTION,
            "query": query_vector,
            "limit": limit,
            "with_payload": True,
        }
        if threshold and threshold > 0:
            query_kwargs["score_threshold"] = threshold

        response = q_client.query_points(**query_kwargs)

        results = []
        for res in response.points:
            payload = res.payload or {}
            results.append({
                "content": payload.get("text", ""),
                "source": payload.get("source", "Unknown"),
                "score": float(res.score) if res.score is not None else None,
                "source_type": payload.get("source_type", "general"),
            })

        status = RetrievalStatus.SUCCESS if results else RetrievalStatus.NO_RESULTS
        if status == RetrievalStatus.SUCCESS:
            logfire.info(f"✅ Qdrant retrieved {len(results)} points above score threshold ({threshold}).")
        else:
            logfire.info(f"ℹ️ Qdrant search returned 0 documents above score threshold ({threshold}).")

        return RetrievalOutcome(
            status=status,
            documents=results,
            query=query,
            total_candidates=len(results),
            score_threshold=threshold,
        )

    except Exception as e:
        status, safe_msg = classify_retrieval_exception(e)
        logfire.error(f"❌ Qdrant Search Failed [{status.value}]: {e}")
        return RetrievalOutcome(
            status=status,
            documents=[],
            query=query,
            total_candidates=0,
            error_message=safe_msg,
            score_threshold=threshold,
        )


class OutcomeList(list):
    """List subclass that retains retrieval outcome metadata for backward compatibility."""
    def __init__(self, items, outcome: RetrievalOutcome):
        super().__init__(items)
        self.outcome = outcome


def search_enterprise_knowledge(
    query: str,
    limit: int = 8,
    score_threshold: float = None
) -> OutcomeList:
    """
    Backward-compatible search function.
    Returns a list of document dicts with attached `.outcome` metadata.
    """
    outcome = search_enterprise_knowledge_detailed(query, limit=limit, score_threshold=score_threshold)
    return OutcomeList(outcome.documents, outcome)
