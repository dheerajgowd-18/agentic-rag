import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

import os

DEFAULT_SCORE_THRESHOLD = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.30"))


def search_enterprise_knowledge(query: str, limit: int = 8, score_threshold: float = None):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Filters candidates by score_threshold to eliminate low-relevance noise.
    """
    try:
        query_vector = embed_query(query)
        threshold = score_threshold if score_threshold is not None else DEFAULT_SCORE_THRESHOLD

        # Using query_points - the modern standard for Qdrant
        query_kwargs = {
            "collection_name": settings.QDRANT_COLLECTION,
            "query": query_vector,
            "limit": limit,
            "with_payload": True,
        }
        if threshold and threshold > 0:
            query_kwargs["score_threshold"] = threshold

        response = client.query_points(**query_kwargs)

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score
            })

        logfire.info(f"Qdrant retrieved {len(results)} points above score threshold ({threshold}).")
        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []

