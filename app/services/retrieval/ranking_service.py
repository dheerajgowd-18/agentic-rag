import os
import tempfile
import time
import threading
import asyncio
import logfire
from typing import Optional, List, Any
from flashrank import Ranker, RerankRequest

# Lazy thread-safe initialization
_ranker = None
_ranker_lock = threading.Lock()


class RerankOutcome:
    """Explicitly represents the result of reranking, including degradation/fallback states."""
    def __init__(
        self,
        documents: List[Any],
        reranked: bool,
        original_count: int,
        final_count: int,
        duration: float,
        fallback_used: bool = False,
        original_order_returned: bool = False,
        error: Optional[str] = None,
    ):
        self.documents = documents
        self.reranked = reranked
        self.original_count = original_count
        self.final_count = final_count
        self.duration = duration
        self.fallback_used = fallback_used
        self.original_order_returned = original_order_returned
        self.error = error

    def to_dict(self) -> dict:
        return {
            "reranked": self.reranked,
            "original_count": self.original_count,
            "final_count": self.final_count,
            "duration": round(self.duration, 4),
            "fallback_used": self.fallback_used,
            "original_order_returned": self.original_order_returned,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return (
            f"<RerankOutcome reranked={self.reranked} fallback={self.fallback_used} "
            f"orig_order={self.original_order_returned} docs={self.final_count}/{self.original_count}>"
        )


class RerankList(list):
    """List subclass preserving reranking outcome metadata."""
    def __init__(self, items, outcome: RerankOutcome):
        super().__init__(items)
        self.outcome = outcome


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily in a thread-safe manner.
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is not None:
        return _ranker

    with _ranker_lock:
        if _ranker is not None:
            return _ranker

        logfire.info("🧠 Initializing FlashRank Model (TinyBERT) locally...")
        try:
            cache_path = os.path.join(tempfile.gettempdir(), "flashrank")
            _ranker = Ranker(cache_dir=cache_path)
        except Exception:
            _ranker = Ranker()
        return _ranker


def rerank_documents_detailed(query: str, documents: list, top_n: int = 5) -> RerankOutcome:
    """
    Refines retrieval results with explicit tracking of success, degradation, and fallback.
    Does NOT log document content during errors to preserve confidentiality.
    """
    if not documents:
        return RerankOutcome(
            documents=[],
            reranked=False,
            original_count=0,
            final_count=0,
            duration=0.0,
            fallback_used=False,
            original_order_returned=False,
        )

    start_time = time.time()
    original_count = len(documents)
    logfire.info(f"📡 [Reranker] Sending {original_count} docs to FlashRank Cross-Encoder...")

    try:
        ranker = _get_ranker()

        # Build passages for FlashRank with original index
        passages = []
        is_dict = isinstance(documents[0], dict)
        for i, doc in enumerate(documents):
            text = doc.get("content", "") if is_dict else str(doc)
            passages.append({"id": i, "text": text})

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        # Map back to original document items with updated scores
        reranked_docs = []
        for res in results[:top_n]:
            orig_idx = int(res["id"])
            orig_doc = documents[orig_idx]
            if is_dict:
                updated_doc = dict(orig_doc)
                raw_score = res.get("score")
                updated_doc["score"] = float(raw_score) if raw_score is not None else None
                reranked_docs.append(updated_doc)
            else:
                reranked_docs.append(res["text"])

        duration = time.time() - start_time
        top_score = results[0]["score"] if results else "N/A"
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s. Selected {len(reranked_docs)}/{original_count}. Top score: {top_score}")

        return RerankOutcome(
            documents=reranked_docs,
            reranked=True,
            original_count=original_count,
            final_count=len(reranked_docs),
            duration=duration,
            fallback_used=False,
            original_order_returned=False,
        )

    except Exception as e:
        duration = time.time() - start_time
        safe_err = f"{type(e).__name__}: {str(e)}"
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {safe_err} (reverting to original retrieval ordering)")

        # Degraded fallback: retain original retrieval order and slice to top_n
        fallback_docs = documents[:top_n]
        return RerankOutcome(
            documents=fallback_docs,
            reranked=False,
            original_count=original_count,
            final_count=len(fallback_docs),
            duration=duration,
            fallback_used=True,
            original_order_returned=True,
            error=safe_err,
        )


def rerank_documents(query: str, documents: list, top_n: int = 5) -> RerankList:
    """
    Backward-compatible reranking function returning list with .outcome attached.
    """
    outcome = rerank_documents_detailed(query, documents, top_n=top_n)
    return RerankList(outcome.documents, outcome)


async def rerank_documents_async(query: str, documents: list, top_n: int = 5) -> RerankList:
    """Non-blocking async wrapper around rerank_documents."""
    return await asyncio.to_thread(rerank_documents, query, documents, top_n)
