import time
import logfire
from flashrank import Ranker, RerankRequest

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily. 
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing FlashRank Model (TinyBERT) locally...")
        try:
            # We use a specific cache directory to avoid permission issues in production
            _ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            _ranker = Ranker()
    return _ranker



def rerank_documents(query: str, documents: list, top_n: int = 5) -> list:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    Supports either list of dicts (with 'content', 'source', etc.) or list of strings.
    Always returns items with full metadata preserved if dicts are provided.
    """
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")

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
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}")

        return reranked_docs

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fallback to the original order to ensure the user still gets an answer
        return documents[:top_n]


import asyncio

async def rerank_documents_async(query: str, documents: list, top_n: int = 5) -> list:
    """
    Non-blocking async wrapper around rerank_documents.
    Executes CPU-bound ONNX model inference in a background worker thread.
    """
    return await asyncio.to_thread(rerank_documents, query, documents, top_n)


