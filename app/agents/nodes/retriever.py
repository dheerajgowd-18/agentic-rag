import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge, RetrievalStatus
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    Captures structured retrieval and reranking degradation outcomes.
    """
    query = state["current_query"]
    existing_meta = dict(state.get("execution_metadata") or {})
    current_plan = list(state.get("plan") or [])

    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        outcome = getattr(raw_results, "outcome", None)
        status_val = outcome.status.value if outcome else ("SUCCESS" if raw_results else "NO_RESULTS")
        candidate_count = len(raw_results)

        logfire.info(f"Retrieved {candidate_count} candidates from Vector DB [Status: {status_val}]")

        if not raw_results:
            if outcome and outcome.status == RetrievalStatus.NO_RESULTS:
                msg = f"No documents met the relevance threshold for query: '{query}'."
                logfire.warning(msg)
                plan_entry = "Context Retrieval: No relevant documentation found (below threshold)"
                status_msg = "No sufficiently relevant documentation found."
            else:
                err_msg = outcome.error_message if outcome else "Vector search failed"
                logfire.error(f"Retrieval infrastructure failure [{status_val}]: {err_msg}")
                plan_entry = f"Context Retrieval Error: {status_val}"
                status_msg = f"Knowledge base search degraded: {status_val}"

            existing_meta.update({
                "retrieval_executed": True,
                "retrieval_status": status_val,
                "candidate_count": 0,
                "candidates_count": 0,
                "reranking_executed": False,
                "selected_passage_count": 0,
                "selected_passages_count": 0,
                "fallback_used": False,
            })

            return {
                "documents": [],
                "status": status_msg,
                "plan": current_plan + [plan_entry],
                "execution_metadata": existing_meta,
            }

        with logfire.span("⚖️ Semantic Reranking"):
            reranked_docs = rerank_documents(query, raw_results, top_n=5)
            rerank_outcome = getattr(reranked_docs, "outcome", None)
            rerank_meta = rerank_outcome.to_dict() if rerank_outcome else {"reranked": True}
            logfire.info(f"Reranking complete. Kept top {len(reranked_docs)} chunks. Reranked: {rerank_meta.get('reranked')}")

    plan_step = "Context Retrieved"
    if rerank_meta.get("fallback_used"):
        plan_step += " (Reranker fallback to vector order)"

    existing_meta.update({
        "retrieval_executed": True,
        "retrieval_status": status_val,
        "candidate_count": candidate_count,
        "candidates_count": candidate_count,
        "reranking_executed": True,
        "rerank_metadata": rerank_meta,
        "selected_passage_count": len(reranked_docs),
        "selected_passages_count": len(reranked_docs),
        "fallback_used": rerank_meta.get("fallback_used", False),
    })

    return {
        "documents": reranked_docs,
        "status": "Found technical context.",
        "plan": current_plan + [plan_step],
        "execution_metadata": existing_meta,
    }
