# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import json
import asyncio
import logfire
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Safe to import app modules - logfire is already active
from fastapi import FastAPI, Response, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.config import settings
from app.agents.graph import rag_agent, get_active_persistence_mode, simplify_search_query
from app.agents.nodes.planner import evaluate_planner, PlannerDecision
from app.agents.nodes.responder import build_responder_prompt, validate_citations
from app.guardrails import initialize_rails, guard, guard_async, guard_detailed, guard_async_detailed
from app.gateway import portkey_client
from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge,
    search_enterprise_knowledge_detailed,
    validate_collection_compatibility,
    RetrievalStatus,
)
from app.services.retrieval.ranking_service import rerank_documents_async, rerank_documents


# Initialize FastAPI
app = FastAPI(
    title="Enterprise Agentic RAG API",
    description="Production Agentic RAG with LangGraph, Portkey, Qdrant, and NeMo Guardrails",
    version="1.0.0"
)

# Secure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Mount static files for modern HTML/CSS/JS frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup_event():
    logfire.info("🚀 Starting Enterprise Agentic RAG Service...")
    initialize_rails()
    config_report = settings.validate_configuration()
    if not config_report["valid"]:
        logfire.warning(f"⚠️ Configuration issues detected on startup: {config_report['issues']}")
    if config_report["warnings"]:
        logfire.info(f"ℹ️ Configuration warnings: {config_report['warnings']}")


class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=4000, description="User question")
    thread_id: Optional[str] = Field(
        default="default_user",
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="Alphanumeric thread ID for session memory"
    )


class ClearMemoryRequest(BaseModel):
    thread_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="Thread ID to clear"
    )


# ── UI & Health Routes ────────────────────────────────────────────────────────
@app.get("/")
def home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/api/health")
def health():
    """Lightweight liveness probe."""
    return {
        "status": "online",
        "service": "enterprise-agentic-rag",
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "checkpoint_persistence": get_active_persistence_mode(),
    }


@app.get("/api/ready")
def readiness():
    """Readiness probe checking database connectivity without invoking full LLM generation."""
    try:
        config_status = settings.validate_configuration()
        if not config_status["valid"]:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": config_status["issues"]}
            )

        col_compat = validate_collection_compatibility()
        if not col_compat.get("compatible"):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": col_compat.get("reason")}
            )

        return {
            "status": "ready",
            "collection": settings.QDRANT_COLLECTION,
            "dimension": col_compat.get("actual_dimension"),
            "embedding_provider": settings.EMBEDDING_PROVIDER,
        }
    except Exception as e:
        logfire.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "message": "Readiness probe failed"}
        )


@app.get("/graph")
def get_graph_image():
    """Returns the Mermaid image of the agent's workflow."""
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


# ── Shared Retrieval Pipeline Helper ─────────────────────────────────────────
async def execute_shared_retrieval(search_query: str) -> tuple[list, list, dict]:
    """
    Centralizes the vector search, reranking, and source deduplication
    shared between the synchronous and streaming execution paths.
    """
    raw_results = await asyncio.to_thread(search_enterprise_knowledge, search_query, limit=15)
    outcome = getattr(raw_results, "outcome", None)
    retrieval_status = outcome.status.value if outcome else ("SUCCESS" if raw_results else "NO_RESULTS")

    reranked_docs = []
    sources = []
    rerank_meta = {}

    if raw_results:
        reranked_docs = await rerank_documents_async(search_query, raw_results, top_n=5)
        rerank_outcome = getattr(reranked_docs, "outcome", None)
        rerank_meta = rerank_outcome.to_dict() if rerank_outcome else {"reranked": True}

        # Deduplicate sources preserving filenames and scores
        seen_texts = set()
        for doc in reranked_docs:
            content = doc.get("content", "")
            if content not in seen_texts:
                seen_texts.add(content)
                raw_score = doc.get("score")
                sources.append({
                    "source": doc.get("source", "Document"),
                    "content": content,
                    "score": float(raw_score) if raw_score is not None else None,
                })

    meta = {
        "retrieval_status": retrieval_status,
        "candidate_count": len(raw_results),
        "candidates_count": len(raw_results),
        "rerank_metadata": rerank_meta,
        "reranking_executed": bool(rerank_meta.get("reranked") or rerank_meta.get("fallback_used")),
        "selected_passage_count": len(reranked_docs),
        "selected_passages_count": len(reranked_docs),
        "fallback_used": rerank_meta.get("fallback_used", False),
    }

    return reranked_docs, sources, meta


# ── Streaming Endpoint (SSE) ──────────────────────────────────────────────────
@app.post("/query/stream")
async def query_stream(request: QueryRequest, http_request: Request):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Emits real-time thoughts, retrieved sources, synthesized tokens, and completion metadata.
    Enforces clear cancellation semantics: incomplete streams are not saved as valid answers.
    """
    q = request.q
    thread_id = request.thread_id or "default_user"
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        stream_completed = False
        full_answer = ""
        decision_intent = "CONVERSATIONAL"
        decision_query = ""
        sources = []
        reranked_docs = []

        try:
            # Gate 1: NeMo Guardrails
            yield f"data: {json.dumps({'type': 'thought', 'step': '🛡️ Checking NeMo Guardrails safety gate...'})}\n\n"
            guard_result = await guard_async_detailed(q)

            if guard_result.is_blocked or guard_result.is_error:
                logfire.info(f"🛡️ Guardrails blocked query: {q[:60]}")
                yield f"data: {json.dumps({'type': 'thought', 'step': '🛡️ Guardrails filter triggered — direct safety response'})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': guard_result.content or 'Request blocked by safety policy.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'status': 'Blocked by guardrails', 'sources': []})}\n\n"
                stream_completed = True
                return

            if guard_result.is_dialog:
                logfire.info(f"💬 Guardrails dialog turn: {q[:60]}")
                yield f"data: {json.dumps({'type': 'thought', 'step': '💬 Conversational greeting/dialog flow handled directly'})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': guard_result.content})}\n\n"
                rag_agent.update_state(config, {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": guard_result.content}
                    ]
                })
                yield f"data: {json.dumps({'type': 'done', 'status': 'complete', 'sources': []})}\n\n"
                stream_completed = True
                return

            yield f"data: {json.dumps({'type': 'thought', 'step': '🛡️ Guardrails passed — evaluating intent'})}\n\n"

            # Gate 2: Fetch history and prepare turn
            state_checkpoint = rag_agent.get_state(config)
            history_msgs = list(state_checkpoint.values.get("messages", [])) if state_checkpoint else []
            turn_messages = history_msgs + [{"role": "user", "content": q}]

            # Gate 3: Planner Node Evaluation
            yield f"data: {json.dumps({'type': 'thought', 'step': '🧠 Planner node: analyzing conversation context...'})}\n\n"
            decision: PlannerDecision = evaluate_planner(turn_messages)
            decision_intent = decision.intent
            decision_query = decision.search_query

            if decision.intent == "CONVERSATIONAL":
                yield f"data: {json.dumps({'type': 'thought', 'step': 'Intent: Conversational / Memory (Vector search skipped)'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'thought', 'step': f'Intent: Technical inquiry (Searching for: {decision.search_query})'})}\n\n"
                yield f"data: {json.dumps({'type': 'thought', 'step': '🔍 Searching Qdrant Vector Cloud for matching documentation...'})}\n\n"

                reranked_docs, sources, ret_meta = await execute_shared_retrieval(decision.search_query)

                # Bounded refinement alignment with graph: if 0 candidates found, attempt bounded rewrite
                if not reranked_docs and ret_meta.get("candidate_count", 0) == 0:
                    refined_q = simplify_search_query(decision.search_query)
                    if refined_q and refined_q != decision.search_query:
                        yield f"data: {json.dumps({'type': 'thought', 'step': f'⚠️ Context Grader: 0 matches — refining search to: {refined_q}'})}\n\n"
                        reranked_docs, sources, ret_meta = await execute_shared_retrieval(refined_q)
                        ret_meta["retry_executed"] = True
                        ret_meta["rewritten_query"] = refined_q

                candidate_count = ret_meta.get("candidate_count", 0)
                yield f"data: {json.dumps({'type': 'thought', 'step': f'Retrieved {candidate_count} candidates. Running FlashRank cross-encoder...'})}\n\n"
                yield f"data: {json.dumps({'type': 'thought', 'step': f'⚖️ FlashRank reranking complete. Selected top {len(reranked_docs)} semantic chunks.'})}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Gate 4: Streaming LLM Synthesis
            prompt_query = "CONVERSATIONAL" if decision.intent == "CONVERSATIONAL" else decision.search_query
            prompt = build_responder_prompt(prompt_query, turn_messages, reranked_docs)
            yield f"data: {json.dumps({'type': 'thought', 'step': f'✍️ Synthesizing response via {settings.GROQ_MODEL}...' })}\n\n"

            stream = portkey_client.chat.completions.create(
                model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                stream=True
            )

            for chunk in stream:
                if await http_request.is_disconnected():
                    logfire.warning(f"⚠️ SSE stream disconnected by client | thread={thread_id}")
                    if hasattr(stream, "close"):
                        try:
                            stream.close()
                        except Exception:
                            pass
                    raise asyncio.CancelledError()
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_answer += delta
                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
                    await asyncio.sleep(0.005)

            # Validate citations deterministically on completed answer
            sanitized_answer, cite_meta = validate_citations(full_answer, len(reranked_docs))

            # Update LangGraph memory state with completed turn
            rag_agent.update_state(config, {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": sanitized_answer}
                ],
                "current_query": prompt_query,
                "documents": reranked_docs,
                "status": "Response generated."
            })

            stream_completed = True
            yield f"data: {json.dumps({'type': 'done', 'status': 'complete', 'sources': sources})}\n\n"

        except asyncio.CancelledError:
            logfire.warning(f"⚠️ SSE stream disconnected by client | thread={thread_id}")
            # Policy on client cancellation: discard incomplete answer from memory
            # Record an explicit interrupted marker rather than a partial answer
            rag_agent.update_state(config, {
                "status": "Interrupted by client cancellation.",
            })
            raise

        except Exception as e:
            logfire.error(f"❌ Streaming error: {e}")
            err_text = "\n\n[An internal error occurred while processing your request. Please try again later.]"
            yield f"data: {json.dumps({'type': 'token', 'content': err_text})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'status': 'error', 'sources': []})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Synchronous Endpoint ──────────────────────────────────────────────────────
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the unified LangGraph RAG flow with memory.
    Returns structured execution metadata for evaluation and diagnostics.
    """
    q = request.q
    thread_id = request.thread_id or "default_user"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "retry_count": 0,
        "execution_metadata": {},
    }

    try:
        guard_result = guard_detailed(q)
        if guard_result.is_blocked or guard_result.is_error:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": guard_result.content or "I am an Enterprise IT Assistant. I cannot assist with that request.",
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
                "execution_metadata": {
                    "guardrail_decision": "BLOCKED",
                    "retrieval_executed": False,
                    "reason": guard_result.reason,
                }
            }

        if guard_result.is_dialog:
            logfire.info(f"💬 Conversational dialog flow handled | thread={thread_id}")
            rag_agent.update_state(config, {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": guard_result.content}
                ]
            })
            return {
                "question": q,
                "answer": guard_result.content,
                "thought_process": ["Intent: Conversational Greeting", "Retrieval: Skipped"],
                "status": "Handled conversationally.",
                "sources": [],
                "execution_metadata": {
                    "guardrail_decision": "DIALOG",
                    "planner_intent": "CONVERSATIONAL",
                    "retrieval_executed": False,
                }
            }

        # Run compiled LangGraph
        final_output = rag_agent.invoke(initial_state, config=config)

        # Normalize sources for response
        raw_docs = final_output.get("documents", [])
        sources = []
        for d in raw_docs:
            if isinstance(d, dict):
                sources.append({
                    "source": d.get("source", "Document"),
                    "content": d.get("content", ""),
                    "score": d.get("score"),
                })
            else:
                sources.append({"source": "Document", "content": str(d), "score": None})

        exec_meta = dict(final_output.get("execution_metadata") or {})
        exec_meta["guardrail_decision"] = "ALLOWED"

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan", []),
            "status": final_output.get("status"),
            "sources": sources,
            "execution_metadata": exec_meta,
        }

    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "question": q,
                "answer": "I apologize, but an internal error occurred while processing your request. Please try again later.",
                "thought_process": ["Error encountered during execution."],
                "status": "error",
                "sources": [],
                "execution_metadata": {
                    "error": "Internal execution error",
                    "error_type": type(e).__name__,
                }
            }
        )


# ── Clear Memory Endpoint ─────────────────────────────────────────────────────
@app.post("/memory/clear")
def clear_memory(request: ClearMemoryRequest):
    """Clears LangGraph conversational memory for a given thread_id."""
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        rag_agent.update_state(config, {"messages": [{"role": "system", "content": "__CLEAR__"}]})
        logfire.info(f"🗑️ Cleared memory for thread: {request.thread_id}")
        return {"status": "cleared", "thread_id": request.thread_id}
    except Exception as e:
        logfire.error(f"Failed to clear memory: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "Failed to clear memory"}
        )
