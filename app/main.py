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
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from app.config import settings
from app.agents.graph import rag_agent
from app.agents.nodes.responder import build_responder_prompt
from app.guardrails import initialize_rails, guard, guard_async
from app.gateway import portkey_client
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents



# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")

# Mount static files for modern HTML/CSS/JS frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup_event():
    initialize_rails()


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


class ClearMemoryRequest(BaseModel):
    thread_id: str


# ── UI Route ──────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/api/health")
def health():
    return {"message": "Enterprise LangGraph RAG API is live.", "status": "online"}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


# ── Streaming Endpoint (SSE) with Real-Time Thoughts, Sources & Tokens ────────
@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Emits real-time thoughts, retrieved sources, synthesized tokens, and done events.
    Supports instant cancellation via client AbortController.
    """
    q = request.q
    thread_id = request.thread_id or "default_user"
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            # Gate 1: NeMo Guardrails
            yield f"data: {json.dumps({'type': 'thought', 'step': '🛡️ Checking NeMo Guardrails safety gate...'})}\n\n"
            is_blocked, is_dialog, rail_response = await guard_async(q)
            if is_blocked:
                logfire.info(f"🛡️ Guardrails blocked query: {q[:60]}")
                yield f"data: {json.dumps({'type': 'thought', 'step': '🛡️ Guardrails filter triggered — direct safety response'})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': rail_response})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'status': 'Blocked by guardrails', 'sources': []})}\n\n"
                return

            if is_dialog:
                logfire.info(f"💬 Guardrails dialog turn: {q[:60]}")
                yield f"data: {json.dumps({'type': 'thought', 'step': '💬 Conversational greeting/dialog flow handled directly'})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': rail_response})}\n\n"
                # Update memory so the greeting is remembered
                rag_agent.update_state(config, {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": rail_response}
                    ]
                })
                yield f"data: {json.dumps({'type': 'done', 'status': 'complete', 'sources': []})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'thought', 'step': '🛡️ Guardrails passed — evaluating intent'})}\n\n"


            # Gate 2: Fetch history and prepare turn
            state_checkpoint = rag_agent.get_state(config)
            history_msgs = list(state_checkpoint.values.get("messages", [])) if state_checkpoint else []
            turn_messages = history_msgs + [{"role": "user", "content": q}]

            # Gate 3: Planner Node Evaluation
            yield f"data: {json.dumps({'type': 'thought', 'step': '🧠 Planner node: analyzing conversation context...'})}\n\n"
            from app.agents.nodes.planner import evaluate_planner
            decision = evaluate_planner(turn_messages)

            sources = []
            reranked_docs = []

            if decision == "CONVERSATIONAL":
                yield f"data: {json.dumps({'type': 'thought', 'step': 'Intent: Conversational / Memory (Vector search skipped)'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'thought', 'step': f'Intent: Technical inquiry (Searching for: {decision})'})}\n\n"
                yield f"data: {json.dumps({'type': 'thought', 'step': '🔍 Searching Qdrant Vector Cloud for matching documentation...'})}\n\n"
                raw_results = search_enterprise_knowledge(decision, limit=15)

                yield f"data: {json.dumps({'type': 'thought', 'step': f'Retrieved {len(raw_results)} candidates. Running FlashRank cross-encoder...'})}\n\n"
                reranked_docs = rerank_documents(decision, raw_results, top_n=5)
                yield f"data: {json.dumps({'type': 'thought', 'step': f'⚖️ FlashRank reranking complete. Selected top {len(reranked_docs)} semantic chunks.'})}\n\n"

                # Extract deduplicated sources preserving filenames and scores
                seen_texts = set()
                for doc in reranked_docs:
                    content = doc.get("content", "")
                    if content not in seen_texts:
                        seen_texts.add(content)
                        sources.append({
                            "source": doc.get("source", "Document"),
                            "content": content,
                            "score": doc.get("score")
                        })

                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Gate 4: Streaming LLM Synthesis using shared prompt builder
            prompt = build_responder_prompt(decision, turn_messages, reranked_docs)
            yield f"data: {json.dumps({'type': 'thought', 'step': f'✍️ Synthesizing response via {settings.GROQ_MODEL}...' })}\n\n"

            full_answer = ""
            stream = portkey_client.chat.completions.create(
                model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                stream=True
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_answer += delta
                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
                    await asyncio.sleep(0.005)

            # Update LangGraph memory state with completed turn
            rag_agent.update_state(config, {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": full_answer}
                ],
                "current_query": decision,
                "documents": reranked_docs,
                "status": "Response generated."
            })

            yield f"data: {json.dumps({'type': 'done', 'status': 'complete', 'sources': sources})}\n\n"

        except Exception as e:
            logfire.error(f"❌ Streaming error: {e}")
            err_text = f"\n\n[Error processing request: {e}]"
            err_payload = json.dumps({"type": "token", "content": err_text})
            done_payload = json.dumps({"type": "done", "status": "error", "sources": []})
            yield f"data: {err_payload}\n\n"
            yield f"data: {done_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")



# ── Clear Memory Endpoint ─────────────────────────────────────────────────────
@app.post("/memory/clear")
def clear_memory(request: ClearMemoryRequest):
    """
    Clears LangGraph conversational memory for a given thread_id.
    """
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        rag_agent.update_state(config, {"messages": [{"role": "system", "content": "__CLEAR__"}]})
        logfire.info(f"🗑️ Cleared memory for thread: {request.thread_id}")
        return {"status": "cleared", "thread_id": request.thread_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}



# ── Legacy Synchronous Endpoint ──────────────────────────────────────────────
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id or "default_user"

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        is_blocked, is_dialog, rail_response = guard(q)
        if is_blocked:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        if is_dialog:
            logfire.info(f"💬 Conversational dialog flow handled | thread={thread_id}")
            rag_agent.update_state(config, {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": rail_response}
                ]
            })
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Conversational Greeting", "Retrieval: Skipped"],
                "status": "Handled conversationally.",
                "sources": []
            }

        final_output = rag_agent.invoke(initial_state, config=config)

        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }
