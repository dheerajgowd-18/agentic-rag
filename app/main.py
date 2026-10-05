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
from app.guardrails import initialize_rails, guard, guard_async
from app.gateway import portkey_client, get_langchain_llm
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
            rail_fired, rail_response = await guard_async(q)
            if rail_fired:
                logfire.info(f"🛡️ Guardrails blocked query: {q[:60]}")
                yield f"data: {json.dumps({'type': 'thought', 'step': '🛡️ Guardrails filter triggered — direct safety response'})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': rail_response})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'status': 'Blocked by guardrails', 'sources': []})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'thought', 'step': '🛡️ Guardrails passed — evaluating intent'})}\n\n"

            # Gate 2: Fetch history from LangGraph checkpointer
            state_checkpoint = rag_agent.get_state(config)
            history_msgs = state_checkpoint.values.get("messages", []) if state_checkpoint else []

            history_str = ""
            for msg in history_msgs:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

            # Gate 3: Planner Node
            yield f"data: {json.dumps({'type': 'thought', 'step': '🧠 Planner node: analyzing conversation context...'})}\n\n"
            planner_prompt = f"""
            You are an intelligent Assistant Planner. 
            Analyze the conversation history and the latest user message.
            
            CONVERSATION HISTORY:
            {history_str}
            
            LATEST MESSAGE:
            "{q}"
            
            Task:
            1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above, respond with 'CONVERSATIONAL'.
            2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.
            
            Output ONLY 'CONVERSATIONAL' or the search query.
            """
            planner_llm = get_langchain_llm(feature="planner")
            decision = planner_llm.invoke(planner_prompt).content.strip()

            sources = []
            if decision == "CONVERSATIONAL":
                yield f"data: {json.dumps({'type': 'thought', 'step': 'Intent: Conversational / Memory (Vector search skipped)'})}\n\n"
                prompt = f"""
                You are a friendly and helpful Enterprise AI Assistant.
                Answer the user's latest message using the CONVERSATION HISTORY below.

                CONVERSATION HISTORY:
                {history_str}

                LATEST MESSAGE:
                "{q}"
                """
            else:
                yield f"data: {json.dumps({'type': 'thought', 'step': f'Intent: Technical inquiry (Searching for: {decision})'})}\n\n"
                yield f"data: {json.dumps({'type': 'thought', 'step': '🔍 Searching Qdrant Vector Cloud for matching documentation...'})}\n\n"
                raw_results = search_enterprise_knowledge(decision, limit=15)
                doc_contents = [doc['content'] for doc in raw_results]

                yield f"data: {json.dumps({'type': 'thought', 'step': f'Retrieved {len(doc_contents)} candidates. Running FlashRank cross-encoder...'})}\n\n"
                reranked_contents = rerank_documents(decision, doc_contents, top_n=5)
                yield f"data: {json.dumps({'type': 'thought', 'step': '⚖️ FlashRank reranking complete. Selected top 5 semantic chunks.'})}\n\n"

                for raw in raw_results:
                    if raw['content'] in reranked_contents:
                        sources.append({"source": raw.get("source", "Document"), "content": raw['content']})

                # Deduplicate sources
                seen_texts = set()
                unique_sources = []
                for s in sources:
                    if s['content'] not in seen_texts:
                        seen_texts.add(s['content'])
                        unique_sources.append(s)
                sources = unique_sources

                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

                max_context_chars = 25000
                full_context = ""
                for doc in reranked_contents:
                    if len(full_context) + len(doc) < max_context_chars:
                        full_context += f"CONTENT: {doc}\n\n"

                prompt = f"""
                You are a Senior Technical Architect.
                Answer the question using the TECHNICAL CONTEXT provided.

                TECHNICAL CONTEXT:
                {full_context}

                CONVERSATION HISTORY:
                {history_str}

                USER QUESTION:
                "{q}"
                """

            # Gate 4: Streaming LLM Synthesis
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
                ]
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
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
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
