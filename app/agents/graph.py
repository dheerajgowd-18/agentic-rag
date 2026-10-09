import os
import re
import logfire
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node

MAX_RETRIEVAL_RETRIES = 1


def context_grader_node(state: AgentState):
    """
    Evaluates whether retrieved documents contain sufficient evidence.
    If empty or insufficient, allows a single bounded query rewrite attempt.
    """
    docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    current_plan = list(state.get("plan") or [])
    meta = dict(state.get("execution_metadata") or {})

    # Check if we have documents with substantive content
    has_substantive_content = any(
        (d.get("content", "").strip() if isinstance(d, dict) else str(d).strip())
        for d in docs
    )

    if has_substantive_content:
        logfire.info(f"✅ Context Grader: Sufficient evidence found ({len(docs)} chunks).")
        meta["grader_outcome"] = "sufficient"
        return {
            "plan": current_plan + ["Context Grader: Sufficient evidence"],
            "execution_metadata": meta,
        }

    # Insufficient evidence
    if retry_count < MAX_RETRIEVAL_RETRIES:
        logfire.warning(f"⚠️ Context Grader: Insufficient context. Triggering query refinement (attempt {retry_count + 1}/{MAX_RETRIEVAL_RETRIES}).")
        meta["grader_outcome"] = "retry_needed"
        return {
            "plan": current_plan + [f"Context Grader: Insufficient evidence — initiating rewrite"],
            "execution_metadata": meta,
        }

    logfire.info("ℹ️ Context Grader: Max retries reached. Proceeding to responder with empty context.")
    meta["grader_outcome"] = "exhausted"
    return {
        "plan": current_plan + ["Context Grader: Max retries reached — proceeding without context"],
        "execution_metadata": meta,
    }


STOP_WORDS = {
    "what", "is", "the", "how", "do", "i", "can", "you", "tell", "me", "about",
    "in", "on", "at", "for", "a", "an", "to", "of", "and", "or", "with", "does",
    "are", "why", "which", "where", "when", "who", "should", "would", "could", "please",
    "need", "want", "help"
}


def simplify_search_query(query: str) -> str:
    """
    Deterministic query refiner to extract core technical keywords for bounded retry.
    Strips conversational stop words and punctuation so vector search matches salient terms.
    """
    if not query:
        return ""
    clean = re.sub(r"[^\w\s\-]", " ", query).strip()
    words = clean.split()
    salient = [w for w in words if w.lower() not in STOP_WORDS]
    if salient:
        return " ".join(salient[:6])
    return " ".join(words[:6]) if words else query


def query_rewriter_node(state: AgentState):
    """
    Deterministic query refiner to broaden search keywords for bounded retry.
    Avoids unnecessary extra LLM latency while stripping overly restrictive constraints.
    """
    query = state.get("current_query", "")
    retry_count = state.get("retry_count", 0) + 1
    current_plan = list(state.get("plan") or [])
    meta = dict(state.get("execution_metadata") or {})

    rewritten_query = simplify_search_query(query)

    logfire.info(f"🔄 Query Rewriter: '{query}' -> '{rewritten_query}' (retry {retry_count})")
    meta["retry_executed"] = True
    meta["rewritten_query"] = rewritten_query

    return {
        "current_query": rewritten_query,
        "retry_count": retry_count,
        "plan": current_plan + [f"Query Rewriter: Refined search query to '{rewritten_query}'"],
        "execution_metadata": meta,
    }


def route_planner(state: AgentState):
    """Routes the workflow based on the planner's decision."""
    if state.get("current_query") == "CONVERSATIONAL":
        return "responder"
    return "retriever"


def route_grader(state: AgentState):
    """Decides whether to retry retrieval or proceed to synthesis."""
    meta = state.get("execution_metadata") or {}
    grader_outcome = meta.get("grader_outcome")
    retry_count = state.get("retry_count", 0)

    if grader_outcome == "retry_needed" and retry_count < MAX_RETRIEVAL_RETRIES:
        return "query_rewriter"
    return "responder"


# 1. Initialize the State Graph
workflow = StateGraph(AgentState)

# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("context_grader", context_grader_node)
workflow.add_node("query_rewriter", query_rewriter_node)
workflow.add_node("responder", generate_node)

# 3. Connect Edges
workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder"
    }
)

workflow.add_edge("retriever", "context_grader")

workflow.add_conditional_edges(
    "context_grader",
    route_grader,
    {
        "query_rewriter": "query_rewriter",
        "responder": "responder"
    }
)

workflow.add_edge("query_rewriter", "retriever")
workflow.add_edge("responder", END)


# --- CHECKPOINT PERSISTENCE ---
_active_persistence_mode = "memory"


def get_active_persistence_mode() -> str:
    """Returns the actual active checkpoint persistence mode."""
    return _active_persistence_mode


def _build_checkpointer():
    global _active_persistence_mode
    persistence_mode = getattr(settings, "CHECKPOINT_PERSISTENCE", "memory").lower()
    db_path = getattr(settings, "CHECKPOINT_DB_PATH", "checkpoints.sqlite")
    strict_required = (
        getattr(settings, "CHECKPOINT_PERSISTENCE_REQUIRED", False)
        or os.getenv("CHECKPOINT_PERSISTENCE_REQUIRED", "false").lower() == "true"
    )

    if persistence_mode in ("sqlite", "disk"):
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn = sqlite3.connect(db_path, check_same_thread=False)
            logfire.info(f"💾 Checkpoint persistence: SQLite active at '{db_path}'")
            _active_persistence_mode = "sqlite"
            return SqliteSaver(conn)
        except Exception as e:
            if strict_required:
                logfire.error(f"❌ Strict durable checkpoint persistence required but initialization failed: {e}")
                raise RuntimeError(
                    f"Durable SQLite checkpoint persistence is required (CHECKPOINT_PERSISTENCE_REQUIRED=true) "
                    f"but failed to initialize: {e}"
                ) from e
            logfire.warning(f"⚠️ SQLite checkpoint persistence unavailable ({e}). Running with in-memory MemorySaver (degraded).")
            _active_persistence_mode = "memory (degraded fallback)"
            return MemorySaver()

    logfire.info("💾 Checkpoint persistence: In-memory MemorySaver active")
    _active_persistence_mode = "memory"
    return MemorySaver()


checkpointer = _build_checkpointer()

# 4. Compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)
