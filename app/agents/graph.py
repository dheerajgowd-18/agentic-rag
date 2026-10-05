from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.config import settings
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


# 1. Initialize the State Graph
workflow = StateGraph(AgentState)


# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)

# 3. Define the Edges & Routing Logic
def route_planner(state: AgentState):
    """
    Routes the workflow based on the planner's decision.
    """
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"

workflow.set_entry_point("planner")


# Conditional Edge: Planner -> Router -> (Retriever OR Responder)
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder"
    }
)


workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)


# --- CHECKPOINT PERSISTENCE ---
# Supports configurable SQLite persistence or standard in-memory MemorySaver
def _build_checkpointer():
    persistence_mode = getattr(settings, "CHECKPOINT_PERSISTENCE", "memory")
    db_path = getattr(settings, "CHECKPOINT_DB_PATH", "checkpoints.sqlite")

    if persistence_mode in ("sqlite", "disk"):
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn = sqlite3.connect(db_path, check_same_thread=False)
            return SqliteSaver(conn)
        except Exception as e:
            # Fall back gracefully if sqlite extension or setup is unavailable
            return MemorySaver()

    return MemorySaver()


checkpointer = _build_checkpointer()


# 4. Compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)



