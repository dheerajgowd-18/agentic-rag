from typing import TypedDict, List, Annotated, Any, Dict, Optional


def reduce_messages(existing: List[dict], update: List[dict]) -> List[dict]:
    """
    Custom reducer for conversation messages.
    Appends new messages during normal flow.
    Clears history only when an explicit clear marker is present:
    either a message with content '__CLEAR__'.
    If update is None or empty list, preserves existing messages.
    """
    if update is None:
        return existing or []
    if any(isinstance(m, dict) and m.get("content") == "__CLEAR__" for m in update):
        return []
    if not update:
        return existing or []
    return (existing or []) + update


class AgentState(TypedDict, total=False):
    messages: Annotated[List[dict], reduce_messages]
    current_query: str
    documents: List[Any]
    plan: List[str]
    status: str
    final_answer: str
    execution_metadata: Dict[str, Any]
    retry_count: int
