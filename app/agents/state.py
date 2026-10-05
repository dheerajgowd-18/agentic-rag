from typing import TypedDict, List, Annotated, Any


def reduce_messages(existing: List[dict], update: List[dict]) -> List[dict]:
    """
    Custom reducer for conversation messages.
    Appends new messages during normal flow, but allows clearing history
    if an empty list is passed or if a clear marker is present.
    """
    if not update:
        return []
    if any(m.get("content") == "__CLEAR__" for m in update):
        return []
    return (existing or []) + update


class AgentState(TypedDict):
    messages: Annotated[List[dict], reduce_messages]
    current_query: str
    documents: List[Any]
    plan: List[str]
    status: str
    final_answer: str

