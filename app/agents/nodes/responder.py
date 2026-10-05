import logfire
from app.agents.state import AgentState
from app.config import settings
from app.gateway import portkey_client, extract_cache_status


def build_responder_prompt(query: str, messages: list[dict], documents: list) -> str:
    """
    Builds the synthesis prompt from conversation history and technical documentation.
    Shared by generate_node and streaming endpoints to prevent prompt divergence.
    """
    history_str = ""
    for msg in messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = messages[-1]["content"] if messages else ""

    if query == "CONVERSATIONAL":
        return f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """

    max_context_chars = 25000
    full_context = ""
    for doc in documents:
        text = doc.get("content", "") if isinstance(doc, dict) else str(doc)
        formatted_chunk = f"CONTENT: {text}"
        if len(full_context) + len(formatted_chunk) < max_context_chars:
            full_context += formatted_chunk + "\n\n"
        else:
            logfire.warning("Context truncated to fit Groq TPM limits.")
            break

    return f"""
    You are a Senior Technical Architect.
    Answer the question using the TECHNICAL CONTEXT provided.

    TECHNICAL CONTEXT:
    {full_context}

    CONVERSATION HISTORY:
    {history_str}

    USER QUESTION:
    "{user_msg}"
    """


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]
    prompt = build_responder_prompt(query, state.get("messages", []), state.get("documents", []))


    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
