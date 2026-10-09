import json
import re
from typing import Literal, Optional
from pydantic import BaseModel, Field
import logfire

from app.agents.state import AgentState
from app.gateway import get_langchain_llm

# Portkey-backed LLM: fallback + cache + retry
llm = get_langchain_llm(feature="planner")


class PlannerDecision(BaseModel):
    """Structured and validated representation of the planner's routing decision."""
    intent: Literal["CONVERSATIONAL", "TECHNICAL"] = Field(
        ...,
        description="'CONVERSATIONAL' for greetings and direct conversation memory. 'TECHNICAL' for technical questions requiring documentation retrieval."
    )
    search_query: str = Field(
        default="",
        description="Refined keyword-dense search query. Required for TECHNICAL intent; empty for CONVERSATIONAL."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the routing classification."
    )


def build_planner_prompt(messages: list[dict]) -> str:
    history = ""
    for msg in messages[:-1]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        history += f"{role}: {content}\n"

    user_message = messages[-1].get("content", "") if messages else ""

    return f"""
You are an intelligent Assistant Planner in an enterprise knowledge system.
Analyze the conversation history and the latest user message to route the query.

CONVERSATION HISTORY:
{history.strip() if history.strip() else "(No prior conversation)"}

LATEST USER MESSAGE:
"{user_message}"

ROUTING RULES:
1. Intent: CONVERSATIONAL
   - Pure greetings (e.g. "hi", "hello", "hey", "good morning")
   - Pure conversational memory questions that can be answered SOLELY using the conversation history above (e.g. "what is my name?", "what did I just ask?").
   - For this intent, the search query MUST be empty.

2. Intent: TECHNICAL
   - ANY technical question about Kubernetes, Intel hardware, networking, system architecture, Linux, or cloud infrastructure.
   - ANY technical follow-up question (e.g. "how do I configure that?", "what about for CPU limits?", "can it scale horizontally?").
   - NEVER classify a technical follow-up as CONVERSATIONAL just because it refers to previous context.
   - For TECHNICAL intent, output a standalone, refined search query resolving any pronouns or coreferences from history.
   - The search query MUST NOT be empty.

Output your decision as a valid JSON object with the following keys:
{{
  "intent": "CONVERSATIONAL" or "TECHNICAL",
  "search_query": "refined search query or empty string",
  "reasoning": "brief explanation"
}}
"""


def parse_planner_decision(raw_output: str, latest_user_msg: str) -> PlannerDecision:
    """
    Robust parser for planner output. Handles structured JSON, plain text fallback,
    and guarantees non-empty search queries for TECHNICAL requests.
    """
    clean_text = raw_output.strip()

    # 1. Try JSON parsing (direct or embedded markdown code block, non-greedy first)
    for pattern in (r"\{.*?\}", r"\{.*\}"):
        json_match = re.search(pattern, clean_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict):
                    intent = str(parsed.get("intent") or "").upper().strip()
                    query = str(parsed.get("search_query") or "").strip()
                    reasoning = str(parsed.get("reasoning") or "").strip()

                    if intent in ("CONVERSATIONAL", "TECHNICAL"):
                        if intent == "TECHNICAL" and not query:
                            query = latest_user_msg.strip()
                        if intent == "TECHNICAL" and not query:
                            intent = "CONVERSATIONAL"
                        return PlannerDecision(
                            intent=intent,
                            search_query=query if intent == "TECHNICAL" else "",
                            reasoning=reasoning or "Parsed from JSON output"
                        )
            except Exception:
                pass

    # 2. Handle partial or malformed JSON via regex extraction
    intent_match = re.search(r'["\']?intent["\']?\s*:\s*["\']?(TECHNICAL|CONVERSATIONAL)["\']?', clean_text, re.IGNORECASE)
    if intent_match:
        intent = intent_match.group(1).upper()
        query_match = re.search(r'["\']?search_query["\']?\s*:\s*["\']([^"\'\n\r]+)["\']', clean_text, re.IGNORECASE)
        query = query_match.group(1).strip() if query_match else ""
        if intent == "TECHNICAL" and not query:
            query = latest_user_msg.strip()
        if intent == "TECHNICAL" and not query:
            intent = "CONVERSATIONAL"
        return PlannerDecision(
            intent=intent,
            search_query=query if intent == "TECHNICAL" else "",
            reasoning="Extracted from partial or malformed JSON"
        )

    # 3. Text heuristics for fallback models
    upper_text = clean_text.upper()
    if upper_text == "CONVERSATIONAL" or (upper_text.startswith("CONVERSATIONAL") and len(clean_text) < 30):
        return PlannerDecision(intent="CONVERSATIONAL", search_query="", reasoning="Direct text match: CONVERSATIONAL")

    # If the text has JSON syntax tokens but couldn't be parsed, fallback safely to user question
    if clean_text.startswith("{") or "{" in clean_text or "search_query" in clean_text.lower():
        fallback_query = latest_user_msg.strip()
        if not fallback_query:
            return PlannerDecision(
                intent="CONVERSATIONAL",
                search_query="",
                reasoning="Fallback to conversational due to empty user query"
            )
        return PlannerDecision(
            intent="TECHNICAL",
            search_query=fallback_query,
            reasoning="Fallback to user query from unparseable JSON-like output"
        )

    # 4. Clean search query string if model returned plain query
    candidate_query = clean_text.replace('"', '').replace("'", "").strip()
    # Strip any "Search query:" or "Refined query:" prefix
    candidate_query = re.sub(r"^(refined\s+)?(search\s+)?query:\s*", "", candidate_query, flags=re.IGNORECASE).strip()

    if not candidate_query:
        candidate_query = latest_user_msg.strip()

    if not candidate_query:
        return PlannerDecision(
            intent="CONVERSATIONAL",
            search_query="",
            reasoning="Empty query inferred as conversational"
        )

    return PlannerDecision(
        intent="TECHNICAL",
        search_query=candidate_query,
        reasoning="Inferred technical query from text output"
    )


def evaluate_planner(messages: list[dict]) -> PlannerDecision:
    """Invokes the planner LLM to determine intent and query with structured fallback."""
    if not messages:
        return PlannerDecision(intent="CONVERSATIONAL", search_query="", reasoning="Empty message history")

    latest_user_msg = messages[-1].get("content", "")
    prompt = build_planner_prompt(messages)

    with logfire.span("🧠 Planner Decision"):
        try:
            raw_response = llm.invoke(prompt).content
            decision = parse_planner_decision(raw_response, latest_user_msg)
            logfire.info(f"Intent identified: {decision.intent} | Query: '{decision.search_query}'")
            return decision
        except Exception as e:
            logfire.error(f"Planner LLM invocation failed: {e}. Falling back to rule-based routing.")
            # Deterministic heuristic fallback
            cleaned = latest_user_msg.lower().strip()
            if cleaned in ("hi", "hello", "hey", "howdy", "good morning", "good afternoon"):
                return PlannerDecision(intent="CONVERSATIONAL", search_query="", reasoning="Heuristic greeting fallback")
            return PlannerDecision(
                intent="TECHNICAL",
                search_query=latest_user_msg.strip(),
                reasoning="Heuristic fallback due to planner failure"
            )


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    Returns structured execution metadata and updates the workflow plan.
    """
    decision = evaluate_planner(state.get("messages", []))

    if decision.intent == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
            "execution_metadata": {
                "planner_intent": "CONVERSATIONAL",
                "retrieval_executed": False,
                "search_query": "",
                "reasoning": decision.reasoning,
            }
        }

    return {
        "current_query": decision.search_query,
        "status": f"Technical research needed. Searching for: {decision.search_query}",
        "plan": ["Intent: Technical", f"Search Term: {decision.search_query}"],
        "execution_metadata": {
            "planner_intent": "TECHNICAL",
            "retrieval_executed": True,
            "search_query": decision.search_query,
            "reasoning": decision.reasoning,
        }
    }
