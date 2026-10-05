import nest_asyncio
nest_asyncio.apply()

import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    YAML_CONTENT,
    SAFETY_BLOCKED_INDICATORS,
    DIALOG_INDICATORS,
    RAIL_INDICATORS,
)


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    """
    global _rails

    guard_model = getattr(settings, "GROQ_GUARD_MODEL", "openai/gpt-oss-20b")
    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=guard_model,
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised ({guard_model}).")


async def guard_async(message: str) -> tuple[bool, bool, str | None]:
    """
    Asynchronously run a user message through the NeMo rails gate.

    Returns:
        (is_blocked, is_dialog, content)
        - (True, False, text) : Safety block (jailbreak / off-topic).
        - (False, True, text) : Conversational dialog rail (greeting, capabilities, farewell).
        - (False, False, None): Clean query proceeding to agent graph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = await _rails.generate_async(messages=[{"role": "user", "content": message}])
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        if any(indicator in content for indicator in SAFETY_BLOCKED_INDICATORS):
            logfire.info(f"🛡️ Guardrails safety block fired | query='{message[:80]}'")
            return True, False, content

        if any(indicator in content for indicator in DIALOG_INDICATORS):
            logfire.info(f"💬 Guardrails dialog flow triggered | query='{message[:80]}'")
            return False, True, content

        logfire.info("✅ Guardrails passed.")
        return False, False, None


def guard(message: str) -> tuple[bool, bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (is_blocked, is_dialog, content)
        - (True, False, text) : Safety block (jailbreak / off-topic).
        - (False, True, text) : Conversational dialog rail (greeting, capabilities, farewell).
        - (False, False, None): Clean query proceeding to agent graph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        content = result.get("content", "") if isinstance(result, dict) else str(result)

        if any(indicator in content for indicator in SAFETY_BLOCKED_INDICATORS):
            logfire.info(f"🛡️ Guardrails safety block fired | query='{message[:80]}'")
            return True, False, content

        if any(indicator in content for indicator in DIALOG_INDICATORS):
            logfire.info(f"💬 Guardrails dialog flow triggered | query='{message[:80]}'")
            return False, True, content

        logfire.info("✅ Guardrails passed.")
        return False, False, None

