import os
import nest_asyncio
nest_asyncio.apply()

from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
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


class GuardrailOutcome(str, Enum):
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"
    DIALOG = "DIALOG"
    ERROR = "ERROR"


class GuardrailResult:
    """Structured representation of internal guardrail decision."""
    def __init__(
        self,
        outcome: GuardrailOutcome,
        content: Optional[str] = None,
        reason: Optional[str] = None,
        raw_events: Optional[List[Dict[str, Any]]] = None,
    ):
        self.outcome = outcome
        self.content = content
        self.reason = reason
        self.raw_events = raw_events or []

    @property
    def is_blocked(self) -> bool:
        return self.outcome == GuardrailOutcome.BLOCKED

    @property
    def is_dialog(self) -> bool:
        return self.outcome == GuardrailOutcome.DIALOG

    @property
    def is_allowed(self) -> bool:
        return self.outcome == GuardrailOutcome.ALLOWED

    @property
    def is_error(self) -> bool:
        return self.outcome == GuardrailOutcome.ERROR

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome.value,
            "content": self.content,
            "reason": self.reason,
            "is_blocked": self.is_blocked,
            "is_dialog": self.is_dialog,
            "is_allowed": self.is_allowed,
        }

    def __repr__(self) -> str:
        return f"<GuardrailResult outcome={self.outcome.value} reason={self.reason}>"


_rails: LLMRails | None = None
_rails_init_error: Exception | None = None

# Fail-closed policy: If guardrails fails to initialize or execute, block the request rather than silently bypassing
FAIL_CLOSED = os.getenv("GUARDRAILS_FAIL_CLOSED", "true").lower() == "true"


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Records any initialization error explicitly.
    """
    global _rails, _rails_init_error

    try:
        guard_model = getattr(settings, "GROQ_GUARD_MODEL", "openai/gpt-oss-20b")
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured; cannot initialize Guardrails LLM.")

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
        _rails_init_error = None
        logfire.info(f"🛡️ NeMo Guardrails initialised ({guard_model}).")
    except Exception as e:
        _rails = None
        _rails_init_error = e
        logfire.error(f"❌ NeMo Guardrails initialization failed: {e}")


def _classify_rail_response(content: str, raw_result: Any) -> GuardrailResult:
    """Classifies model and rail output into structured internal decisions."""
    # Check for known safety blocked indicators
    for indicator in SAFETY_BLOCKED_INDICATORS:
        if indicator.lower() in content.lower():
            return GuardrailResult(
                outcome=GuardrailOutcome.BLOCKED,
                content=content,
                reason="Triggered safety policy (jailbreak or off-topic)",
            )

    # Check for dialog flow indicators
    for indicator in DIALOG_INDICATORS:
        if indicator.lower() in content.lower():
            return GuardrailResult(
                outcome=GuardrailOutcome.DIALOG,
                content=content,
                reason="Triggered standard dialog conversation flow",
            )

    return GuardrailResult(
        outcome=GuardrailOutcome.ALLOWED,
        content=None,
        reason="Query within allowable operating scope",
    )


async def guard_async_detailed(message: str) -> GuardrailResult:
    """Asynchronously runs a query through guardrails returning a structured GuardrailResult."""
    global _rails, _rails_init_error

    if _rails is None:
        if FAIL_CLOSED:
            err_detail = f"Guardrails uninitialized: {_rails_init_error}" if _rails_init_error else "Guardrails not initialized"
            logfire.error(f"🛡️ Guardrails gate failed closed: {err_detail}")
            return GuardrailResult(
                outcome=GuardrailOutcome.ERROR,
                content="Safety processing unavailable. Request blocked under fail-closed security policy.",
                reason=err_detail,
            )
        else:
            logfire.warning("⚠️ Guardrails uninitialized — skipping gate (fail-open configured).")
            return GuardrailResult(outcome=GuardrailOutcome.ALLOWED, reason="Fail-open fallback")

    with logfire.span("🛡️ Guardrails Check"):
        try:
            result = await _rails.generate_async(messages=[{"role": "user", "content": message}])
            content = result.get("content", "") if isinstance(result, dict) else str(result)
            decision = _classify_rail_response(content, result)
            logfire.info(f"Guardrail Decision: {decision.outcome.value} | Reason: {decision.reason}")
            return decision
        except Exception as e:
            logfire.error(f"Guardrails runtime error: {e}")
            if FAIL_CLOSED:
                return GuardrailResult(
                    outcome=GuardrailOutcome.ERROR,
                    content="Safety verification encountered an error. Request blocked.",
                    reason=f"Runtime error: {type(e).__name__}",
                )
            return GuardrailResult(outcome=GuardrailOutcome.ALLOWED, reason="Fail-open on runtime error")


def guard_detailed(message: str) -> GuardrailResult:
    """Synchronously runs a query through guardrails returning a structured GuardrailResult."""
    global _rails, _rails_init_error

    if _rails is None:
        if FAIL_CLOSED:
            err_detail = f"Guardrails uninitialized: {_rails_init_error}" if _rails_init_error else "Guardrails not initialized"
            logfire.error(f"🛡️ Guardrails gate failed closed: {err_detail}")
            return GuardrailResult(
                outcome=GuardrailOutcome.ERROR,
                content="Safety processing unavailable. Request blocked under fail-closed security policy.",
                reason=err_detail,
            )
        else:
            logfire.warning("⚠️ Guardrails uninitialized — skipping gate (fail-open configured).")
            return GuardrailResult(outcome=GuardrailOutcome.ALLOWED, reason="Fail-open fallback")

    with logfire.span("🛡️ Guardrails Check"):
        try:
            result = _rails.generate(messages=[{"role": "user", "content": message}])
            content = result.get("content", "") if isinstance(result, dict) else str(result)
            decision = _classify_rail_response(content, result)
            logfire.info(f"Guardrail Decision: {decision.outcome.value} | Reason: {decision.reason}")
            return decision
        except Exception as e:
            logfire.error(f"Guardrails runtime error: {e}")
            if FAIL_CLOSED:
                return GuardrailResult(
                    outcome=GuardrailOutcome.ERROR,
                    content="Safety verification encountered an error. Request blocked.",
                    reason=f"Runtime error: {type(e).__name__}",
                )
            return GuardrailResult(outcome=GuardrailOutcome.ALLOWED, reason="Fail-open on runtime error")


async def guard_async(message: str) -> Tuple[bool, bool, Optional[str]]:
    """
    Backward-compatible asynchronous guard function.
    Returns: (is_blocked, is_dialog, content)
    """
    result = await guard_async_detailed(message)
    if result.outcome == GuardrailOutcome.BLOCKED or result.outcome == GuardrailOutcome.ERROR:
        return True, False, result.content
    if result.outcome == GuardrailOutcome.DIALOG:
        return False, True, result.content
    return False, False, None


def guard(message: str) -> Tuple[bool, bool, Optional[str]]:
    """
    Backward-compatible synchronous guard function.
    Returns: (is_blocked, is_dialog, content)
    """
    result = guard_detailed(message)
    if result.outcome == GuardrailOutcome.BLOCKED or result.outcome == GuardrailOutcome.ERROR:
        return True, False, result.content
    if result.outcome == GuardrailOutcome.DIALOG:
        return False, True, result.content
    return False, False, None
