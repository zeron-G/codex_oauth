"""Local Codex and Grok OAuth-to-LLM transport helpers."""
from __future__ import annotations

from typing import Any

from .auth import (
    CodexAuth,
    CodexAuthError,
    CodexTokens,
    load_codex_tokens,
    token_expires_at,
    token_is_expired,
)
from .client import (
    CodexAPIError,
    CodexOAuthClient,
    CodexOAuthError,
    extract_text,
    extract_tool_calls,
    messages_to_response_input,
)
from .grok import GrokAPIError, GrokOAuthClient
from .grok_auth import GrokAuth, GrokAuthError, GrokTokens, load_grok_tokens
from .types import LLMResponse, ToolCall, Usage


def create_client(provider: str = "codex", **kwargs: Any) -> CodexOAuthClient | GrokOAuthClient:
    """Select a transport explicitly; credentials never route by model-name guessing."""
    if provider == "codex":
        return CodexOAuthClient(**kwargs)
    if provider == "grok":
        return GrokOAuthClient(**kwargs)
    raise ValueError("provider must be 'codex' or 'grok'.")


__all__ = [
    "CodexAPIError",
    "CodexAuth",
    "CodexAuthError",
    "CodexOAuthClient",
    "CodexOAuthError",
    "CodexTokens",
    "GrokAPIError",
    "GrokAuth",
    "GrokAuthError",
    "GrokOAuthClient",
    "GrokTokens",
    "LLMResponse",
    "ToolCall",
    "Usage",
    "create_client",
    "extract_text",
    "extract_tool_calls",
    "load_codex_tokens",
    "load_grok_tokens",
    "messages_to_response_input",
    "token_expires_at",
    "token_is_expired",
]
