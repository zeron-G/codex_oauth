"""Explicitly opt-in: consumes the caller's real Grok account allowance."""
from __future__ import annotations

import os

import pytest

from codex_oauth import GrokOAuthClient


@pytest.mark.asyncio
async def test_live_grok_oauth_llm_call() -> None:
    if os.environ.get("GROK_OAUTH_LIVE") != "1":
        pytest.skip("Set GROK_OAUTH_LIVE=1 to run a real Grok OAuth call.")
    # Unlike offline fixtures, never inject fake tokens or hide missing credentials.
    model = os.environ.get("GROK_OAUTH_LIVE_MODEL", "grok-build")
    async with GrokOAuthClient(model=model) as client:
        response = await client.complete(input="Reply with exactly: grok_oauth_live_ok")
    assert "grok_oauth_live_ok" in response.content
    assert response.model
