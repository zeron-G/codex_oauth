from __future__ import annotations

import os

import pytest

from codex_oauth import CodexOAuthClient


@pytest.mark.asyncio
async def test_live_codex_oauth_llm_call() -> None:
    if os.environ.get("CODEX_OAUTH_LIVE") != "1":
        pytest.skip("Set CODEX_OAUTH_LIVE=1 to run the live Codex OAuth smoke test.")
    if not os.environ.get("CODEX_OAUTH_ACCESS_TOKEN") and not os.environ.get("CODEX_OAUTH_AUTH_JSON"):
        default_auth = os.path.expanduser("~/.codex/auth.json")
        if not os.path.exists(default_auth):
            pytest.skip("No Codex OAuth token env vars or ~/.codex/auth.json available.")

    model = os.environ.get("CODEX_OAUTH_LIVE_MODEL", "gpt-5.5")
    async with CodexOAuthClient(model=model) as client:
        response = await client.complete(
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: codex_oauth_live_ok",
                }
            ]
        )

    assert "codex_oauth_live_ok" in response.content
    assert response.model

