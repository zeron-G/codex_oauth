from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from codex_oauth.auth import CodexAuth, CodexTokens
from codex_oauth.client import CodexOAuthClient, messages_to_response_input


class StaticAuth(CodexAuth):
    def __init__(self) -> None:
        self.tokens = CodexTokens(access_token="access", account_id="acct", source="test")

    async def get_tokens(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> CodexTokens:
        return self.tokens


def test_messages_to_response_input() -> None:
    instructions, input_items = messages_to_response_input(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert instructions == "Be brief."
    assert input_items == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_complete_collects_streaming_response() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["payload"] = json.loads(request.content.decode())
        data = "\n\n".join(
            [
                'data: {"type":"response.output_text.delta","delta":"hel"}',
                'data: {"type":"response.output_text.delta","delta":"lo"}',
                'data: {"type":"response.completed","response":{"model":"gpt-5.5","output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}}',
            ]
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=data)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = CodexOAuthClient(
            model="codex/gpt-5.5",
            auth=StaticAuth(),
            base_url="https://codex.example/backend-api/codex",
            http_client=http_client,
        )
        response = await client.complete(messages=[{"role": "user", "content": "Hi"}])

    assert response.content == "hello"
    assert response.model == "gpt-5.5"
    assert response.usage.total_tokens == 3
    assert seen["payload"]["model"] == "gpt-5.5"
    assert seen["payload"]["stream"] is True
    assert seen["headers"]["authorization"] == "Bearer access"
    assert seen["headers"]["chatgpt-account-id"] == "acct"


@pytest.mark.asyncio
async def test_create_response_returns_raw_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        assert payload["input"] == "ping"
        assert payload["stream"] is False
        return httpx.Response(200, json={"id": "resp_1", "output": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = CodexOAuthClient(
            auth=StaticAuth(),
            base_url="https://codex.example/backend-api/codex",
            http_client=http_client,
        )
        raw = await client.create_response(input="ping")

    assert raw["id"] == "resp_1"

