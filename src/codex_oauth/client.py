"""Responses-style client for the local Codex OAuth session."""
from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from .auth import CodexAuth, CodexAuthError, CodexTokens
from .types import LLMResponse, ToolCall, Usage


DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_INSTRUCTIONS = "You are a helpful assistant."


class CodexOAuthError(RuntimeError):
    """Base error for this package."""


class CodexAPIError(CodexOAuthError):
    """Raised when the Codex backend returns an error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _base_url() -> str:
    return os.environ.get("CODEX_OAUTH_BASE_URL", DEFAULT_CODEX_BASE_URL).rstrip("/")


def _strip_model_prefix(model: str) -> str:
    return model.removeprefix("codex/").strip()


def messages_to_response_input(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert chat-style messages to a Responses-style instructions/input pair."""
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")

        if role == "system":
            if content:
                instructions.append(_stringify_content(content))
            continue

        if isinstance(content, list):
            input_items.extend(_content_blocks_to_input_items(role, content))
        elif content:
            input_items.append({"role": role, "content": str(content)})

    return "\n".join(part for part in instructions if part).strip(), input_items


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                text = block.get("text", block.get("content", ""))
                if text:
                    parts.append(str(text))
            elif block:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _content_blocks_to_input_items(role: str, blocks: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            if block:
                items.append({"role": role, "content": str(block)})
            continue

        block_type = str(block.get("type", ""))
        if block_type == "tool_use":
            items.append(
                {
                    "type": "function_call",
                    "name": str(block.get("name", "")),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    "call_id": str(block.get("id", "")),
                }
            )
        elif block_type == "tool_result":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(block.get("tool_use_id", "")),
                    "output": str(block.get("content", "")),
                }
            )
        else:
            text = block.get("text", block.get("content", ""))
            if text:
                items.append({"role": role, "content": str(text)})
    return items


def _normalize_tools(tools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools or []:
        if tool.get("type") == "function" and "name" in tool:
            normalized.append(dict(tool))
            continue

        if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
            fn = tool["function"]
            normalized.append(
                {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
            continue

        if "name" in tool:
            normalized.append(
                {
                    "type": "function",
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", tool.get("parameters", {})),
                }
            )
    return normalized


class CodexOAuthClient:
    """Small async client for Codex's ChatGPT OAuth-backed Responses transport."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        auth: CodexAuth | None = None,
        auth_path: str | os.PathLike[str] | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: httpx.Timeout | float | None = None,
        store: bool = False,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.auth = auth or CodexAuth(auth_path)
        self.base_url = (base_url or _base_url()).rstrip("/")
        self.store = store
        self.extra_headers = dict(extra_headers or {})
        self._external_client = http_client
        self._client: httpx.AsyncClient | None = http_client
        self._timeout = timeout or httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)

    async def __aenter__(self) -> "CodexOAuthClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None and self._external_client is None:
            await self._client.aclose()
        self._client = self._external_client

    async def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]] | None = None,
        input: str | list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        model: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Create a response and collect streaming SSE events into one result."""
        events: list[dict[str, Any]] = []
        async for event in self.stream_events(
            messages=messages,
            input=input,
            instructions=instructions,
            model=model,
            tools=tools,
            extra_body=extra_body,
        ):
            events.append(event)
        return _events_to_llm_response(events, model=_strip_model_prefix(model or self.model))

    async def create_response(
        self,
        *,
        messages: Sequence[Mapping[str, Any]] | None = None,
        input: str | list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        model: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        stream: bool = False,
        extra_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a raw response."""
        payload = self._build_payload(
            messages=messages,
            input=input,
            instructions=instructions,
            model=model,
            tools=tools,
            stream=stream,
            extra_body=extra_body,
        )
        return await self._post_json(payload)

    async def stream_events(
        self,
        *,
        messages: Sequence[Mapping[str, Any]] | None = None,
        input: str | list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        model: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed Responses SSE events as dictionaries."""
        payload = self._build_payload(
            messages=messages,
            input=input,
            instructions=instructions,
            model=model,
            tools=tools,
            stream=True,
            extra_body=extra_body,
        )
        try:
            async for event in self._stream_once(payload):
                yield event
        except CodexAPIError as exc:
            if exc.status_code != 401:
                raise
            await self.auth.get_tokens(http_client=await self._get_client(), force_refresh=True)
            async for event in self._stream_once(payload):
                yield event

    def _build_payload(
        self,
        *,
        messages: Sequence[Mapping[str, Any]] | None,
        input: str | list[dict[str, Any]] | None,
        instructions: str | None,
        model: str | None,
        tools: Sequence[Mapping[str, Any]] | None,
        stream: bool,
        extra_body: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if messages is not None:
            converted_instructions, converted_input = messages_to_response_input(messages)
            instructions = instructions if instructions is not None else converted_instructions
            input = converted_input

        if input is None:
            raise ValueError("Either messages or input must be provided.")

        payload: dict[str, Any] = {
            "model": _strip_model_prefix(model or self.model),
            "input": input,
            "stream": stream,
            "store": self.store,
        }
        payload["instructions"] = instructions or DEFAULT_INSTRUCTIONS

        normalized_tools = _normalize_tools(tools)
        if normalized_tools:
            payload["tools"] = normalized_tools

        if extra_body:
            payload.update(extra_body)
        return payload

    async def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._post_json_once(payload)
        except CodexAPIError as exc:
            if exc.status_code != 401:
                raise
            await self.auth.get_tokens(http_client=await self._get_client(), force_refresh=True)
            return await self._post_json_once(payload)

    async def _post_json_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = await self._get_client()
        tokens = await self.auth.get_tokens(http_client=client)
        response = await client.post(
            f"{self.base_url}/responses",
            headers=self._headers(tokens),
            json=payload,
        )
        if response.status_code >= 400:
            raise CodexAPIError(
                f"Codex API error {response.status_code}: {_safe_error_text(response)}",
                status_code=response.status_code,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise CodexAPIError("Codex API returned invalid JSON.") from exc

    async def _stream_once(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        client = await self._get_client()
        tokens = await self.auth.get_tokens(http_client=client)
        async with client.stream(
            "POST",
            f"{self.base_url}/responses",
            headers=self._headers(tokens, stream=True),
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = await _safe_stream_error_text(response)
                raise CodexAPIError(
                    f"Codex API error {response.status_code}: {body}",
                    status_code=response.status_code,
                )

            async for event in _aiter_sse_events(response):
                yield event

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _headers(self, tokens: CodexTokens, *, stream: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {tokens.access_token}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        if tokens.account_id:
            headers["ChatGPT-Account-ID"] = tokens.account_id
        headers.update(self.extra_headers)
        return headers


async def _aiter_sse_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


async def _safe_stream_error_text(response: httpx.Response) -> str:
    parts: list[str] = []
    async for chunk in response.aiter_text():
        parts.append(chunk)
        if sum(len(part) for part in parts) >= 500:
            break
    return "".join(parts)[:500]


def _safe_error_text(response: httpx.Response) -> str:
    return response.text[:500]


def _events_to_llm_response(events: Sequence[Mapping[str, Any]], *, model: str) -> LLMResponse:
    content_parts: list[str] = []
    completed_response: dict[str, Any] = {}

    for event in events:
        event_type = str(event.get("type", ""))
        if event_type == "response.output_text.delta":
            content_parts.append(str(event.get("delta", "")))
        elif event_type == "response.output_text.done" and not content_parts:
            content_parts.append(str(event.get("text", "")))
        elif event_type == "response.completed":
            response = event.get("response", {})
            if isinstance(response, dict):
                completed_response = response

    content = "".join(content_parts)
    if not content and completed_response:
        content = extract_text(completed_response)

    tool_calls = extract_tool_calls(completed_response)
    usage = Usage.from_response_usage(completed_response.get("usage") if completed_response else {})
    response_model = str(completed_response.get("model") or model)
    return LLMResponse(
        content=content,
        model=response_model,
        tool_calls=tool_calls,
        usage=usage,
        raw=completed_response or {"events": list(events)},
    )


def extract_text(response: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for block in item.get("content", []) or []:
            if isinstance(block, Mapping) and block.get("type") == "output_text":
                parts.append(str(block.get("text", "")))
    return "".join(parts)


def extract_tool_calls(response: Mapping[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, Mapping) or item.get("type") != "function_call":
            continue
        calls.append(
            ToolCall(
                id=str(item.get("call_id", item.get("id", ""))),
                name=str(item.get("name", "")),
                arguments=str(item.get("arguments", "{}")),
            )
        )
    return calls


__all__ = [
    "CodexAPIError",
    "CodexAuthError",
    "CodexOAuthClient",
    "CodexOAuthError",
    "extract_text",
    "extract_tool_calls",
    "messages_to_response_input",
]
