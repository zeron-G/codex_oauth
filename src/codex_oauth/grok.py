"""Experimental Grok CLI OAuth-backed Chat Completions transport."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

from .grok_auth import GrokAuth, GrokTokens, _validate_url
from .types import LLMResponse, ToolCall, Usage

DEFAULT_GROK_BASE_URL = "https://cli-chat-proxy.grok.com/v1"
DEFAULT_GROK_MODEL = "grok-build"


class GrokAPIError(RuntimeError):
    """Grok transport/protocol error. Response bodies and credentials are not echoed."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _client_version(configured: str | None) -> str:
    value = configured or os.environ.get("GROK_OAUTH_CLIENT_VERSION", "").strip()
    if not value:
        executable = shutil.which("grok")
        if executable:
            try:
                result = subprocess.run(
                    [executable, "--version"], capture_output=True, text=True,
                    timeout=5, check=True,
                )
                match = re.search(r"\b(\d+\.\d+\.\d+(?:[-+][\w.-]+)?)\b", result.stdout)
                value = match.group(1) if match else ""
            except (OSError, subprocess.SubprocessError):
                pass
    if not value or not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][\w.-]+)?", value):
        raise GrokAPIError(
            "Cannot determine Grok CLI version. Install/update the official CLI, or set "
            "GROK_OAUTH_CLIENT_VERSION to the version reported by `grok --version`."
        )
    return value


def _http_error(status: int) -> GrokAPIError:
    details = {
        401: "OAuth session rejected; run `grok login --oauth` again.",
        403: "Account/model entitlement or policy denied. No API-key fallback was attempted.",
        426: "CLI version rejected; update Grok and check GROK_OAUTH_CLIENT_VERSION.",
        429: "Rate/quota limit reached. No automatic retry or account switching was attempted.",
    }
    return GrokAPIError(
        f"Grok API error (HTTP {status}). " + details.get(status, "Request failed; upstream body omitted."),
        status_code=status,
    )


def _tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        if tool.get("type", "function") != "function":
            raise ValueError("Grok adapter supports function tools only.")
        if isinstance(tool.get("function"), Mapping):
            function = dict(tool["function"])
        else:
            function = {k: tool[k] for k in ("name", "description", "parameters", "strict") if k in tool}
            if "input_schema" in tool and "parameters" not in function:
                function["parameters"] = tool["input_schema"]
        if not isinstance(function.get("name"), str) or not function["name"].strip():
            raise ValueError("Each function tool must have a nonempty name.")
        result.append({"type": "function", "function": function})
    return result


async def _sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Parse SSE frames, including multiline data and a final unframed event."""
    lines: list[str] = []
    finished = False
    async for line in response.aiter_lines():
        line = line.removeprefix("\ufeff")
        if line.startswith("data:"):
            value = line[5:]
            lines.append(value[1:] if value.startswith(" ") else value)
        elif line == "" and lines:
            data = "\n".join(lines)
            lines.clear()
            if data.strip() == "[DONE]":
                return
            event = _decode_event(data)
            finished = finished or any(
                isinstance(c, dict) and c.get("finish_reason") is not None
                for c in event.get("choices", [])
            )
            yield event
    if lines:
        data = "\n".join(lines)
        if data.strip() == "[DONE]":
            return
        event = _decode_event(data)
        finished = finished or any(
            isinstance(c, dict) and c.get("finish_reason") is not None
            for c in event.get("choices", [])
        )
        yield event
    if not finished:
        raise GrokAPIError("Grok stream ended without a completion marker; response may be incomplete.")


def _decode_event(data: str) -> dict[str, Any]:
    try:
        event = json.loads(data)
    except ValueError:
        raise GrokAPIError("Grok stream returned malformed JSON.") from None
    if not isinstance(event, dict) or "error" in event or event.get("type") == "error":
        raise GrokAPIError("Grok stream returned an error or invalid event; body omitted.")
    if not isinstance(event.get("choices", []), list):
        raise GrokAPIError("Grok stream returned invalid choices.")
    return event


class GrokOAuthClient:
    """Async Grok client. complete() shares Codex's LLMResponse return type.

    create_response() returns native Chat Completion JSON; stream_events() yields
    native Chat Completion chunks, NOT Responses API events. Use one client/auth
    instance per event loop. This class is an SDK, not an HTTP proxy server.
    """

    def __init__(
        self, *, model: str = DEFAULT_GROK_MODEL, auth: GrokAuth | None = None,
        auth_path: str | os.PathLike[str] | None = None, base_url: str | None = None,
        client_version: str | None = None, http_client: httpx.AsyncClient | None = None,
        timeout: httpx.Timeout | float | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.auth = auth or GrokAuth(auth_path)
        self.base_url = _validate_url(
            base_url or os.environ.get("GROK_OAUTH_BASE_URL", "").strip() or DEFAULT_GROK_BASE_URL
        )
        self.client_version = client_version
        self.extra_headers = dict(extra_headers or {})
        reserved = {"authorization", "proxy-authorization", "cookie", "host", "chatgpt-account-id"}
        if reserved.intersection(key.lower() for key in self.extra_headers):
            raise ValueError("Do not override credential/routing headers in extra_headers.")
        self._external_client = http_client
        self._client = http_client
        self._timeout = timeout if timeout is not None else httpx.Timeout(
            connect=10.0, read=300.0, write=30.0, pool=10.0,
        )

    async def __aenter__(self) -> GrokOAuthClient:
        await self._get_client()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._external_client is None:
            await self._client.aclose()
        self._client = self._external_client

    def _headers(self, tokens: GrokTokens, *, stream: bool = False) -> dict[str, str]:
        headers = httpx.Headers(self.extra_headers)
        headers.update({
            "Authorization": f"Bearer {tokens.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        })
        if urlsplit(self.base_url).hostname == "cli-chat-proxy.grok.com":
            self.client_version = _client_version(self.client_version)
            headers["x-grok-client-version"] = self.client_version
            headers["X-XAI-Token-Auth"] = "xai-grok-cli"
            headers["x-authenticateresponse"] = "authenticate-response"
            headers["x-grok-client-identifier"] = "codex-oauth"
            headers["x-grok-client-mode"] = "headless"
        return dict(headers)

    def _payload(
        self, *, messages: Sequence[Mapping[str, Any]] | None = None,
        input: str | None = None, instructions: str | None = None,
        model: str | None = None, tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None, stream: bool = False,
    ) -> dict[str, Any]:
        if (messages is None) == (input is None):
            raise ValueError("Provide exactly one of messages or input.")
        if input is not None and not isinstance(input, str):
            raise ValueError("Grok input accepts a string; use Chat-style messages for tool history.")
        selected = (model or self.model).strip().removeprefix("grok/")
        if not selected or selected.startswith(("codex/", "gpt-")):
            raise ValueError("Select a Grok model; Codex/GPT models need the Codex provider.")
        history = [dict(message) for message in messages] if messages is not None else [
            {"role": "user", "content": input},
        ]
        if instructions is not None:
            history.insert(0, {"role": "system", "content": instructions})
        if not history:
            raise ValueError("messages must not be empty.")
        payload: dict[str, Any] = {"model": selected, "messages": history, "stream": stream}
        if tools is not None:
            payload["tools"] = _tools(tools)
        if extra_body:
            if {"model", "messages", "input", "instructions", "tools", "stream"}.intersection(extra_body):
                raise ValueError("Use named arguments instead of overriding routing/input fields in extra_body.")
            if extra_body.get("n", 1) != 1:
                raise ValueError("This adapter supports one completion per request (n=1).")
            payload.update(extra_body)
        return payload

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        client = await self._get_client()
        tokens = await self.auth.get_tokens(http_client=client)
        for attempt in range(2):
            try:
                response = await client.request(
                    method, f"{self.base_url}{path}", headers=self._headers(tokens), json=payload,
                    follow_redirects=False, auth=None, timeout=self._timeout,
                )
            except httpx.HTTPError:
                raise GrokAPIError("Grok request transport failed; no automatic replay was attempted.") from None
            if response.status_code == 401 and attempt == 0:
                tokens = await self.auth.get_tokens(
                    http_client=client, force_refresh=True, rejected_access_token=tokens.access_token,
                )
                continue
            if not 200 <= response.status_code < 300:
                raise _http_error(response.status_code)
            try:
                data = response.json()
            except ValueError:
                raise GrokAPIError("Grok API returned invalid JSON.") from None
            if not isinstance(data, dict) or "error" in data:
                raise GrokAPIError("Grok API returned an invalid response or an error; body omitted.")
            return data
        raise GrokAPIError("Grok authentication retry exhausted.")

    async def list_models(self) -> dict[str, Any]:
        """Return the account-visible /models payload; availability is server-side."""
        return await self._request("GET", "/models")

    async def create_response(
        self, *, messages: Sequence[Mapping[str, Any]] | None = None,
        input: str | None = None, instructions: str | None = None,
        model: str | None = None, tools: Sequence[Mapping[str, Any]] | None = None,
        stream: bool = False, extra_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return native Chat Completion JSON. Use stream_events for streaming."""
        if stream:
            raise ValueError("Use stream_events() for streaming responses.")
        payload = self._payload(
            messages=messages, input=input, instructions=instructions, model=model,
            tools=tools, extra_body=extra_body,
        )
        return await self._request("POST", "/chat/completions", payload)

    async def complete(
        self, *, messages: Sequence[Mapping[str, Any]] | None = None,
        input: str | None = None, instructions: str | None = None,
        model: str | None = None, tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        data = await self.create_response(
            messages=messages, input=input, instructions=instructions, model=model,
            tools=tools, extra_body=extra_body,
        )
        try:
            message = data["choices"][0]["message"]
            text = message.get("content") or ""
            if not isinstance(text, str):
                raise ValueError
            calls = []
            for call in message.get("tool_calls") or []:
                function = call["function"]
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                calls.append(ToolCall(id=call["id"], name=function["name"], arguments=arguments))
            usage = Usage.from_response_usage(data.get("usage"))
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise GrokAPIError("Grok returned a malformed Chat Completion.") from None
        return LLMResponse(
            content=text, model=data.get("model") or (model or self.model).removeprefix("grok/"),
            tool_calls=calls, usage=usage, raw=data,
        )

    async def stream_events(
        self, *, messages: Sequence[Mapping[str, Any]] | None = None,
        input: str | None = None, instructions: str | None = None,
        model: str | None = None, tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield native Chat Completion chunks, including tool_calls and usage."""
        payload = self._payload(
            messages=messages, input=input, instructions=instructions, model=model,
            tools=tools, extra_body=extra_body, stream=True,
        )
        client = await self._get_client()
        tokens = await self.auth.get_tokens(http_client=client)
        for attempt in range(2):
            try:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", headers=self._headers(tokens, stream=True),
                    json=payload, follow_redirects=False, auth=None, timeout=self._timeout,
                ) as response:
                    if response.status_code != 401 or attempt == 1:
                        if not 200 <= response.status_code < 300:
                            raise _http_error(response.status_code)
                        if "text/event-stream" not in response.headers.get("content-type", "").lower():
                            raise GrokAPIError("Expected a Grok SSE response, not a JSON/HTML body.")
                        async for event in _sse(response):
                            yield event
                        return
                # Only retry an HTTP 401 before any event is delivered; close first.
                tokens = await self.auth.get_tokens(
                    http_client=client, force_refresh=True, rejected_access_token=tokens.access_token,
                )
            except httpx.HTTPError:
                raise GrokAPIError("Grok stream transport failed; partial output was not replayed.") from None
