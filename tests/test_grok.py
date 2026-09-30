from __future__ import annotations

import json
import os
from types import SimpleNamespace

import httpx
import pytest

from codex_oauth import CodexOAuthClient, GrokAPIError, GrokOAuthClient, create_client
from codex_oauth.grok import DEFAULT_GROK_BASE_URL
from codex_oauth.grok_auth import GrokTokens


@pytest.fixture(autouse=True)
def auth_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("GROK_OAUTH_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GROK_OAUTH_ACCESS_TOKEN", "grok-access-secret")
    monkeypatch.setenv("GROK_OAUTH_REFRESH_TOKEN", "grok-refresh-secret")
    monkeypatch.setenv("CODEX_OAUTH_ACCESS_TOKEN", "never-use-codex")
    monkeypatch.setenv("CODEX_OAUTH_ACCOUNT_ID", "never-use-account")
    monkeypatch.setenv("GROK_OAUTH_CLIENT_VERSION", "0.2.16")  # test fixture, not a current-version claim


def completion(content="hello", **fields):
    return {"id": "chat-test", "model": "grok-build", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": content, **fields}, "finish_reason": "stop"},
    ], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}


def chunk(delta=None, finish=None):
    return {"id": "chat-test", "model": "grok-build", "choices": [
        {"index": 0, "delta": delta or {}, "finish_reason": finish},
    ]}


def sse(events, done=True):
    text = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          text=text + ("data: [DONE]\n\n" if done else ""))


@pytest.mark.parametrize("tool", [
    {"type": "function", "function": {"name": "search", "parameters": {}, "strict": True}},
    {"type": "function", "name": "search", "parameters": {}, "strict": True},
    {"name": "search", "input_schema": {}, "strict": True},
])
async def test_complete_headers_tools_and_usage(tool):
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == DEFAULT_GROK_BASE_URL + "/chat/completions"
        assert request.headers["authorization"] == "Bearer grok-access-secret"
        assert "chatgpt-account-id" not in request.headers
        assert request.headers["x-grok-client-version"] == "0.2.16"
        assert request.headers["x-xai-token-auth"] == "xai-grok-cli"
        assert request.headers["x-authenticateresponse"] == "authenticate-response"
        assert request.headers["x-grok-client-identifier"] == "codex-oauth"
        assert request.headers["x-grok-client-mode"] == "headless"
        payload = json.loads(request.content)
        assert payload["model"] == "grok-build"
        assert payload["stream"] is False
        assert "input" not in payload and "store" not in payload
        assert payload["messages"][0] == {"role": "system", "content": "Be brief"}
        assert payload["tools"][0]["function"] == {"name": "search", "parameters": {}, "strict": True}
        assert payload["tool_choice"] == "auto"
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with GrokOAuthClient(model="grok/grok-build", http_client=http) as client:
            result = await client.complete(input="Hi", instructions="Be brief", tools=[tool],
                                           extra_body={"tool_choice": "auto"})
        assert not http.is_closed
    assert len(calls) == 1
    assert result.content == "hello" and result.usage.total_tokens == 5
    assert result.raw["id"] == "chat-test"


async def test_tool_call_round_trip_preserves_chat_history():
    history = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "call-1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
    ]}, {"role": "tool", "tool_call_id": "call-1", "content": "found"}]
    original = json.dumps(history)

    def handler(request):
        assert json.loads(request.content)["messages"] == history
        return httpx.Response(200, json=completion(None, tool_calls=[
            {"id": "call-2", "type": "function", "function": {"name": "next", "arguments": '{"x":1}'}},
        ], reasoning_content="separate reasoning"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await GrokOAuthClient(http_client=http).complete(messages=history)
    assert json.dumps(history) == original
    assert result.content == ""
    assert result.tool_calls[0].id == "call-2"
    assert result.tool_calls[0].arguments == '{"x":1}'
    assert result.raw["choices"][0]["message"]["reasoning_content"] == "separate reasoning"


@pytest.mark.parametrize("always_reject", [False, True])
async def test_401_refreshes_once(always_reject):
    sent, refreshes = [], []

    def handler(request):
        if request.url.host == "auth.x.ai":
            refreshes.append(request)
            return httpx.Response(200, json={"access_token": "new-secret", "expires_in": 3600,
                                            "refresh_token": "new-refresh-secret"})
        sent.append(request.headers["authorization"])
        if always_reject or len(sent) == 1:
            return httpx.Response(401, text="grok-access-secret")
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GrokOAuthClient(http_client=http)
        if always_reject:
            with pytest.raises(GrokAPIError) as error:
                await client.complete(input="hi")
            assert error.value.status_code == 401
        else:
            assert (await client.complete(input="hi")).content == "hello"
    assert sent == ["Bearer grok-access-secret", "Bearer new-secret"]
    assert len(refreshes) == 1


@pytest.mark.parametrize("status", [403, 426, 429, 500, 307])
async def test_no_retry_or_body_echo_for_other_statuses(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"location": "https://untrusted.example"},
                              text="grok-access-secret grok-refresh-secret")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as http:
        with pytest.raises(GrokAPIError) as error:
            await GrokOAuthClient(http_client=http).complete(input="hi")
    assert error.value.status_code == status
    assert "grok-access-secret" not in str(error.value)
    assert "grok-refresh-secret" not in str(error.value)
    assert len(calls) == 1


async def test_list_models_uses_grok_session():
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "grok-build"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert (await GrokOAuthClient(http_client=http).list_models())["data"][0]["id"] == "grok-build"


async def test_stream_preserves_text_tool_deltas_and_usage_after_401():
    events = [chunk({"content": "hi"}), chunk({"tool_calls": [
        {"index": 0, "id": "call-1", "type": "function", "function": {"name": "search", "arguments": '{"q":'}},
    ]}), chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"hi"}'}}]}, "tool_calls"),
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}]
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.host == "auth.x.ai":
            return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})
        assert json.loads(request.content)["stream"] is True
        if len(calls) == 1:
            return httpx.Response(401)
        return sse(events)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        actual = [event async for event in GrokOAuthClient(http_client=http).stream_events(input="hi")]
    assert actual == events
    assert len(calls) == 3


@pytest.mark.parametrize("text", [
    ': comment\r\nevent: message\r\ndata:{"choices":\r\ndata:[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\r\n\r\ndata:[DONE]\r\n\r\n',
    'data:{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}',
])
async def test_sse_multiline_no_space_and_final_unframed_event(text):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, text=text,
    ))) as http:
        events = [e async for e in GrokOAuthClient(http_client=http).stream_events(input="hi")]
    assert events[0]["choices"][0]["delta"]["content"] == "hi"


@pytest.mark.parametrize("tail", [
    'data: {"error":{"message":"grok-access-secret"}}\n\n',
    'data: {broken-json}\n\n',
    '',
])
async def test_stream_error_or_truncation_never_replays(tail):
    count = []
    text = "data:" + json.dumps(chunk({"content": "partial"})) + "\n\n" + tail

    def handler(request):
        count.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)

    received = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(GrokAPIError) as error:
            async for event in GrokOAuthClient(http_client=http).stream_events(input="hi"):
                received.append(event)
    assert len(received) == 1 and len(count) == 1
    assert "grok-access-secret" not in str(error.value)


async def test_interrupted_network_stream_closes_without_replay():
    class BrokenStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield ("data:" + json.dumps(chunk({"content": "partial"})) + "\n\n").encode()
            raise httpx.ReadError("connection dropped")

        async def aclose(self):
            self.closed = True

    stream = BrokenStream()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(GrokAPIError, match="not replayed"):
            _ = [e async for e in GrokOAuthClient(http_client=http).stream_events(input="hi")]
    assert len(calls) == 1 and stream.closed


@pytest.mark.parametrize("body", [[], {}, {"choices": []}, {"error": {"message": "secret"}}])
async def test_malformed_completion_raises(body):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))) as http:
        with pytest.raises(GrokAPIError):
            await GrokOAuthClient(http_client=http).complete(input="hi")


@pytest.mark.parametrize("kwargs", [
    {}, {"input": "x", "messages": []}, {"messages": []}, {"input": []},
    {"input": "x", "model": "gpt-5.5"}, {"input": "x", "model": "codex/gpt-5.5"},
    {"input": "x", "extra_body": {"stream": True}}, {"input": "x", "extra_body": {"n": 2}},
    {"input": "x", "tools": [{"type": "web_search"}]}, {"input": "x", "tools": [{}]},
])
def test_bad_payloads_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        GrokOAuthClient()._payload(**kwargs)


def test_factory_preserves_default_codex():
    assert isinstance(create_client(), CodexOAuthClient)
    assert isinstance(create_client("grok"), GrokOAuthClient)
    with pytest.raises(ValueError):
        create_client("unknown")


@pytest.mark.parametrize("header", ["Authorization", "authorization", "Cookie", "ChatGPT-Account-ID", "Host"])
def test_credential_and_routing_headers_cannot_be_overridden(header):
    with pytest.raises(ValueError):
        GrokOAuthClient(extra_headers={header: "secret"})


def test_version_detection_and_missing_version(monkeypatch):
    monkeypatch.delenv("GROK_OAUTH_CLIENT_VERSION")
    monkeypatch.setattr("codex_oauth.grok.shutil.which", lambda name: None)
    with pytest.raises(GrokAPIError, match="version"):
        GrokOAuthClient()._headers(GrokTokens("access"))
    monkeypatch.setattr("codex_oauth.grok.shutil.which", lambda name: "/test/grok")
    monkeypatch.setattr("codex_oauth.grok.subprocess.run", lambda *a, **k: SimpleNamespace(stdout="grok 0.2.16 (test)"))
    assert GrokOAuthClient()._headers(GrokTokens("access"))["x-grok-client-version"] == "0.2.16"


async def test_raw_stream_flag_is_not_misleading():
    with pytest.raises(ValueError, match="stream_events"):
        await GrokOAuthClient().create_response(input="hi", stream=True)
