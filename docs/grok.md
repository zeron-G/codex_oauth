# Grok OAuth provider

## Status and scope

This is an **experimental local SDK adapter**, not an official xAI SDK or a hosted OpenAI-compatible proxy server. It reuses an existing **Grok CLI OAuth login**, just as the Codex provider reuses an existing Codex login. Initial browser/device authentication remains the responsibility of the official CLI; this package does not implement another PKCE login server.

The Grok provider calls `https://cli-chat-proxy.grok.com/v1/chat/completions` using the selected xAI OAuth access token. It does not use Codex credentials, browser cookies, `XAI_API_KEY`, or silently fall back to the public paid API. Account access, model availability, rate limits and quota accounting are controlled by xAI. A SuperGrok or X subscription alone is **not a guarantee** that a particular CLI model is enabled. No live xAI account was used to validate this change; the automated coverage is offline/mock-based unless you explicitly enable the live test below.

Existing `CodexOAuthClient` imports and CLI commands keep their default Codex behavior. The two providers normalize `complete()` to the same `LLMResponse`, but their **raw** response and streaming protocols differ.

## Sign in and make a first request

Install the official Grok CLI using its [current installation documentation](https://github.com/xai-org/grok-build). Sign in on the same machine/user account that runs this SDK:

```sh
grok login --oauth
grok --version
python -m pip install -e .
codex-oauth --provider grok --list-models
codex-oauth --provider grok --model grok-build "Reply with exactly: hello"
codex-oauth --provider grok --stream "Explain SSE briefly."
```

For SSH/headless login, the official CLI also documents:

```sh
grok login --device-auth
```

Use a model ID actually returned for your account by `--list-models`. `grok-build` is the adapter's default alias, **not a promise of availability**. Select a different returned model with `--model` when needed.

The adapter sends `x-grok-client-version`, the proxy's OAuth middleware headers, an honest `codex-oauth` client identifier, and `headless` client mode. It determines the version from `grok --version`, or from `GROK_OAUTH_CLIENT_VERSION` / `--grok-client-version`. A version override must be the version reported by your installed CLI, not an invented version to evade an upgrade requirement. There is no hard-coded "latest version" fallback. If version detection fails, the request fails with setup guidance.

An alternate cache can be selected without copying its contents into code:

```sh
codex-oauth --provider grok --auth-path /private/path/auth.json "Hello"
```

## Python: shared normalized interface

```python
import asyncio
from codex_oauth import create_client

async def main() -> None:
    async with create_client("grok", model="grok-build") as client:
        result = await client.complete(
            messages=[{"role": "user", "content": "Explain a Python async lock."}]
        )
        print(result.content)
        print(result.usage.total_tokens)

asyncio.run(main())
```

`GrokOAuthClient(...)` is also directly importable. `create_client("codex", ...)` returns the original Codex client. Provider selection is explicit: changing a model name does not move credentials between services.

### Tool calls and multi-turn agents

`complete()` accepts OpenAI Chat-style messages and preserves assistant `tool_calls`, tool-result messages, multimodal content blocks and provider-specific message fields for the upstream. It does not execute tools. Upstream support for particular content types is not guaranteed by this pass-through.

Tool definitions may be native Chat Completions functions, flat Responses-style function definitions, or named definitions with `input_schema`. Non-function built-in tools are rejected instead of being silently dropped. The adapter preserves `strict` on function definitions.

```python
result = await client.complete(
    messages=history,
    tools=[{
        "type": "function",
        "function": {
            "name": "lookup_item",
            "description": "Look up an item by ID.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    }],
    extra_body={"tool_choice": "auto"},
)

# Preserve the original assistant message (including tool-call metadata).
history.append(result.raw["choices"][0]["message"])
for call in result.tool_calls:
    # Your application must validate/allowlist the name and arguments and run
    # the appropriate tool. Do not execute arbitrary model-generated commands.
    output = await run_allowed_tool(call.name, call.arguments)
    history.append({"role": "tool", "tool_call_id": call.id, "content": output})
```

Here `run_allowed_tool` is application code, not a function supplied by this package. Tool results should be strings (serialize structured results as JSON). Responses-style **input arrays** and Anthropic `tool_use`/`tool_result` message conversion are not implemented for Grok; use Chat-style `messages` for agent history. String `input="..."` is supported as a convenience.

### Raw calls and streaming

`create_response()` on **Grok** returns a native Chat Completion (`choices`), whereas on **Codex** it returns a Responses object (`output`). `create_response(stream=True)` is rejected for Grok; use `stream_events()` instead.

```python
raw = await client.create_response(input="Hello")

async for event in client.stream_events(
    input="Explain generators.",
    extra_body={"stream_options": {"include_usage": True}},
):
    for choice in event.get("choices", []):
        print(choice.get("delta", {}).get("content") or "", end="", flush=True)
```

Grok streaming events are **native Chat Completion chunks**, not Codex `response.output_text.delta` events. Tool-call argument fragments, reasoning fields and usage-only chunks are passed through. Requesting usage depends on upstream support. Consumers assembling streamed tool arguments must merge fragments by tool-call index. `complete()` uses a non-streaming request and already returns normalized final tool calls. The CLI `--stream` displays text only; use Python for streamed tool calls.

`extra_body` is available for provider-supported options, but cannot override named routing/input fields or `stream`. This adapter supports one completion per request (`n=1`).

## Credentials and configuration

Lookup order: explicit `GROK_OAUTH_ACCESS_TOKEN` environment credentials, then an explicit `auth_path`, `GROK_OAUTH_AUTH_JSON`, `$GROK_HOME/auth.json`, and finally `~/.grok/auth.json`.

The official cache is scoped by issuer and public client ID:

```json
{
  "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828": {
    "auth_mode": "oidc",
    "oidc_issuer": "https://auth.x.ai",
    "oidc_client_id": "b1a00492-073a-47ea-816f-4c329264a828",
    "key": "REDACTED_ACCESS_TOKEN",
    "refresh_token": "REDACTED_REFRESH_TOKEN",
    "expires_at": "2026-01-01T12:00:00Z"
  }
}
```

This is a redacted schema illustration, not a usable token file. Flat `access_token` documents and `{ "tokens": { ... } }` documents are supported too. The loader never chooses an arbitrary other issuer/account from a multi-entry file. Enterprise OIDC, external-auth providers and OS credential stores are not supported by this adapter.

| Variable | Purpose |
| --- | --- |
| `GROK_OAUTH_AUTH_JSON` | Cache file override |
| `GROK_HOME` | CLI home directory; default `~/.grok` |
| `GROK_OAUTH_ACCESS_TOKEN` | Explicit OAuth access token; takes precedence over files |
| `GROK_OAUTH_REFRESH_TOKEN` | Optional refresh token accompanying the explicit access token |
| `GROK_OAUTH_EXPIRES_AT` | Optional Unix timestamp or timezone-aware ISO expiry |
| `GROK_OAUTH_CLIENT_ID` | Explicit public-client ID/cache scope override |
| `GROK_OAUTH_CLIENT_VERSION` | Installed CLI semantic version override |
| `GROK_OAUTH_BASE_URL` | Trusted inference endpoint override |
| `GROK_OAUTH_REFRESH_URL` | Trusted refresh endpoint override; default `https://auth.x.ai/oauth2/token` |

**Custom endpoints receive credentials. Only configure endpoints you control/trust.** The adapter requires HTTPS except for loopback HTTP, and disables redirects for refresh and inference requests even when an injected `httpx.AsyncClient` enables redirects. Do not inject an HTTP client with unrelated default credentials/cookies/auth hooks.

## Refresh, errors and security

Refresh uses a form-encoded `refresh_token` grant. An instance-local `asyncio.Lock` coalesces parallel refreshes and stale 401s. A successful rotation is written with a temporary file and atomic replacement, preserving unrelated cache entries and metadata; Unix permissions are restricted to `0600`. Windows users should keep the cache in a private user directory with appropriate ACLs. Environment-provided rotations stay in memory; environment variables are not rewritten.

Share one `GrokAuth`/client per event loop. Atomic persistence is **not** a cross-process token-rotation lock. Do not have the CLI and multiple SDK workers refreshing the same file concurrently. A detected cache change during a refresh is not overwritten, but this check cannot remove every external-writer race. Do not copy refresh credentials into several independently refreshing caches either. File changes from a later external login are picked up on a subsequent call.

On disk-write failure, fresh rotated credentials remain in that auth instance, and an error is raised. Resolve persistence before exiting: the old disk refresh token may no longer work. A symlink cache is not overwritten; select the real private file path. `persist_refresh=False` is useful for controlled read-only setups but can leave a rotating CLI cache stale after process exit.

The adapter does not print tokens or echo arbitrary upstream error bodies. Avoid dumping environment variables, HTTP debug logs, token dataclasses via `asdict`, or auth-cache contents yourself. These are user-session credentials, not distributable service API keys. Use only your own authorized session and permitted provider use; this package does not remove subscription rules or grant additional entitlements.

| Failure | Behavior |
| --- | --- |
| HTTP 401 | Refresh and retry once, only before any stream event is delivered |
| HTTP 403 | Fail with entitlement/policy guidance; no paid-API fallback |
| HTTP 426 | Ask you to update/check the real CLI version |
| HTTP 429 | Fail without automatic quota retry/account switching |
| Malformed or interrupted stream | Raise instead of silently returning success; never replay partial output |

## Verification

Offline tests do not require accounts or network access:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q src tests examples
```

Run an actual subscription-backed smoke test **on your machine after login**. It consumes your account allowance and may be denied by account/model entitlements:

```sh
GROK_OAUTH_LIVE=1 python -m pytest -q tests/test_grok_live.py
```

PowerShell:

```powershell
$env:GROK_OAUTH_LIVE = "1"
python -m pytest -q tests/test_grok_live.py
Remove-Item Env:GROK_OAUTH_LIVE
```

`GROK_OAUTH_LIVE_MODEL` selects a model from your account's model list. The existing Codex live smoke test and release workflow are unchanged. Default CI skips both live tests. Passing mock tests establishes local protocol handling, **not** successful xAI login, available quota, or end-to-end model access.

## Protocol references

The implementation was checked against the following source material; these internal CLI surfaces may change:

- [Official Grok authentication guide](https://github.com/xai-org/grok-build/blob/main/crates/codegen/xai-grok-pager/docs/user-guide/02-authentication.md)
- [Official OAuth configuration](https://github.com/xai-org/grok-build/blob/2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8/crates/codegen/xai-grok-login/src/config.rs)
- [Official proxy headers](https://github.com/xai-org/grok-build/blob/2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8/crates/codegen/xai-grok-shell/src/agent/proxy_headers.rs)
- [Official CLI inference transport documentation](https://github.com/xai-org/grok-build/blob/2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8/crates/codegen/xai-grok-shell/README.md)
- [Independent, dated OAuth interoperability observations](https://github.com/code-yeongyu/gorky/blob/main/docs/reverse-engineering-notes.md) (not an xAI support commitment)
