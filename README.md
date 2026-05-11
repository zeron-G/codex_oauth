# codex_oauth

Experimental Python client for using the local Codex ChatGPT sign-in cache as an LLM transport.

This is meant for local prototypes that already run Codex and want a small Responses-style client without copying OAuth tokens around. It keeps auth local, redacts token values in debug representations, and never prints tokens from the CLI.

## Important boundary

OpenAI's supported production API path is the official OpenAI API with API keys and the Responses API. The Codex OAuth path used here talks to the Codex/ChatGPT backend that Codex itself uses, so treat it as an experimental local-development convenience. APIs, headers, models, and rate limits can change.

Use this package only with your own local Codex login session. For shared apps, CI, servers, or production, use the official OpenAI SDK and `OPENAI_API_KEY`.

## Official docs this package tracks

- Codex authentication docs say Codex caches login details locally at `~/.codex/auth.json` or in an OS credential store, and refreshes ChatGPT session tokens during Codex use: <https://developers.openai.com/codex/auth#login-caching>
- The latest model guide currently points to `gpt-5.5` and recommends the Responses API for reasoning, tool-calling, and multi-turn use cases: <https://developers.openai.com/api/docs/guides/latest-model.md>
- The code generation guide says Codex works best with latest GPT-5 family models such as `gpt-5.5`; Codex-specific models such as `gpt-5.3-codex` remain available for coding agents: <https://developers.openai.com/api/docs/guides/code-generation#use-codex>
- The Responses API reference describes the official request and response shape for text/image inputs, tools, and streaming: <https://platform.openai.com/docs/api-reference/responses/create>

## Install

```powershell
git clone https://github.com/zeron-G/codex_oauth.git
cd codex_oauth
python -m pip install -e .
```

Make sure you have signed in to Codex with ChatGPT on the same machine:

```powershell
codex login
```

If your Codex config stores credentials in the OS keychain instead of `~/.codex/auth.json`, switch Codex to file-backed credentials or pass explicit token environment variables. This package does not read OS keychains.

## CLI

```powershell
codex-oauth "Write a tiny haiku about local tools."
codex-oauth --model gpt-5.3-codex --system "You are terse." "Summarize this repo."
codex-oauth --stream "Explain SSE in one paragraph."
```

The CLI reads stdin when no prompt argument is provided.

## Python

```python
import asyncio

from codex_oauth import CodexOAuthClient


async def main() -> None:
    async with CodexOAuthClient(model="gpt-5.5") as client:
        response = await client.complete(
            messages=[
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Give me one practical use for prompt caching."},
            ],
        )
        print(response.content)


asyncio.run(main())
```

Raw Responses-style calls are also available:

```python
raw = await client.create_response(
    input="Return JSON with one key named ok.",
    model="gpt-5.5",
    extra_body={"text": {"format": {"type": "json_object"}}},
)
```

## Auth lookup

Lookup order:

1. `CODEX_OAUTH_ACCESS_TOKEN` plus optional `CODEX_OAUTH_REFRESH_TOKEN` and `CODEX_OAUTH_ACCOUNT_ID`
2. `CODEX_OAUTH_AUTH_JSON`
3. `~/.codex/auth.json`

Other useful environment variables:

- `CODEX_OAUTH_BASE_URL`: default `https://chatgpt.com/backend-api/codex`
- `CODEX_OAUTH_REFRESH_URL`: default `https://auth.openai.com/oauth/token`
- `CODEX_OAUTH_CLIENT_ID`: OAuth client id override

## Security notes

- `~/.codex/auth.json` can contain plaintext bearer and refresh tokens. Do not commit it, upload it, log it, or bake it into images.
- Treat Codex OAuth tokens as user-session credentials, not service credentials.
- Prefer short local runs. For shared apps, CI, servers, or production, use the official OpenAI API key flow.

