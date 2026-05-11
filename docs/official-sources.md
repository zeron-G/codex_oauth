# Official Sources

This package tracks only the public documentation needed to explain its boundaries.

## Codex Auth

Codex authentication documentation says Codex caches sign-in details locally at `~/.codex/auth.json` or in an OS-specific credential store. It also says ChatGPT sign-in sessions are refreshed automatically during active Codex use.

Source: <https://developers.openai.com/codex/auth#login-caching>

## Latest GPT Guidance

The latest-model guide identifies `gpt-5.5` as the current latest model. It recommends the Responses API for reasoning, tool-calling, and multi-turn workflows, with `reasoning.effort` tuned by latency and quality needs.

Source: <https://developers.openai.com/api/docs/guides/latest-model.md>

## Codex Model Guidance

The code generation guide describes Codex as OpenAI's coding agent. It says Codex works best with latest GPT-5 family models such as `gpt-5.5`, while Codex-specific models such as `gpt-5.3-codex` are available for coding-agent workflows.

Source: <https://developers.openai.com/api/docs/guides/code-generation#use-codex>

## Responses API

The official Responses API reference defines the supported public request and response shape, including text and image input, function tools, and streaming events.

Source: <https://platform.openai.com/docs/api-reference/responses/create>

## Unsupported Local Bridge

The docs above support Codex login caching and the official Responses API model shape. They do not document the ChatGPT/Codex backend endpoint as a public OpenAI API. The OAuth bridge in this package is therefore an experimental local convenience, not a production API contract.

