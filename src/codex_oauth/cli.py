"""Command line entry point for the Codex and Grok OAuth clients."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from . import create_client
from .auth import CodexAuthError
from .client import CodexAPIError, DEFAULT_MODEL
from .grok import DEFAULT_GROK_MODEL, GrokAPIError, GrokOAuthClient
from .grok_auth import GrokAuthError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex-oauth",
        description="Call an LLM through a local Codex or Grok CLI OAuth session.",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text. Reads stdin when omitted.")
    parser.add_argument("--provider", choices=("codex", "grok"), default="codex")
    parser.add_argument("--model", default=None, help="Model name; default is provider-specific.")
    parser.add_argument("--system", default="", help="Optional system instructions.")
    parser.add_argument("--auth-path", type=Path, default=None, help="Path to the provider's auth.json.")
    parser.add_argument("--base-url", default=None, help="Trusted upstream override (receives credentials).")
    parser.add_argument("--grok-client-version", default=None, help="Version reported by grok --version.")
    parser.add_argument("--list-models", action="store_true", help="List account-visible Grok models.")
    parser.add_argument("--stream", action="store_true", help="Print text deltas as they arrive.")
    parser.add_argument("--json", action="store_true", help="Print the normalized response as JSON.")
    return parser


async def run(args: argparse.Namespace) -> int:
    provider = args.provider
    if provider != "grok" and (args.list_models or args.grok_client_version):
        raise ValueError("--list-models and --grok-client-version require --provider grok.")
    prompt = " ".join(args.prompt).strip()
    if not prompt and not args.list_models:
        prompt = sys.stdin.read().strip()
    if not prompt and not args.list_models:
        print("No prompt provided.", file=sys.stderr)
        return 2

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})
    options: dict[str, Any] = {
        "model": args.model or (DEFAULT_GROK_MODEL if provider == "grok" else DEFAULT_MODEL),
        "auth_path": args.auth_path,
        "base_url": args.base_url,
    }
    if provider == "grok":
        options["client_version"] = args.grok_client_version
    async with create_client(provider, **options) as client:
        if args.list_models:
            assert isinstance(client, GrokOAuthClient)
            print(json.dumps(await client.list_models(), ensure_ascii=False, indent=2))
            return 0
        if args.stream and not args.json:
            async for event in client.stream_events(messages=messages):
                if provider == "grok":
                    for choice in event.get("choices", []):
                        if choice.get("index", 0) == 0:
                            print(choice.get("delta", {}).get("content") or "", end="", flush=True)
                elif event.get("type") == "response.output_text.delta":
                    print(event.get("delta", ""), end="", flush=True)
            print()
            return 0

        response = await client.complete(messages=messages)
        if args.json:
            print(json.dumps({
                "content": response.content,
                "model": response.model,
                "tool_calls": [asdict(call) for call in response.tool_calls],
                "usage": asdict(response.usage),
            }, ensure_ascii=False, indent=2))
        else:
            print(response.content)
    return 0


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except httpx.HTTPError:
        print("codex-oauth: HTTP transport failed.", file=sys.stderr)
        raise SystemExit(1) from None
    except (CodexAuthError, CodexAPIError, GrokAuthError, GrokAPIError, OSError, ValueError) as exc:
        print(f"codex-oauth: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
