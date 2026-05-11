"""Command line entry point."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .auth import CodexAuth, CodexAuthError
from .client import CodexAPIError, CodexOAuthClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex-oauth",
        description="Call an LLM through the local Codex ChatGPT OAuth session.",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text. Reads stdin when omitted.")
    parser.add_argument("--model", default="gpt-5.5", help="Model name, e.g. gpt-5.5.")
    parser.add_argument("--system", default="", help="Optional system instructions.")
    parser.add_argument("--auth-path", type=Path, default=None, help="Path to Codex auth.json.")
    parser.add_argument("--stream", action="store_true", help="Print text deltas as they arrive.")
    parser.add_argument("--json", action="store_true", help="Print the normalized response as JSON.")
    return parser


async def run(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        return 2

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    auth = CodexAuth(args.auth_path) if args.auth_path else None
    async with CodexOAuthClient(model=args.model, auth=auth) as client:
        if args.stream and not args.json:
            async for event in client.stream_events(messages=messages):
                if event.get("type") == "response.output_text.delta":
                    print(event.get("delta", ""), end="", flush=True)
            print()
            return 0

        response = await client.complete(messages=messages)
        if args.json:
            print(
                json.dumps(
                    {
                        "content": response.content,
                        "model": response.model,
                        "tool_calls": [asdict(call) for call in response.tool_calls],
                        "usage": asdict(response.usage),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(response.content)
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except (CodexAuthError, CodexAPIError, OSError) as exc:
        print(f"codex-oauth: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

