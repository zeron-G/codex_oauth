from __future__ import annotations

import asyncio

from codex_oauth import CodexOAuthClient


async def main() -> None:
    async with CodexOAuthClient(model="gpt-5.5") as client:
        response = await client.complete(
            messages=[
                {"role": "system", "content": "You are concise and practical."},
                {"role": "user", "content": "Give me one reason to use the Responses API."},
            ],
        )
        print(response.content)


if __name__ == "__main__":
    asyncio.run(main())

