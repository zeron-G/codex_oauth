"""Public response dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_response_usage(cls, data: dict[str, Any] | None) -> "Usage":
        data = data or {}
        input_tokens = int(data.get("input_tokens", data.get("prompt_tokens", 0)) or 0)
        output_tokens = int(data.get("output_tokens", data.get("completion_tokens", 0)) or 0)
        total_tokens = int(data.get("total_tokens", input_tokens + output_tokens) or 0)
        return cls(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class LLMResponse:
    content: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)

