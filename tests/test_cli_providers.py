from __future__ import annotations

import io
import json

import pytest

from codex_oauth import LLMResponse
from codex_oauth import cli
from codex_oauth.grok import GrokOAuthClient


class FakeGrok(GrokOAuthClient):
    def __init__(self, provider):
        self.provider = provider

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def complete(self, **kwargs):
        return LLMResponse(content="hello", model="test-model")

    async def list_models(self):
        return {"data": [{"id": "test-model"}]}

    async def stream_events(self, **kwargs):
        if self.provider == "grok":
            yield {"choices": [{"index": 0, "delta": {"content": "hello"}}]}
            yield {"choices": [], "usage": {"completion_tokens": 1}}
        else:
            yield {"type": "response.output_text.delta", "delta": "hello"}


@pytest.mark.parametrize("provider,model", [("codex", "gpt-5.5"), ("grok", "grok-build")])
@pytest.mark.parametrize("mode", ["text", "stream", "json"])
async def test_cli_routes_provider_and_preserves_output(monkeypatch, capsys, provider, model, mode):
    selected = []

    def factory(name, **kwargs):
        selected.append((name, kwargs))
        return FakeGrok(name)

    monkeypatch.setattr(cli, "create_client", factory)
    argv = ["--provider", provider, "hi"] + ([] if mode == "text" else ["--" + mode])
    assert await cli.run(cli.build_parser().parse_args(argv)) == 0
    assert selected[0][0] == provider
    assert selected[0][1]["model"] == model
    output = capsys.readouterr().out
    assert (json.loads(output)["content"] if mode == "json" else output.strip()) == "hello"


async def test_list_models_does_not_read_stdin(monkeypatch, capsys):
    monkeypatch.setattr(cli, "create_client", lambda *a, **k: FakeGrok("grok"))
    assert await cli.run(cli.build_parser().parse_args(["--provider", "grok", "--list-models"])) == 0
    assert json.loads(capsys.readouterr().out)["data"][0]["id"] == "test-model"


async def test_empty_stdin_returns_usage_error(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(""))
    assert await cli.run(cli.build_parser().parse_args([])) == 2
    assert "No prompt" in capsys.readouterr().err


async def test_grok_only_flags_rejected_for_codex():
    with pytest.raises(ValueError):
        await cli.run(cli.build_parser().parse_args(["--list-models"]))


def test_default_provider_unchanged():
    assert cli.build_parser().parse_args(["hi"]).provider == "codex"
