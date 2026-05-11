from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from codex_oauth.auth import CodexAuth, CodexAuthError, load_codex_tokens, token_is_expired


def _jwt(exp: int) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def test_token_is_expired_with_skew() -> None:
    soon = int((datetime.now(UTC) + timedelta(seconds=60)).timestamp())
    later = int((datetime.now(UTC) + timedelta(hours=1)).timestamp())

    assert token_is_expired(_jwt(soon), skew_seconds=300)
    assert not token_is_expired(_jwt(later), skew_seconds=300)
    assert not token_is_expired("not-a-jwt")


def test_load_tokens_from_auth_json(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CODEX_OAUTH_ACCESS_TOKEN", raising=False)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "access-token-secret-12345",
                    "refresh_token": "refresh-token-secret-12345",
                    "id_token": "id-token-secret-12345",
                    "account_id": "acct",
                },
            }
        ),
        encoding="utf-8",
    )

    tokens = load_codex_tokens(auth_path)

    assert tokens.access_token == "access-token-secret-12345"
    assert tokens.refresh_token == "refresh-token-secret-12345"
    assert tokens.account_id == "acct"
    token_repr = repr(tokens)
    assert "access-token-secret-12345" not in token_repr
    assert "refresh-token-secret-12345" not in token_repr


def test_load_tokens_rejects_api_key_auth(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CODEX_OAUTH_ACCESS_TOKEN", raising=False)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"auth_mode": "api-key"}), encoding="utf-8")

    with pytest.raises(CodexAuthError):
        load_codex_tokens(auth_path)


@pytest.mark.asyncio
async def test_refresh_persists_new_tokens(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CODEX_OAUTH_ACCESS_TOKEN", raising=False)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _jwt(0),
                    "refresh_token": "refresh-old",
                    "id_token": "id-old",
                    "account_id": "acct",
                },
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["grant_type"] == "refresh_token"
        assert body["refresh_token"] == "refresh-old"
        return httpx.Response(
            200,
            json={
                "access_token": "access-new",
                "refresh_token": "refresh-new",
                "id_token": "id-new",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        auth = CodexAuth(auth_path, refresh_url="https://auth.example/token")
        tokens = await auth.get_tokens(http_client=http_client)

    assert tokens.access_token == "access-new"
    saved = json.loads(auth_path.read_text(encoding="utf-8"))
    assert saved["tokens"]["access_token"] == "access-new"
    assert saved["tokens"]["refresh_token"] == "refresh-new"
    assert saved["tokens"]["id_token"] == "id-new"

