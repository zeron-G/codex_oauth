from __future__ import annotations

import asyncio
import json
import os
import time
from urllib.parse import parse_qs

import httpx
import pytest

from codex_oauth.grok_auth import (
    DEFAULT_GROK_CLIENT_ID,
    DEFAULT_GROK_ISSUER,
    DEFAULT_GROK_REFRESH_URL,
    GrokAuth,
    GrokAuthError,
    load_grok_tokens,
)

SCOPE = f"{DEFAULT_GROK_ISSUER}::{DEFAULT_GROK_CLIENT_ID}"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("GROK_OAUTH_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GROK_HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_OAUTH_ACCESS_TOKEN", "codex-must-never-be-used")
    monkeypatch.setenv("XAI_API_KEY", "api-key-must-never-be-used")


def cache(tmp_path, shape="official", expired=True):
    entry = {
        "key": "access-old-secret", "refresh_token": "refresh-old-secret",
        "expires_at": 0 if expired else time.time() + 3600,
        "auth_mode": "oidc", "oidc_client_id": DEFAULT_GROK_CLIENT_ID,
        "oidc_issuer": DEFAULT_GROK_ISSUER,
        "coding_data_retention_opt_out": True, "team_id": "test-team",
    }
    if shape == "official":
        data = {SCOPE: entry, "other-issuer": {"key": "other-token"}}
    else:
        entry["access_token"] = entry.pop("key")
        data = {"tokens": entry} if shape == "nested" else entry
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def refreshed(**kwargs):
    return {"access_token": "access-new-secret", "refresh_token": "refresh-new-secret",
            "expires_in": 3600, "token_type": "Bearer", **kwargs}


@pytest.mark.parametrize("shape", ["official", "flat", "nested"])
def test_load_supported_cache_shapes(tmp_path, shape):
    tokens = load_grok_tokens(cache(tmp_path, shape))
    assert tokens.access_token == "access-old-secret"
    assert tokens.expires_at == 0
    assert tokens.refresh_token == "refresh-old-secret"
    assert "access-old-secret" not in repr(tokens)
    assert "refresh-old-secret" not in repr(tokens)


def test_no_codex_or_api_key_fallback(tmp_path):
    with pytest.raises(GrokAuthError, match="grok login"):
        load_grok_tokens(tmp_path / "missing.json")


def test_environment_precedence_and_epoch_zero(tmp_path, monkeypatch):
    path = cache(tmp_path)
    monkeypatch.setenv("GROK_OAUTH_ACCESS_TOKEN", "explicit-access")
    monkeypatch.setenv("GROK_OAUTH_EXPIRES_AT", "0")
    tokens = load_grok_tokens(path)
    assert tokens.access_token == "explicit-access"
    assert tokens.source == "env"
    assert tokens.expires_at == 0


def test_auth_json_env_path(tmp_path, monkeypatch):
    path = cache(tmp_path)
    monkeypatch.setenv("GROK_OAUTH_AUTH_JSON", str(path))
    assert load_grok_tokens().access_token == "access-old-secret"


@pytest.mark.parametrize("value", [[], None, "bad", {"other": {"key": "secret"}},
                                    {"access_token": None}, {SCOPE: "bad"}])
def test_reject_malformed_cache(tmp_path, value):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(GrokAuthError):
        load_grok_tokens(path)


@pytest.mark.parametrize("updates", [
    {"auth_mode": "api-key"}, {"auth_mode": "external"},
    {"oidc_issuer": "https://untrusted.example"}, {"oidc_client_id": "another-client"},
    {"expires_at": "not-a-date"}, {"expires_at": True}, {"expires_at": "nan"},
    {"expires_at": "2026-01-01T00:00:00"},
])
def test_reject_unsupported_metadata(tmp_path, updates):
    path = cache(tmp_path)
    data = json.loads(path.read_text())
    data[SCOPE].update(updates)
    path.write_text(json.dumps(data))
    with pytest.raises(GrokAuthError):
        load_grok_tokens(path)


def test_outer_api_key_auth_mode_is_rejected(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"auth_mode": "api-key", "tokens": {"access_token": "secret"}}))
    with pytest.raises(GrokAuthError):
        load_grok_tokens(path)


@pytest.mark.parametrize("shape", ["official", "flat", "nested"])
async def test_atomic_refresh_preserves_cache_and_rotates(tmp_path, shape):
    path = cache(tmp_path, shape)
    before = json.loads(path.read_text())
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == DEFAULT_GROK_REFRESH_URL
        assert request.headers["content-type"].startswith("application/x-www-form-urlencoded")
        form = parse_qs(request.content.decode())
        assert form == {"grant_type": ["refresh_token"], "client_id": [DEFAULT_GROK_CLIENT_ID],
                        "refresh_token": ["refresh-old-secret"]}
        return httpx.Response(200, json=refreshed(id_token="new-id-secret"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        auth = GrokAuth(path)
        tokens = await auth.get_tokens(http_client=http)
        assert (await auth.get_tokens(http_client=http)) is tokens
    assert len(calls) == 1
    after = json.loads(path.read_text())
    entry = after[SCOPE] if shape == "official" else after.get("tokens", after)
    assert entry["key" if shape == "official" else "access_token"] == "access-new-secret"
    assert entry["refresh_token"] == "refresh-new-secret"
    assert entry["id_token"] == "new-id-secret"
    assert entry["coding_data_retention_opt_out"] is True
    assert entry["team_id"] == "test-team"
    assert load_grok_tokens(path).expires_at > time.time() + 3500
    if shape == "official":
        assert after["other-issuer"] == before["other-issuer"]
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.tmp"))


async def test_concurrent_refresh_and_stale_401_are_coalesced(tmp_path):
    auth = GrokAuth(cache(tmp_path))
    calls = []

    async def handler(request):
        calls.append(request)
        await asyncio.sleep(0.01)
        return httpx.Response(200, json=refreshed())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        results = await asyncio.gather(*(auth.get_tokens(http_client=http) for _ in range(20)))
        await asyncio.gather(*(auth.get_tokens(
            http_client=http, force_refresh=True, rejected_access_token="access-old-secret",
        ) for _ in range(20)))
    assert len(calls) == 1
    assert all(t.access_token == "access-new-secret" for t in results)


async def test_cache_hot_reload(tmp_path):
    path = cache(tmp_path, expired=False)
    auth = GrokAuth(path)
    assert (await auth.get_tokens()).access_token == "access-old-secret"
    data = json.loads(path.read_text())
    data[SCOPE]["key"] = "external-login-new-longer-secret"
    path.write_text(json.dumps(data))
    assert (await auth.get_tokens()).access_token == "external-login-new-longer-secret"


async def test_environment_rotation_is_memory_only(tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_OAUTH_ACCESS_TOKEN", "env-old")
    monkeypatch.setenv("GROK_OAUTH_REFRESH_TOKEN", "env-refresh")
    auth = GrokAuth(tmp_path / "missing.json")
    data = refreshed()
    data.pop("refresh_token")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=data))) as http:
        result = await auth.get_tokens(http_client=http, force_refresh=True)
        assert result.refresh_token == "env-refresh"
        assert (await auth.get_tokens(http_client=http)).access_token == "access-new-secret"
    assert not (tmp_path / "missing.json").exists()
    assert os.environ["GROK_OAUTH_ACCESS_TOKEN"] == "env-old"


@pytest.mark.parametrize("data", [
    [], {}, {"access_token": None}, refreshed(refresh_token=None),
    refreshed(token_type="MAC"), refreshed(expires_in=-1), refreshed(expires_in="nan"),
    refreshed(expires_in=True), refreshed(expires_in="bad"),
])
async def test_invalid_refresh_never_overwrites_cache(tmp_path, data):
    path = cache(tmp_path)
    original = path.read_bytes()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=data))) as http:
        with pytest.raises(GrokAuthError):
            await GrokAuth(path).get_tokens(http_client=http)
    assert path.read_bytes() == original


async def test_refresh_redirect_and_error_body_are_not_followed_or_echoed(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(307, headers={"location": "https://untrusted.example"}, text="refresh-old-secret")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as http:
        with pytest.raises(GrokAuthError) as error:
            await GrokAuth(cache(tmp_path)).get_tokens(http_client=http)
    assert len(calls) == 1
    assert "refresh-old-secret" not in str(error.value)


async def test_failed_persistence_retains_rotation_in_memory(tmp_path, monkeypatch):
    path = cache(tmp_path)
    original = path.read_bytes()
    auth = GrokAuth(path)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=refreshed())

    def fail_replace(*args):
        raise PermissionError("denied")

    monkeypatch.setattr("codex_oauth.grok_auth.os.replace", fail_replace)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(GrokAuthError, match="atomically"):
            await auth.get_tokens(http_client=http)
        assert (await auth.get_tokens(http_client=http)).refresh_token == "refresh-new-secret"
    assert len(calls) == 1
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


async def test_refuse_overwriting_external_login(tmp_path):
    path = cache(tmp_path)

    def handler(request):
        data = json.loads(path.read_text())
        data[SCOPE]["key"] = "external-winner"
        path.write_text(json.dumps(data))
        return httpx.Response(200, json=refreshed())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(GrokAuthError, match="changed during refresh"):
            await GrokAuth(path).get_tokens(http_client=http)
    assert json.loads(path.read_text())[SCOPE]["key"] == "external-winner"


@pytest.mark.parametrize("url", ["http://remote.example", "https://user:password@example.com", "https://example.com/?q=1"])
def test_reject_unsafe_refresh_urls(url):
    with pytest.raises(ValueError):
        GrokAuth(refresh_url=url)
