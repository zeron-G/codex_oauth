"""Codex OAuth token loading and refresh helpers."""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


DEFAULT_AUTH_PATH = Path.home() / ".codex" / "auth.json"
DEFAULT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_REFRESH_URL = "https://auth.openai.com/oauth/token"
DEFAULT_SCOPE = "openid profile email offline_access"


class CodexAuthError(RuntimeError):
    """Raised when local Codex authentication cannot be loaded or refreshed."""


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _default_auth_path() -> Path:
    configured = os.environ.get("CODEX_OAUTH_AUTH_JSON", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_AUTH_PATH


def _redact(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 12:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


@dataclass(slots=True)
class CodexTokens:
    access_token: str
    refresh_token: str = ""
    id_token: str = ""
    account_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    source: str = "unknown"

    def __repr__(self) -> str:
        return (
            "CodexTokens("
            f"access_token={_redact(self.access_token)!r}, "
            f"refresh_token={_redact(self.refresh_token)!r}, "
            f"id_token={_redact(self.id_token)!r}, "
            f"account_id={self.account_id!r}, "
            f"source={self.source!r})"
        )

    def with_updates(self, data: dict[str, Any]) -> "CodexTokens":
        return CodexTokens(
            access_token=data.get("access_token", self.access_token),
            refresh_token=data.get("refresh_token", self.refresh_token),
            id_token=data.get("id_token", self.id_token),
            account_id=data.get("account_id", self.account_id),
            raw={**self.raw, **data},
            source=self.source,
        )


def _json_b64url_decode(value: str) -> dict[str, Any]:
    padded = value + "=" * (-len(value) % 4)
    return json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))


def token_expires_at(access_token: str) -> int | None:
    """Return the JWT exp claim if available.

    This does not validate signatures; it is only used to decide whether a local
    access token should be refreshed before a request.
    """
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        claims = _json_b64url_decode(parts[1])
        exp = claims.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None


def token_is_expired(access_token: str, *, skew_seconds: int = 300) -> bool:
    if not access_token:
        return True
    exp = token_expires_at(access_token)
    if exp is None:
        return False
    return datetime.now(UTC).timestamp() >= exp - skew_seconds


def load_codex_tokens(auth_path: str | Path | None = None) -> CodexTokens:
    """Load Codex tokens from env vars or Codex's local auth cache."""
    env_access = os.environ.get("CODEX_OAUTH_ACCESS_TOKEN", "").strip()
    if env_access:
        return CodexTokens(
            access_token=env_access,
            refresh_token=os.environ.get("CODEX_OAUTH_REFRESH_TOKEN", "").strip(),
            id_token=os.environ.get("CODEX_OAUTH_ID_TOKEN", "").strip(),
            account_id=os.environ.get("CODEX_OAUTH_ACCOUNT_ID", "").strip(),
            source="env",
        )

    path = Path(auth_path).expanduser() if auth_path else _default_auth_path()
    if not path.exists():
        raise CodexAuthError(
            f"Codex auth cache not found at {path}. Run `codex login` or set "
            "CODEX_OAUTH_ACCESS_TOKEN."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CodexAuthError(f"Could not read Codex auth cache at {path}: {exc}") from exc

    auth_mode = data.get("auth_mode")
    if auth_mode and auth_mode != "chatgpt":
        raise CodexAuthError(
            f"Codex auth cache at {path} is auth_mode={auth_mode!r}; expected ChatGPT login."
        )

    token_data = data.get("tokens") if isinstance(data.get("tokens"), dict) else data
    access_token = str(token_data.get("access_token", "")).strip()
    if not access_token:
        raise CodexAuthError(f"Codex auth cache at {path} does not contain an access token.")

    return CodexTokens(
        access_token=access_token,
        refresh_token=str(token_data.get("refresh_token", "")).strip(),
        id_token=str(token_data.get("id_token", "")).strip(),
        account_id=str(token_data.get("account_id", "")).strip(),
        raw=dict(token_data),
        source=str(path),
    )


class CodexAuth:
    """Stateful local Codex auth loader with refresh support."""

    def __init__(
        self,
        auth_path: str | Path | None = None,
        *,
        client_id: str | None = None,
        refresh_url: str | None = None,
        scope: str = DEFAULT_SCOPE,
        persist_refresh: bool = True,
    ) -> None:
        self.auth_path = Path(auth_path).expanduser() if auth_path else _default_auth_path()
        self.client_id = client_id or _env("CODEX_OAUTH_CLIENT_ID", DEFAULT_CLIENT_ID)
        self.refresh_url = refresh_url or _env("CODEX_OAUTH_REFRESH_URL", DEFAULT_REFRESH_URL)
        self.scope = scope
        self.persist_refresh = persist_refresh
        self._tokens: CodexTokens | None = None

    async def get_tokens(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> CodexTokens:
        tokens = self._tokens or load_codex_tokens(self.auth_path)
        self._tokens = tokens
        if force_refresh or token_is_expired(tokens.access_token):
            tokens = await self.refresh(tokens, http_client=http_client)
            self._tokens = tokens
        return tokens

    async def refresh(
        self,
        tokens: CodexTokens | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> CodexTokens:
        tokens = tokens or self._tokens or load_codex_tokens(self.auth_path)
        if not tokens.refresh_token:
            raise CodexAuthError("Codex refresh token is not available.")

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": tokens.refresh_token,
            "client_id": self.client_id,
            "scope": self.scope,
        }

        async def _post(client: httpx.AsyncClient) -> httpx.Response:
            return await client.post(self.refresh_url, json=payload)

        try:
            if http_client is None:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await _post(client)
            else:
                response = await _post(http_client)
        except httpx.HTTPError as exc:
            raise CodexAuthError(f"Codex token refresh failed: {exc}") from exc

        if response.status_code != 200:
            body = response.text[:300].replace(tokens.refresh_token, "***")
            raise CodexAuthError(f"Codex token refresh failed ({response.status_code}): {body}")

        try:
            refreshed = response.json()
        except ValueError as exc:
            raise CodexAuthError("Codex token refresh returned invalid JSON.") from exc

        new_tokens = tokens.with_updates(refreshed)
        if self.persist_refresh and tokens.source != "env":
            self._persist(new_tokens)
        self._tokens = new_tokens
        return new_tokens

    def _persist(self, tokens: CodexTokens) -> None:
        if not self.auth_path.exists():
            return
        try:
            data = json.loads(self.auth_path.read_text(encoding="utf-8"))
            if isinstance(data.get("tokens"), dict):
                data["tokens"].update(
                    {
                        "access_token": tokens.access_token,
                        "refresh_token": tokens.refresh_token,
                        "id_token": tokens.id_token,
                    }
                )
                if tokens.account_id:
                    data["tokens"]["account_id"] = tokens.account_id
            else:
                data.update(
                    {
                        "access_token": tokens.access_token,
                        "refresh_token": tokens.refresh_token,
                        "id_token": tokens.id_token,
                    }
                )
                if tokens.account_id:
                    data["account_id"] = tokens.account_id
            data["last_refresh"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            self.auth_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            raise CodexAuthError(f"Could not persist refreshed Codex tokens: {exc}") from exc

