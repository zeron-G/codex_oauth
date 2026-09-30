"""Local Grok CLI OAuth cache loading and refresh-token rotation.

Initial sign-in is delegated to the official ``grok login --oauth`` command.
No browser cookies, API-key fallback, or enterprise credential discovery.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .auth import token_expires_at

DEFAULT_GROK_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
DEFAULT_GROK_ISSUER = "https://auth.x.ai"
DEFAULT_GROK_REFRESH_URL = f"{DEFAULT_GROK_ISSUER}/oauth2/token"


class GrokAuthError(RuntimeError):
    """Grok OAuth credentials cannot be loaded, refreshed, or safely persisted."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _auth_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    if configured := _env("GROK_OAUTH_AUTH_JSON"):
        return Path(configured).expanduser()
    return Path(_env("GROK_HOME", str(Path.home() / ".grok"))).expanduser() / "auth.json"


def _validate_url(url: str) -> str:
    parsed = urlsplit(url)
    local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if (
        not parsed.hostname or parsed.username or parsed.password
        or parsed.query or parsed.fragment
        or (parsed.scheme != "https" and not (parsed.scheme == "http" and local))
    ):
        raise ValueError("Use an HTTPS endpoint (HTTP is allowed only on loopback).")
    return url.rstrip("/")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _expiry(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, bool):
            raise TypeError
        try:
            result = float(value)
        except (TypeError, ValueError):
            dt = datetime.fromisoformat(str(value))
            if dt.tzinfo is None:
                raise ValueError
            result = dt.timestamp()
        if not math.isfinite(result):
            raise ValueError
        return result
    except (ValueError, TypeError, OverflowError):
        raise GrokAuthError("Grok expires_at must be a Unix timestamp or timezone-aware ISO date.") from None


@dataclass(slots=True)
class GrokTokens:
    access_token: str = field(repr=False)
    refresh_token: str = field(default="", repr=False)
    expires_at: float | None = None
    client_id: str = DEFAULT_GROK_CLIENT_ID
    source: str = "unknown"
    cache_key: str | None = field(default=None, repr=False)
    token_field: str = field(default="access_token", repr=False)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        raise GrokAuthError(
            "Cannot read Grok auth cache. Run `grok login --oauth` or check GROK_OAUTH_AUTH_JSON."
        ) from None
    if not isinstance(value, dict):
        raise GrokAuthError("Grok auth cache must contain a JSON object.")
    return value


def _entry(data: dict[str, Any], client_id: str) -> tuple[dict[str, Any], str | None]:
    # Never select an arbitrary account/issuer from a multi-account cache.
    if data.get("auth_mode") not in (None, "oidc", "oauth", "oauth2"):
        raise GrokAuthError("Expected a Grok OAuth session, not API-key or external-provider auth.")
    scope = f"{DEFAULT_GROK_ISSUER}::{client_id}"
    if scope in data:
        value, key = data[scope], scope
    elif isinstance(data.get("tokens"), dict):
        value, key = data["tokens"], "tokens"
    elif "key" in data or "access_token" in data:
        value, key = data, None
    else:
        raise GrokAuthError("No matching xAI OAuth session. Run `grok login --oauth`.")
    if not isinstance(value, dict):
        raise GrokAuthError("Malformed Grok OAuth cache entry.")
    if value.get("auth_mode") not in (None, "oidc", "oauth", "oauth2"):
        raise GrokAuthError("Expected a Grok OAuth session, not API-key or external-provider auth.")
    if value.get("oidc_issuer", DEFAULT_GROK_ISSUER) != DEFAULT_GROK_ISSUER:
        raise GrokAuthError("Enterprise/custom OIDC issuers are not supported by this adapter.")
    if value.get("oidc_client_id", client_id) != client_id:
        raise GrokAuthError("Grok OAuth client ID does not match the selected cache entry.")
    return value, key


def load_grok_tokens(
    auth_path: str | os.PathLike[str] | None = None, *, client_id: str | None = None,
) -> GrokTokens:
    """Load explicit GROK_OAUTH_* tokens or the official CLI file-backed cache."""
    client_id = client_id or _env("GROK_OAUTH_CLIENT_ID", DEFAULT_GROK_CLIENT_ID)
    if access := _env("GROK_OAUTH_ACCESS_TOKEN"):
        expiration = _expiry(_env("GROK_OAUTH_EXPIRES_AT"))
        return GrokTokens(
            access_token=access, refresh_token=_env("GROK_OAUTH_REFRESH_TOKEN"),
            expires_at=expiration if expiration is not None else token_expires_at(access),
            client_id=client_id, source="env",
        )
    path = _auth_path(auth_path)
    data = _read_json(path)
    value, key = _entry(data, client_id)
    token_field = "key" if "key" in value else "access_token"
    access = _text(value.get(token_field))
    if not access:
        raise GrokAuthError("Grok OAuth cache has no nonempty access token.")
    expiration = _expiry(value.get("expires_at"))
    return GrokTokens(
        access_token=access, refresh_token=_text(value.get("refresh_token")),
        expires_at=expiration if expiration is not None else token_expires_at(access),
        client_id=client_id, source=str(path), cache_key=key, token_field=token_field,
    )


class GrokAuth:
    """Refresh with an instance-local async lock; share one instance per event loop.

    Atomic writes prevent partial JSON. They are NOT a cross-process refresh lock:
    do not let the CLI and multiple SDK processes rotate the same cache concurrently.
    """

    def __init__(
        self, auth_path: str | os.PathLike[str] | None = None, *,
        client_id: str | None = None, refresh_url: str | None = None,
        persist_refresh: bool = True,
    ) -> None:
        self.auth_path = _auth_path(auth_path)
        self.client_id = client_id or _env("GROK_OAUTH_CLIENT_ID", DEFAULT_GROK_CLIENT_ID)
        self.refresh_url = _validate_url(
            refresh_url or _env("GROK_OAUTH_REFRESH_URL", DEFAULT_GROK_REFRESH_URL)
        )
        self.persist_refresh = persist_refresh
        self._tokens: GrokTokens | None = None
        self._stamp: tuple[int, int] | None = None
        self._lock = asyncio.Lock()

    def _file_stamp(self) -> tuple[int, int] | None:
        try:
            stat = self.auth_path.stat()
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    async def get_tokens(
        self, *, http_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False, rejected_access_token: str | None = None,
    ) -> GrokTokens:
        async with self._lock:
            if self._tokens is None or (
                self._tokens.source != "env" and self._stamp != self._file_stamp()
            ):
                self._tokens = load_grok_tokens(self.auth_path, client_id=self.client_id)
                self._stamp = self._file_stamp()
            tokens = self._tokens
            # A concurrent request may already have replaced the rejected token.
            if rejected_access_token is not None and tokens.access_token != rejected_access_token:
                force_refresh = False
            now = time.time()
            needs_refresh = tokens.expires_at is not None and (
                tokens.expires_at <= now or (
                    bool(tokens.refresh_token) and tokens.expires_at <= now + 300
                )
            )
            if force_refresh or needs_refresh:
                if http_client is None:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        tokens = await self._refresh(tokens, client)
                else:
                    tokens = await self._refresh(tokens, http_client)
            return tokens

    async def _refresh(self, old: GrokTokens, client: httpx.AsyncClient) -> GrokTokens:
        if not old.refresh_token:
            raise GrokAuthError("Grok refresh token is unavailable. Run `grok login --oauth`.")
        try:
            response = await client.post(
                self.refresh_url,
                data={"grant_type": "refresh_token", "client_id": self.client_id,
                      "refresh_token": old.refresh_token},
                follow_redirects=False, timeout=15.0,
            )
        except httpx.HTTPError:
            raise GrokAuthError("Grok OAuth refresh transport failed; credentials were not logged.") from None
        if response.status_code != 200:
            # Do not echo a provider body: it can contain credentials or personal data.
            raise GrokAuthError(
                f"Grok OAuth refresh failed (HTTP {response.status_code}). "
                "Check connectivity or run `grok login --oauth` again."
            )
        try:
            data = response.json()
        except ValueError:
            raise GrokAuthError("Grok OAuth refresh returned invalid JSON.") from None
        if not isinstance(data, dict) or not _text(data.get("access_token")):
            raise GrokAuthError("Grok OAuth refresh returned no valid access_token.")
        if "refresh_token" in data and not _text(data["refresh_token"]):
            raise GrokAuthError("Grok OAuth refresh returned an invalid refresh_token.")
        if str(data.get("token_type", "Bearer")).lower() != "bearer":
            raise GrokAuthError("Grok OAuth returned an unsupported token_type.")
        access = data["access_token"].strip()
        expiration = token_expires_at(access)
        if "expires_in" in data:
            try:
                lifetime = float(data["expires_in"])
                if isinstance(data["expires_in"], bool) or not math.isfinite(lifetime) or lifetime <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise GrokAuthError("Grok OAuth refresh returned an invalid expires_in.") from None
            expiration = time.time() + lifetime
        new = replace(
            old, access_token=access,
            refresh_token=data.get("refresh_token", old.refresh_token), expires_at=expiration,
        )
        # Retain rotated credentials in memory even if disk persistence fails.
        self._tokens = new
        if self.persist_refresh and old.source != "env":
            self._persist(old, new, data)
            self._stamp = self._file_stamp()
        return new

    def _persist(self, old: GrokTokens, new: GrokTokens, refreshed: dict[str, Any]) -> None:
        data = _read_json(self.auth_path)
        entry, key = _entry(data, self.client_id)
        if key != old.cache_key or (
            _text(entry.get(old.token_field)) != old.access_token
            or _text(entry.get("refresh_token")) != old.refresh_token
        ):
            raise GrokAuthError("Grok cache changed during refresh; refusing to overwrite another login.")
        entry[old.token_field] = new.access_token
        entry["refresh_token"] = new.refresh_token
        entry["create_time"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if new.expires_at is None:
            entry.pop("expires_at", None)
        else:
            entry["expires_at"] = datetime.fromtimestamp(new.expires_at, UTC).isoformat().replace("+00:00", "Z")
        if _text(refreshed.get("id_token")):
            entry["id_token"] = refreshed["id_token"]
        temporary: str | None = None
        try:
            if self.auth_path.is_symlink():
                raise OSError("symlink cache")
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.auth_path.parent,
                prefix=f".{self.auth_path.name}.", suffix=".tmp", delete=False,
            ) as output:
                temporary = output.name
                os.chmod(temporary, 0o600)
                json.dump(data, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.auth_path)
        except OSError:
            raise GrokAuthError(
                "Could not atomically save refreshed Grok tokens; check file permissions. "
                "Fresh credentials remain in this auth instance only."
            ) from None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
