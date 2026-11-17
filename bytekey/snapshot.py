"""Persistent, tamper-checked snapshot and pagination tokens for collections.

Both token kinds are self-describing HMAC-signed envelopes, so they survive
connection close and process exit: any process that can open the collection
file can verify and replay them.  They never carry row data, only identities
and resume positions.

* A **snapshot token** binds the collection identity, the pinned generation
  and a server-side snapshot id (``sid``).  The row for ``sid`` lives in the
  collection database, so ``release_snapshot`` can revoke the token and its
  cursors centrally -- a released or forged token is always refused.

* A **collection cursor** additionally binds the exact snapshot id (so a
  cursor from one snapshot cannot be replayed against another snapshot of the
  same generation), the canonical query spec, the page size and the last full
  encoded key delivered.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .planner import canonical_query_spec

TOKEN_VERSION = 1


class TokenError(ValueError):
    """Raised for malformed, forged, foreign, released or expired tokens."""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    if not isinstance(text, str):
        raise TokenError("token must be a string")
    try:
        padding = "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text + padding)
    except Exception as exc:
        raise TokenError("token is not valid base64url") from exc


def _sign(secret: str, raw: bytes) -> str:
    return _b64url(hmac.new(secret.encode("ascii"), raw, hashlib.sha256).digest())


def _pack(body: Mapping[str, Any], secret: str) -> str:
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _b64url(raw) + "." + _sign(secret, raw)


def _unpack(token: str, secret: str, kind: str) -> dict:
    if not isinstance(token, str) or token.count(".") != 1:
        raise TokenError(f"malformed {kind} token")
    body_b64, sig_b64 = token.split(".")
    raw = _b64url_decode(body_b64)
    sig = _b64url_decode(sig_b64)
    expected = hmac.new(
        secret.encode("ascii"), raw, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(sig, expected):
        raise TokenError(f"{kind} token signature mismatch (forged or foreign)")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenError(f"{kind} token body is not valid JSON") from exc
    if not isinstance(body, dict) or body.get("v") != TOKEN_VERSION or body.get("k") != kind:
        raise TokenError(f"unsupported or malformed {kind} token")
    return body


# ---------------------------------------------------------------------------
# Snapshot tokens
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SnapshotTicket:
    collection_id: str
    gen: int
    sid: str
    db_path: str


def encode_snapshot_token(
    *, secret: str, collection_id: str, gen: int, sid: str, db_path: str
) -> str:
    return _pack(
        {
            "v": TOKEN_VERSION,
            "k": "snap",
            "cid": collection_id,
            "gen": int(gen),
            "sid": sid,
            "path": db_path,
        },
        secret,
    )


def decode_snapshot_token(token: str, secret: str) -> SnapshotTicket:
    body = _unpack(token, secret, "snap")
    try:
        gen = int(body["gen"])
        sid = str(body["sid"])
        cid = str(body["cid"])
        path = str(body["path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TokenError("snapshot token is missing fields") from exc
    return SnapshotTicket(collection_id=cid, gen=gen, sid=sid, db_path=path)


# ---------------------------------------------------------------------------
# Collection pagination cursors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CollectionCursor:
    collection_id: str
    sid: str
    gen: int
    query_spec: dict
    page_size: int
    last_key: bytes
    db_path: str


def encode_collection_cursor(
    *,
    secret: str,
    collection_id: str,
    sid: str,
    gen: int,
    query_spec: Mapping[str, Any],
    page_size: int,
    last_key: bytes,
    db_path: str,
) -> str:
    if not isinstance(page_size, int) or page_size <= 0:
        raise TokenError("page_size must be a positive integer")
    if not isinstance(last_key, (bytes, bytearray)) or not last_key:
        raise TokenError("cursor requires a non-empty last_key")
    return _pack(
        {
            "v": TOKEN_VERSION,
            "k": "cur",
            "cid": collection_id,
            "sid": sid,
            "gen": int(gen),
            "q": canonical_query_spec(query_spec),
            "ps": int(page_size),
            "lk": _b64url(bytes(last_key)),
            "path": db_path,
        },
        secret,
    )


def decode_collection_cursor(token: str, secret: str) -> CollectionCursor:
    body = _unpack(token, secret, "cur")
    try:
        return CollectionCursor(
            collection_id=str(body["cid"]),
            sid=str(body["sid"]),
            gen=int(body["gen"]),
            query_spec=dict(body["q"]),
            page_size=int(body["ps"]),
            last_key=_b64url_decode(str(body["lk"])),
            db_path=str(body["path"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TokenError("cursor token is missing or carries invalid fields") from exc
