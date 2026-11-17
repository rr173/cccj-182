"""Opaque, tamper-checked pagination cursors.

A cursor binds four identities together:

* the index it was built from (schema fingerprint + build secret + path);
* the exact query specification (canonical JSON);
* the page-size semantics;
* the last fully delivered encoded key (plus the index walk direction).

Replaying it on another index, with another query, or with a different page
size raises :class:`CursorError` before any database scan.  Because the
resume predicate is strictly ``key > last_key`` (or ``key < last_key`` when
the walk is reversed), every row_id sharing the last business tuple remains
reachable, while the already-delivered key itself is never repeated even
when it sits exactly on a range endpoint.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .planner import canonical_query_spec

CURSOR_VERSION = 1


class CursorError(ValueError):
    """Raised for malformed, foreign or mismatched cursors."""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    try:
        padding = "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text + padding)
    except Exception as exc:  # pragma: no cover - exercised via tests
        raise CursorError("cursor is not valid base64url") from exc


@dataclass(frozen=True)
class Cursor:
    index_fingerprint: str
    query_spec: dict
    page_size: int
    reverse_delivery: bool
    last_key: bytes | None
    build_secret: str
    db_path: str

    def to_token(self) -> str:
        body = {
            "v": CURSOR_VERSION,
            "index_fingerprint": self.index_fingerprint,
            "query_spec": self.query_spec,
            "page_size": self.page_size,
            "reverse_delivery": self.reverse_delivery,
            "last_key": _b64url(self.last_key) if self.last_key is not None else None,
            "build_secret": self.build_secret,
            "db_path": self.db_path,
        }
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sig = hmac.new(
            self.build_secret.encode("ascii"), raw, hashlib.sha256
        ).digest()
        return _b64url(raw) + "." + _b64url(sig)


def encode_cursor(
    *,
    index_fingerprint: str,
    query_spec: Mapping[str, Any],
    page_size: int,
    reverse_delivery: bool,
    last_key: bytes | None,
    build_secret: str,
    db_path: str,
) -> str:
    if not isinstance(page_size, int) or page_size <= 0:
        raise CursorError("page_size must be a positive integer")
    cursor = Cursor(
        index_fingerprint=index_fingerprint,
        query_spec=canonical_query_spec(query_spec),
        page_size=page_size,
        reverse_delivery=bool(reverse_delivery),
        last_key=bytes(last_key) if last_key is not None else None,
        build_secret=build_secret,
        db_path=db_path,
    )
    return cursor.to_token()


def decode_cursor(token: str) -> Cursor:
    if not isinstance(token, str) or token.count(".") != 1:
        raise CursorError("malformed cursor token")
    body_b64, sig_b64 = token.split(".")
    raw = _b64url_decode(body_b64)
    sig = _b64url_decode(sig_b64)
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CursorError("cursor body is not valid JSON") from exc
    required = {
        "v",
        "index_fingerprint",
        "query_spec",
        "page_size",
        "reverse_delivery",
        "last_key",
        "build_secret",
        "db_path",
    }
    if not required.issubset(body):
        raise CursorError("cursor is missing fields")
    if body["v"] != CURSOR_VERSION:
        raise CursorError(f"unsupported cursor version {body['v']!r}")
    expected = hmac.new(
        body["build_secret"].encode("ascii"), raw, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(sig, expected):
        raise CursorError("cursor signature mismatch (tampered or foreign token)")
    last_key = _b64url_decode(body["last_key"]) if body["last_key"] is not None else None
    if not isinstance(body["page_size"], int) or body["page_size"] <= 0:
        raise CursorError("cursor carries an invalid page_size")
    return Cursor(
        index_fingerprint=body["index_fingerprint"],
        query_spec=body["query_spec"],
        page_size=body["page_size"],
        reverse_delivery=bool(body["reverse_delivery"]),
        last_key=last_key,
        build_secret=body["build_secret"],
        db_path=body["db_path"],
    )
