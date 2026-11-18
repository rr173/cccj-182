"""Disk-backed two-source equi-join over pinned collection snapshots.

A **join read** combines two bytekey collections (or one collection with
itself) that were each pinned to a generation with an existing snapshot
token.  Each side independently chooses its access path -- the primary
ordering or one *ready* named secondary index -- and an ordinary bytekey
range query that filters its rows first.  The two schemas, column counts,
orderings and index column orders may differ arbitrarily; the join columns
do not have to be a prefix of either access path.

Execution model (no external database, one SQLite work file)
------------------------------------------------------------

``prepare_join`` copies the two *filtered, generation-pinned* inputs into a
caller-specified **work file** exactly once:

``bytekey_join_left(lseq, jkey, has_null, payload)``
    every visible left row in left path order (``lseq`` 1..L).  ``payload``
    is the row's full encoded main key, so delivery decodes the complete
    business tuple in schema order plus the original row_id -- non-join
    columns are never dropped.

``bytekey_join_right(rseq, jkey, has_null, matched, payload)``
    every visible right row in right path order (``rseq`` 1..R).

``jkey`` is a *neutral* type-tagged encoding of the mapped raw values
(ascending, nulls-first components built with the unchanged
:func:`bytekey.encoding.encode_component`).  It is never either side's
storage key: direction inversion and the row_id tie-break of the sources
play no role in equality.  Null, empty text and empty bytes keep distinct
tags, and text is strict UTF-8 with no normalization.  Equality of ``jkey``
byte strings is exactly component-wise business equality.

Two indexes make the right side seekable: ``(jkey, rseq)`` -- a duplicate
key group is one contiguous, positionally seekable run -- and
``(matched, rseq)`` for the ``right_only`` phase of ``full`` joins.  For
``full`` mode ``matched`` is precomputed once at prepare time with a set
operation over the two stored inputs (never a stored pair set).

Paging is a pure function of the immutable work file: left rows are
advanced with a rowid seek, matches are produced with a
``jkey = ? AND rseq > ?`` seek that lands *inside* a duplicate group
without re-reading it, and ``right_only`` rows are delivered from the
``matched = 0`` index range.  Nothing is held in Python containers beyond
one page plus one look-ahead row; a group larger than memory and a product
larger than both inputs simply stream.  The pairing result is never
materialized on disk either -- only the two inputs live in the work file.

Result order is the *left path order*: each left row's matches arrive in
right path order, an unmatched left row is delivered in place
(``left_only``), and ``full`` appends the unmatched right rows
(``right_only``) in right path order.  Direction, null placement and the
row_id tie-break all come from the source paths, not from the join key.

Identity, cursors and source eligibility
----------------------------------------

The work file carries a persistent ``join_id`` and a random secret.  The
**handle token** returned by ``prepare_join`` and every **page cursor** are
HMAC-signed envelopes bound to that identity and to a declaration hash
covering both collection identities, both snapshot ids and generations,
both access-path instances (index name + immutable instance id +
declaration hash), both canonical range queries, the column mapping, the
mode, the null rule and the page size.  A cursor from another join read --
even one declared identically -- cannot verify.

The caller owns the snapshot tokens; the join never releases them.  Source
eligibility is re-verified at prepare start, at prepare commit and on every
page: a released token, a deleted or same-name-replaced index instance, or
a collection file replaced by another collection is refused with the
offending side named.  A prepare that is killed mid-way rolls back to the
``building`` state, which ``open_join`` rejects and ``join_cleanup``
removes; a completed work file is never overwritten.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .collection import (
    META_TABLE as COLL_META_TABLE,
    SNAPSHOT_TABLE,
    Collection,
    open_collection,
)
from .encoding import encode_component, key_decode, key_encode
from .planner import plan_query
from .schema import Column, Schema
from .secondary import (
    SEC_INDEX_TABLE,
    CoverageError,
    SecondaryDeclaration,
    plan_index_query,
    resolve_declaration,
    validate_query_columns,
)
from .snapshot import TokenError, decode_snapshot_token

JOIN_FORMAT = "bytekey-join/v1"
DECLARATION_FORMAT = "bytekey-join-decl/v1"
JOIN_APPLICATION_ID = 0x626A6F69  # 'bjoi'
META_TABLE = "bytekey_join_meta"
LEFT_TABLE = "bytekey_join_left"
RIGHT_TABLE = "bytekey_join_right"
TOKEN_VERSION = 1
BUSY_TIMEOUT_MS = 10000
SCAN_CHUNK = 1000

BUILDING = "building"
READY = "ready"
CLOSED = "closed"

INNER = "inner"
LEFT = "left"
FULL = "full"
MODES = (INNER, LEFT, FULL)

MATCHED = "matched"
LEFT_ONLY = "left_only"
RIGHT_ONLY = "right_only"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class JoinError(RuntimeError):
    """Base class for every join-side failure."""


class JoinDeclarationError(JoinError, ValueError):
    """The join declaration is invalid (mapping, mode, types, columns)."""


class JoinStateError(JoinError):
    """The work file is missing, foreign, unfinished, closed or protected."""


class JoinSourceError(JoinError):
    """A source collection, token or index lost eligibility; names the side."""

    def __init__(self, side: str, message: str):
        self.side = side
        super().__init__(f"{side} source: {message}")


class JoinCredentialError(JoinError, ValueError):
    """A handle or cursor does not belong to this join read."""


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JoinSide:
    """The present side of a join result: identity + full business tuple."""

    row_id: bytes
    values: tuple


@dataclass(frozen=True)
class JoinRow:
    """One join result.  The absent side is ``None`` -- never a null row."""

    kind: str                     # MATCHED | LEFT_ONLY | RIGHT_ONLY
    left: JoinSide | None
    right: JoinSide | None


@dataclass
class JoinPage:
    rows: list[JoinRow]
    next_cursor: str | None
    join_id: str
    declaration_hash: str


@dataclass
class JoinPageStats:
    left_rows_examined: int        # left rows pulled from the work file
    right_candidates_examined: int  # right rows pulled by probes / right_only
    full_row_reads: int            # payloads decoded for delivered rows
    pairs_delivered: int
    left_only_delivered: int
    right_only_delivered: int
    has_more: bool
    source_checks: dict            # per-side eligibility evidence
    explain_query_plan: dict       # work-file seek plans for the 3 scan shapes

    def to_dict(self) -> dict:
        return {
            "left_rows_examined": self.left_rows_examined,
            "right_candidates_examined": self.right_candidates_examined,
            "full_row_reads": self.full_row_reads,
            "pairs_delivered": self.pairs_delivered,
            "left_only_delivered": self.left_only_delivered,
            "right_only_delivered": self.right_only_delivered,
            "has_more": self.has_more,
            "source_checks": self.source_checks,
            "explain_query_plan": self.explain_query_plan,
        }


@dataclass(frozen=True)
class JoinSource:
    """One side of a join: collection file, snapshot token, path and filter.

    ``index=None`` selects the collection's primary ordering; otherwise it
    names a *ready* secondary access index.  ``query`` is an ordinary
    bytekey range query (``full_tuple`` or ``prefix_range``) interpreted on
    that path.
    """

    collection: str
    token: str
    query: Mapping[str, Any]
    index: str | None = None


# ---------------------------------------------------------------------------
# Signed token helpers (handle + page cursor), mirroring bytekey.snapshot
# ---------------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    if not isinstance(text, str):
        raise JoinCredentialError("token must be a string")
    try:
        padding = "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text + padding)
    except Exception as exc:
        raise JoinCredentialError("token is not valid base64url") from exc


def _pack(body: Mapping[str, Any], secret: str) -> str:
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(secret.encode("ascii"), raw, hashlib.sha256).digest()
    return _b64url(raw) + "." + _b64url(sig)


def _unpack(token: str, secret: str, kind: str) -> dict:
    if not isinstance(token, str) or token.count(".") != 1:
        raise JoinCredentialError(f"malformed {kind} token")
    body_b64, sig_b64 = token.split(".")
    raw = _b64url_decode(body_b64)
    sig = _b64url_decode(sig_b64)
    expected = hmac.new(secret.encode("ascii"), raw, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise JoinCredentialError(
            f"{kind} token signature mismatch (forged or foreign)"
        )
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JoinCredentialError(f"{kind} token body is not valid JSON") from exc
    if not isinstance(body, dict) or body.get("v") != TOKEN_VERSION or body.get("k") != kind:
        raise JoinCredentialError(f"unsupported or malformed {kind} token")
    return body


def _encode_handle(*, secret: str, join_id: str, path: str) -> str:
    return _pack(
        {"v": TOKEN_VERSION, "k": "jhandle", "jid": join_id, "path": path},
        secret,
    )


def _decode_handle(token: str, secret: str) -> dict:
    body = _unpack(token, secret, "jhandle")
    if not isinstance(body.get("jid"), str) or not isinstance(body.get("path"), str):
        raise JoinCredentialError("handle token is missing fields")
    return body


def _encode_cursor(
    *, secret: str, join_id: str, declaration_hash: str, page_size: int,
    pos: Mapping[str, Any], path: str,
) -> str:
    body = {
        "v": TOKEN_VERSION,
        "k": "jcur",
        "jid": join_id,
        "dh": declaration_hash,
        "ps": int(page_size),
        "ph": pos["ph"],
        "path": path,
    }
    if pos["ph"] in ("m", "l"):
        body["ls"] = int(pos["ls"])
    if pos["ph"] == "m":
        body["rs"] = int(pos["rs"])
    if pos["ph"] == "r":
        body["rs"] = int(pos["rs"])
    return _pack(body, secret)


def _decode_cursor(token: str, secret: str) -> dict:
    body = _unpack(token, secret, "jcur")
    try:
        ph = body["ph"]
        if ph not in ("m", "l", "r"):
            raise KeyError("ph")
        pos = {"ph": ph}
        if ph in ("m", "l"):
            pos["ls"] = int(body["ls"])
        if ph in ("m", "r"):
            pos["rs"] = int(body["rs"])
        return {
            "jid": str(body["jid"]),
            "dh": str(body["dh"]),
            "ps": int(body["ps"]),
            "path": str(body["path"]),
            "pos": pos,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise JoinCredentialError(
            "cursor token is missing or carries invalid fields"
        ) from exc


# ---------------------------------------------------------------------------
# Declaration validation
# ---------------------------------------------------------------------------

def _normalize_mapping(mapping: Any) -> list[tuple[str, str]]:
    """Accept ``[[left, right], ...]`` or ``{"left": [...], "right": [...]}``."""
    pairs: list[tuple[Any, Any]] = []
    if isinstance(mapping, Mapping):
        lcols = mapping.get("left")
        rcols = mapping.get("right")
        for label, cols in (("left", lcols), ("right", rcols)):
            if not isinstance(cols, Sequence) or isinstance(cols, (str, bytes)):
                raise JoinDeclarationError(
                    f"mapping.{label} must be a list of column names"
                )
        if len(lcols) != len(rcols):
            raise JoinDeclarationError(
                "mapping.left and mapping.right must have the same length"
            )
        pairs = list(zip(lcols, rcols))
    elif isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes)):
        for i, item in enumerate(mapping):
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) != 2
            ):
                raise JoinDeclarationError(
                    f"mapping entry {i} must be a [left_column, right_column] pair"
                )
            pairs.append((item[0], item[1]))
    else:
        raise JoinDeclarationError(
            "column mapping must be a non-empty ordered list of pairs"
        )
    if not pairs:
        raise JoinDeclarationError(
            "column mapping must declare at least one column pair"
        )
    for lname, rname in pairs:
        for label, name in (("left", lname), ("right", rname)):
            if not isinstance(name, str) or not name:
                raise JoinDeclarationError(
                    f"{label} mapping column must be a non-empty string, "
                    f"got {name!r}"
                )
    lnames = [p[0] for p in pairs]
    rnames = [p[1] for p in pairs]
    if len(set(lnames)) != len(lnames):
        raise JoinDeclarationError("left mapping columns must not repeat")
    if len(set(rnames)) != len(rnames):
        raise JoinDeclarationError("right mapping columns must not repeat")
    return [(str(l), str(r)) for l, r in pairs]


def _bind_mapping(
    pairs: list[tuple[str, str]], lschema: Schema, rschema: Schema
) -> tuple[list[int], list[int], tuple[Column, ...]]:
    """Positions per side plus the neutral join-key component declarations."""
    lpos: list[int] = []
    rpos: list[int] = []
    join_cols: list[Column] = []
    for i, (lname, rname) in enumerate(pairs):
        try:
            lcol = lschema.column(lname)
        except KeyError:
            raise JoinDeclarationError(
                f"left column {lname!r} is not in the left schema"
            ) from None
        try:
            rcol = rschema.column(rname)
        except KeyError:
            raise JoinDeclarationError(
                f"right column {rname!r} is not in the right schema"
            ) from None
        if lcol.kind != rcol.kind:
            raise JoinDeclarationError(
                f"mapped columns {lname!r} ({lcol.kind}) and {rname!r} "
                f"({rcol.kind}) have different kinds"
            )
        lpos.append(next(i for i, c in enumerate(lschema.columns) if c.name == lname))
        rpos.append(next(i for i, c in enumerate(rschema.columns) if c.name == rname))
        # Neutral components: ascending, nulls-first.  Only equality is ever
        # asked of a jkey, so the direction choice is immaterial -- but it is
        # fixed so equal business keys always produce equal bytes.
        join_cols.append(Column(f"jk{i}", lcol.kind, asc=True, nulls_first=True))
    return lpos, rpos, tuple(join_cols)


def _join_key(
    join_cols: tuple[Column, ...], values: Sequence[Any], positions: Sequence[int]
) -> tuple[bytes, bool]:
    """Neutral type-tagged join key + 'has a null component' flag."""
    parts = []
    has_null = False
    for col, pos in zip(join_cols, positions):
        value = values[pos]
        if value is None:
            has_null = True
        parts.append(encode_component(col, value))
    return b"".join(parts), has_null


# ---------------------------------------------------------------------------
# Source resolution and eligibility
# ---------------------------------------------------------------------------

def _resolve_side(source: JoinSource, side: str) -> dict:
    """Open the collection and pin down token, access path and query plan."""
    if not isinstance(source, JoinSource):
        raise JoinDeclarationError(f"{side} source must be a JoinSource")
    if not isinstance(source.query, Mapping):
        raise JoinDeclarationError(f"{side} query must be an object")
    coll = open_collection(source.collection)
    try:
        try:
            ticket = coll._resolve_snapshot(source.token)
        except TokenError as exc:
            raise JoinSourceError(side, f"snapshot token refused: {exc}") from None
        if source.index is None:
            plan = plan_query(coll.schema, source.query)
            access = {"kind": "main"}
        else:
            info = coll._resolve_ready_index(source.index)
            decl = resolve_declaration(
                coll.schema, SecondaryDeclaration.from_json(info["declaration_json"])
            )
            if ticket.gen < info["start_gen"]:
                raise CoverageError(
                    f"{side} snapshot generation {ticket.gen} is older than "
                    f"index {source.index!r}'s coverage start "
                    f"{info['start_gen']}; the index cannot answer that "
                    "generation"
                )
            validate_query_columns(coll.schema, decl, source.query)
            plan = plan_index_query(coll.schema, decl, source.query)
            access = {
                "kind": "index",
                "name": source.index,
                "instance_id": info["instance_id"],
                "declaration_hash": info["declaration_hash"],
            }
        return {
            "coll": coll,
            "schema": coll.schema,
            "token": source.token,
            "sid": ticket.sid,
            "gen": ticket.gen,
            "access": access,
            "query": source.query,
            "query_spec": plan.query_spec,
        }
    except BaseException:
        coll.close()
        raise


def _verify_source(vdecl: Mapping[str, Any], side: str) -> dict:
    """Re-check one source's eligibility from a fresh read-only connection.

    Called at prepare start (inside :func:`_resolve_side`), before the
    prepare commit, and on every page.  Returns the evidence dict.
    """
    path = vdecl["collection"]
    if not os.path.exists(path):
        raise JoinSourceError(side, f"collection file is missing: {path}")
    uri = "file:" + path.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise JoinSourceError(side, f"cannot open collection file: {exc}") from None
    try:
        conn.execute("PRAGMA busy_timeout = %d" % BUSY_TIMEOUT_MS)
        try:
            meta = dict(
                conn.execute(
                    f"SELECT name, value FROM {COLL_META_TABLE}"
                ).fetchall()
            )
        except sqlite3.DatabaseError:
            raise JoinSourceError(
                side, "file is not a bytekey collection (replaced by another file?)"
            ) from None
        if meta.get("collection_id") != vdecl["collection_id"]:
            raise JoinSourceError(
                side, "collection file was replaced by another collection"
            )
        try:
            ticket = decode_snapshot_token(vdecl["token"], meta.get("secret", ""))
        except TokenError as exc:
            raise JoinSourceError(
                side, f"snapshot token no longer verifies: {exc}"
            ) from None
        if (
            ticket.collection_id != vdecl["collection_id"]
            or ticket.sid != vdecl["sid"]
            or ticket.gen != vdecl["gen"]
        ):
            raise JoinSourceError(side, "snapshot token identity changed")
        row = conn.execute(
            f"SELECT gen, status FROM {SNAPSHOT_TABLE} WHERE sid = ?",
            (vdecl["sid"],),
        ).fetchone()
        if row is None:
            raise JoinSourceError(
                side, "snapshot token is unknown on the collection"
            )
        if row[1] != "active":
            raise JoinSourceError(
                side, f"snapshot token has been {row[1]}"
            )
        if row[0] != vdecl["gen"]:
            raise JoinSourceError(
                side, "snapshot generation changed on the collection"
            )
        access = vdecl["access"]
        evidence = {
            "collection": path,
            "collection_id": vdecl["collection_id"],
            "sid": vdecl["sid"],
            "gen": vdecl["gen"],
            "access": access["kind"],
            "ok": True,
        }
        if access["kind"] == "index":
            row = conn.execute(
                f"SELECT name, status, declaration_hash FROM {SEC_INDEX_TABLE} "
                "WHERE instance_id = ?",
                (access["instance_id"],),
            ).fetchone()
            if row is None:
                raise JoinSourceError(
                    side,
                    f"secondary index instance {access['instance_id']} vanished",
                )
            name, status, decl_hash = row
            if status != "ready":
                raise JoinSourceError(
                    side,
                    f"secondary index {name!r} is now {status!r} "
                    "(deleted or replaced under the same name)",
                )
            if name != access["name"] or decl_hash != access["declaration_hash"]:
                raise JoinSourceError(
                    side, "secondary index declaration changed"
                )
            evidence["access"] = f"index:{name}"
            evidence["instance_id"] = access["instance_id"]
        return evidence
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Work file
# ---------------------------------------------------------------------------

def _connect_work(path: str, *, create: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000.0)
    conn.execute("PRAGMA busy_timeout = %d" % BUSY_TIMEOUT_MS)
    if create:
        # Persistent header markers, set once at creation: the journal mode
        # and the application id that identifies an interrupted file as a
        # join work file even before its declaration commits.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = FULL")
        conn.execute(f"PRAGMA application_id = {JOIN_APPLICATION_ID}")
    return conn


def _work_application_id(path: str) -> int | None:
    """The file's application_id, or None when it cannot be read."""
    uri = "file:" + path.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return None
    try:
        return int(conn.execute("PRAGMA application_id").fetchone()[0])
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()


def _read_work_meta(path: str) -> dict:
    """Read the work-file meta table; empty dict when it is not a join file."""
    uri = "file:" + path.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return {}
    try:
        rows = conn.execute(
            f"SELECT name, value FROM {META_TABLE}"
        ).fetchall()
        meta = dict(rows)
        if meta.get("format") != JOIN_FORMAT:
            return {}
        return meta
    except sqlite3.DatabaseError:
        return {}
    finally:
        conn.close()


def _describe_existing(path: str) -> str:
    meta = _read_work_meta(path)
    if not meta:
        if _work_application_id(path) == JOIN_APPLICATION_ID:
            return (
                f"work file {path!r} is an interrupted join prepare whose "
                "declaration never committed; clean it up with join_cleanup "
                "before re-preparing"
            )
        return (
            f"work file {path!r} exists and is not a join work file; "
            "refusing to overwrite it"
        )
    state = meta.get("state")
    jid = meta.get("join_id")
    if state == READY:
        return (
            f"work file {path!r} is a completed join read (join_id {jid}); "
            "refusing to overwrite it -- close it and clean it up first"
        )
    if state == BUILDING:
        return (
            f"work file {path!r} is an unfinished join prepare (join_id "
            f"{jid}); clean it up with join_cleanup before re-preparing"
        )
    if state == CLOSED:
        return (
            f"work file {path!r} is a closed join read (join_id {jid}); "
            "clean it up with join_cleanup before reusing the path"
        )
    return f"work file {path!r} exists; refusing to overwrite it"


def _remove_work_files(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        stray = path + suffix
        if os.path.exists(stray):
            try:
                os.remove(stray)
            except OSError:
                pass


def _declaration_hash(declaration: Mapping[str, Any]) -> str:
    raw = json.dumps(
        declaration, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hook(crash_hook, phase: str, conn) -> None:
    if crash_hook is not None:
        crash_hook(phase, conn)


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def prepare_join(
    work_path: str,
    *,
    left: JoinSource,
    right: JoinSource,
    mapping: Any,
    mode: str,
    nulls_equal: bool = False,
    page_size: int,
    chunk_size: int = SCAN_CHUNK,
    crash_hook=None,
) -> dict:
    """Build the work file for one join read and return its handle.

    Both filtered inputs are streamed from their pinned generations into the
    work file exactly once (bounded memory, chunked inserts).  The pairing
    result is never stored.  On success the work file is ``ready`` and the
    returned handle token plus the work file path open it in any process.
    """
    if not isinstance(work_path, str) or not work_path:
        raise JoinDeclarationError("work_path must be a non-empty string")
    pairs = _normalize_mapping(mapping)
    if mode not in MODES:
        raise JoinDeclarationError(f"mode must be one of {MODES}, got {mode!r}")
    if not isinstance(nulls_equal, bool):
        raise JoinDeclarationError("nulls_equal must be a bool")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
        raise JoinDeclarationError("page_size must be a positive integer")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise JoinDeclarationError("chunk_size must be a positive integer")
    if not isinstance(left, JoinSource) or not isinstance(right, JoinSource):
        raise JoinDeclarationError("left and right must be JoinSource objects")

    work_path = os.path.abspath(work_path)
    if os.path.exists(work_path):
        raise JoinStateError(_describe_existing(work_path))
    directory = os.path.dirname(work_path) or "."
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise JoinStateError(
            f"cannot create work file directory {directory!r}: {exc}"
        ) from None

    # Resolve (and thereby verify) both sources before any work-file write.
    lres = _resolve_side(left, "left")
    try:
        rres = _resolve_side(right, "right")
    except BaseException:
        lres["coll"].close()
        raise
    try:
        lpos, rpos, join_cols = _bind_mapping(
            pairs, lres["schema"], rres["schema"]
        )
        declaration = {
            "format": DECLARATION_FORMAT,
            "left": {
                "collection": lres["coll"].path,
                "collection_id": lres["coll"].collection_id,
                "token": left.token,
                "sid": lres["sid"],
                "gen": lres["gen"],
                "access": lres["access"],
                "query": lres["query_spec"],
                "schema": lres["schema"].to_dict(),
            },
            "right": {
                "collection": rres["coll"].path,
                "collection_id": rres["coll"].collection_id,
                "token": right.token,
                "sid": rres["sid"],
                "gen": rres["gen"],
                "access": rres["access"],
                "query": rres["query_spec"],
                "schema": rres["schema"].to_dict(),
            },
            "mapping": [[l, r] for l, r in pairs],
            "mode": mode,
            "nulls_equal": nulls_equal,
            "page_size": page_size,
        }
        dh = _declaration_hash(declaration)
        join_id = secrets.token_hex(16)
        secret = secrets.token_hex(32)

        conn = None
        try:
            conn = _connect_work(work_path, create=True)
        except sqlite3.Error as exc:
            raise JoinStateError(
                f"cannot create join work file {work_path!r}: {exc}"
            ) from None
        try:
            conn.execute(
                f"CREATE TABLE {META_TABLE} (name TEXT PRIMARY KEY, "
                "value TEXT NOT NULL) WITHOUT ROWID"
            )
            conn.execute(
                f"CREATE TABLE {LEFT_TABLE} ("
                "lseq INTEGER PRIMARY KEY, jkey BLOB NOT NULL, "
                "has_null INTEGER NOT NULL, payload BLOB NOT NULL)"
            )
            conn.execute(
                f"CREATE TABLE {RIGHT_TABLE} ("
                "rseq INTEGER PRIMARY KEY, jkey BLOB NOT NULL, "
                "has_null INTEGER NOT NULL, matched INTEGER NOT NULL DEFAULT 0, "
                "payload BLOB NOT NULL)"
            )
            meta = {
                "format": JOIN_FORMAT,
                "join_id": join_id,
                "secret": secret,
                "state": BUILDING,
                "declaration_json": json.dumps(
                    declaration, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ),
                "declaration_hash": dh,
                "stats_json": "",
            }
            conn.executemany(
                f"INSERT INTO {META_TABLE}(name, value) VALUES (?, ?)",
                list(meta.items()),
            )
            conn.commit()
            _hook(crash_hook, "after_meta", conn)
            stats = _load_and_finalize(
                conn, work_path, lres, rres, lpos, rpos, join_cols,
                mode, nulls_equal, chunk_size, crash_hook,
            )
        except BaseException:
            conn.close()
            raise
        else:
            conn.close()
        _hook(crash_hook, "post_commit", None)
        return {
            "status": "ready",
            "join_id": join_id,
            "handle_token": _encode_handle(
                secret=secret, join_id=join_id, path=work_path
            ),
            "declaration_hash": dh,
            "work_path": work_path,
            "stats": stats,
        }
    finally:
        lres["coll"].close()
        rres["coll"].close()


def _scan_side_into(
    conn: sqlite3.Connection,
    res: Mapping[str, Any],
    table: str,
    positions: Sequence[int],
    join_cols: tuple[Column, ...],
    chunk_size: int,
    seq_column: str,
) -> dict:
    """Stream one filtered source into its work-file table in path order."""
    coll: Collection = res["coll"]
    schema: Schema = res["schema"]
    access = res["access"]
    candidates = 0
    invisible = 0
    lookups = 0
    seq = 0
    eqp_sample: list[dict] = []
    intervals: list[dict] = []
    buf: list[tuple] = []
    insert_sql = (
        f"INSERT INTO {table}({seq_column}, jkey, has_null, payload) "
        "VALUES (?, ?, ?, ?)"
    )

    def flush() -> None:
        if buf:
            conn.executemany(insert_sql, buf)
            buf.clear()

    cursor_token = None
    while True:
        if access["kind"] == "main":
            page, st, _plan = coll.scan_snapshot_page(
                res["token"], res["query"], chunk_size, cursor_token=cursor_token
            )
            candidates += st.candidates_examined
        else:
            page, st, _plan = coll.scan_index_page(
                res["token"], access["name"], res["query"], chunk_size,
                cursor_token=cursor_token,
            )
            candidates += st.index_candidates_examined
            lookups += st.full_row_lookups
        invisible += st.invisible_versions_skipped
        if not eqp_sample:
            eqp_sample = st.explain_query_plan
            intervals = st.intervals
        for values, rid in page.rows:
            seq += 1
            jkey, has_null = _join_key(join_cols, values, positions)
            payload = key_encode(schema, values, rid)
            buf.append((seq, jkey, 1 if has_null else 0, payload))
            if len(buf) >= chunk_size:
                flush()
        cursor_token = page.next_cursor
        if cursor_token is None:
            break
    flush()
    return {
        "access": access["kind"] if access["kind"] == "main"
        else f"index:{access['name']}",
        "gen": res["gen"],
        "sid": res["sid"],
        "candidates_examined": candidates,
        "invisible_versions_skipped": invisible,
        "full_row_lookups": lookups,
        "rows_written": seq,
        "intervals": intervals,
        "explain_query_plan": eqp_sample,
    }


def _load_and_finalize(
    conn, work_path, lres, rres, lpos, rpos, join_cols, mode, nulls_equal,
    chunk_size, crash_hook,
) -> dict:
    """One atomic transaction: copy both inputs, index them, mark ready.

    A kill anywhere before the commit rolls everything back to the durable
    ``building`` marker, so a half-loaded work file can never serve pages.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        left_stats = _scan_side_into(
            conn, lres, LEFT_TABLE, lpos, join_cols, chunk_size, "lseq"
        )
        _hook(crash_hook, "after_left", conn)
        right_stats = _scan_side_into(
            conn, rres, RIGHT_TABLE, rpos, join_cols, chunk_size, "rseq"
        )
        _hook(crash_hook, "after_right", conn)
        conn.execute(
            f"CREATE INDEX bytekey_join_left_jkey ON {LEFT_TABLE}(jkey, lseq)"
        )
        conn.execute(
            f"CREATE INDEX bytekey_join_right_jkey ON {RIGHT_TABLE}(jkey, rseq)"
        )
        conn.execute(
            f"CREATE INDEX bytekey_join_right_unmatched "
            f"ON {RIGHT_TABLE}(matched, rseq)"
        )
        matched_right = None
        if mode == FULL:
            if nulls_equal:
                conn.execute(
                    f"UPDATE {RIGHT_TABLE} SET matched = 1 "
                    f"WHERE jkey IN (SELECT jkey FROM {LEFT_TABLE})"
                )
            else:
                conn.execute(
                    f"UPDATE {RIGHT_TABLE} SET matched = 1 WHERE has_null = 0 "
                    f"AND jkey IN (SELECT jkey FROM {LEFT_TABLE} "
                    "WHERE has_null = 0)"
                )
            matched_right = conn.execute(
                f"SELECT COUNT(*) FROM {RIGHT_TABLE} WHERE matched = 1"
            ).fetchone()[0]
        _hook(crash_hook, "after_matched", conn)
        groups = {}
        for table, label in ((LEFT_TABLE, "left"), (RIGHT_TABLE, "right")):
            mx, dups = conn.execute(
                f"SELECT COALESCE(MAX(c), 0), "
                f"COALESCE(SUM(CASE WHEN c > 1 THEN 1 ELSE 0 END), 0) "
                f"FROM (SELECT COUNT(*) AS c FROM {table} GROUP BY jkey)"
            ).fetchone()
            groups[f"max_{label}_group"] = mx
            groups[f"{label}_duplicate_groups"] = dups
        # A duplicate group is never materialized in memory: it is a
        # contiguous, positionally seekable run of the (jkey, seq) index.
        groups["groups_disk_resident"] = True

        # Final eligibility check before the ready switch commits.
        checks = {
            "left": _verify_source(_verify_decl_of(lres), "left"),
            "right": _verify_source(_verify_decl_of(rres), "right"),
        }
        stats = {
            "left": left_stats,
            "right": right_stats,
            "groups": groups,
            "matched_right_rows": matched_right,
            "source_checks": checks,
        }
        _hook(crash_hook, "pre_commit", conn)
        conn.execute(
            f"UPDATE {META_TABLE} SET value = ? WHERE name = 'state'",
            (READY,),
        )
        conn.execute(
            f"UPDATE {META_TABLE} SET value = ? WHERE name = 'stats_json'",
            (json.dumps(stats, sort_keys=True),),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    # Disk footprint is reported after the commit so it reflects the real
    # file; the WAL is checkpointed first for an accurate number.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    stats["work_file_bytes"] = os.path.getsize(work_path)
    conn.execute(
        f"UPDATE {META_TABLE} SET value = ? WHERE name = 'stats_json'",
        (json.dumps(stats, sort_keys=True),),
    )
    conn.commit()
    return stats


def _verify_decl_of(res: Mapping[str, Any]) -> dict:
    """The verification view of a resolved source (matches the declaration)."""
    coll: Collection = res["coll"]
    return {
        "collection": coll.path,
        "collection_id": coll.collection_id,
        "token": res["token"],
        "sid": res["sid"],
        "gen": res["gen"],
        "access": res["access"],
    }


# ---------------------------------------------------------------------------
# Join handle
# ---------------------------------------------------------------------------

class Join:
    """An open handle on one prepared join read (work file + handle token)."""

    def __init__(self, work_path: str, handle_token: str):
        if not isinstance(work_path, str) or not work_path:
            raise JoinStateError("work_path must be a non-empty string")
        self.path = os.path.abspath(work_path)
        if not os.path.exists(self.path):
            raise JoinStateError(f"no join work file at {self.path!r}")
        self._conn = _connect_work(self.path)
        try:
            meta = dict(
                self._conn.execute(
                    f"SELECT name, value FROM {META_TABLE}"
                ).fetchall()
            )
        except sqlite3.DatabaseError:
            self._conn.close()
            if _work_application_id(self.path) == JOIN_APPLICATION_ID:
                raise JoinStateError(
                    "join prepare was interrupted before its declaration "
                    "committed; the partial file is not a usable join -- "
                    "clean up with join_cleanup or re-prepare"
                ) from None
            raise JoinStateError(
                f"{self.path!r} is not a join work file"
            ) from None
        if meta.get("format") != JOIN_FORMAT:
            self._conn.close()
            raise JoinStateError(f"{self.path!r} is not a join work file")
        self.join_id = meta["join_id"]
        self._secret = meta["secret"]
        self._declaration_hash = meta["declaration_hash"]
        self._declaration = json.loads(meta["declaration_json"])
        state = meta["state"]
        if state == BUILDING:
            self._conn.close()
            raise JoinStateError(
                "join prepare did not complete (state=building); the partial "
                "inputs are not a usable join -- clean up with join_cleanup "
                "or re-prepare"
            )
        if state == CLOSED:
            self._conn.close()
            raise JoinStateError("join read is closed")
        if state != READY:
            self._conn.close()
            raise JoinStateError(f"join work file has unknown state {state!r}")
        try:
            handle = _decode_handle(handle_token, self._secret)
            if handle["jid"] != self.join_id:
                raise JoinCredentialError(
                    "handle belongs to a different join read"
                )
            if handle["path"] != self.path:
                raise JoinCredentialError(
                    "handle was created against a different work file path"
                )
        except BaseException:
            self._conn.close()
            raise
        self._left_schema = Schema.from_dict(self._declaration["left"]["schema"])
        self._right_schema = Schema.from_dict(self._declaration["right"]["schema"])
        self._closed = False

    # -- introspection -----------------------------------------------------

    @property
    def declaration(self) -> dict:
        return dict(self._declaration)

    @property
    def declaration_hash(self) -> str:
        return self._declaration_hash

    def status(self) -> dict:
        return join_status(self.path)

    # -- internals ----------------------------------------------------------

    def _require_open_ready(self) -> None:
        if self._closed:
            raise JoinStateError("join handle object is closed")
        row = self._conn.execute(
            f"SELECT value FROM {META_TABLE} WHERE name = 'state'"
        ).fetchone()
        if row is None or row[0] != READY:
            raise JoinStateError(
                "join read is no longer ready (closed concurrently?)"
            )

    def _eqp(self, sql: str, params: Sequence[Any]) -> list[dict]:
        return [
            {"id": r[0], "parent": r[1], "notused": r[2], "detail": r[3]}
            for r in self._conn.execute("EXPLAIN QUERY PLAN " + sql, params)
        ]

    def _left_from(self, start_lseq: int, counters: dict) -> Iterable[tuple]:
        for row in self._conn.execute(
            f"SELECT lseq, jkey, has_null, payload FROM {LEFT_TABLE} "
            "WHERE lseq >= ? ORDER BY lseq",
            (start_lseq,),
        ):
            counters["left_rows_examined"] += 1
            yield row

    def _probe(self, jkey: bytes, after_rseq: int, counters: dict) -> Iterable[tuple]:
        # Seek inside the (jkey, rseq) index: lands in the middle of a
        # duplicate group without re-reading its head.
        for row in self._conn.execute(
            f"SELECT rseq, payload FROM {RIGHT_TABLE} "
            "WHERE jkey = ? AND rseq > ? ORDER BY rseq",
            (jkey, after_rseq),
        ):
            counters["right_candidates_examined"] += 1
            yield row

    def _right_only_from(self, after_rseq: int, counters: dict) -> Iterable[tuple]:
        for row in self._conn.execute(
            f"SELECT rseq, payload FROM {RIGHT_TABLE} "
            "WHERE matched = 0 AND rseq > ? ORDER BY rseq",
            (after_rseq,),
        ):
            counters["right_candidates_examined"] += 1
            yield row

    def _iter_results(self, pos: Mapping[str, Any] | None, counters: dict):
        """Yield ``(kind, lpayload, rpayload, cursor_pos)`` strictly after ``pos``.

        The generator is lazy on every side: left rows stream from a rowid
        seek, matches from an index seek, ``right_only`` rows from the
        unmatched index.  Only the page's look-ahead row is ever fetched
        beyond what is delivered.
        """
        decl = self._declaration
        mode = decl["mode"]
        nulls_equal = decl["nulls_equal"]
        start_lseq: int | None = None
        if pos is None:
            start_lseq = 1
        elif pos["ph"] == "m":
            row = self._conn.execute(
                f"SELECT jkey, has_null, payload FROM {LEFT_TABLE} "
                "WHERE lseq = ?",
                (pos["ls"],),
            ).fetchone()
            counters["left_rows_examined"] += 1
            if row is None:
                raise JoinStateError(
                    "work file is corrupt: cursor points past the left stream"
                )
            jkey, has_null, lpayload = row
            if nulls_equal or not has_null:
                for rseq, rpayload in self._probe(jkey, pos["rs"], counters):
                    yield (
                        MATCHED, lpayload, rpayload,
                        {"ph": "m", "ls": pos["ls"], "rs": rseq},
                    )
            start_lseq = pos["ls"] + 1
        elif pos["ph"] == "l":
            start_lseq = pos["ls"] + 1
        if start_lseq is not None:
            after_rseq = 0
            for lseq, jkey, has_null, lpayload in self._left_from(
                start_lseq, counters
            ):
                delivered = False
                if nulls_equal or not has_null:
                    for rseq, rpayload in self._probe(jkey, 0, counters):
                        delivered = True
                        yield (
                            MATCHED, lpayload, rpayload,
                            {"ph": "m", "ls": lseq, "rs": rseq},
                        )
                if not delivered and mode in (LEFT, FULL):
                    yield (LEFT_ONLY, lpayload, None, {"ph": "l", "ls": lseq})
        else:
            after_rseq = pos["rs"]
        if mode == FULL:
            for rseq, rpayload in self._right_only_from(after_rseq, counters):
                yield (RIGHT_ONLY, None, rpayload, {"ph": "r", "rs": rseq})

    # -- paging --------------------------------------------------------------

    def page(self, cursor_token: str | None = None) -> tuple[JoinPage, JoinPageStats]:
        """Deliver exactly one page; a pure function of the work file.

        Both sources' eligibility is re-verified on every call.  The same
        cursor always yields the same page and the same next cursor, no
        matter how often it is replayed, in this or another process.
        """
        self._require_open_ready()
        decl = self._declaration
        page_size = decl["page_size"]
        pos = None
        if cursor_token is not None:
            cursor = _decode_cursor(cursor_token, self._secret)
            if cursor["jid"] != self.join_id:
                raise JoinCredentialError(
                    "cursor belongs to a different join read"
                )
            if cursor["dh"] != self._declaration_hash:
                raise JoinCredentialError(
                    "cursor declaration does not match this join read"
                )
            if cursor["ps"] != page_size:
                raise JoinCredentialError(
                    f"cursor page_size is {cursor['ps']}, but this join read "
                    f"uses {page_size}"
                )
            if cursor["path"] != self.path:
                raise JoinCredentialError(
                    "cursor was created against a different work file path"
                )
            pos = cursor["pos"]

        # Source eligibility on every page, before any row is delivered.
        checks = {
            "left": _verify_source(decl["left"], "left"),
            "right": _verify_source(decl["right"], "right"),
        }

        counters = {
            "left_rows_examined": 0,
            "right_candidates_examined": 0,
            "full_row_reads": 0,
        }
        collected: list[tuple] = []
        for item in self._iter_results(pos, counters):
            collected.append(item)
            if len(collected) > page_size:
                break
        has_more = len(collected) > page_size
        emitted = collected[:page_size]

        rows: list[JoinRow] = []
        pairs = left_only = right_only = 0
        for kind, lpayload, rpayload, _pos in emitted:
            left_side = right_side = None
            if lpayload is not None:
                values, rid = key_decode(self._left_schema, lpayload)
                counters["full_row_reads"] += 1
                left_side = JoinSide(row_id=rid, values=values)
            if rpayload is not None:
                values, rid = key_decode(self._right_schema, rpayload)
                counters["full_row_reads"] += 1
                right_side = JoinSide(row_id=rid, values=values)
            if kind == MATCHED:
                pairs += 1
            elif kind == LEFT_ONLY:
                left_only += 1
            else:
                right_only += 1
            rows.append(JoinRow(kind=kind, left=left_side, right=right_side))

        next_cursor = None
        if has_more and emitted:
            next_cursor = _encode_cursor(
                secret=self._secret,
                join_id=self.join_id,
                declaration_hash=self._declaration_hash,
                page_size=page_size,
                pos=emitted[-1][3],
                path=self.path,
            )
        eqp = {
            "left_scan": self._eqp(
                f"SELECT lseq, jkey, has_null, payload FROM {LEFT_TABLE} "
                "WHERE lseq >= ? ORDER BY lseq",
                (1,),
            ),
            "right_probe": self._eqp(
                f"SELECT rseq, payload FROM {RIGHT_TABLE} "
                "WHERE jkey = ? AND rseq > ? ORDER BY rseq",
                (b"\x01", 0),
            ),
            "right_only_scan": self._eqp(
                f"SELECT rseq, payload FROM {RIGHT_TABLE} "
                "WHERE matched = 0 AND rseq > ? ORDER BY rseq",
                (0,),
            ),
        }
        stats = JoinPageStats(
            left_rows_examined=counters["left_rows_examined"],
            right_candidates_examined=counters["right_candidates_examined"],
            full_row_reads=counters["full_row_reads"],
            pairs_delivered=pairs,
            left_only_delivered=left_only,
            right_only_delivered=right_only,
            has_more=has_more,
            source_checks=checks,
            explain_query_plan=eqp,
        )
        page = JoinPage(
            rows=rows,
            next_cursor=next_cursor,
            join_id=self.join_id,
            declaration_hash=self._declaration_hash,
        )
        return page, stats

    def iter_pages(self) -> Iterable[tuple[JoinPage, JoinPageStats]]:
        """Yield pages until the stream is exhausted (bounded memory)."""
        cursor = None
        while True:
            page, stats = self.page(cursor)
            yield page, stats
            if page.next_cursor is None:
                return
            cursor = page.next_cursor

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Release this handle object's connection.

        This only closes the Python object; the join read itself (its
        identity, handle token and cursors) keeps working in this or any
        other process.  Use :meth:`terminate` to end the join read.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def terminate(self, *, reclaim: bool = False) -> dict:
        """End the join read: the handle token and every cursor stop verifying.

        With ``reclaim=True`` the work file is deleted as well; other joins
        and the caller's snapshot tokens are unaffected.
        """
        if self._closed:
            raise JoinStateError(
                "join handle object is closed; terminate via terminate_join()"
            )
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                f"SELECT value FROM {META_TABLE} WHERE name = 'state'"
            ).fetchone()
            if row is None or row[0] != READY:
                self._conn.rollback()
                raise JoinStateError(
                    f"join read is already {row[0] if row else 'missing'}"
                )
            self._conn.execute(
                f"UPDATE {META_TABLE} SET value = ? WHERE name = 'state'",
                (CLOSED,),
            )
            self._conn.commit()
        except BaseException:
            try:
                self._conn.rollback()
            except sqlite3.Error:
                pass
            raise
        path = self.path
        self.close()
        if reclaim:
            _remove_work_files(path)
        return {"terminated": self.join_id, "reclaimed": bool(reclaim)}

    def __enter__(self) -> "Join":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_join(work_path: str, handle_token: str) -> Join:
    """Open a prepared join read in this (or any new) process."""
    return Join(work_path, handle_token)


def terminate_join(work_path: str, handle_token: str, *, reclaim: bool = False) -> dict:
    """End a join read (the explicit "close the join" operation).

    The handle token and every cursor of this join read stop verifying; the
    disk work data is kept unless ``reclaim=True``.  Closing Python objects
    (:meth:`Join.close`) is *not* this operation and never invalidates a
    join read.
    """
    join = Join(work_path, handle_token)
    return join.terminate(reclaim=reclaim)


# ---------------------------------------------------------------------------
# Status / cleanup
# ---------------------------------------------------------------------------

def join_status(work_path: str) -> dict:
    """Describe a join work file (any state) without needing the handle."""
    if not isinstance(work_path, str) or not work_path:
        raise JoinStateError("work_path must be a non-empty string")
    path = os.path.abspath(work_path)
    if not os.path.exists(path):
        raise JoinStateError(f"no join work file at {path!r}")
    meta = _read_work_meta(path)
    if not meta:
        if _work_application_id(path) == JOIN_APPLICATION_ID:
            return {
                "work_path": path,
                "format": JOIN_FORMAT,
                "join_id": None,
                "declaration_hash": None,
                "state": "building",
                "incomplete": True,
                "detail": "prepare was interrupted before its declaration "
                          "committed; clean up with join_cleanup",
            }
        raise JoinStateError(f"{path!r} is not a join work file")
    declaration = json.loads(meta["declaration_json"])
    stats = json.loads(meta["stats_json"]) if meta.get("stats_json") else None

    def side_summary(side: str) -> dict:
        s = declaration[side]
        access = s["access"]
        return {
            "collection": s["collection"],
            "collection_id": s["collection_id"],
            "sid": s["sid"],
            "gen": s["gen"],
            "access": access["kind"] if access["kind"] == "main"
            else f"index:{access['name']}",
            "query": s["query"],
        }

    out = {
        "work_path": path,
        "format": meta["format"],
        "join_id": meta["join_id"],
        "declaration_hash": meta["declaration_hash"],
        "state": meta["state"],
        "mode": declaration["mode"],
        "nulls_equal": declaration["nulls_equal"],
        "page_size": declaration["page_size"],
        "mapping": declaration["mapping"],
        "left": side_summary("left"),
        "right": side_summary("right"),
    }
    if stats is not None:
        out["stats"] = stats
    return out


def join_cleanup(work_path: str) -> dict:
    """Delete an unfinished (``building``) or ``closed`` join work file.

    A ``ready`` work file belongs to an active join read and is refused;
    close it first.  This is the explicit recovery operation after a failed
    or killed prepare, including one interrupted before its declaration
    committed (the file is recognized by its application id).
    """
    if not isinstance(work_path, str) or not work_path:
        raise JoinStateError("work_path must be a non-empty string")
    path = os.path.abspath(work_path)
    if not os.path.exists(path):
        raise JoinStateError(f"no join work file at {path!r}")
    meta = _read_work_meta(path)
    if not meta:
        if _work_application_id(path) == JOIN_APPLICATION_ID:
            _remove_work_files(path)
            return {
                "removed": True,
                "was_state": "building",
                "join_id": None,
                "work_path": path,
            }
        raise JoinStateError(f"{path!r} is not a join work file")
    state = meta.get("state")
    if state == READY:
        raise JoinStateError(
            "join read is still active (ready); close it before cleanup"
        )
    _remove_work_files(path)
    return {
        "removed": True,
        "was_state": state,
        "join_id": meta.get("join_id"),
        "work_path": path,
    }
