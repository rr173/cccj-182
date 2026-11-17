"""Updatable collections with durable, multi-generation read snapshots.

A collection keeps the existing byte-key encoding, schema declarations and
the two query shapes, but turns the single immutable index into an append-only
*versioned* store.  Every successful :meth:`Collection.apply` publishes one
new, consecutively numbered generation (``head`` advances by exactly one);
snapshots pin a generation and read it forever, independently of later
publishes, connection close or process exit.

Storage model (one SQLite file, all real state on disk)
-------------------------------------------------------

``bytekey_kv(key, rid, cgen, dgen, PRIMARY KEY(key, rid, cgen)) WITHOUT ROWID``

* ``key``  -- the full encoded business key *including* the row_id component,
  produced by the unchanged :func:`bytekey.encoding.key_encode`;
* ``rid``  -- the non-empty row_id bytes (identity of a record);
* ``cgen`` -- generation that created this entry;
* ``dgen`` -- generation that superseded/deleted it, or NULL while live.

A partial unique index enforces at most one live (``dgen IS NULL``) entry per
row_id.  An entry is **visible at snapshot generation g** iff
``cgen <= g AND (dgen IS NULL OR dgen > g)``.  Because the primary key leads
with ``key``, every query is still a concrete BLOB range seek over the same
intervals the planner emits; in-range historical candidates are filtered by
the visibility predicate, rows outside the intervals are never decoded.

Small corrections never touch the whole table: an apply writes only the
superseded/inserted entries of the rows named in the batch -- O(batch) index
work, no full-table scan, no per-generation index copy.

Transactions, idempotency and crash safety
------------------------------------------

Each writer action (apply / snapshot begin / release / collect) takes
``BEGIN IMMEDIATE``, so all commits have one total serialization order across
processes.  A batch is fully validated into an on-disk staging table inside
the transaction before any collection row changes; any malformed source, bad
type, duplicate identity or ``expected_head`` mismatch rolls back and leaves
no partial update and no receipt (the batch_id stays free).  The receipt
table makes successful requests safely re-sendable.

History collection
------------------

``collect_history(R)`` refuses to move the retention boundary backwards and
physically deletes exactly the dead entries no retained generation and no
active snapshot can read.  Active snapshot tokens pin every entry their
generation can see; only after the last protecting token is released can a
later collection remove them.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .encoding import EncodingError, key_decode, key_encode
from .planner import Interval, QueryPlan, plan_query
from .schema import Schema
from .snapshot import (
    CollectionCursor,
    SnapshotTicket,
    TokenError,
    decode_collection_cursor,
    decode_snapshot_token,
    encode_collection_cursor,
    encode_snapshot_token,
)

META_TABLE = "bytekey_meta"
KV_TABLE = "bytekey_kv"
RECEIPT_TABLE = "bytekey_receipts"
SNAPSHOT_TABLE = "bytekey_snapshots"
STAGING_TABLE = "bytekey_staging"
COLLECTION_FORMAT = "bytekey-collection/v1"
APPLICATION_ID = 0x6279746B  # 'bytk', shared with the static index
DIGEST_VERSION = "bytekey-batch-digest/v1"
BUSY_TIMEOUT_MS = 10000


class CollectionError(RuntimeError):
    """Base class for every collection-side failure."""


class BatchError(CollectionError):
    """A batch is invalid (bad source, bad type, duplicate identity)."""


class ConflictError(CollectionError):
    """Head mismatch, competing writer, or idempotency-key conflict."""


class HistoryError(CollectionError):
    """Illegal retention-boundary move or generation outside retention."""


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class ApplyReceipt:
    batch_id: bytes
    head: int
    expected_head: int
    digest_hex: str
    operations: int
    applied: int
    noop: int
    replayed: bool

    def to_dict(self) -> dict:
        return {
            "batch_id": {"__hex__": self.batch_id.hex()},
            "head": self.head,
            "expected_head": self.expected_head,
            "digest": self.digest_hex,
            "operations": self.operations,
            "applied": self.applied,
            "noop": self.noop,
            "replayed": self.replayed,
        }


@dataclass
class SnapshotPage:
    rows: list[tuple[tuple, bytes]]
    next_cursor: str | None
    last_key: bytes | None
    gen: int
    sid: str


@dataclass
class SnapshotScanStats:
    candidates_examined: int       # rows physically pulled from storage
    invisible_versions_skipped: int
    rows_delivered: int
    has_more: bool
    used_index: bool
    explain_query_plan: list[dict]
    intervals: list[dict]


@dataclass
class HistoryStats:
    retain_from: int
    previous_retain_from: int
    entries_removed: int
    active_snapshots: int
    head: int


@dataclass
class CollectionStatus:
    path: str
    collection_id: str
    head: int
    retain_from: int
    schema: Schema
    row_id_name: str
    live_entries: int
    total_version_entries: int
    dead_entries: int
    receipt_count: int
    active_snapshots: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "collection_id": self.collection_id,
            "head": self.head,
            "retain_from": self.retain_from,
            "schema": self.schema.to_dict(),
            "row_id_name": self.row_id_name,
            "live_entries": self.live_entries,
            "total_version_entries": self.total_version_entries,
            "dead_entries": self.dead_entries,
            "receipt_count": self.receipt_count,
            "active_snapshots": list(self.active_snapshots),
        }


# ---------------------------------------------------------------------------
# Canonical batch content / digest
# ---------------------------------------------------------------------------

def _typed_json_value(value: Any) -> dict:
    """Type-tagged canonical JSON value.

    The tag makes null, empty text, empty bytes, bool and integers
    non-confusable in the digest, even when their plain JSON spelling would
    collide (``""`` vs ``""``) or be ambiguous (``true`` vs ``1``).
    """
    if value is None:
        return {"t": "null"}
    if isinstance(value, bool):
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        return {"t": "i64", "v": str(value)}
    if isinstance(value, str):
        return {"t": "txt", "v": value}
    if isinstance(value, (bytes, bytearray)):
        return {"t": "bin", "v": bytes(value).hex()}
    raise BatchError(f"value of type {type(value).__name__} is not digestible")


def _canonical_op_json(kind: str, rid: bytes, values: Sequence | None) -> bytes:
    if kind == "d":
        body = {"op": "delete", "rid": rid.hex()}
    else:
        body = {
            "op": "put",
            "rid": rid.hex(),
            "v": [_typed_json_value(v) for v in values],
        }
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _check_batch_id(batch_id: Any) -> bytes:
    if not isinstance(batch_id, (bytes, bytearray)):
        raise BatchError(f"batch_id must be bytes, got {type(batch_id).__name__}")
    batch_id = bytes(batch_id)
    if not batch_id:
        raise BatchError("batch_id must be a non-empty bytes value")
    return batch_id


# ---------------------------------------------------------------------------
# Database creation / opening
# ---------------------------------------------------------------------------

def _connect_rw(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000.0)
    conn.execute("PRAGMA busy_timeout = %d" % BUSY_TIMEOUT_MS)
    conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
    return conn


def _create_schema_objects(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"CREATE TABLE {META_TABLE} (name TEXT PRIMARY KEY, value TEXT NOT NULL) "
        "WITHOUT ROWID"
    )
    # PK leads with key: business queries are BLOB range seeks on the primary
    # key index; (rid, cgen) tail columns let version lookups share it, and
    # the separate partial unique index guards the one-live-entry invariant.
    conn.execute(
        f"CREATE TABLE {KV_TABLE} ("
        "key BLOB NOT NULL, rid BLOB NOT NULL, "
        "cgen INTEGER NOT NULL, dgen INTEGER, "
        "PRIMARY KEY(key, rid, cgen)) WITHOUT ROWID"
    )
    conn.execute(
        f"CREATE UNIQUE INDEX bytekey_kv_live_rid ON {KV_TABLE}(rid) "
        "WHERE dgen IS NULL"
    )
    conn.execute(
        f"CREATE TABLE {RECEIPT_TABLE} ("
        "batch_id BLOB PRIMARY KEY, head INTEGER NOT NULL, expected_head INTEGER "
        "NOT NULL, digest TEXT NOT NULL, op_count INTEGER NOT NULL, "
        "applied_count INTEGER NOT NULL, noop_count INTEGER NOT NULL) WITHOUT ROWID"
    )
    conn.execute(
        f"CREATE TABLE {SNAPSHOT_TABLE} ("
        "sid TEXT PRIMARY KEY, gen INTEGER NOT NULL, status TEXT NOT NULL, "
        "created_seq INTEGER NOT NULL) WITHOUT ROWID"
    )
    # Staging is a regular table but only ever holds rows inside one writer
    # transaction; commit or rollback always empties it.
    conn.execute(
        f"CREATE TABLE {STAGING_TABLE} ("
        "batch_id BLOB NOT NULL, seq INTEGER NOT NULL, line_no INTEGER NOT NULL, "
        "rid BLOB NOT NULL, kind TEXT NOT NULL, encoded_key BLOB, "
        "canon BLOB NOT NULL, PRIMARY KEY(batch_id, rid)) WITHOUT ROWID"
    )


def create_collection(path: str, schema: Schema) -> str:
    """Create an empty updatable collection.  On success ``head = 0``."""
    if os.path.exists(path):
        raise CollectionError(
            f"refusing to overwrite existing collection {path!r}; delete it first"
        )
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(
        directory, f".bytekey-coll.tmp.{os.getpid()}.{secrets.token_hex(8)}.db"
    )
    try:
        conn = _connect_rw(tmp_path)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")
            _create_schema_objects(conn)
            collection_id = secrets.token_hex(16)
            secret = secrets.token_hex(32)
            fingerprint = _fingerprint(schema)
            meta = {
                "format": COLLECTION_FORMAT,
                "collection_id": collection_id,
                "secret": secret,
                "schema_json": schema.canonical_json().decode("utf-8"),
                "fingerprint": fingerprint,
                "row_id_name": schema.row_id_name,
                "head": "0",
                "retain_from": "0",
                "snapshot_seq": "0",
            }
            conn.execute("BEGIN")
            conn.executemany(
                f"INSERT INTO {META_TABLE}(name, value) VALUES (?, ?)",
                list(meta.items()),
            )
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
        os.replace(tmp_path, path)
    except BaseException:
        for suffix in ("", "-wal", "-shm"):
            stray = tmp_path + suffix
            if os.path.exists(stray):
                try:
                    os.remove(stray)
                except OSError:
                    pass
        raise
    return collection_id


def import_from_index(
    index_path: str, collection_path: str, *, chunk_size: int = 1000
) -> tuple[str, int]:
    """Import a finished static index as generation 0 of a new collection.

    The static index file is opened read-only and never modified; its rows
    are streamed through a bounded chunk buffer.  The new collection starts
    at ``head = 0`` with every imported entry live (``cgen = 0``).
    """
    from .db import open_index  # local import: static layer stays independent

    if os.path.exists(collection_path):
        raise CollectionError(
            f"refusing to overwrite existing collection {collection_path!r}"
        )
    static = open_index(index_path)
    try:
        create_collection(collection_path, static.schema)
        conn = _connect_rw(collection_path)
        total = 0
        cid = ""
        try:
            conn.execute("PRAGMA synchronous = FULL")
            conn.execute("BEGIN IMMEDIATE")
            buf: list[tuple[bytes, bytes]] = []

            def flush() -> None:
                nonlocal total
                if not buf:
                    return
                conn.executemany(
                    f"INSERT INTO {KV_TABLE}(key, rid, cgen, dgen) "
                    "VALUES (?, ?, 0, NULL)",
                    [(k, r) for k, r in buf],
                )
                total += len(buf)
                buf.clear()

            # Static index table is a single BLOB primary key; stream it in
            # stored key order via a server-side cursor.
            from .db import KEY_TABLE

            for (key,) in static.conn.execute(
                f"SELECT key FROM {KEY_TABLE} ORDER BY key ASC"
            ):
                _values, rid = key_decode(static.schema, key)
                buf.append((key, rid))
                if len(buf) >= chunk_size:
                    flush()
            flush()
            cid = conn.execute(
                f"SELECT value FROM {META_TABLE} WHERE name = 'collection_id'"
            ).fetchone()[0]
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except BaseException:
            conn.rollback()
            conn.close()
            raise
        else:
            conn.close()
    finally:
        static.close()
    return cid, total


def _fingerprint(schema: Schema) -> str:
    h = hashlib.sha256()
    h.update(b"bytekey-index/v1\n")
    h.update(schema.canonical_json())
    return h.hexdigest()


def open_collection(path: str) -> "Collection":
    if not os.path.exists(path):
        raise CollectionError(f"collection file does not exist: {path}")
    return Collection(path)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class Collection:
    """An updatable, snapshot-readable byte-key collection."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.writer = _connect_rw(self.path)
        # A separate connection for reads; WAL lets scans proceed while the
        # writer holds an immediate transaction.
        self.reader = sqlite3.connect(
            "file:"
            + self.path.replace("?", "%3f").replace("#", "%23")
            + "?mode=ro",
            uri=True,
        )
        self.reader.execute("PRAGMA busy_timeout = %d" % BUSY_TIMEOUT_MS)
        try:
            meta_rows = self.reader.execute(
                f"SELECT name, value FROM {META_TABLE}"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            self._close_connections()
            raise CollectionError(f"not a bytekey database: {path}") from exc
        meta = dict(meta_rows)
        if meta.get("format") != COLLECTION_FORMAT:
            self._close_connections()
            raise CollectionError(
                f"{path!r} is not an updatable collection "
                f"(static indexes stay read-only via open_index)"
            )
        self.meta = meta
        self.collection_id = meta["collection_id"]
        self.secret = meta["secret"]
        self.fingerprint = meta["fingerprint"]
        self.schema = Schema.from_json(meta["schema_json"])
        self.row_id_name = meta["row_id_name"]
        self._in_txn = False

    def _close_connections(self) -> None:
        try:
            self.writer.close()
        except sqlite3.Error:
            pass
        try:
            self.reader.close()
        except sqlite3.Error:
            pass

    def close(self) -> None:
        self._close_connections()

    def close_and_get_id(self) -> str:
        cid = self.collection_id
        self.close()
        return cid

    def __enter__(self) -> "Collection":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- low level helpers -------------------------------------------------

    def _meta_int(self, name: str) -> int:
        return int(
            self.writer.execute(
                f"SELECT value FROM {META_TABLE} WHERE name = ?", (name,)
            ).fetchone()[0]
        )

    def _set_meta(self, name: str, value: int) -> None:
        self.writer.execute(
            f"UPDATE {META_TABLE} SET value = ? WHERE name = ?",
            (str(value), name),
        )

    def _begin_write(self) -> None:
        try:
            self.writer.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise ConflictError(f"collection is locked by another writer: {exc}") from exc
        self._in_txn = True

    def _rollback(self) -> None:
        if self._in_txn:
            self.writer.rollback()
            self._in_txn = False

    def _commit(self) -> None:
        self.writer.commit()
        self._in_txn = False

    # -- validation --------------------------------------------------------

    def _validate_operation(self, item: Any, line_no: int) -> tuple[str, bytes, bytes | None, tuple | None]:
        if not isinstance(item, Mapping):
            raise BatchError(f"line {line_no}: operation must be a JSON object")
        op = item.get("op")
        if op not in ("put", "delete"):
            raise BatchError(
                f"line {line_no}: op must be 'put' or 'delete', got {op!r}"
            )
        rid_name = self.row_id_name
        if rid_name not in item:
            raise BatchError(f"line {line_no}: operation is missing {rid_name!r}")
        rid = item[rid_name]
        if not isinstance(rid, (bytes, bytearray)):
            raise BatchError(
                f"line {line_no}: {rid_name} must be bytes, got {type(rid).__name__}"
            )
        rid = bytes(rid)
        if not rid:
            raise BatchError(f"line {line_no}: {rid_name} must be non-empty")
        known = {"op", rid_name} | {c.name for c in self.schema.columns}
        unknown = set(item) - known
        if unknown:
            raise BatchError(
                f"line {line_no}: unknown field(s) {sorted(unknown)}"
            )
        if op == "delete":
            extra = set(item) - {"op", rid_name}
            if extra:
                raise BatchError(
                    f"line {line_no}: delete carries only {rid_name}, "
                    f"got extra field(s) {sorted(extra)}"
                )
            return "d", rid, None, None
        values = []
        for col in self.schema.columns:
            if col.name not in item:
                raise BatchError(
                    f"line {line_no}: put is missing column {col.name!r}"
                )
            values.append(item[col.name])
        try:
            key = key_encode(self.schema, values, rid)
        except EncodingError as exc:
            raise BatchError(f"line {line_no}: {exc}") from None
        return "p", rid, key, tuple(values)

    # -- apply -------------------------------------------------------------

    def apply(
        self,
        batch_id: Any,
        expected_head: Any,
        operations: Iterable[Any],
        *,
        crash_hook: Callable[[str, sqlite3.Connection], None] | None = None,
    ) -> ApplyReceipt:
        """Validate and publish one batch atomically.

        ``operations`` is a *streaming* iterable of operation mappings; it is
        never materialized into a Python list or dict.  On any validation
        failure nothing is published and no receipt is kept.  A successfully
        applied batch can be re-sent with the same ``batch_id``: identical
        content and ``expected_head`` return the original receipt without
        changing anything, while any difference raises :class:`ConflictError`.
        """
        batch_id = _check_batch_id(batch_id)
        if isinstance(expected_head, bool) or not isinstance(expected_head, int):
            raise BatchError("expected_head must be an integer generation")
        if expected_head < 0:
            raise BatchError("expected_head must be >= 0")
        if not isinstance(operations, Iterable):
            raise BatchError("operations must be an iterable of mappings")

        self._begin_write()
        try:
            # 1) Idempotency lookup happens before any staging work.
            prior = self.writer.execute(
                f"SELECT head, expected_head, digest, op_count, applied_count, "
                f"noop_count FROM {RECEIPT_TABLE} WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()

            head = self._meta_int("head")
            if prior is None and head != expected_head:
                raise ConflictError(
                    f"expected_head {expected_head} does not match current "
                    f"head {head}"
                )

            # 2) Stream-validate into the on-disk staging table.
            op_count = 0
            try:
                for line_no, item in enumerate(operations, start=1):
                    kind, rid, key, values = self._validate_operation(item, line_no)
                    canon = _canonical_op_json(
                        kind, rid, values if kind == "p" else None
                    )
                    try:
                        self.writer.execute(
                            f"INSERT INTO {STAGING_TABLE}"
                            "(batch_id, seq, line_no, rid, kind, encoded_key, canon) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (batch_id, op_count, line_no, rid, kind, key, canon),
                        )
                    except sqlite3.IntegrityError:
                        # UNIQUE(batch_id, rid): repeated identity inside batch.
                        dup = self.writer.execute(
                            f"SELECT line_no FROM {STAGING_TABLE} "
                            "WHERE batch_id = ? AND rid = ? ORDER BY line_no",
                            (batch_id, rid),
                        ).fetchall()
                        lines = [str(r[0]) for r in dup] + [str(line_no)]
                        raise BatchError(
                            f"line {line_no}: row_id {rid.hex()} appears more "
                            f"than once in the batch (lines {', '.join(lines)}); "
                            "last-write-wins is not accepted"
                        ) from None
                    op_count += 1
            except BatchError:
                raise
            except Exception as exc:  # corrupt stream tail, generator failure...
                raise BatchError(f"batch source failed while reading: {exc}") from exc

            # 3) Content digest over the canonical operation *set*: ordering
            #    is normalized here, so reordered lines / JSON whitespace /
            #    hex-case differences collapse to the same digest.  Rows are
            #    pulled one at a time from the staging table, never listed.
            digest = hashlib.sha256()
            digest.update(
                ("%s\n%s\n%d\n" % (DIGEST_VERSION, self.collection_id, expected_head))
                .encode("ascii")
            )
            for (canon,) in self.writer.execute(
                f"SELECT canon FROM {STAGING_TABLE} WHERE batch_id = ? "
                "ORDER BY rid, seq",
                (batch_id,),
            ):
                digest.update(canon)
                digest.update(b"\n")
            digest_hex = digest.hexdigest()

            # 4) Idempotent replay handling.
            if prior is not None:
                prior_head, prior_eh, prior_digest, pc, pa, pn = prior
                if prior_eh != expected_head or prior_digest != digest_hex:
                    raise ConflictError(
                        f"batch_id {batch_id.hex()} was already applied with "
                        "different expected_head or content"
                    )
                # Same request: return the stored receipt, change nothing.
                self._rollback()
                return ApplyReceipt(
                    batch_id=batch_id,
                    head=prior_head,
                    expected_head=prior_eh,
                    digest_hex=prior_digest,
                    operations=pc,
                    applied=pa,
                    noop=pn,
                    replayed=True,
                )

            # 5) Publish the mutation.  Only named row_ids are touched.
            new_head = head + 1
            applied = 0
            noop = 0
            # Stream staged operations one row at a time: the batch source is
            # never materialized into a Python list/dict, so writer memory is
            # bounded by the staging cursor plus fixed bookkeeping.  The
            # staging table itself is only deleted after this cursor closes.
            staged_cursor = self.writer.execute(
                f"SELECT rid, kind, encoded_key FROM {STAGING_TABLE} "
                "WHERE batch_id = ? ORDER BY rid",
                (batch_id,),
            )
            for rid, kind, encoded_key in staged_cursor:
                live = self.writer.execute(
                    f"SELECT key FROM {KV_TABLE} WHERE rid = ? AND dgen IS NULL",
                    (rid,),
                ).fetchone()
                if kind == "d":
                    if live is None:
                        noop += 1  # deleting a missing identity is a success no-op
                        continue
                    self.writer.execute(
                        f"UPDATE {KV_TABLE} SET dgen = ? WHERE rid = ? AND dgen IS NULL",
                        (new_head, rid),
                    )
                    applied += 1
                else:
                    if live is None:
                        self.writer.execute(
                            f"INSERT INTO {KV_TABLE}"
                            "(key, rid, cgen, dgen) VALUES (?, ?, ?, NULL)",
                            (encoded_key, rid, new_head),
                        )
                        applied += 1
                    elif live[0] == encoded_key:
                        noop += 1  # identical full replacement
                    else:
                        self.writer.execute(
                            f"UPDATE {KV_TABLE} SET dgen = ? WHERE rid = ? "
                            "AND dgen IS NULL",
                            (new_head, rid),
                        )
                        self.writer.execute(
                            f"INSERT INTO {KV_TABLE}"
                            "(key, rid, cgen, dgen) VALUES (?, ?, ?, NULL)",
                            (encoded_key, rid, new_head),
                        )
                        applied += 1

            self.writer.execute(
                f"INSERT INTO {RECEIPT_TABLE}"
                "(batch_id, head, expected_head, digest, op_count, "
                "applied_count, noop_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (batch_id, new_head, expected_head, digest_hex,
                 op_count, applied, noop),
            )
            self._set_meta("head", new_head)
            self.writer.execute(
                f"DELETE FROM {STAGING_TABLE} WHERE batch_id = ?", (batch_id,)
            )
            if crash_hook is not None:
                crash_hook("pre_commit", self.writer)
            self._commit()
            if crash_hook is not None:
                crash_hook("post_commit", self.writer)
            return ApplyReceipt(
                batch_id=batch_id,
                head=new_head,
                expected_head=expected_head,
                digest_hex=digest_hex,
                operations=op_count,
                applied=applied,
                noop=noop,
                replayed=False,
            )
        except ConflictError:
            self._rollback()
            raise
        except BatchError:
            self._rollback()
            raise
        except sqlite3.OperationalError as exc:
            self._rollback()
            raise ConflictError(f"write failed: {exc}") from exc
        except BaseException:
            self._rollback()
            raise

    # -- snapshots ---------------------------------------------------------

    def _resolve_snapshot(self, token: str) -> SnapshotTicket:
        ticket = decode_snapshot_token(token, self.secret)
        if ticket.collection_id != self.collection_id:
            raise TokenError("snapshot token belongs to a different collection")
        if ticket.db_path != self.path:
            raise TokenError("snapshot token was created against a different file path")
        row = self.reader.execute(
            f"SELECT gen, status FROM {SNAPSHOT_TABLE} WHERE sid = ?",
            (ticket.sid,),
        ).fetchone()
        if row is None:
            raise TokenError("snapshot token is unknown (never issued or forged)")
        gen, status = row
        if status != "active":
            raise TokenError(f"snapshot token has been {status} and cannot be used")
        if gen != ticket.gen:
            raise TokenError("snapshot token generation does not match server state")
        return ticket

    def begin_snapshot(self, gen: int | None = None) -> str:
        """Pin a generation and return a durable, revocable snapshot token."""
        if gen is not None and (isinstance(gen, bool) or not isinstance(gen, int)):
            raise BatchError("snapshot generation must be an integer or None")
        self._begin_write()
        try:
            head = self._meta_int("head")
            retain = self._meta_int("retain_from")
            target = head if gen is None else gen
            if target < 0 or target > head:
                raise HistoryError(
                    f"snapshot generation {target} does not exist "
                    f"(current head {head})"
                )
            if target < retain:
                raise HistoryError(
                    f"generation {target} is below the retention boundary "
                    f"{retain}; it can only be read by tokens already active "
                    "before collection"
                )
            seq = self._meta_int("snapshot_seq") + 1
            sid = secrets.token_hex(16)
            self.writer.execute(
                f"INSERT INTO {SNAPSHOT_TABLE}(sid, gen, status, created_seq) "
                "VALUES (?, ?, 'active', ?)",
                (sid, target, seq),
            )
            self._set_meta("snapshot_seq", seq)
            self._commit()
        except BaseException:
            self._rollback()
            raise
        return encode_snapshot_token(
            secret=self.secret,
            collection_id=self.collection_id,
            gen=target,
            sid=sid,
            db_path=self.path,
        )

    def release_snapshot(self, token: str) -> dict:
        """Revoke a snapshot token and every cursor derived from it."""
        ticket = decode_snapshot_token(token, self.secret)
        if ticket.collection_id != self.collection_id or ticket.db_path != self.path:
            raise TokenError("snapshot token belongs to a different collection or file")
        self._begin_write()
        try:
            row = self.writer.execute(
                f"SELECT gen, status FROM {SNAPSHOT_TABLE} WHERE sid = ?",
                (ticket.sid,),
            ).fetchone()
            if row is None:
                raise TokenError("snapshot token is unknown (never issued or forged)")
            gen, status = row
            if status != "active":
                raise TokenError(f"snapshot token was already {status}")
            if gen != ticket.gen:
                raise TokenError("snapshot token generation does not match server state")
            self.writer.execute(
                f"UPDATE {SNAPSHOT_TABLE} SET status = 'released' WHERE sid = ?",
                (ticket.sid,),
            )
            self._commit()
        except BaseException:
            self._rollback()
            raise
        return {"released": ticket.sid, "gen": gen}

    # -- snapshot scans ----------------------------------------------------

    def _scan_sql(
        self, interval: Interval, last_key: bytes | None
    ) -> tuple[str, list[bytes]]:
        clauses: list[str] = []
        params: list[bytes] = []
        if interval.lower is not None:
            clauses.append("key >= ?")
            params.append(interval.lower)
        if interval.upper is not None:
            clauses.append("key < ?")
            params.append(interval.upper)
        if last_key is not None:
            clauses.append("key > ?")
            params.append(last_key)
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = (
            f"SELECT key, rid, cgen, dgen FROM {KV_TABLE} WHERE {where} "
            "ORDER BY key ASC, rid ASC, cgen ASC"
        )
        return sql, params

    def explain_snapshot_scan(
        self, query: Mapping[str, Any], page_size: int
    ) -> dict:
        plan = plan_query(self.schema, query)
        if not plan.intervals:
            return {"plan": plan.explain(), "explain_query_plan": []}
        sql, params = self._scan_sql(plan.intervals[0], None)
        eqp = self.reader.execute(
            "EXPLAIN QUERY PLAN " + sql, params
        ).fetchall()
        return {
            "plan": plan.explain(),
            "explain_query_plan": [
                {"id": r[0], "parent": r[1], "notused": r[2], "detail": r[3]}
                for r in eqp
            ],
            "page_size": page_size,
        }

    def scan_snapshot_page(
        self,
        snapshot_token: str,
        query: Mapping[str, Any],
        page_size: int,
        *,
        cursor_token: str | None = None,
    ) -> tuple[SnapshotPage, SnapshotScanStats, QueryPlan]:
        """Read one bounded page from a pinned generation.

        The snapshot token and cursor are verified against the database on
        every call, so released/forged credentials are refused and no read
        ever silently follows a newer generation.
        """
        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
            raise TokenError("page_size must be a positive integer")
        ticket = self._resolve_snapshot(snapshot_token)
        plan = plan_query(self.schema, query)

        cursor: CollectionCursor | None = None
        last_key: bytes | None = None
        if cursor_token is not None:
            cursor = decode_collection_cursor(cursor_token, self.secret)
            if cursor.collection_id != self.collection_id:
                raise TokenError("cursor belongs to a different collection")
            if cursor.db_path != self.path:
                raise TokenError("cursor was created against a different file path")
            if cursor.sid != ticket.sid:
                raise TokenError(
                    "cursor belongs to a different snapshot token "
                    "(same generation is not enough)"
                )
            if cursor.gen != ticket.gen:
                raise TokenError("cursor generation does not match snapshot")
            if cursor.query_spec != plan.query_spec:
                raise TokenError("cursor query specification does not match current query")
            if cursor.page_size != page_size:
                raise TokenError(
                    f"cursor page_size is {cursor.page_size}, but {page_size} requested"
                )
            last_key = cursor.last_key

        used_index = True
        eqp: list[dict] = []
        if plan.intervals:
            sql0, params0 = self._scan_sql(plan.intervals[0], last_key)
            eqp_rows = self.reader.execute(
                "EXPLAIN QUERY PLAN " + sql0, params0
            ).fetchall()
            eqp = [
                {"id": r[0], "parent": r[1], "notused": r[2], "detail": r[3]}
                for r in eqp_rows
            ]
            used_index = any(
                "USING PRIMARY KEY" in r["detail"] or "USING INDEX" in r["detail"]
                for r in eqp
            )

        g = ticket.gen
        candidates = 0
        invisible = 0
        delivered: list[tuple[bytes, tuple, bytes]] = []
        if plan.intervals:
            for interval in plan.intervals:
                if len(delivered) > page_size:
                    break
                sql, params = self._scan_sql(interval, last_key)
                # Server-side cursor: memory is bounded by the page plus one
                # key-group, never by the table.  Every pulled row is counted
                # as an examined storage candidate.  Entries are grouped by
                # full encoded key because the same key can belong to several
                # row_ids over time (a moved record can land on another
                # record's key): each key contributes exactly the entries
                # visible at the pinned generation.  An equal-key group is
                # indivisible: the page may overfill by at most one group so
                # that equal-tuple siblings are never split.
                current_key: bytes | None = None
                current_group: list[tuple[bytes, tuple, bytes]] = []
                group_invisible = 0
                stop = False

                def close_group() -> None:
                    nonlocal current_key, current_group, group_invisible, invisible
                    nonlocal stop
                    if current_key is None:
                        return
                    if current_group:
                        delivered.extend(current_group)
                        if len(delivered) > page_size:
                            stop = True
                    else:
                        invisible += group_invisible
                    current_key = None
                    current_group = []
                    group_invisible = 0

                for key, rid, cgen, dgen in self.reader.execute(sql, params):
                    candidates += 1
                    if key != current_key:
                        close_group()
                        if stop:
                            break
                        current_key = key
                    visible = not (cgen > g or (dgen is not None and dgen <= g))
                    if not visible:
                        group_invisible += 1
                        continue
                    values, decoded_rid = key_decode(self.schema, key)
                    if decoded_rid != rid:
                        raise TokenError("stored key/row_id pair is inconsistent")
                    current_group.append((key, values, rid))
                else:
                    close_group()

        # Groups are indivisible, so the collected list can overshoot the
        # page size by part of one equal-key group.  Emit exactly the first
        # page_size rows and remember the last *delivered* key; the next page
        # resumes with key > last_key, which re-reads the un-delivered tail
        # of the same key group (counted again as examined candidates).
        has_more = len(delivered) > page_size
        page_rows = delivered[:page_size]
        page_last_key = page_rows[-1][0] if page_rows else None
        next_cursor = None
        if has_more and page_last_key is not None:
            next_cursor = encode_collection_cursor(
                secret=self.secret,
                collection_id=self.collection_id,
                sid=ticket.sid,
                gen=ticket.gen,
                query_spec=plan.query_spec,
                page_size=page_size,
                last_key=page_last_key,
                db_path=self.path,
            )
        stats = SnapshotScanStats(
            candidates_examined=candidates,
            invisible_versions_skipped=invisible,
            rows_delivered=len(page_rows),
            has_more=has_more,
            used_index=used_index,
            explain_query_plan=eqp,
            intervals=[
                {
                    "lower_hex": i.lower.hex() if i.lower is not None else None,
                    "upper_hex": i.upper.hex() if i.upper is not None else None,
                }
                for i in plan.intervals
            ],
        )
        page = SnapshotPage(
            rows=[(values, rid) for _key, values, rid in page_rows],
            next_cursor=next_cursor,
            last_key=page_last_key,
            gen=ticket.gen,
            sid=ticket.sid,
        )
        return page, stats, plan

    def iter_snapshot(
        self,
        snapshot_token: str,
        query: Mapping[str, Any],
        page_size: int,
    ) -> Iterable[tuple[tuple, bytes]]:
        """Yield every match of a snapshot page by page (bounded memory)."""
        cursor_token: str | None = None
        while True:
            page, _stats, _plan = self.scan_snapshot_page(
                snapshot_token, query, page_size, cursor_token=cursor_token
            )
            for item in page.rows:
                yield item
            if page.next_cursor is None:
                return
            cursor_token = page.next_cursor

    # -- history collection ------------------------------------------------

    def collect_history(self, retain_from: int) -> HistoryStats:
        """Move the retention boundary to ``retain_from`` and reclaim history.

        The boundary is monotonic and may not pass ``head``.  Every entry an
        active snapshot can still see is kept, regardless of age; entries with
        no possible reader at or above the new boundary are physically
        deleted.  Receipts and head are never removed.
        """
        if isinstance(retain_from, bool) or not isinstance(retain_from, int):
            raise HistoryError("retain_from must be an integer generation")
        self._begin_write()
        try:
            head = self._meta_int("head")
            previous = self._meta_int("retain_from")
            if retain_from < previous:
                raise HistoryError(
                    f"retention boundary cannot move backwards: {retain_from} "
                    f"< {previous}"
                )
            if retain_from > head:
                raise HistoryError(
                    f"retain_from {retain_from} is above current head {head}"
                )
            # An entry is reclaimable exactly when it is already dead at the
            # new boundary AND no active snapshot's generation lies inside its
            # visibility window [cgen, dgen).
            cur = self.writer.execute(
                f"DELETE FROM {KV_TABLE} "
                "WHERE dgen IS NOT NULL AND dgen <= ? "
                "AND NOT EXISTS ("
                f"  SELECT 1 FROM {SNAPSHOT_TABLE} s "
                "  WHERE s.status = 'active' AND s.gen >= bytekey_kv.cgen "
                "  AND s.gen < bytekey_kv.dgen)"
                ,
                (retain_from,),
            )
            removed = cur.rowcount
            self._set_meta("retain_from", retain_from)
            active = self.writer.execute(
                f"SELECT COUNT(*) FROM {SNAPSHOT_TABLE} WHERE status = 'active'"
            ).fetchone()[0]
            self._commit()
        except BaseException:
            self._rollback()
            raise
        return HistoryStats(
            retain_from=retain_from,
            previous_retain_from=previous,
            entries_removed=removed,
            active_snapshots=active,
            head=head,
        )

    # -- status ------------------------------------------------------------

    def status(self) -> CollectionStatus:
        head = self._meta_int_reader("head")
        retain = self._meta_int_reader("retain_from")
        live = self.reader.execute(
            f"SELECT COUNT(*) FROM {KV_TABLE} WHERE dgen IS NULL"
        ).fetchone()[0]
        total = self.reader.execute(f"SELECT COUNT(*) FROM {KV_TABLE}").fetchone()[0]
        receipts = self.reader.execute(
            f"SELECT COUNT(*) FROM {RECEIPT_TABLE}"
        ).fetchone()[0]
        snaps = [
            {"sid": r[0], "gen": r[1], "status": r[2]}
            for r in self.reader.execute(
                f"SELECT sid, gen, status FROM {SNAPSHOT_TABLE} "
                "ORDER BY created_seq"
            ).fetchall()
        ]
        active = [s for s in snaps if s["status"] == "active"]
        return CollectionStatus(
            path=self.path,
            collection_id=self.collection_id,
            head=head,
            retain_from=retain,
            schema=self.schema,
            row_id_name=self.row_id_name,
            live_entries=live,
            total_version_entries=total,
            dead_entries=total - live,
            receipt_count=receipts,
            active_snapshots=active,
        )

    def _meta_int_reader(self, name: str) -> int:
        return int(
            self.reader.execute(
                f"SELECT value FROM {META_TABLE} WHERE name = ?", (name,)
            ).fetchone()[0]
        )

    def lookup_receipt(self, batch_id: bytes) -> dict | None:
        row = self.reader.execute(
            f"SELECT batch_id, head, expected_head, digest, op_count, "
            f"applied_count, noop_count FROM {RECEIPT_TABLE} WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "batch_id": {"__hex__": row[0].hex()},
            "head": row[1],
            "expected_head": row[2],
            "digest": row[3],
            "operations": row[4],
            "applied": row[5],
            "noop": row[6],
        }
