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
later collection remove them.  Secondary-index versions and journal rows obey
the same rule, and an in-flight build protects its own inputs.

Named secondary access indexes
------------------------------

``register_index`` / ``advance_index`` / ``scan_index_page`` add a second,
named access order over a subset of the existing business columns (see
``bytekey.secondary`` for the declaration and storage model).  An index is
built in budgeted, resumable steps under short transactions, becomes queryable
only at an explicit ready switch that is guaranteed to cover every generation
from its start ``B`` to the then-current head, and is maintained by every
later apply in the same commit as the main version, head and receipt.
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
from .secondary import (
    BUILDING,
    READY,
    SEC_ENTRY_TABLE,
    SEC_INDEX_TABLE,
    SEC_QUEUE_TABLE,
    IndexConflictError,
    IndexCursorError,
    IndexNotFoundError,
    IndexNotReadyError,
    CoverageError,
    SecondaryDeclaration,
    encode_index_key,
    plan_index_query,
    resolve_declaration,
    validate_query_columns,
)
from .snapshot import (
    CollectionCursor,
    SnapshotTicket,
    TokenError,
    decode_collection_cursor,
    decode_secondary_cursor,
    decode_snapshot_token,
    encode_collection_cursor,
    encode_secondary_cursor,
    encode_snapshot_token,
)

META_TABLE = "bytekey_meta"
KV_TABLE = "bytekey_kv"
RECEIPT_TABLE = "bytekey_receipts"
SNAPSHOT_TABLE = "bytekey_snapshots"
STAGING_TABLE = "bytekey_staging"
COLLECTION_FORMAT = "bytekey-collection/v2"
LEGACY_COLLECTION_FORMAT = "bytekey-collection/v1"
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
    secondary_entries_removed: int = 0
    queue_entries_removed: int = 0

    def to_dict(self) -> dict:
        return {
            "retain_from": self.retain_from,
            "previous_retain_from": self.previous_retain_from,
            "entries_removed": self.entries_removed,
            "main_entries_removed": self.entries_removed,
            "secondary_entries_removed": self.secondary_entries_removed,
            "queue_entries_removed": self.queue_entries_removed,
            "active_snapshots": self.active_snapshots,
            "head": self.head,
        }


@dataclass
class IndexBuildStatus:
    """Progress of one named secondary index instance."""

    name: str
    instance_id: str
    build_id: str
    status: str
    start_gen: int
    declaration: dict
    declaration_hash: str
    head: int
    processed_seq: int
    next_seq: int
    baseline_done: bool
    baseline_cursor_key: bytes | None
    baseline_candidates_read: int      # this step
    catchup_changes_processed: int     # this step
    cumulative_baseline_candidates: int
    cumulative_catchup_changes: int
    covered_from: int
    ready_gen: int | None
    secondary_entries: int
    queue_entries: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instance_id": self.instance_id,
            "build_id": self.build_id,
            "status": self.status,
            "start_gen": self.start_gen,
            "covered_from": self.covered_from,
            "ready_gen": self.ready_gen,
            "head": self.head,
            "declaration": self.declaration,
            "declaration_hash": self.declaration_hash,
            "processed_seq": self.processed_seq,
            "next_seq": self.next_seq,
            "baseline_done": self.baseline_done,
            "baseline_cursor_key_hex": (
                self.baseline_cursor_key.hex()
                if self.baseline_cursor_key is not None
                else None
            ),
            "step": {
                "baseline_candidates_read": self.baseline_candidates_read,
                "catchup_changes_processed": self.catchup_changes_processed,
            },
            "cumulative": {
                "baseline_candidates_read": self.cumulative_baseline_candidates,
                "catchup_changes_processed": self.cumulative_catchup_changes,
            },
            "secondary_entries": self.secondary_entries,
            "queue_entries": self.queue_entries,
        }


@dataclass
class IndexScanStats:
    index_candidates_examined: int    # secondary entries pulled from storage
    invisible_versions_skipped: int
    full_row_lookups: int             # main-version locator seeks
    rows_delivered: int
    has_more: bool
    used_index: bool
    explain_query_plan: list[dict]
    intervals: list[dict]

    def to_dict(self) -> dict:
        return {
            "index_candidates_examined": self.index_candidates_examined,
            "invisible_versions_skipped": self.invisible_versions_skipped,
            "full_row_lookups": self.full_row_lookups,
            "rows_delivered": self.rows_delivered,
            "has_more": self.has_more,
            "used_index": self.used_index,
            "explain_query_plan": self.explain_query_plan,
            "intervals": self.intervals,
        }


@dataclass
class IndexPage:
    rows: list[tuple[tuple, bytes]]
    next_cursor: str | None
    last_index_key: bytes | None
    gen: int
    sid: str
    index_name: str
    instance_id: str


@dataclass
class IndexInfo:
    name: str
    instance_id: str
    build_id: str
    status: str
    start_gen: int
    declaration: dict
    declaration_hash: str
    ready_gen: int | None
    baseline_done: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instance_id": self.instance_id,
            "build_id": self.build_id,
            "status": self.status,
            "start_gen": self.start_gen,
            "covered_from": self.start_gen,
            "ready_gen": self.ready_gen,
            "declaration": self.declaration,
            "declaration_hash": self.declaration_hash,
            "baseline_done": self.baseline_done,
        }


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
    indexes: list[dict] = field(default_factory=list)

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
            "indexes": list(self.indexes),
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
    _ensure_secondary_schema(conn)


def _ensure_secondary_schema(conn: sqlite3.Connection) -> None:
    """Create the secondary-index tables if absent (idempotent, recoverable).

    Also used as the v1 -> v2 format upgrade: running it repeatedly is safe,
    so an interrupted migration simply runs again on the next open.
    """
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {SEC_INDEX_TABLE} ("
        "instance_id TEXT PRIMARY KEY, name TEXT NOT NULL, "
        "build_id TEXT NOT NULL, status TEXT NOT NULL, "
        "start_gen INTEGER NOT NULL, declaration_json TEXT NOT NULL, "
        "declaration_hash TEXT NOT NULL, "
        "next_seq INTEGER NOT NULL DEFAULT 1, "
        "processed_seq INTEGER NOT NULL DEFAULT 0, "
        "baseline_cursor_key BLOB, baseline_cursor_rid BLOB, "
        "baseline_cursor_cgen INTEGER, "
        "baseline_done INTEGER NOT NULL DEFAULT 0, "
        "ready_gen INTEGER, created_seq INTEGER NOT NULL, "
        "baseline_candidates INTEGER NOT NULL DEFAULT 0, "
        "catchup_changes INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID"
    )
    # One *active* index per name; a deleted/cancelled instance frees the name
    # for a different declaration while its old identity remains resolvable
    # (and therefore always rejected by old cursors).
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS bytekey_sec_active_name "
        f"ON {SEC_INDEX_TABLE}(name) WHERE status IN ('building','ready')"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS bytekey_sec_build_id "
        f"ON {SEC_INDEX_TABLE}(build_id)"
    )
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {SEC_ENTRY_TABLE} ("
        "instance_id TEXT NOT NULL, index_key BLOB NOT NULL, "
        "cgen INTEGER NOT NULL, dgen INTEGER, rid BLOB NOT NULL, "
        "main_key BLOB NOT NULL, main_cgen INTEGER NOT NULL, "
        "PRIMARY KEY(instance_id, index_key, cgen)) WITHOUT ROWID"
    )
    # Cleanup helper only; queries use the primary key range.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS bytekey_sec_entries_dead "
        f"ON {SEC_ENTRY_TABLE}(instance_id, dgen)"
    )
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {SEC_QUEUE_TABLE} ("
        "instance_id TEXT NOT NULL, seq INTEGER NOT NULL, "
        "kind TEXT NOT NULL, index_key BLOB NOT NULL, rid BLOB NOT NULL, "
        "cgen INTEGER NOT NULL, main_key BLOB NOT NULL, close_gen INTEGER, "
        "PRIMARY KEY(instance_id, seq)) WITHOUT ROWID"
    )


def _migrate_collection(conn: sqlite3.Connection, fmt: str) -> None:
    """Recoverably upgrade a legacy collection in place.

    Collection identity, head, receipts, snapshots and every version row are
    untouched; only new (initially empty) secondary-index objects are added
    and the format marker is advanced.
    """
    if fmt == COLLECTION_FORMAT:
        return
    if fmt != LEGACY_COLLECTION_FORMAT:
        raise CollectionError(f"unsupported collection format {fmt!r}")
    try:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_secondary_schema(conn)
        conn.execute(
            f"UPDATE {META_TABLE} SET value = ? WHERE name = 'format'",
            (COLLECTION_FORMAT,),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


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
        fmt = meta.get("format")
        if fmt == LEGACY_COLLECTION_FORMAT:
            # Recoverable in-place upgrade: identity, head, receipts and
            # snapshots are preserved; only empty secondary objects are added.
            try:
                _migrate_collection(self.writer, fmt)
            except BaseException:
                self._close_connections()
                raise
            meta = dict(
                self.writer.execute(f"SELECT name, value FROM {META_TABLE}").fetchall()
            )
            fmt = meta.get("format")
        if fmt != COLLECTION_FORMAT:
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
            # Building secondary indexes must cover everything published from
            # their start generation B on.  The journal is written in this
            # very transaction (same commit as the main version + receipt),
            # and it is derived from the *declaration* of every building
            # instance -- never from a connection-time directory, so a writer
            # connection opened before registration still maintains it.
            building = self.writer.execute(
                f"SELECT instance_id, declaration_json FROM {SEC_INDEX_TABLE} "
                "WHERE status = 'building' ORDER BY instance_id"
            ).fetchall()
            building_indexes = [
                (iid, resolve_declaration(self.schema, SecondaryDeclaration.from_json(dj)))
                for iid, dj in building
            ]
            queue_seq: dict[str, int] = {}
            if building_indexes:
                for iid, _decl in building_indexes:
                    queue_seq[iid] = self.writer.execute(
                        f"SELECT next_seq FROM {SEC_INDEX_TABLE} "
                        "WHERE instance_id = ?",
                        (iid,),
                    ).fetchone()[0]

            def journal_open(iid: str, decl, key: bytes, rid: bytes,
                             values: tuple, gen: int) -> None:
                idx_key = encode_index_key(self.schema, decl, values, rid)
                seq = queue_seq[iid]
                queue_seq[iid] = seq + 1
                self.writer.execute(
                    f"INSERT INTO {SEC_QUEUE_TABLE}"
                    "(instance_id, seq, kind, index_key, rid, cgen, main_key, "
                    "close_gen) VALUES (?, ?, 'open', ?, ?, ?, ?, NULL)",
                    (iid, seq, idx_key, rid, gen, key),
                )

            def journal_close(iid: str, decl, key: bytes, rid: bytes,
                              version_cgen: int, values: tuple,
                              closed_gen: int) -> None:
                idx_key = encode_index_key(self.schema, decl, values, rid)
                seq = queue_seq[iid]
                queue_seq[iid] = seq + 1
                self.writer.execute(
                    f"INSERT INTO {SEC_QUEUE_TABLE}"
                    "(instance_id, seq, kind, index_key, rid, cgen, main_key, "
                    "close_gen) VALUES (?, ?, 'close', ?, ?, ?, ?, ?)",
                    (iid, seq, idx_key, rid, version_cgen, key, closed_gen),
                )

            # Ready secondary indexes are maintained *synchronously in the
            # same transaction*: the switch to ready is an explicit commit and
            # every later apply must keep the index exact.
            ready = self.writer.execute(
                f"SELECT instance_id, declaration_json FROM {SEC_INDEX_TABLE} "
                "WHERE status = 'ready' ORDER BY instance_id"
            ).fetchall()
            ready_indexes = [
                (iid, resolve_declaration(self.schema, SecondaryDeclaration.from_json(dj)))
                for iid, dj in ready
            ]

            def ready_insert(iid: str, decl, key: bytes, rid: bytes,
                             values: tuple, gen: int) -> None:
                idx_key = encode_index_key(self.schema, decl, values, rid)
                self.writer.execute(
                    f"INSERT INTO {SEC_ENTRY_TABLE}"
                    "(instance_id, index_key, cgen, dgen, rid, main_key, "
                    "main_cgen) VALUES (?, ?, ?, NULL, ?, ?, ?)",
                    (iid, idx_key, gen, rid, key, gen),
                )

            def ready_close(iid: str, decl, key: bytes, rid: bytes,
                            version_cgen: int, values: tuple,
                            closed_gen: int) -> None:
                idx_key = encode_index_key(self.schema, decl, values, rid)
                self.writer.execute(
                    f"UPDATE {SEC_ENTRY_TABLE} SET dgen = ? "
                    "WHERE instance_id = ? AND index_key = ? AND cgen = ? "
                    "AND dgen IS NULL",
                    (closed_gen, iid, idx_key, version_cgen),
                )

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
                    f"SELECT key, cgen FROM {KV_TABLE} WHERE rid = ? AND dgen IS NULL",
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
                    if building_indexes or ready_indexes:
                        old_values, _ = key_decode(self.schema, live[0])
                        for iid, decl in building_indexes:
                            journal_close(iid, decl, live[0], rid, live[1],
                                          old_values, new_head)
                        for iid, decl in ready_indexes:
                            ready_close(iid, decl, live[0], rid, live[1],
                                       old_values, new_head)
                    applied += 1
                else:
                    if live is None:
                        self.writer.execute(
                            f"INSERT INTO {KV_TABLE}"
                            "(key, rid, cgen, dgen) VALUES (?, ?, ?, NULL)",
                            (encoded_key, rid, new_head),
                        )
                        if building_indexes or ready_indexes:
                            values, _ = key_decode(self.schema, encoded_key)
                            for iid, decl in building_indexes:
                                journal_open(iid, decl, encoded_key, rid,
                                             values, new_head)
                            for iid, decl in ready_indexes:
                                ready_insert(iid, decl, encoded_key, rid,
                                             values, new_head)
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
                        if building_indexes or ready_indexes:
                            old_values, _ = key_decode(self.schema, live[0])
                            new_values, _ = key_decode(self.schema, encoded_key)
                            for iid, decl in building_indexes:
                                journal_close(iid, decl, live[0], rid, live[1],
                                              old_values, new_head)
                                journal_open(iid, decl, encoded_key, rid,
                                             new_values, new_head)
                            for iid, decl in ready_indexes:
                                ready_close(iid, decl, live[0], rid, live[1],
                                           old_values, new_head)
                                ready_insert(iid, decl, encoded_key, rid,
                                             new_values, new_head)
                        applied += 1

            for iid, seq in queue_seq.items():
                self.writer.execute(
                    f"UPDATE {SEC_INDEX_TABLE} SET next_seq = ? "
                    "WHERE instance_id = ?",
                    (seq, iid),
                )

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

    # -- secondary index lifecycle ----------------------------------------

    def _index_row(self, name: str) -> tuple:
        row = self.reader.execute(
            f"SELECT instance_id, name, build_id, status, start_gen, "
            "declaration_json, declaration_hash, next_seq, processed_seq, "
            "baseline_cursor_key, baseline_cursor_rid, baseline_cursor_cgen, "
            "baseline_done, ready_gen, created_seq, baseline_candidates, "
            "catchup_changes "
            f"FROM {SEC_INDEX_TABLE} WHERE name = ? AND status IN "
            "('building','ready')",
            (name,),
        ).fetchone()
        if row is None:
            raise IndexNotFoundError(f"no live secondary index named {name!r}")
        return row

    def _index_row_by_instance(self, instance_id: str) -> tuple | None:
        return self.reader.execute(
            f"SELECT instance_id, name, build_id, status, start_gen, "
            "declaration_json, declaration_hash, next_seq, processed_seq, "
            "baseline_cursor_key, baseline_cursor_rid, baseline_cursor_cgen, "
            "baseline_done, ready_gen, created_seq, baseline_candidates, "
            "catchup_changes "
            f"FROM {SEC_INDEX_TABLE} WHERE instance_id = ?",
            (instance_id,),
        ).fetchone()

    @staticmethod
    def _index_row_dict(row: tuple) -> dict:
        (iid, name, build_id, status, start_gen, dj, dh, next_seq, processed,
         cur_key, cur_rid, cur_cgen, baseline_done, ready_gen, created_seq,
         base_count, catchup) = row
        return {
            "instance_id": iid,
            "name": name,
            "build_id": build_id,
            "status": status,
            "start_gen": start_gen,
            "declaration_json": dj,
            "declaration_hash": dh,
            "next_seq": next_seq,
            "processed_seq": processed,
            "baseline_cursor_key": cur_key,
            "baseline_cursor_rid": cur_rid,
            "baseline_cursor_cgen": cur_cgen,
            "baseline_done": bool(baseline_done),
            "ready_gen": ready_gen,
            "created_seq": created_seq,
            "baseline_candidates": base_count,
            "catchup_changes": catchup,
        }

    def register_index(
        self,
        declaration: SecondaryDeclaration | Mapping[str, Any],
        *,
        build_id: bytes | None = None,
    ) -> dict:
        """Register a named secondary index and pin its coverage start B.

        One short transaction mints an immutable instance identity and the
        resumable build handle, records ``start_gen = head`` and leaves the
        instance in ``building``.  No baseline row is read or copied here.
        Duplicate names among *active* indexes fail explicitly; reuse the
        returned ``build_id`` to resume the original work.
        """
        decl = resolve_declaration(self.schema, declaration)
        if build_id is not None:
            if not isinstance(build_id, (bytes, bytearray)) or not build_id:
                raise BatchError("build_id must be non-empty bytes")
            build_hex = bytes(build_id).hex()
        else:
            build_hex = secrets.token_hex(16)
        self._begin_write()
        try:
            head = self._meta_int("head")
            active = self.writer.execute(
                f"SELECT instance_id, build_id FROM {SEC_INDEX_TABLE} "
                "WHERE name = ? AND status IN ('building','ready')",
                (decl.name,),
            ).fetchone()
            if active is not None:
                raise IndexConflictError(
                    f"an active secondary index named {decl.name!r} already "
                    f"exists (instance {active[0]}, build "
                    f"{bytes.fromhex(active[1]).hex()}); resume it with its "
                    "build_id instead of registering again"
                )
            seq = self.writer.execute(
                f"SELECT COALESCE(MAX(created_seq), 0) + 1 FROM {SEC_INDEX_TABLE}"
            ).fetchone()[0]
            instance_id = secrets.token_hex(16)
            self.writer.execute(
                f"INSERT INTO {SEC_INDEX_TABLE}"
                "(instance_id, name, build_id, status, start_gen, "
                "declaration_json, declaration_hash, next_seq, processed_seq, "
                "baseline_cursor_key, baseline_cursor_rid, baseline_cursor_cgen, "
                "baseline_done, ready_gen, created_seq) "
                "VALUES (?, ?, ?, 'building', ?, ?, ?, 1, 0, NULL, NULL, NULL, "
                "0, NULL, ?)",
                (
                    instance_id,
                    decl.name,
                    build_hex,
                    head,
                    decl.canonical_json().decode("utf-8"),
                    decl.declaration_hash(),
                    seq,
                ),
            )
            self._commit()
        except BaseException:
            self._rollback()
            raise
        return {
            "name": decl.name,
            "instance_id": instance_id,
            "build_id": {"__hex__": build_hex},
            "build_id_hex": build_hex,
            "start_gen": head,
            "covered_from": head,
            "status": BUILDING,
            "declaration": decl.to_dict(),
            "declaration_hash": decl.declaration_hash(),
        }

    def index_status(self, name: str) -> IndexBuildStatus:
        row = self._index_row(name)
        info = self._index_row_dict(row)
        head = self._meta_int_reader("head")
        entries = self.reader.execute(
            f"SELECT COUNT(*) FROM {SEC_ENTRY_TABLE} WHERE instance_id = ?",
            (info["instance_id"],),
        ).fetchone()[0]
        queued = self.reader.execute(
            f"SELECT COUNT(*) FROM {SEC_QUEUE_TABLE} WHERE instance_id = ?",
            (info["instance_id"],),
        ).fetchone()[0]
        return IndexBuildStatus(
            name=info["name"],
            instance_id=info["instance_id"],
            build_id=info["build_id"],
            status=info["status"],
            start_gen=info["start_gen"],
            declaration=json.loads(info["declaration_json"]),
            declaration_hash=info["declaration_hash"],
            head=head,
            processed_seq=info["processed_seq"],
            next_seq=info["next_seq"],
            baseline_done=info["baseline_done"],
            baseline_cursor_key=info["baseline_cursor_key"],
            baseline_candidates_read=0,
            catchup_changes_processed=0,
            cumulative_baseline_candidates=info["baseline_candidates"],
            cumulative_catchup_changes=info["catchup_changes"],
            covered_from=info["start_gen"],
            ready_gen=info["ready_gen"],
            secondary_entries=entries,
            queue_entries=queued,
        )

    def list_indexes(self, *, include_deleted: bool = False) -> list[IndexInfo]:
        where = "" if include_deleted else (
            " WHERE status IN ('building','ready')"
        )
        rows = self.reader.execute(
            f"SELECT instance_id, name, build_id, status, start_gen, "
            "declaration_json, declaration_hash, next_seq, processed_seq, "
            "baseline_cursor_key, baseline_cursor_rid, baseline_cursor_cgen, "
            "baseline_done, ready_gen, created_seq, baseline_candidates, "
            "catchup_changes "
            f"FROM {SEC_INDEX_TABLE}{where} ORDER BY created_seq"
        ).fetchall()
        out = []
        for row in rows:
            info = self._index_row_dict(row)
            out.append(
                IndexInfo(
                    name=info["name"],
                    instance_id=info["instance_id"],
                    build_id=info["build_id"],
                    status=info["status"],
                    start_gen=info["start_gen"],
                    declaration=json.loads(info["declaration_json"]),
                    declaration_hash=info["declaration_hash"],
                    ready_gen=info["ready_gen"],
                    baseline_done=info["baseline_done"],
                )
            )
        return out

    def _load_instance(
        self, name: str | None = None, instance_id: str | None = None,
        *, require_building: bool = False, require_ready: bool = False,
    ) -> dict:
        row = (
            self._index_row_by_instance(instance_id)
            if instance_id is not None
            else self._index_row(name)
        )
        if row is None:
            raise IndexNotFoundError(
                f"secondary index instance {instance_id!r} does not exist"
            )
        info = self._index_row_dict(row)
        if require_building and info["status"] != BUILDING:
            raise IndexConflictError(
                f"index {info['name']!r} ({info['instance_id']}) is "
                f"{info['status']}, not building"
            )
        if require_ready and info["status"] != READY:
            raise IndexNotReadyError(
                f"index {info['name']!r} is {info['status']}, not ready"
            )
        return info

    def advance_index(
        self,
        name: str,
        budget: int,
        *,
        expected_build_id: bytes | str | None = None,
        crash_hook: Callable[[str, sqlite3.Connection], None] | None = None,
    ) -> IndexBuildStatus:
        """Perform at most ``budget`` units of index work, then commit.

        Baseline candidate reads and catch-up journal rows each count as one
        unit; a single call never processes more than ``budget`` combined.
        All progress is written to disk inside one short transaction, so the
        build can be resumed by another process and no write lock survives
        the call.
        """
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise BatchError("budget must be an integer")
        if budget < 0:
            raise BatchError("budget must be >= 0")
        if isinstance(expected_build_id, (bytes, bytearray)):
            expected_build_id = bytes(expected_build_id).hex()
        row = self._index_row(name)
        info = self._index_row_dict(row)
        if expected_build_id is not None and info["build_id"] != expected_build_id:
            raise IndexConflictError(
                f"build_id does not match the active build for {name!r} "
                f"(expected {info['build_id']}, got {expected_build_id})"
            )
        if info["status"] == READY:
            return self.index_status(name)
        if info["status"] != BUILDING:
            raise IndexConflictError(
                f"index {name!r} is {info['status']} and cannot advance"
            )

        decl = resolve_declaration(self.schema, SecondaryDeclaration.from_json(
            info["declaration_json"]
        ))
        start_gen = info["start_gen"]
        this_baseline = 0
        this_catchup = 0
        baseline_done = info["baseline_done"]
        cur_key = info["baseline_cursor_key"]
        cur_rid = info["baseline_cursor_rid"]
        cur_cgen = info["baseline_cursor_cgen"]
        processed_seq = info["processed_seq"]
        ready_gen: int | None = None

        self._begin_write()
        try:
            # Re-read under the write lock (another process may have advanced).
            fresh = self._index_row_dict(
                self.writer.execute(
                    f"SELECT instance_id, name, build_id, status, start_gen, "
                    "declaration_json, declaration_hash, next_seq, processed_seq, "
                    "baseline_cursor_key, baseline_cursor_rid, baseline_cursor_cgen, "
                    "baseline_done, ready_gen, created_seq, baseline_candidates, "
                    "catchup_changes "
                    f"FROM {SEC_INDEX_TABLE} WHERE instance_id = ?",
                    (info["instance_id"],),
                ).fetchone()
            )
            if fresh["status"] != BUILDING:
                if fresh["status"] == READY:
                    self._commit()
                    return self.index_status(name)
                raise IndexConflictError(
                    f"index {name!r} changed to {fresh['status']} concurrently"
                )
            baseline_done = fresh["baseline_done"]
            cur_key = fresh["baseline_cursor_key"]
            cur_rid = fresh["baseline_cursor_rid"]
            cur_cgen = fresh["baseline_cursor_cgen"]
            processed_seq = fresh["processed_seq"]

            remaining = budget
            # ---- Phase 1: baseline candidates from the main store ----
            # Candidates are exactly the versions that can be visible in some
            # generation >= B: created at or before B and not already dead at
            # B.  A later apply only ever closes such a version with
            # dgen > B, so the candidate set is stable under concurrent
            # publication and keyset pagination cannot skip a row that moved
            # from live to dead mid-build.  Versions created after B are
            # covered by the catch-up journal instead.
            while not baseline_done and remaining > 0:
                sql = (
                    f"SELECT key, rid, cgen, dgen FROM {KV_TABLE} "
                    "WHERE cgen <= ? AND (dgen IS NULL OR dgen > ?)"
                )
                params: list = [start_gen, start_gen]
                if cur_key is not None:
                    sql += " AND (key, rid, cgen) > (?, ?, ?)"
                    params.extend([cur_key, cur_rid, cur_cgen])
                sql += " ORDER BY key, rid, cgen LIMIT ?"
                params.append(remaining)
                batch = self.writer.execute(sql, params).fetchall()
                if not batch:
                    baseline_done = True
                    break
                for key, rid, cgen, _dgen in batch:
                    values, decoded_rid = key_decode(self.schema, key)
                    if decoded_rid != rid:
                        raise CollectionError(
                            f"stored key/row_id pair is inconsistent for {rid.hex()}"
                        )
                    self._insert_index_entry(
                        info["instance_id"], decl, key, rid, values, cgen
                    )
                    cur_key, cur_rid, cur_cgen = key, rid, cgen
                    this_baseline += 1
                    remaining -= 1
                if remaining <= 0:
                    break

            # ---- Phase 2: catch-up journal (ordered by apply publication) ----
            if baseline_done and remaining > 0:
                qsql = (
                    f"SELECT seq, kind, index_key, rid, cgen, main_key, close_gen "
                    f"FROM {SEC_QUEUE_TABLE} WHERE instance_id = ? AND seq > ? "
                    "ORDER BY seq LIMIT ?"
                )
                rows = self.writer.execute(
                    qsql, (info["instance_id"], processed_seq, remaining)
                ).fetchall()
                for seq, kind, index_key, rid, cgen, main_key, close_gen in rows:
                    if kind == "open":
                        # A version opening at cgen > B can never coincide with
                        # a baseline entry (which only covers cgen <= B) or with
                        # another open, so a conflict would be a real bug and
                        # must surface rather than be silently ignored.
                        self.writer.execute(
                            f"INSERT INTO {SEC_ENTRY_TABLE}"
                            "(instance_id, index_key, cgen, dgen, rid, main_key, "
                            "main_cgen) VALUES (?, ?, ?, NULL, ?, ?, ?)",
                            (info["instance_id"], index_key, cgen, rid, main_key,
                             cgen),
                        )
                    else:  # close
                        self.writer.execute(
                            f"UPDATE {SEC_ENTRY_TABLE} SET dgen = ? "
                            "WHERE instance_id = ? AND index_key = ? "
                            "AND cgen = ? AND dgen IS NULL",
                            (close_gen, info["instance_id"], index_key, cgen),
                        )
                    processed_seq = seq
                    this_catchup += 1
                    remaining -= 1

            # ---- Phase 3: commit the switch when everything is covered ----
            # The journal's next_seq is only bumped by applies; anything with
            # seq < next_seq is durable input.  processed_seq >= next_seq - 1
            # means every such event has been folded in below this write lock.
            final_head = self._meta_int("head")
            current_next = self.writer.execute(
                f"SELECT next_seq FROM {SEC_INDEX_TABLE} WHERE instance_id = ?",
                (info["instance_id"],),
            ).fetchone()[0]
            if baseline_done and processed_seq >= current_next - 1:
                # Every version created at or after B is present, and every
                # later change has been folded in.  The explicit switch.
                ready_gen = final_head
                self.writer.execute(
                    f"UPDATE {SEC_INDEX_TABLE} SET status = 'ready', "
                    "baseline_cursor_key = NULL, baseline_cursor_rid = NULL, "
                    "baseline_cursor_cgen = NULL, baseline_done = 1, "
                    "ready_gen = ?, processed_seq = ?, "
                    "baseline_candidates = baseline_candidates + ?, "
                    "catchup_changes = catchup_changes + ? "
                    "WHERE instance_id = ?",
                    (ready_gen, processed_seq, this_baseline, this_catchup,
                     info["instance_id"]),
                )
                self.writer.execute(
                    f"DELETE FROM {SEC_QUEUE_TABLE} WHERE instance_id = ?",
                    (info["instance_id"],),
                )
            else:
                self.writer.execute(
                    f"UPDATE {SEC_INDEX_TABLE} SET baseline_cursor_key = ?, "
                    "baseline_cursor_rid = ?, baseline_cursor_cgen = ?, "
                    "baseline_done = ?, processed_seq = ?, "
                    "baseline_candidates = baseline_candidates + ?, "
                    "catchup_changes = catchup_changes + ? "
                    "WHERE instance_id = ?",
                    (
                        cur_key if not baseline_done else None,
                        cur_rid if not baseline_done else None,
                        cur_cgen if not baseline_done else None,
                        1 if baseline_done else 0,
                        processed_seq,
                        this_baseline,
                        this_catchup,
                        info["instance_id"],
                    ),
                )
            if crash_hook is not None:
                crash_hook("pre_commit", self.writer)
            self._commit()
            if crash_hook is not None:
                crash_hook("post_commit", self.writer)
        except BaseException:
            self._rollback()
            raise

        status = self.index_status(name)
        status.baseline_candidates_read = this_baseline
        status.catchup_changes_processed = this_catchup
        return status

    def _insert_index_entry(
        self, instance_id: str, decl: SecondaryDeclaration,
        main_key: bytes, rid: bytes, values: tuple, cgen: int,
    ) -> None:
        index_key = encode_index_key(self.schema, decl, values, rid)
        # The baseline keyset visits each (key, rid, cgen) at most once, so a
        # primary-key conflict here would mean a pagination bug; do not mask
        # it with INSERT OR IGNORE.
        self.writer.execute(
            f"INSERT INTO {SEC_ENTRY_TABLE}"
            "(instance_id, index_key, cgen, dgen, rid, main_key, main_cgen) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?)",
            (instance_id, index_key, cgen, rid, main_key, cgen),
        )

    def cancel_index(self, name: str) -> dict:
        """Cancel a building index, releasing only its build-only material."""
        self._begin_write()
        try:
            row = self.writer.execute(
                f"SELECT instance_id, status, start_gen FROM {SEC_INDEX_TABLE} "
                "WHERE name = ? AND status IN ('building','ready')",
                (name,),
            ).fetchone()
            if row is None:
                raise IndexNotFoundError(f"no live secondary index named {name!r}")
            iid, status, start_gen = row
            if status == READY:
                raise IndexConflictError(
                    f"index {name!r} is ready; use delete_index to retire it"
                )
            entries = self.writer.execute(
                f"DELETE FROM {SEC_ENTRY_TABLE} WHERE instance_id = ?", (iid,)
            ).rowcount
            queued = self.writer.execute(
                f"DELETE FROM {SEC_QUEUE_TABLE} WHERE instance_id = ?", (iid,)
            ).rowcount
            self.writer.execute(
                f"UPDATE {SEC_INDEX_TABLE} SET status = 'cancelled' "
                "WHERE instance_id = ?",
                (iid,),
            )
            self._commit()
        except BaseException:
            self._rollback()
            raise
        return {
            "cancelled": name,
            "instance_id": iid,
            "start_gen": start_gen,
            "secondary_entries_removed": entries,
            "queue_entries_removed": queued,
        }

    def delete_index(self, name: str, *, instance_id: str | None = None) -> dict:
        """Retire a named index instance immediately.

        The instance can no longer advance or page; the name is free for a
        *different* declaration.  Old cursors keep their old instance id and
        are rejected.  Neither the main store, other builds, receipts nor
        snapshots are touched.
        """
        self._begin_write()
        try:
            row = self.writer.execute(
                f"SELECT instance_id, status, start_gen FROM {SEC_INDEX_TABLE} "
                "WHERE name = ? AND status IN ('building','ready')",
                (name,),
            ).fetchone()
            if row is None:
                raise IndexNotFoundError(f"no live secondary index named {name!r}")
            iid, status, start_gen = row
            if instance_id is not None and instance_id != iid:
                raise IndexConflictError(
                    f"index name {name!r} now refers to a different instance "
                    f"({iid}); the supplied identity {instance_id} is stale"
                )
            entries = self.writer.execute(
                f"DELETE FROM {SEC_ENTRY_TABLE} WHERE instance_id = ?", (iid,)
            ).rowcount
            queued = self.writer.execute(
                f"DELETE FROM {SEC_QUEUE_TABLE} WHERE instance_id = ?", (iid,)
            ).rowcount
            self.writer.execute(
                f"UPDATE {SEC_INDEX_TABLE} SET status = 'deleted' "
                "WHERE instance_id = ?",
                (iid,),
            )
            self._commit()
        except BaseException:
            self._rollback()
            raise
        return {
            "deleted": name,
            "instance_id": iid,
            "previous_status": status,
            "start_gen": start_gen,
            "secondary_entries_removed": entries,
            "queue_entries_removed": queued,
        }


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

    # -- secondary-index snapshot scans -----------------------------------

    def _index_scan_sql(
        self, instance_id: str, interval: Interval,
        after: tuple[bytes, bytes, int] | None,
    ) -> tuple[str, list]:
        # ``index_key`` ends with the ascending row_id component, so within a
        # single index_key the rid is constant and there is exactly one rid.
        # Ordering/seek on (index_key, cgen) is therefore the same total order
        # as (index_key, rid, cgen) and matches the primary-key prefix
        # (instance_id, index_key, cgen): SQLite streams the seek in key order
        # with no temp sort and no full-range materialization.
        clauses = ["instance_id = ?"]
        params: list = [instance_id]
        if interval.lower is not None:
            clauses.append("index_key >= ?")
            params.append(interval.lower)
        if interval.upper is not None:
            clauses.append("index_key < ?")
            params.append(interval.upper)
        if after is not None:
            clauses.append("(index_key, cgen) > (?, ?)")
            params.extend([after[0], after[2]])
        sql = (
            f"SELECT index_key, cgen, dgen, rid, main_key, main_cgen "
            f"FROM {SEC_ENTRY_TABLE} WHERE " + " AND ".join(clauses) +
            " ORDER BY index_key ASC, cgen ASC"
        )
        return sql, params

    def _resolve_ready_index(self, name: str) -> dict:
        info = self._load_instance(name=name, require_ready=False)
        if info["status"] == BUILDING:
            raise IndexNotReadyError(
                f"index {name!r} is still building; it cannot serve reads "
                "before its ready switch commits"
            )
        if info["status"] != READY:
            raise IndexNotFoundError(f"no live secondary index named {name!r}")
        return info

    def scan_index_page(
        self,
        snapshot_token: str,
        index_name: str,
        query: Mapping[str, Any],
        page_size: int,
        *,
        cursor_token: str | None = None,
    ) -> tuple[IndexPage, IndexScanStats, QueryPlan]:
        """Read one page of the *full* collection records through an index.

        The snapshot token still selects the read generation; the index only
        supplies a different access order.  The token's generation must be at
        least the index's coverage start B -- an older token is refused
        instead of silently falling back to a main-table scan.  Every returned
        record is the complete main-schema tuple and original row_id.
        """
        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
            raise TokenError("page_size must be a positive integer")
        ticket = self._resolve_snapshot(snapshot_token)
        info = self._resolve_ready_index(index_name)
        decl = resolve_declaration(
            self.schema, SecondaryDeclaration.from_json(info["declaration_json"])
        )
        if ticket.gen < info["start_gen"]:
            raise CoverageError(
                f"snapshot generation {ticket.gen} is older than index "
                f"{index_name!r}'s coverage start {info['start_gen']}; this "
                "index cannot answer that generation (a main-table scan is "
                "not substituted)"
            )
        validate_query_columns(self.schema, decl, query)
        plan = plan_index_query(self.schema, decl, query)

        after: tuple[bytes, bytes, int] | None = None
        if cursor_token is not None:
            cursor = decode_secondary_cursor(cursor_token, self.secret)
            if cursor.collection_id != self.collection_id:
                raise IndexCursorError("cursor belongs to a different collection")
            if cursor.db_path != self.path:
                raise IndexCursorError(
                    "cursor was created against a different file path"
                )
            if cursor.sid != ticket.sid:
                raise IndexCursorError(
                    "cursor belongs to a different snapshot token"
                )
            if cursor.gen != ticket.gen:
                raise IndexCursorError("cursor generation does not match snapshot")
            if cursor.index_name != index_name:
                raise IndexCursorError(
                    f"cursor belongs to index {cursor.index_name!r}, not "
                    f"{index_name!r}"
                )
            if cursor.instance_id != info["instance_id"]:
                raise IndexCursorError(
                    "cursor belongs to a different index instance "
                    "(a same-name rebuild cannot continue it)"
                )
            if cursor.declaration_hash != info["declaration_hash"]:
                raise IndexCursorError(
                    "cursor declaration does not match the index declaration"
                )
            if cursor.query_spec != plan.query_spec:
                raise IndexCursorError(
                    "cursor query specification does not match current query"
                )
            if cursor.page_size != page_size:
                raise IndexCursorError(
                    f"cursor page_size is {cursor.page_size}, but {page_size} "
                    "was requested"
                )
            after = (cursor.last_index_key, cursor.last_rid, cursor.last_cgen)

        eqp: list[dict] = []
        used_index = True
        if plan.intervals:
            sql0, params0 = self._index_scan_sql(
                info["instance_id"], plan.intervals[0], after
            )
            eqp = [
                {"id": r[0], "parent": r[1], "notused": r[2], "detail": r[3]}
                for r in self.reader.execute(
                    "EXPLAIN QUERY PLAN " + sql0, params0
                ).fetchall()
            ]
            used_index = any(
                "USING PRIMARY KEY" in r["detail"] or "USING INDEX" in r["detail"]
                for r in eqp
            )

        g = ticket.gen
        candidates = 0
        invisible = 0
        lookups = 0
        delivered: list[tuple] = []
        if plan.intervals:
            for interval in plan.intervals:
                if len(delivered) > page_size:
                    break
                sql, params = self._index_scan_sql(
                    info["instance_id"], interval, after
                )
                # One streaming storage cursor over the index key range.  At
                # most page_size + 1 visible entries are retained, so memory
                # is bounded by the page, not the range or the table.
                for index_key, cgen, dgen, rid, main_key, main_cgen in \
                        self.reader.execute(sql, params):
                    candidates += 1
                    visible = not (
                        cgen > g or (dgen is not None and dgen <= g)
                    )
                    if not visible:
                        invisible += 1
                        continue
                    # Locate the exact main version (key, rid, cgen), never a
                    # current live row in its place.  This is a primary-key
                    # seek on the main table, one per delivered candidate.
                    lookups += 1
                    main = self.reader.execute(
                        f"SELECT 1 FROM {KV_TABLE} WHERE key = ? AND rid = ? "
                        "AND cgen = ?",
                        (main_key, rid, main_cgen),
                    ).fetchone()
                    if main is None:
                        raise CollectionError(
                            "secondary index points at a missing main version"
                        )
                    values, decoded_rid = key_decode(self.schema, main_key)
                    if decoded_rid != rid:
                        raise CollectionError(
                            f"stored key/row_id pair is inconsistent for "
                            f"{rid.hex()}"
                        )
                    delivered.append((index_key, cgen, rid, main_key, values))
                    if len(delivered) > page_size:
                        break
                if len(delivered) > page_size:
                    break

        page_rows = delivered[:page_size]
        has_more = len(delivered) > page_size
        next_cursor = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_secondary_cursor(
                secret=self.secret,
                collection_id=self.collection_id,
                sid=ticket.sid,
                gen=ticket.gen,
                query_spec=plan.query_spec,
                page_size=page_size,
                index_name=index_name,
                instance_id=info["instance_id"],
                declaration_hash=info["declaration_hash"],
                last_index_key=last[0],
                last_rid=last[2],
                last_cgen=last[1],
                db_path=self.path,
            )
        stats = IndexScanStats(
            index_candidates_examined=candidates,
            invisible_versions_skipped=invisible,
            full_row_lookups=lookups,
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
        page = IndexPage(
            rows=[(row[4], row[2]) for row in page_rows],
            next_cursor=next_cursor,
            last_index_key=page_rows[-1][0] if page_rows else None,
            gen=ticket.gen,
            sid=ticket.sid,
            index_name=index_name,
            instance_id=info["instance_id"],
        )
        return page, stats, plan

    def iter_index(
        self,
        snapshot_token: str,
        index_name: str,
        query: Mapping[str, Any],
        page_size: int,
    ) -> Iterable[tuple[tuple, bytes]]:
        cursor_token: str | None = None
        while True:
            page, _stats, _plan = self.scan_index_page(
                snapshot_token, index_name, query, page_size,
                cursor_token=cursor_token,
            )
            for item in page.rows:
                yield item
            if page.next_cursor is None:
                return
            cursor_token = page.next_cursor

    def explain_index_scan(
        self,
        index_name: str,
        query: Mapping[str, Any],
        page_size: int = 100,
    ) -> dict:
        """Plan evidence for the secondary path without reading any rows."""
        info = self._resolve_ready_index(index_name)
        decl = resolve_declaration(
            self.schema, SecondaryDeclaration.from_json(info["declaration_json"])
        )
        validate_query_columns(self.schema, decl, query)
        plan = plan_index_query(self.schema, decl, query)
        eqp: list[dict] = []
        if plan.intervals:
            sql0, params0 = self._index_scan_sql(
                info["instance_id"], plan.intervals[0], None
            )
            eqp = [
                {"id": r[0], "parent": r[1], "notused": r[2], "detail": r[3]}
                for r in self.reader.execute(
                    "EXPLAIN QUERY PLAN " + sql0, params0
                ).fetchall()
            ]
        return {
            "index": info["name"],
            "instance_id": info["instance_id"],
            "declaration": json.loads(info["declaration_json"]),
            "covered_from": info["start_gen"],
            "ready_gen": info["ready_gen"],
            "plan": plan.explain(),
            "explain_query_plan": eqp,
            "entry_table_seek": bool(eqp) and any(
                "USING PRIMARY KEY" in d["detail"] for d in eqp
            ),
            "page_size": page_size,
        }

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
            # visibility window [cgen, dgen).  In addition, a *building*
            # secondary index protects its own baseline candidates in the main
            # store: those versions are inputs to a future advance, and the
            # user must not have to keep an extra snapshot alive for the
            # build.  Once the build is ready or cancelled the protection
            # disappears (ready indexes only ever need versions an active
            # snapshot can still read, exactly like the main store).
            cur = self.writer.execute(
                f"DELETE FROM {KV_TABLE} "
                "WHERE dgen IS NOT NULL AND dgen <= ? "
                "AND NOT EXISTS ("
                f"  SELECT 1 FROM {SNAPSHOT_TABLE} s "
                "  WHERE s.status = 'active' AND s.gen >= bytekey_kv.cgen "
                "  AND s.gen < bytekey_kv.dgen)"
                " AND NOT EXISTS ("
                f"  SELECT 1 FROM {SEC_INDEX_TABLE} i "
                "  WHERE i.status = 'building' "
                "  AND bytekey_kv.cgen <= i.start_gen "
                "  AND bytekey_kv.dgen > i.start_gen)",
                (retain_from,),
            )
            removed = cur.rowcount
            # Secondary history items obey the same rule, per instance.  A
            # *building* index protects its own material: its baseline
            # candidates and any on-disk journal rows are inputs to a future
            # advance, so history collection must not consume them while the
            # build is in flight.  (Ready indexes keep dead versions only as
            # long as an active snapshot can see them, exactly like main.)
            sec_removed = self.writer.execute(
                f"DELETE FROM {SEC_ENTRY_TABLE} "
                "WHERE dgen IS NOT NULL AND dgen <= ? "
                "AND NOT EXISTS ("
                f"  SELECT 1 FROM {SNAPSHOT_TABLE} s "
                "  WHERE s.status = 'active' AND s.gen >= bytekey_sec_entries.cgen "
                "  AND s.gen < bytekey_sec_entries.dgen)"
                " AND NOT EXISTS ("
                f"  SELECT 1 FROM {SEC_INDEX_TABLE} i "
                "  WHERE i.instance_id = bytekey_sec_entries.instance_id "
                "  AND i.status = 'building')",
                (retain_from,),
            ).rowcount
            # Journal rows belong exclusively to a *building* instance; they
            # are build-only protected input and are released when the build
            # commits ready (or is cancelled/deleted).  A live build keeps all
            # of its journal even if retention moved past it.  Any journal row
            # whose instance is no longer building is stale and removable.
            queue_removed = self.writer.execute(
                f"DELETE FROM {SEC_QUEUE_TABLE} "
                "WHERE NOT EXISTS ("
                f"  SELECT 1 FROM {SEC_INDEX_TABLE} i "
                "  WHERE i.instance_id = bytekey_sec_queue.instance_id "
                "  AND i.status = 'building')"
            ).rowcount
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
            secondary_entries_removed=sec_removed,
            queue_entries_removed=queue_removed,
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
        indexes = [
            {
                "name": i.name,
                "instance_id": i.instance_id,
                "build_id": i.build_id,
                "status": i.status,
                "start_gen": i.start_gen,
                "covered_from": i.start_gen,
                "ready_gen": i.ready_gen,
                "declaration": i.declaration,
                "declaration_hash": i.declaration_hash,
                "baseline_done": i.baseline_done,
            }
            for i in self.list_indexes(include_deleted=False)
        ]
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
            indexes=indexes,
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
