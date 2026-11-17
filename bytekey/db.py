"""SQLite BLOB-primary-key storage, streaming build and paged interval scans.

The index table is declared ``WITHOUT ROWID`` with a single BLOB primary key,
so SQLite stores and walks exactly the encoded byte order -- there is no
second sort key and no business column inside the database.  Builds stream
rows through ``executemany`` inside one transaction; a finished index file is
never overwritten.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .cursor import Cursor, CursorError, decode_cursor, encode_cursor
from .encoding import EncodingError, key_decode, key_encode
from .planner import Interval, QueryPlan, plan_query
from .schema import Schema

META_TABLE = "bytekey_meta"
KEY_TABLE = "bytekey_index"
ROWID_TABLE = "bytekey_rowids"
META_VERSION = 1


class BuildError(RuntimeError):
    """Raised when an index cannot be built or an existing one would be touched."""


@dataclass
class BuildStats:
    rows: int
    bytes_written: int


@dataclass
class Page:
    rows: list[tuple[tuple, bytes]]   # decoded (business_tuple, row_id)
    next_cursor: str | None
    last_key: bytes | None


@dataclass
class ScanStats:
    rows_returned: int
    rows_visited: int
    used_index: bool
    explain_query_plan: list[dict]
    explain_bytecode: list[dict]


# ---------------------------------------------------------------------------
# Fingerprint / metadata
# ---------------------------------------------------------------------------

def compute_fingerprint(schema: Schema) -> str:
    h = hashlib.sha256()
    h.update(b"bytekey-index/v1\n")
    h.update(schema.canonical_json())
    return h.hexdigest()


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    # BLOB lexicographic ordering is a core SQLite feature; guard the engine.
    conn.execute("PRAGMA application_id = 0x6279746b")  # 'bytk'
    return conn


def _open_readonly(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise BuildError(f"index file does not exist: {path}")
    uri = "file:" + os.path.abspath(path).replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    row = conn.execute("PRAGMA application_id").fetchone()
    if row[0] != 0x6279746B:
        conn.close()
        raise BuildError(f"not a bytekey index file: {path}")
    return conn


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_index(
    path: str,
    schema: Schema,
    rows: Iterable[Any],
    *,
    chunk_size: int = 1000,
) -> BuildStats:
    """Stream rows into a new read-only index.

    ``rows`` yields mappings containing every schema column and the row_id
    field.  The database is built at a temporary sibling path and atomically
    renamed only after a clean commit; any bad value aborts the build and the
    target path is left untouched.
    """
    if os.path.exists(path):
        raise BuildError(
            f"refusing to overwrite existing index {path!r}; delete it first"
        )
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(
        directory, f".bytekey.tmp.{os.getpid()}.{secrets.token_hex(8)}.db"
    )
    build_secret = secrets.token_hex(32)
    fingerprint = compute_fingerprint(schema)
    max_size = schema.max_key_bytes

    total = 0
    try:
        conn = _connect(tmp_path)
        try:
            conn.execute(f"CREATE TABLE {KEY_TABLE} (key BLOB PRIMARY KEY) WITHOUT ROWID")
            conn.execute(
                f"CREATE TABLE {ROWID_TABLE} (rid BLOB PRIMARY KEY) WITHOUT ROWID"
            )
            conn.execute(
                f"CREATE TABLE {META_TABLE} (name TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID"
            )
            conn.execute("PRAGMA journal_mode = DELETE")
            conn.execute("PRAGMA synchronous = FULL")
            conn.execute("BEGIN")

            def keyed() -> Iterator[tuple[bytes, bytes]]:
                nonlocal total
                # Streaming by design: no per-row Python set is retained, so
                # build memory stays bounded by the chunk buffer.  Uniqueness
                # is enforced inside SQLite by two PRIMARY KEY constraints:
                # the full encoded key on the index table and the row_id on
                # its own table, both populated inside the same transaction.
                for line_no, row in enumerate(rows, start=1):
                    if not isinstance(row, Mapping):
                        raise BuildError(f"row {line_no} is not an object")
                    known = {c.name for c in schema.columns} | {schema.row_id_name}
                    unknown = set(row) - known
                    if unknown:
                        raise BuildError(f"row {line_no} has unknown field(s): {sorted(unknown)}")
                    if schema.row_id_name not in row:
                        raise BuildError(f"row {line_no} is missing {schema.row_id_name!r}")
                    row_id = row[schema.row_id_name]
                    if not isinstance(row_id, (bytes, bytearray)):
                        raise BuildError(
                            f"row {line_no}: {schema.row_id_name} must be bytes, "
                            f"got {type(row_id).__name__}"
                        )
                    row_id = bytes(row_id)
                    values = []
                    for col in schema.columns:
                        if col.name not in row:
                            raise BuildError(
                                f"row {line_no} is missing column {col.name!r}"
                            )
                        values.append(row[col.name])
                    try:
                        key = key_encode(schema, values, row_id, max_size=max_size)
                    except EncodingError as exc:
                        raise BuildError(f"row {line_no}: {exc}") from None
                    total += 1
                    yield key, row_id

            def insert_chunk(buf: list[tuple[bytes, bytes]]) -> None:
                if not buf:
                    return
                try:
                    conn.executemany(
                        f"INSERT INTO {KEY_TABLE}(key) VALUES (?)",
                        [(k,) for k, _ in buf],
                    )
                    conn.executemany(
                        f"INSERT INTO {ROWID_TABLE}(rid) VALUES (?)",
                        [(r,) for _, r in buf],
                    )
                except sqlite3.IntegrityError as exc:
                    # Collision: reused row_id, or identical tuple + row_id.
                    raise BuildError(
                        f"duplicate row_id or encoded key rejected: {exc}"
                    ) from None

            buf: list[tuple[bytes, bytes]] = []
            for key, row_id in keyed():
                buf.append((key, row_id))
                if len(buf) >= chunk_size:
                    insert_chunk(buf)
                    buf.clear()
            insert_chunk(buf)

            meta = {
                "version": str(META_VERSION),
                "fingerprint": fingerprint,
                "build_secret": build_secret,
                "schema_json": schema.canonical_json().decode("utf-8"),
                "row_id_name": schema.row_id_name,
                "row_count": str(total),
            }
            conn.executemany(
                f"INSERT INTO {META_TABLE}(name, value) VALUES (?, ?)",
                list(meta.items()),
            )
            conn.commit()
        finally:
            conn.close()

        # Atomic publish.  os.replace is atomic on the same filesystem; the
        # target's absence was verified at entry, so a finished index is never
        # overwritten and there is no partially-written file at the path.
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        # A replace failure must never leave a stray destination file.
        raise

    return BuildStats(rows=total, bytes_written=os.path.getsize(path))


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------

class Index:
    """A read-only, immutable byte-key index."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.conn = _open_readonly(self.path)
        meta_rows = self.conn.execute(
            f"SELECT name, value FROM {META_TABLE}"
        ).fetchall()
        meta = dict(meta_rows)
        if int(meta.get("version", "0")) != META_VERSION:
            raise BuildError(f"unsupported index meta version in {path!r}")
        self.schema = Schema.from_json(meta["schema_json"])
        self.fingerprint = meta["fingerprint"]
        self.build_secret = meta["build_secret"]
        self.row_id_name = meta["row_id_name"]
        self.row_count = int(meta["row_count"])

    # -- introspection -----------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def explain(self, query: Mapping[str, Any]) -> dict:
        plan = plan_query(self.schema, query)
        return {"plan": plan.explain(), "query_plan_detail": self._eqp(plan, None)}

    def _eqp(self, plan: QueryPlan, cursor: Cursor | None) -> list[dict]:
        sql, params = self._build_sql(plan.intervals, plan.reverse_delivery, cursor)
        rows = self.conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
        return [
            {"id": r[0], "parent": r[1], "notused": r[2], "detail": r[3]} for r in rows
        ]

    def _explain_bytecode(
        self, plan: QueryPlan, cursor: Cursor | None
    ) -> list[dict]:
        sql, params = self._build_sql(plan.intervals, plan.reverse_delivery, cursor)
        rows = self.conn.execute("EXPLAIN " + sql, params).fetchall()
        cols = ("addr", "opcode", "p1", "p2", "p3", "p4", "p5", "comment")
        return [dict(zip(cols, r)) for r in rows]

    # -- SQL generation ----------------------------------------------------

    @staticmethod
    def _build_sql(
        intervals: Sequence[Interval],
        reverse: bool,
        cursor: Cursor | None,
    ) -> tuple[str, list[bytes]]:
        interval_clauses: list[str] = []
        params: list[bytes] = []
        if not intervals:
            # A plan that compiled to no intervals matches nothing.
            interval_clauses.append("0 = 1")
        elif len(intervals) == 1:
            interval = intervals[0]
            if interval.lower is not None:
                interval_clauses.append("key >= ?")
                params.append(interval.lower)
            if interval.upper is not None:
                interval_clauses.append("key < ?")
                params.append(interval.upper)
        else:
            or_parts: list[str] = []
            for interval in intervals:
                parts: list[str] = []
                if interval.lower is not None:
                    parts.append("key >= ?")
                    params.append(interval.lower)
                if interval.upper is not None:
                    parts.append("key < ?")
                    params.append(interval.upper)
                or_parts.append("(" + (" AND ".join(parts) if parts else "1=1") + ")")
            interval_clauses.append(" OR ".join(or_parts))
        clauses = list(interval_clauses)
        if cursor is not None and cursor.last_key is not None:
            if reverse:
                clauses.append("key < ?")
            else:
                clauses.append("key > ?")
            params.append(cursor.last_key)
        where = " AND ".join(clauses) if clauses else "1=1"
        direction = "DESC" if reverse else "ASC"
        sql = (
            f"SELECT key FROM {KEY_TABLE} WHERE {where} "
            f"ORDER BY key {direction}"
        )
        return sql, params

    # -- scanning ----------------------------------------------------------

    def _check_cursor(
        self,
        plan: QueryPlan,
        page_size: int,
        cursor: Cursor | None,
    ) -> Cursor | None:
        if cursor is None:
            return None
        if cursor.index_fingerprint != self.fingerprint:
            raise CursorError("cursor belongs to a different index identity")
        if cursor.db_path != self.path:
            raise CursorError("cursor was created against a different index file")
        if cursor.build_secret != self.build_secret:
            raise CursorError("cursor build secret does not match this index")
        if cursor.query_spec != plan.query_spec:
            raise CursorError("cursor query specification does not match current query")
        if cursor.reverse_delivery != plan.reverse_delivery:
            raise CursorError("cursor scan direction does not match current query")
        if cursor.page_size != page_size:
            raise CursorError(
                f"cursor page_size is {cursor.page_size}, but {page_size} was requested"
            )
        return cursor

    def scan_page(
        self,
        query: Mapping[str, Any],
        page_size: int,
        *,
        cursor_token: str | None = None,
    ) -> tuple[Page, ScanStats, QueryPlan]:
        if not isinstance(page_size, int) or page_size <= 0:
            raise CursorError("page_size must be a positive integer")
        plan = plan_query(self.schema, query)
        cursor = decode_cursor(cursor_token) if cursor_token else None
        self._check_cursor(plan, page_size, cursor)

        sql, params = self._build_sql(plan.intervals, plan.reverse_delivery, cursor)
        eqp = self._eqp(plan, cursor)
        bytecode = self._explain_bytecode(plan, cursor)
        if not plan.intervals:
            # Compiled to the empty set: nothing is scanned at all.
            used_index = True
        else:
            # A genuine range search references the PRIMARY KEY index; a
            # plain "SCAN table" would mean a full table read.
            used_index = any(
                "USING PRIMARY KEY" in d["detail"] or "USING INDEX" in d["detail"]
                for d in eqp
            )

        # LIMIT one extra row lets us know whether another page exists while
        # keeping every page a bounded, streaming fetch.
        fetch_sql = sql + " LIMIT ?"
        fetch_params = params + [page_size + 1]
        got = self.conn.execute(fetch_sql, fetch_params).fetchall()
        rows: list[tuple[tuple, bytes]] = []
        last_key: bytes | None = None
        for (key,) in got[:page_size]:
            rows.append(key_decode(self.schema, key))
            last_key = key
        has_more = len(got) > page_size
        next_cursor = None
        if has_more:
            next_cursor = encode_cursor(
                index_fingerprint=self.fingerprint,
                query_spec=plan.query_spec,
                page_size=page_size,
                reverse_delivery=plan.reverse_delivery,
                last_key=last_key,
                build_secret=self.build_secret,
                db_path=self.path,
            )
        # Count only rows physically examined inside the bounded scan.
        visited_sql = f"SELECT COUNT(*) FROM ({sql})"
        rows_visited = self.conn.execute(visited_sql, params).fetchone()[0]
        stats = ScanStats(
            rows_returned=len(rows),
            rows_visited=rows_visited,
            used_index=used_index,
            explain_query_plan=eqp,
            explain_bytecode=bytecode,
        )
        return Page(rows=rows, next_cursor=next_cursor, last_key=last_key), stats, plan

    def iter_all(
        self, query: Mapping[str, Any], page_size: int
    ) -> Iterator[tuple[tuple, bytes]]:
        """Yield every match page by page; never materializes all rows."""
        token: str | None = None
        while True:
            page, _stats, _plan = self.scan_page(
                query, page_size, cursor_token=token
            )
            for item in page.rows:
                yield item
            if page.next_cursor is None:
                return
            token = page.next_cursor


def open_index(path: str) -> Index:
    return Index(path)
