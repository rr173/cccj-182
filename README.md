# bytekey — composite-key byte-order index on SQLite BLOBs

`bytekey` encodes ordered business tuples into **byte-comparable BLOB keys**
stored in a SQLite `WITHOUT ROWID` primary-key table, then compiles business
range queries into concrete byte scan intervals.  It is built for stable,
keyset (seek) pagination over warehouse-style composite keys.

## What it guarantees

* **Order-preserving, reversible, collision-free encoding.**  SQLite's BLOB
  lexicographic ordering of the encoded keys equals the declared business
  ordering, including negative integers, strict-prefix text, embedded NUL
  bytes, empty strings, nulls in either position, and descending columns.
  `key_encode` / `key_decode` are exact inverses.  No `repr`, no delimiter
  string concatenation.
* **Fixed column types** `int64`, `text`, `bytes`, `bool`, each with its own
  `asc`/`desc` and `nulls_first`/`nulls_last`.  Text compares by Unicode code
  point with no normalization and rejects lone surrogates; bytes compare
  unsigned; bool is independent of int.
* **Unique non-empty `bytes` row_id** as the final ascending tie-break when
  whole business tuples are equal.
* **Real index range scans.**  Queries become finite, non-overlapping
  `[lower, upper)` byte intervals used directly in
  `key >= ? AND key < ?` against the BLOB primary key.  Rows are never
  fully decoded then filtered.  `EXPLAIN QUERY PLAN` shows
  `SEARCH ... USING PRIMARY KEY` and the bytecode uses `SeekGE`.
* **Two query shapes.**
  * `full_tuple`: lower/upper bounds, each independently open/closed, with no
    row_id; a boundary includes or excludes the *whole* equal-tuple group.
  * `prefix_range`: equality prefix over the first *k* columns (null allowed
    as an equality value), an optional non-null range over the very next
    column, later columns unconstrained.  Endpoints use the column's natural
    non-null domain lo/hi and openness; nulls are excluded from the range;
    missing endpoints mean unbounded.
* **Bound, identity-checked cursors.**  A cursor binds the index identity,
  the exact query spec, the page size, the scan direction and the last full
  encoded key.  Replaying it on another index, query or page size is
  rejected.  Pagination never skips sibling row_ids and never repeats a key
  sitting on a range endpoint.
* **Immutable, streaming builds.**  A finished index is never overwritten; a
  bad source row aborts the build and leaves no file.  Build memory is
  bounded by the insert chunk (uniqueness is enforced inside SQLite), and
  queries decode one page at a time.

## Encoding sketch

Each column becomes a self-delimiting component with a domain tag:

| domain            | raw tag |
|-------------------|---------|
| NULL (first slot) | `0x01`  |
| bool false / true | `0x02` / `0x03` |
| int64             | `0x04` + 8-byte big-endian sign-flipped |
| text              | `0x05` + escaped UTF-8 + terminator |
| bytes             | `0x06` + escaped payload + terminator |
| NULL (last slot)  | `0x07`  |
| row_id slot       | `0x07` + escaped non-empty bytes + terminator |

Variable payloads escape every `0x00` as `0x00 0xFF` and terminate with
`0x00 0x00`, which keeps strict-prefix and embedded-NUL ordering exact.  A
descending column bit-inverts its whole finished component; inversion is a
monotone-reversing bijection, so DESC business order becomes ascending byte
order (and the raw NULL tag is chosen so its final position still honors
`nulls_first`/`nulls_last`).

## Library API

```python
from bytekey import Schema, Column, build_index, open_index, key_encode, key_decode

schema = Schema(columns=(
    Column("region", "text", asc=True, nulls_first=True),
    Column("ts",     "int64", asc=False, nulls_first=True),
))

build_index("idx.db", schema, row_iter)          # streaming, read-only result
idx = open_index("idx.db")

query = {
    "shape": "prefix_range",
    "prefix": [["region", "a"]],
    "range_column": "ts",
    "lo": 1, "hi": 10, "lo_inclusive": True, "hi_inclusive": False,
}
page, stats, plan = idx.scan_page(query, page_size=100)
plan.explain()                 # intervals + residual checks
stats.explain_query_plan       # proof of PRIMARY KEY range search
page.next_cursor               # opaque token; None on the last page
```

Validation errors (`QueryError`, `EncodingError`, `CursorError`,
`BuildError`) are raised in Python before any database scan whenever the
fault is knowable up front.

## Commands

```bash
python bytekey_cli.py build   --schema schema.json --source rows.jsonl --index idx.db
python bytekey_cli.py explain --index idx.db --query query.json --page-size 100
python bytekey_cli.py range   --index idx.db --query query.json --page-size 1
python bytekey_cli.py encode  --schema schema.json --input one_row.json
python bytekey_cli.py decode  --schema schema.json <key_hex>
```

`bytes` values are passed as hex strings in JSON.

## Tests

The independent oracle in `tests/oracle.py` sorts plain tuples with a
comparator that never imports the encoder.  The suites cover the mandated
hand-written fixture (page size 1, line-by-line comparison), randomized
exhaustive prefix/range checks against the oracle, a large library with
narrow-query access counts, int64 extremes, NUL/0xFF bytes, non-ASCII text,
empty string vs null, DESC prefix truncation, open/closed and unbounded
bounds, foreign/tampered cursors, bad types, corrupt sources and write
failures.

```bash
python3 -I tests/test_encoding.py
python3 -I tests/test_index.py
python3 -I tests/test_random_exhaustive.py
python3 -I tests/test_large_library.py
python3 -I tests/test_edge_cases.py
python3 -I tests/test_cli_and_failures.py
```

---

# Updatable collections + persistent read snapshots

The static index above is unchanged and still read-only.  Alongside it,
`bytekey` now ships an **updatable collection** that accepts continuous
upstream corrections (full-row replacements that may move a record to a
different sort position, deletes, and re-creates) while long-running exports
keep reading the exact generation they started on.

## Data model and visibility

A collection lives in one SQLite file with an append-only versioned table:

```
bytekey_kv(key BLOB, rid BLOB, cgen INTEGER, dgen INTEGER,
           PRIMARY KEY(key, rid, cgen)) WITHOUT ROWID
```

`key` is the unchanged full encoded business key (row_id included), `rid` is
the record identity, `cgen` the creating generation and `dgen` the generation
that superseded or deleted the entry (`NULL` while live).  A partial unique
index guarantees at most one live entry per `rid`.

**Visibility at a snapshot pinned to generation `g`:**

```
entry visible  <=>  cgen <= g  AND  (dgen IS NULL  OR  dgen > g)
```

Consequences, all enforced by the storage predicate rather than by rebuilding
anything:

* a moved business key never appears at both its old and new position in one
  generation;
* a delete cannot affect a snapshot taken before the delete;
* a row_id deleted and later re-created is a fresh entry, never confused with
  the deleted one;
* every query is still a concrete BLOB range seek over the planner's byte
  intervals.  In-range *historical candidates* are pulled through the primary
  key index and tested against the visibility rule; rows outside the
  intervals are never decoded.

Generations start at **0**.  A freshly created (or imported) collection has
`head = 0`; every successful `apply` publishes generation `head + 1` and
advances `head` by exactly one — including empty batches and batches whose
operations are all no-ops.

## Apply: atomic, idempotent, streaming batches

`apply(batch_id, expected_head, operations)` validates the whole stream into
an on-disk staging table inside one `BEGIN IMMEDIATE` transaction before any
collection row changes.  Corrupt source tails, bad types, duplicate row_ids
inside a batch (no last-write-wins), and `expected_head` mismatches roll back
with no partial update and no stored receipt, so the `batch_id` stays free.

A successful batch is safely re-sendable: the same `batch_id` with identical
content and `expected_head` returns the original receipt (flagged
`replayed`) without changing anything — even after `head` moved.  Reordering
lines, changing harmless JSON whitespace, or changing hex letter case all
compare as the same content, while null, empty text, empty bytes and bool
stay non-confusable in the type-tagged digest.  Two processes competing with
the same `expected_head` serialize on the write transaction; exactly one
commits and the other receives an explicit conflict.

Batch source is streamed row by row and spooled to the staging table — it is
never held in a Python list or dict.

## Persistent snapshots and cursors

`begin_snapshot(gen=None)` returns a durable HMAC-signed token for the
current head or any retained generation.  The token plus every cursor derived
from it are validated against the collection database on each use, so they
survive connection close and process exit, and a released or forged token is
refused outright instead of silently following the latest generation.

Cursors bind the collection identity, the **specific snapshot id** (a cursor
cannot be replayed through another snapshot of the same generation), the
canonical query, the page size and the last full delivered key.  Another
process needs only the collection path, snapshot token, query and cursor to
continue paging.

`collect_history(retain_from)` moves a monotonic retention boundary and
physically deletes dead entries that no retained generation and no active
snapshot can read.  An active snapshot pins every entry its generation can
see; after the last protecting token is released, a later collection removes
them.  Receipts and `head` are never reclaimed, so re-sending an old batch
still returns its original receipt.

## Collection API

```python
from bytekey import (
    Schema, Column, create_collection, open_collection, import_from_index,
    BatchError, ConflictError, HistoryError, TokenError,
)

schema = Schema(columns=(
    Column("region", "text", asc=True, nulls_first=True),
    Column("ts",     "int64", asc=False, nulls_first=True),
))

coll = open_collection("warehouse.db")                 # or create_collection(...)
receipt = coll.apply(
    batch_id=b"correction-0042",
    expected_head=coll.status().head,
    operations=iter([                                   # any streaming iterable
        {"op": "put",    "row_id": b"r1", "region": "c", "ts": 10},
        {"op": "delete", "row_id": b"r2"},
    ]),
)
token   = coll.begin_snapshot()                          # pin current head
older   = coll.begin_snapshot(7)                         # pin a retained gen
page, stats, plan = coll.scan_snapshot_page(
    token,
    {"shape": "prefix_range", "prefix": [["region", "a"]], "range_column": "ts"},
    page_size=100,
)
stats.candidates_examined      # rows actually pulled from storage
stats.invisible_versions_skipped
stats.rows_delivered           # final deliveries (never a SQL COUNT masquerade)
stats.explain_query_plan       # SEARCH ... USING PRIMARY KEY evidence
coll.release_snapshot(token)
coll.collect_history(retain_from=12)
```

A static index can seed generation 0 without being modified:

```python
import_from_index("old_index.db", "warehouse.db")       # streamed, bounded memory
```

## Collection commands

All JSON `bytes` (row_id, batch_id and bytes columns) use the single-field
object form `{"__hex__": "00ff"}` on input and output.  Domain errors print a
locatable message and exit nonzero (2); `snapshot-range` returns exactly one
page per invocation so scripts can save the token/cursor and exit.

```bash
python bytekey_cli.py collection-create --schema schema.json --collection warehouse.db
python bytekey_cli.py collection-import --index old_index.db --collection warehouse.db
python bytekey_cli.py apply            --collection warehouse.db \
     --batch-id b42.json --expected-head 5 --source corrections.jsonl
python bytekey_cli.py snapshot-begin   --collection warehouse.db [--gen 5]
python bytekey_cli.py snapshot-range   --collection warehouse.db \
     --token "$SNAP" --query query.json --page-size 100 [--cursor "$CUR"]
python bytekey_cli.py snapshot-release --collection warehouse.db --token "$SNAP"
python bytekey_cli.py snapshot-explain --collection warehouse.db --query query.json
python bytekey_cli.py collect-history  --collection warehouse.db --retain-from 12
python bytekey_cli.py status           --collection warehouse.db
```

`b42.json` contains `{"__hex__": "636f7272656374696f6e2d30303432"}`; an
operation stream looks like:

```jsonl
{"op":"put","row_id":{"__hex__":"7231"},"region":"c","ts":10}
{"op":"delete","row_id":{"__hex__":"7232"}}
```

## Complete migration-to-reclamation walk-through

```bash
# 0. Build the old static index exactly as before (still fully supported).
python bytekey_cli.py build --schema schema.json --source rows.json --index old.db

# 1. Import it as generation 0 of an updatable collection (old.db untouched).
python bytekey_cli.py collection-import --index old.db --collection warehouse.db

# 2. An export starts and pins the generation it begins reading.
python bytekey_cli.py snapshot-begin --collection warehouse.db --gen 0 > snap.json
SNAP=$(python3 -c "import json,sys; print(json.load(open('snap.json'))['snapshot_token'])")

# 3. Upstream corrections keep arriving and publishing new generations.
python bytekey_cli.py apply --collection warehouse.db \
    --batch-id b1.json --expected-head 0 --source corrections-1.jsonl
python bytekey_cli.py apply --collection warehouse.db \
    --batch-id b2.json --expected-head 1 --source corrections-2.jsonl

# 4. The original export continues page by page on generation 0 regardless;
#    each invocation is one process and one page, cursor saved between runs.
#    (The page size is part of the cursor binding, so keep it constant.)
python bytekey_cli.py snapshot-range --collection warehouse.db \
    --token "$SNAP" --query query.json --page-size 250 > page1.json
CUR=$(python3 -c "import json; print(json.load(open('page1.json'))['next_cursor'])")
python bytekey_cli.py snapshot-range --collection warehouse.db \
    --token "$SNAP" --query query.json --page-size 250 --cursor "$CUR" > page2.json
# 5. A second, current-generation reader starts while the old one finishes.
python bytekey_cli.py snapshot-begin --collection warehouse.db > snap-latest.json

# 6. Once the old export ends, release its token, then reclaim history.
python bytekey_cli.py snapshot-release --collection warehouse.db --token "$SNAP"
python bytekey_cli.py collect-history --collection warehouse.db --retain-from 2
python bytekey_cli.py status --collection warehouse.db
```

## Collection tests

The reference model in `tests/refmodel.py` keeps each generation as a plain
`dict[row_id -> tuple]` and never imports the production visibility code.
The suites add the mandated three-generation fixture, cross-process cursor
continuation, duplicate/tail/type rejection, idempotent replay, competing
writers, controlled `os._exit` crash recovery, snapshot lifecycle and
retention invariants, randomized multi-generation differential runs, and a
generated large collection proving bounded correction work, key-range narrow
queries and real reclamation:

```bash
python3 -I tests/test_collection_fixture.py    # hand-written 3-generation oracle
python3 -I tests/test_collection_apply.py      # atomicity, idempotency, races, crashes
python3 -I tests/test_collection_snapshots.py  # tokens, cursors, history collection
python3 -I tests/test_collection_random.py     # randomized differential vs oracle
python3 -I tests/test_collection_large.py      # bounded work, range scans, reclamation
python3 -I tests/test_collection_cli.py        # full CLI lifecycle and failure exits
```

