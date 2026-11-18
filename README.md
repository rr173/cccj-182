# bytekey — composite-key byte-order index on SQLite BLOBs

`bytekey` encodes ordered business tuples into **byte-comparable BLOB keys**
stored in a SQLite `WITHOUT ROWID` primary-key table, then compiles business
range queries into concrete byte scan intervals.  It is built for stable,
keyset (seek) pagination over warehouse-style composite keys.

Two layers build on the static index: an **updatable collection** with durable
multi-generation read snapshots (below), and, on top of it, **named secondary
access indexes** that serve the same full records in a second user-chosen
order without touching the main ordering.  SQLite is the only store; no
external service is required.

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

---

# Named secondary access indexes

A collection's primary ordering is fixed by its schema. When a second workload
needs the same records by a different order (say *time window first, then
region*), you can register a **named secondary access index** over a chosen
subset of the collection's *existing* business columns. It is a real ordered
storage path, not a cached Python sort:

* the key is the unchanged reversible `bytekey` encoding of the projected
  columns, each with its own `asc` / `nulls_first`, terminated by the original
  `row_id` ascending. The column *type* is inherited from the main schema and
  no new business column or expression is introduced;
* equal indexed business values for different identities stay distinct keys
  (the row_id is part of the key), so the business projection is never made
  unique;
* the ready index is maintained by every later `apply` in the *same commit* as
  the main version, head and receipt, so it always reflects the whole
  generation history, not just a snapshot taken at the ready switch.

The two existing query shapes are reused. A query is interpreted in the
*index declaration's* column order; the returned `values` are always the
**full main-schema tuple** (all columns) plus the original `row_id` -- never a
projection of just the sorted columns.

## Declaration

```json
{"name": "ts_region",
 "columns": [{"name": "ts", "asc": false, "nulls_first": true},
             {"name": "region", "asc": true, "nulls_first": true}]}
```

Each column must already exist in the collection schema and may appear at most
once; `name` must be non-empty. A duplicate among *active* names fails
explicitly and hands back the existing `build_id` to resume instead.

## Lifecycle: register, advance, ready

```python
from bytekey import SecondaryDeclaration, IndexColumn, open_collection

coll = open_collection("warehouse.db")
reg = coll.register_index("ts_region", columns=[...])   # or a declaration object
# reg -> {"instance_id", "build_id", "start_gen" (B), "status": "building"}

while True:
    st = coll.advance_index("ts_region", budget=500)     # <= 500 units of work
    if st.status == "ready":
        break
```

* `register_index` takes one short transaction. It mints an immutable
  `instance_id` and a resumable `build_id`, records the coverage start
  `B = head`, and returns `building`. It does **no** baseline read or copy.
* `advance_index(name, budget)` performs at most `budget` combined units:
  baseline candidate reads count one each, catch-up journal rows count one
  each. Progress and the work of the step are committed inside one short
  transaction, so the build can be resumed by a different process and no
  write lock outlives the call. After the call another process can always
  publish the next `apply`.
* The return value reports the step's `baseline_candidates_read` /
  `catchup_changes_processed`, cumulative totals, the persisted baseline
  cursor and `processed_seq` / `next_seq`, and on completion `ready_gen` (the
  generation covered at the switch). `status` is `building` or `ready`.
* Advancing an already-`ready` index is a no-op status read.

**Correctness of backfill meeting corrections.** Baseline candidates are
exactly the versions visible in some generation `>= B`: `cgen <= B AND
(dgen IS NULL OR dgen > B)`. Every apply after registration writes, in its own
commit, an `open` / `close` event per touched version into the per-instance
journal. Baseline pagination is a keyset over `(key, rid, cgen)`, and a later
apply only ever *closes* such a version with `dgen > B`, which never changes a
version's key or removes it from the candidate set -- so nothing is skipped.
Versions created after `B` enter only through the journal. The switch to
`ready` commits only when the baseline is complete **and** every journal event
below the current `next_seq` has been folded in, all inside the same write
lock that publishes the ready flag. Thus an early baseline pass can never
overwrite a committed update (opens of new versions are keyed by their own
`cgen`; closes only set `dgen`), a late event cannot resurrect a deleted row
(the close's `dgen` is honoured), and a change to a non-indexed column still
opens a new version whose full main row is returned for its generation.

## Reading through an index

```python
token = coll.begin_snapshot(gen)          # pin the read generation first
page, stats, plan = coll.scan_index_page(
    token, "ts_region",
    {"shape": "prefix_range", "prefix": [["ts", 30]],
     "range_column": "region", "lo": "c", "hi": "c"},
    page_size=100,
)
page.rows                   # [((region, ts, note), b"row-id"), ...] full rows
stats.index_candidates_examined   # secondary entries pulled from storage
stats.invisible_versions_skipped
stats.full_row_lookups            # main-version primary-key locators
stats.rows_delivered              # final deliveries only
stats.explain_query_plan          # SEARCH ... USING PRIMARY KEY on the entries
```

* The snapshot token still selects the generation; the index only changes the
  access order. A token whose generation is below the index's `B` is refused
  with `CoverageError` -- the main table is **never** silently scanned instead.
  A token issued *before* the build finished works once the index is ready, as
  long as its generation is `>= B` and still retained.
* The scan is a concrete BLOB range seek on `(instance_id, index_key)` with
  `ORDER BY index_key, cgen` -- the primary-key order, so it streams without a
  temp sort and never decodes rows outside the intervals.
* Each visible hit is located as an exact main version by the primary-key
  point seek `(main_key, rid, main_cgen)`, so the full row is the row that
  existed at the pinned generation -- never the current live row substituted
  for history.
* A pagination cursor binds, beyond the collection/snapshot/query/page-size
  binding, the index **name**, the immutable **instance id**, the **declaration
  hash**, and the last `(index_key, rid, cgen)`. Another index, a same-name
  rebuild (new instance), the main ordering, or a different page size cannot
  continue it. With `page_size=1` the equal-business-value siblings are still
  delivered by their ascending row_id (the row_id is part of the key), so none
  is skipped.

A brand-new process needs only the collection path, snapshot token, index name,
query and cursor:

```bash
python bytekey_cli.py index-range --collection warehouse.db --name ts_region \
     --token "$SNAP" --query query.json --page-size 100 [--cursor "$CUR"]
```

## Retention, cancellation, deletion

`collect_history(retain_from)` moves the retention boundary monotonically and
now reclaims, with separate counts:

* `entries_removed` -- main versions that lost every reader;
* `secondary_entries_removed` -- secondary history items with the same rule;
* `queue_entries_removed` -- stale journal rows.

A *building* index protects its own inputs automatically: baseline candidates
in the main store and its journal rows survive even if the user raises the
retention boundary mid-build. You do **not** need to keep an extra snapshot
alive for a build. Ready indexes keep only versions an active snapshot can
still read, exactly like the main store. When a build finishes or is
cancelled, its build-only material is released.

* `cancel_index(name)` stops a `building` index and releases its build-only
  material; the name becomes free.
* `delete_index(name)` immediately retires a `building` or `ready` instance:
  it can no longer advance or page, the name is free for a *different*
  declaration, and old cursors (which carry the old instance id) are refused.
  Deleting one index never disturbs another build, the main queries, batch
  receipts or existing snapshots. File size is not required to shrink
  immediately, and there is no automatic index chooser.

## Formats and migration

The collection format is now `bytekey-collection/v2`. An existing v1 file opens
transparently: the first open runs a recoverable in-place upgrade inside one
transaction that only adds the (empty) secondary tables and advances the
marker. Collection identity, `head`, the original receipts and every existing
snapshot token are preserved. If the upgrade is interrupted it simply runs
again on the next open. Static index files are unchanged and stay read-only.

Because applies read the set of `building` / `ready` indexes from the database
*inside their own write transaction*, a writer connection opened **before** a
registration still maintains the new index -- it can never skip maintenance
based on a stale directory.

## Secondary-index API summary

```python
from bytekey import (
    SecondaryDeclaration, IndexColumn,
    IndexDeclarationError, IndexNotFoundError, IndexNotReadyError,
    IndexConflictError, IndexCursorError, CoverageError,
)

coll.register_index(decl_or_mapping, build_id=None)  # -> registration dict
coll.advance_index(name, budget, expected_build_id=None)  # -> IndexBuildStatus
coll.index_status(name) -> IndexBuildStatus
coll.list_indexes(include_deleted=False) -> [IndexInfo]
coll.scan_index_page(token, name, query, page_size, cursor_token=None)
coll.explain_index_scan(name, query, page_size=100)
coll.cancel_index(name); coll.delete_index(name, instance_id=None)
```

## Secondary-index commands

```bash
python bytekey_cli.py index-register --collection warehouse.db --declaration idx.json
python bytekey_cli.py index-advance  --collection warehouse.db --name ts_region \
     --budget 500 [--build-id HEX]
python bytekey_cli.py index-status   --collection warehouse.db [--name ts_region]
python bytekey_cli.py index-explain  --collection warehouse.db --name ts_region --query query.json
python bytekey_cli.py index-range    --collection warehouse.db --name ts_region \
     --token "$SNAP" --query query.json --page-size 100 [--cursor "$CUR"]
python bytekey_cli.py index-cancel   --collection warehouse.db --name ts_region
python bytekey_cli.py index-delete   --collection warehouse.db --name ts_region
```

Errors name the offending index/build or credential and exit nonzero (2).

## Demonstration

`demo/demo_secondary.py` reproduces the mandated fixture from real files
(reopened between steps): a region-priority generation 0, a `ts DESC, region
ASC` index advanced with budget 1, mid-build corrections, and a non-indexed
`note` change, then pages the full rows for the build-start and later
snapshots. It also runs a ts-only index (region change still surfaces), an
operational tour (pre-opened writer, retention during a build, cancel, and a
same-name rebuild that rejects the old cursor), and writes a bounded JSON
report with the query plan and access counts:

```bash
python3 -I demo/demo_secondary.py       # writes demo/secondary_report/report.json
```

## Secondary-index tests

```bash
python3 -I tests/test_secondary_index.py   # mandated fixture, cursors, lifecycle,
                                           # migration, merge invariants, crashes
python3 -I tests/test_secondary_random.py  # randomized multi-index differential
python3 -I tests/test_secondary_large.py   # budget bounds, narrow seek, no rebuild
python3 -I tests/test_collection_cli.py    # CLI index lifecycle and failure exits
```

The randomized suite covers `None`, empty strings, and zero/NUL/0xFF bytes in
the indexed and non-indexed columns, with every ready index drained against
the independent, encoder-free reference model for every generation.


