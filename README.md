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
