"""bytekey: composite-key byte-order index library backed by SQLite BLOB primary keys.

Public API:
    Schema, Column
    key_encode, key_decode
    build_index, open_index, Index
    plan_query, Interval, QueryPlan
    encode_cursor, decode_cursor
    QueryError, EncodingError, CursorError, BuildError
"""
from .schema import Schema, Column
from .encoding import (
    EncodingError,
    key_encode,
    key_decode,
    encode_component,
    decode_component,
    INT64_MIN,
    INT64_MAX,
)
from .planner import (
    QueryError,
    Interval,
    QueryPlan,
    plan_query,
    canonical_query_spec,
    natural_compare,
)
from .cursor import CursorError, encode_cursor, decode_cursor
from .db import (
    BuildError,
    Index,
    build_index,
    open_index,
    Page,
)

__all__ = [
    "Schema",
    "Column",
    "EncodingError",
    "QueryError",
    "CursorError",
    "BuildError",
    "Interval",
    "QueryPlan",
    "Page",
    "Index",
    "build_index",
    "open_index",
    "key_encode",
    "key_decode",
    "encode_component",
    "decode_component",
    "plan_query",
    "canonical_query_spec",
    "natural_compare",
    "encode_cursor",
    "decode_cursor",
    "INT64_MIN",
    "INT64_MAX",
]
