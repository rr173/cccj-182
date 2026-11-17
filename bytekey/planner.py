"""Compile business range queries into concrete SQLite BLOB scan intervals.

Two query shapes are supported (both validated in memory before the database
is touched):

1. Full-tuple bounds ``lower`` / ``upper``, each independently open or
   closed and each lacking a row_id.  At a boundary, every record sharing the
   business tuple is included or excluded together; row_id never decides
   inclusion, only the delivery order within equal business tuples.

2. Equality prefix over the first *k* columns (null allowed as an equality
   value), an optional range over the very next column whose endpoints must
   be non-null, with every later column unconstrained (so nulls there remain
   visible).  Missing endpoints mean unbounded on that side.  The range is
   always interpreted in the column's natural non-null domain (lo/hi and the
   open/closed flags); asc/desc only changes the delivery order.

Output is a finite list of non-overlapping, byte-wise closed-open intervals
plus a human-readable description of the residual checks a caller would need
(empty here: every predicate, including null exclusion, is compiled into the
bounds rather than applied by decoding rows).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .encoding import (
    TAG_BOOL_FALSE,
    TAG_BOOL_TRUE,
    TAG_BYTES,
    TAG_ROW_ID,
    TAG_TEXT,
    EncodingError,
    _encode_int64,
    _encode_variable,
    _invert,
    encode_component,
)
from .schema import Column, Schema

# The smallest byte string that does NOT start with prefix ``p``.
# Returns None when every extension of p is still a legal key (p is all 0xFF).
_DOMAIN_TOP = bytes([0x07])  # raw-space byte immediately above every value tag


class QueryError(ValueError):
    """Raised in memory for malformed queries / wrong endpoint types."""


@dataclass(frozen=True)
class Interval:
    """Closed-open byte interval; None means unbounded on that side."""

    lower: bytes | None
    upper: bytes | None

    def __post_init__(self) -> None:
        if self.lower is not None and self.upper is not None:
            if self.lower >= self.upper:
                raise QueryError(
                    f"empty/inverted interval: lower {self.lower.hex()} >= "
                    f"upper {self.upper.hex()}"
                )

    def is_empty(self) -> bool:
        return self.lower is not None and self.upper is not None and self.lower >= self.upper

    def sql_clause(self, column: str = "key") -> tuple[str, list[bytes]]:
        clauses: list[str] = []
        params: list[bytes] = []
        if self.lower is not None:
            clauses.append(f"{column} >= ?")
            params.append(self.lower)
        if self.upper is not None:
            clauses.append(f"{column} < ?")
            params.append(self.upper)
        return (" AND ".join(clauses) if clauses else "1=1"), params


@dataclass
class QueryPlan:
    kind: str                      # "full_tuple" or "prefix_range"
    intervals: list[Interval]
    residual_checks: list[str]
    query_spec: dict
    reverse_delivery: bool         # SQLite index walk direction
    prefix_columns: tuple[str, ...] = ()
    range_column: str | None = None

    def explain(self) -> dict:
        return {
            "kind": self.kind,
            "reverse_delivery": self.reverse_delivery,
            "interval_count": len(self.intervals),
            "intervals": [
                {
                    "lower_hex": i.lower.hex() if i.lower is not None else None,
                    "upper_hex": i.upper.hex() if i.upper is not None else None,
                }
                for i in self.intervals
            ],
            "residual_checks": list(self.residual_checks),
            "query_spec": self.query_spec,
            "prefix_columns": list(self.prefix_columns),
            "range_column": self.range_column,
        }


# ---------------------------------------------------------------------------
# Byte helpers
# ---------------------------------------------------------------------------

def prefix_successor(p: bytes) -> bytes | None:
    """Smallest byte string strictly greater than every string with prefix p.

    Returns ``None`` when ``p`` is all ``0xFF``: no finite byte string is
    greater, which the caller treats as "unbounded on that side".
    """
    arr = bytearray(p)
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] != 0xFF:
            arr[i] += 1
            return bytes(arr[: i + 1])
    return None  # p is all 0xFF: no successor inside finite byte strings


def _extend(prefix: bytes, suffix: bytes | None) -> bytes | None:
    """Append ``suffix`` to ``prefix``; ``None`` means that side is unbounded."""
    if suffix is None:
        return None
    return prefix + suffix


def _component_prefix(col: Column, value) -> bytes:
    """Encoded form of a component, asserted to be a prefix of its full key."""
    return encode_component(col, value)


# ---------------------------------------------------------------------------
# Natural (non-null) domain comparison for bound sanity checks
# ---------------------------------------------------------------------------

def natural_compare(a, b) -> int:
    """Compare two non-null scalar values in their natural domain order."""
    if type(a) is not type(b):
        raise QueryError(
            f"cannot compare values of different types: "
            f"{type(a).__name__} vs {type(b).__name__}"
        )
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Query-spec canonicalization (also the cursor's query identity)
# ---------------------------------------------------------------------------

def _canonical_value(value) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"__bytekey_bytes__": bytes(value).hex()}
    if isinstance(value, bool):
        return value
    if isinstance(value, tuple):
        return [_canonical_value(v) for v in value]
    if isinstance(value, list):
        # Plain value lists, plus nested [column, value] prefix pairs whose
        # second element may itself be bytes / null / a scalar.
        return [_canonical_value(v) for v in value]
    if isinstance(value, dict):
        # Already-canonical markers (e.g. produced while descending into a
        # nested prefix list) are stable under re-canonicalization.
        if set(value) == {"__bytekey_bytes__"} and isinstance(
            value["__bytekey_bytes__"], str
        ):
            return dict(value)
        raise QueryError("nested objects are not valid query values")
    if value is None or isinstance(value, (str, int, float)):
        return value
    raise QueryError(f"value of type {type(value).__name__} is not serializable")


def canonical_query_spec(spec: Mapping[str, Any]) -> dict:
    out: dict[str, Any] = {}
    for key in sorted(spec):
        out[key] = _canonical_value(spec[key])
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_endpoint_scalar(col: Column, value, endpoint: str) -> None:
    if value is None:
        raise QueryError(
            f"range endpoint {endpoint!r} on column {col.name!r} must be non-null"
        )
    try:
        # Reuse the encoder's strict type / range / surrogate validation.
        from .encoding import _check_scalar

        _check_scalar(col, value)
    except EncodingError as exc:
        raise QueryError(str(exc)) from None


def _check_full_tuple(schema: Schema, values: Sequence[Any], endpoint: str) -> tuple:
    if not isinstance(values, (list, tuple)):
        raise QueryError(f"{endpoint} must be a list/tuple of column values")
    if len(values) != schema.width:
        raise QueryError(
            f"{endpoint} has {len(values)} values, schema has {schema.width} columns"
        )
    out = []
    for col, value in zip(schema.columns, values):
        try:
            from .encoding import _check_scalar

            _check_scalar(col, value)
        except EncodingError as exc:
            raise QueryError(str(exc)) from None
        out.append(value)
    return tuple(out)


def _encode_full_tuple_prefix(schema: Schema, values: Sequence[Any]) -> bytes:
    return b"".join(
        encode_component(col, value) for col, value in zip(schema.columns, values)
    )


# ---------------------------------------------------------------------------
# Full-tuple bound planning
# ---------------------------------------------------------------------------

def _plan_full_tuple(schema: Schema, query: Mapping[str, Any]) -> QueryPlan:
    lower = query.get("lower")
    upper = query.get("upper")
    lower_inclusive = bool(query.get("lower_inclusive", True))
    upper_inclusive = bool(query.get("upper_inclusive", True))

    # Every real key is "<business-prefix P><row_id component>", where the
    # row_id component starts with tag byte 0x07 and is non-empty/terminated.
    # Hence, writing S = P + 0x07:
    #   * S is a strict prefix of every key in the group;
    #   * the smallest key in the group is S itself followed by the escaped
    #     bytes of the smallest row_id, so key >= S selects the whole group;
    #   * no key in the group is >= successor(P), because every such key
    #     starts with P and successor(P) is the first string past prefix P.
    # These facts hold under DESC inversion because they are purely
    # lexicographic.
    lo_key: bytes | None = None
    hi_key: bytes | None = None
    if lower is not None:
        lo_tuple = _check_full_tuple(schema, lower, "lower")
        p = _encode_full_tuple_prefix(schema, lo_tuple)
        group_start = p + bytes([TAG_ROW_ID])
        lo_key = group_start if lower_inclusive else prefix_successor(p)
    if upper is not None:
        hi_tuple = _check_full_tuple(schema, upper, "upper")
        p = _encode_full_tuple_prefix(schema, hi_tuple)
        group_start = p + bytes([TAG_ROW_ID])
        hi_key = group_start if not upper_inclusive else prefix_successor(p)

    intervals: list[Interval] = []
    if not (lo_key is not None and hi_key is not None and lo_key >= hi_key):
        intervals.append(Interval(lo_key, hi_key))

    spec = canonical_query_spec(
        {
            "shape": "full_tuple",
            "lower": list(lower) if lower is not None else None,
            "upper": list(upper) if upper is not None else None,
            "lower_inclusive": lower_inclusive,
            "upper_inclusive": upper_inclusive,
        }
    )
    return QueryPlan(
        kind="full_tuple",
        intervals=intervals,
        residual_checks=[],
        query_spec=spec,
        reverse_delivery=False,
    )


# ---------------------------------------------------------------------------
# Prefix + next-column range planning
# ---------------------------------------------------------------------------

def _raw_value_prefix(col: Column, value) -> bytes:
    """Raw (ascending-domain) component bytes for a non-null endpoint value."""
    if col.kind == "bool":
        return bytes([TAG_BOOL_TRUE if value else TAG_BOOL_FALSE])
    if col.kind == "int64":
        return _encode_int64(value)
    if col.kind == "text":
        return _encode_variable(TAG_TEXT, value.encode("utf-8", errors="strict"))
    return _encode_variable(TAG_BYTES, bytes(value))


def _domain_window(col: Column) -> tuple[bytes, bytes]:
    """Byte-space [lo, hi) covering exactly the non-null values of a column.

    The window is expressed in *final encoded* byte space.

    ASC: raw value tags occupy 0x02..0x06, so values span ``[0x02, 0x07)``;
    NULL (raw 0x01 or 0x07) sits strictly outside.

    DESC: every byte is bit-inverted, so value tags become 0xF9..0xFD.  The
    inverted NULL tag is 0xFE when nulls_first (raw 0x01) or 0xF8 when
    nulls_last (raw 0x07).  The value window ``[0xF9, 0xFE)`` excludes NULL
    in both placements because 0xF8 lies below it and 0xFE is its open end.
    """
    if col.asc:
        return bytes([TAG_BOOL_FALSE]), _DOMAIN_TOP  # [0x02, 0x07)
    return bytes([0xF9]), bytes([0xFE])


def _plan_prefix_range(schema: Schema, query: Mapping[str, Any]) -> QueryPlan:
    prefix = query.get("prefix")
    if prefix is None:
        prefix = []
    if not isinstance(prefix, (list, tuple)):
        raise QueryError("prefix must be a list of (column, value) pairs")
    k = len(prefix)
    if not (0 <= k < schema.width):
        raise QueryError(
            f"prefix length {k} must be in [0, {schema.width - 1}] "
            f"(a full-tuple equality is a full_tuple query)"
        )

    fixed_bytes = b""
    fixed_names: list[str] = []
    for (item, col) in zip(prefix, schema.columns):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise QueryError("each prefix entry must be a [column, value] pair")
        name, value = item
        if name != col.name:
            raise QueryError(
                f"prefix entry {name!r} does not match declared order; "
                f"expected {col.name!r} at position {len(fixed_names)}"
            )
        try:
            from .encoding import _check_scalar

            _check_scalar(col, value)  # null is a legal equality value here
        except EncodingError as exc:
            raise QueryError(str(exc)) from None
        fixed_bytes += encode_component(col, value)
        fixed_names.append(name)

    range_col = schema.columns[k]
    range_name = query.get("range_column", range_col.name)
    if range_name != range_col.name:
        raise QueryError(
            f"range_column must be the first column after the prefix: "
            f"{range_col.name!r}, got {range_name!r}"
        )

    # A key absent from the query means "unbounded"; an explicit None is an
    # illegal endpoint (nulls are excluded from the range domain).
    lo_present = "lo" in query
    hi_present = "hi" in query
    lo = query["lo"] if lo_present else None
    hi = query["hi"] if hi_present else None
    lo_inclusive = bool(query.get("lo_inclusive", True))
    hi_inclusive = bool(query.get("hi_inclusive", True))

    if lo_present:
        _check_endpoint_scalar(range_col, lo, "lo")
    if hi_present:
        _check_endpoint_scalar(range_col, hi, "hi")
    if lo_present and hi_present:
        cmp = natural_compare(lo, hi)
        if cmp > 0 or (cmp == 0 and not (lo_inclusive and hi_inclusive)):
            raise QueryError("range endpoints define an empty or inverted range")

    # Bounds are built directly in final encoded byte space.  Bit inversion
    # already maps DESC business order onto ascending byte order, so every
    # scan walks storage ASC (reverse_delivery stays False): a larger business
    # value simply has a smaller encoded byte string.  For DESC the business
    # hi therefore becomes the byte-space lower bound.  The openness
    # adjustment (strict successor) is applied to the final bytes so it is
    # always a true lexicographic successor in storage order.
    dom_lo, dom_hi = _domain_window(range_col)

    def _final_prefix(value) -> bytes:
        raw = _raw_value_prefix(range_col, value)
        return raw if range_col.asc else _invert(raw)

    if range_col.asc:
        if not lo_present:
            lower = fixed_bytes + dom_lo
        else:
            p = _final_prefix(lo)
            lower = fixed_bytes + p if lo_inclusive else _extend(
                fixed_bytes, prefix_successor(p)
            )
        if not hi_present:
            upper = fixed_bytes + dom_hi
        else:
            p = _final_prefix(hi)
            upper = _extend(fixed_bytes, prefix_successor(p)) if hi_inclusive \
                else fixed_bytes + p
    else:
        if not hi_present:
            lower = fixed_bytes + dom_lo
        else:
            p = _final_prefix(hi)
            lower = fixed_bytes + p if hi_inclusive else _extend(
                fixed_bytes, prefix_successor(p)
            )
        if not lo_present:
            upper = fixed_bytes + dom_hi
        else:
            p = _final_prefix(lo)
            upper = _extend(fixed_bytes, prefix_successor(p)) if lo_inclusive \
                else fixed_bytes + p
    reverse = False

    intervals: list[Interval] = []
    if not (lower is not None and upper is not None and lower >= upper):
        intervals.append(Interval(lower, upper))

    residual = [
        "none: null exclusion, endpoint openness and prefix equality are all "
        "expressed by the byte bounds; rows outside the intervals are never "
        "decoded"
    ]
    spec = canonical_query_spec(
        {
            "shape": "prefix_range",
            "prefix": [[name, value] for name, value in prefix],
            "range_column": range_col.name,
            "lo": lo,
            "hi": hi,
            "lo_inclusive": lo_inclusive,
            "hi_inclusive": hi_inclusive,
        }
    )
    return QueryPlan(
        kind="prefix_range",
        intervals=intervals,
        residual_checks=residual,
        query_spec=spec,
        reverse_delivery=reverse,
        prefix_columns=tuple(fixed_names),
        range_column=range_col.name,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_ALLOWED_QUERY_FIELDS = {
    "shape",
    "lower",
    "upper",
    "lower_inclusive",
    "upper_inclusive",
    "prefix",
    "range_column",
    "lo",
    "hi",
    "lo_inclusive",
    "hi_inclusive",
}


def plan_query(schema: Schema, query: Mapping[str, Any]) -> QueryPlan:
    """Compile a validated query specification into a byte-scan plan."""
    if not isinstance(query, Mapping):
        raise QueryError("query must be an object")
    unknown = set(query) - _ALLOWED_QUERY_FIELDS
    if unknown:
        raise QueryError(f"unknown query field(s): {sorted(unknown)}")
    shape = query.get("shape")
    if shape == "full_tuple":
        return _plan_full_tuple(schema, query)
    if shape == "prefix_range":
        return _plan_prefix_range(schema, query)
    raise QueryError(
        f"query.shape must be 'full_tuple' or 'prefix_range', got {shape!r}"
    )
