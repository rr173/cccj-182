"""Independent business comparator used as the test oracle.

This module never imports the encoder: ordering expectations are derived
directly from the declared schema and plain Python value comparisons.
"""
from __future__ import annotations

from typing import Any, Sequence

from bytekey.schema import Column, Schema


def _natural_cmp(a: Any, b: Any) -> int:
    """Compare two non-null scalar values in their natural domain."""
    assert a is not None and b is not None
    # bool stays distinct from int: every declared column has a fixed kind,
    # so cross-kind comparisons never occur in generated test data.
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def compare_tuples(
    schema: Schema,
    a: Sequence[Any],
    a_row_id: bytes,
    b: Sequence[Any],
    b_row_id: bytes,
) -> int:
    """Full business comparison, including the row_id tie-break."""
    for col, va, vb in zip(schema.columns, a, b):
        c = _compare_column(col, va, vb)
        if c != 0:
            return c
    # Final tie-break is always row_id ascending, regardless of column dirs.
    if a_row_id < b_row_id:
        return -1
    if a_row_id > b_row_id:
        return 1
    return 0


def _compare_column(col: Column, a: Any, b: Any) -> int:
    if a is None and b is None:
        return 0
    if a is None:
        return -1 if col.nulls_first else 1
    if b is None:
        return 1 if col.nulls_first else -1
    c = _natural_cmp(a, b)
    if c == 0:
        return 0
    return c if col.asc else -c


def expected_order(schema: Schema, records: list[tuple[Sequence[Any], bytes]]):
    """Return records sorted by the independent comparator."""
    import functools

    return sorted(
        records,
        key=functools.cmp_to_key(
            lambda x, y: compare_tuples(schema, x[0], x[1], y[0], y[1])
        ),
    )


def matches_full_tuple(
    schema: Schema,
    values: Sequence[Any],
    lower: Sequence[Any] | None,
    upper: Sequence[Any] | None,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> bool:
    """Reference full-tuple bound filter (row_id excluded from bounds)."""
    if lower is not None:
        c = _tuple_cmp_no_rowid(schema, values, lower)
        if c < 0 or (c == 0 and not lower_inclusive):
            return False
    if upper is not None:
        c = _tuple_cmp_no_rowid(schema, values, upper)
        if c > 0 or (c == 0 and not upper_inclusive):
            return False
    return True


def _tuple_cmp_no_rowid(schema: Schema, a: Sequence[Any], b: Sequence[Any]) -> int:
    for col, va, vb in zip(schema.columns, a, b):
        c = _compare_column(col, va, vb)
        if c != 0:
            return c
    return 0


def matches_prefix_range(
    schema: Schema,
    values: Sequence[Any],
    prefix: list[tuple[str, Any]],
    range_index: int,
    lo: Any,
    hi: Any,
    lo_inclusive: bool,
    hi_inclusive: bool,
) -> bool:
    """Reference prefix + next-column range filter, null-aware."""
    for idx, (_name, want) in enumerate(prefix):
        if not _scalar_equal(values[idx], want):
            return False
    v = values[range_index]
    if v is None:  # range column always excludes null
        return False
    col = schema.columns[range_index]
    if lo is not None:
        c = _natural_cmp(v, lo)
        if c < 0 or (c == 0 and not lo_inclusive):
            return False
    if hi is not None:
        c = _natural_cmp(v, hi)
        if c > 0 or (c == 0 and not hi_inclusive):
            return False
    # Later columns are unconstrained (nulls included) -- nothing to check.
    return True


def _scalar_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return type(a) is type(b) and a == b
