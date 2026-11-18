"""Independent reference model for updatable collections.

The oracle models every generation as a plain ``dict[row_id -> tuple]`` plus
an explicit per-generation version table.  It deliberately never imports the
production storage or visibility code: the only bytekey pieces it touches are
the *schema declaration* and the existing independent comparator in
``tests/oracle.py`` (which itself never imports the encoder).  Visibility,
deletion, re-creation and query results are therefore derived independently
from ordinary dictionary semantics.

Reference visibility rule (independent of the SQL implementation):
    a row_id is visible at generation g iff it was created at some c <= g
    and has not been deleted at any d with d <= g -- equivalently, the row_id
    is a key of ``states[g]``.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey.schema import Column, Schema  # noqa: E402
from tests.oracle import (  # noqa: E402
    compare_tuples,
    expected_order,
    matches_full_tuple,
    matches_prefix_range,
)

# The genuinely unbounded whole-table query (both tuple bounds absent).  An
# empty prefix_range still constrains its range column to the non-null domain
# per the planner contract, so full-generation drains use this shape.
WHOLE_TABLE_QUERY = {
    "shape": "full_tuple",
    "lower": None,
    "upper": None,
    "lower_inclusive": True,
    "upper_inclusive": True,
}


class RefError(Exception):
    pass


class RefCollection:
    """Plain-dictionary, in-memory mirror of the production collection."""

    def __init__(self, schema: Schema):
        self.schema = schema
        # states[g] = {rid: business tuple} for the whole generation g.
        self.states: list[dict[bytes, tuple]] = [{}]
        # Receipts: batch_id -> (expected_head, canonical op-set, result head).
        self.receipts: dict[bytes, tuple[int, tuple, int]] = {}
        # Snapshots: token -> generation.
        self.snapshots: dict[str, int] = {}
        self.released: set[str] = set()
        self._sid = 0

    @property
    def head(self) -> int:
        return len(self.states) - 1

    def _canonical(self, operations: list[dict]) -> tuple:
        # Order-independent canonical form: sorted by raw row_id.
        normalized = []
        seen = set()
        for item in operations:
            rid = item["row_id"]
            if rid in seen:
                raise RefError("duplicate row_id in batch")
            seen.add(rid)
            if item["op"] == "delete":
                normalized.append((rid, "d", None))
            else:
                vals = tuple(item[c.name] for c in self.schema.columns)
                normalized.append((rid, "p", vals))
        return tuple(sorted(normalized, key=lambda x: x[0]))

    def apply(self, batch_id: bytes, expected_head: int, operations: list[dict]) -> dict:
        canon = self._canonical(operations)
        if batch_id in self.receipts:
            old_eh, old_canon, old_head = self.receipts[batch_id]
            if old_eh != expected_head or old_canon != canon:
                raise RefError("idempotency conflict: batch_id reused with different content")
            return {"head": old_head, "replayed": True}
        if expected_head != self.head:
            raise RefError(
                f"expected_head {expected_head} != current head {self.head}"
            )
        current = dict(self.states[self.head])
        applied = 0
        noop = 0
        for rid, kind, vals in canon:
            if kind == "d":
                if rid in current:
                    del current[rid]
                    applied += 1
                else:
                    noop += 1
            else:
                if rid not in current:
                    current[rid] = vals
                    applied += 1
                elif current[rid] == vals:
                    noop += 1
                else:
                    current[rid] = vals
                    applied += 1
        # Every valid batch -- even an empty / all-noop one -- consumes a gen.
        self.states.append(current)
        new_head = self.head
        self.receipts[batch_id] = (expected_head, canon, new_head)
        return {"head": new_head, "replayed": False, "applied": applied, "noop": noop}

    def seed(self, rows: list[tuple[bytes, tuple]]) -> None:
        """Seed generation 0 directly (mirrors an import / initial dataset)."""
        assert self.head == 0 and not self.states[0]
        self.states[0] = {rid: tuple(vals) for rid, vals in rows}

    def begin_snapshot(self, gen: int | None = None) -> str:
        target = self.head if gen is None else gen
        if not 0 <= target <= self.head:
            raise RefError(f"generation {target} does not exist")
        self._sid += 1
        token = f"ref-snap-{self._sid}"
        self.snapshots[token] = target
        return token

    def release(self, token: str) -> None:
        if token not in self.snapshots or token in self.released:
            raise RefError("unknown or released snapshot token")
        self.released.add(token)

    def snapshot_gen(self, token: str) -> int:
        if token not in self.snapshots or token in self.released:
            raise RefError("unknown or released snapshot token")
        return self.snapshots[token]

    # -- reference queries -------------------------------------------------

    def records_at(self, gen: int) -> list[tuple[tuple, bytes]]:
        state = self.states[gen]
        return expected_order(
            self.schema, [(vals, rid) for rid, vals in state.items()]
        )

    def query_at(self, gen: int, query: dict) -> list[tuple[tuple, bytes]]:
        records = self.records_at(gen)
        return [r for r in records if _matches(self.schema, r[0], query)]

    def query_snapshot(self, token: str, query: dict) -> list[tuple[tuple, bytes]]:
        return self.query_at(self.snapshot_gen(token), query)


def _matches(schema: Schema, values: tuple, query: dict) -> bool:
    shape = query["shape"]
    if shape == "full_tuple":
        return matches_full_tuple(
            schema,
            values,
            tuple(query["lower"]) if query.get("lower") is not None else None,
            tuple(query["upper"]) if query.get("upper") is not None else None,
            bool(query.get("lower_inclusive", True)),
            bool(query.get("upper_inclusive", True)),
        )
    if shape == "prefix_range":
        prefix = [(name, value) for name, value in query.get("prefix", [])]
        range_index = len(prefix)
        lo = query.get("lo")
        hi = query.get("hi")
        return matches_prefix_range(
            schema,
            values,
            prefix,
            range_index,
            lo,
            hi,
            bool(query.get("lo_inclusive", True)),
            bool(query.get("hi_inclusive", True)),
        )
    raise AssertionError(shape)


def drain_snapshot(coll, token: str, query: dict, page_size: int):
    """Page through a production snapshot exactly as a client would."""
    out = []
    cursor = None
    pages = 0
    while True:
        page, stats, plan = coll.scan_snapshot_page(
            token, query, page_size, cursor_token=cursor
        )
        out.extend(page.rows)
        pages += 1
        if page.next_cursor is None:
            return out, pages, plan
        cursor = page.next_cursor


# ---------------------------------------------------------------------------
# Secondary-index reference helpers (independent of production planning)
# ---------------------------------------------------------------------------

def index_projection(schema, declaration):
    """Projected (declared-order) tuples for a full main-schema tuple.

    Only ordering flags are taken from the declaration; the column *kind* and
    the values always come from the main schema.  No encoder is involved.
    """
    pos = {c.name: i for i, c in enumerate(schema.columns)}
    out = []
    for item in declaration["columns"]:
        out.append(_ProjectedValue(
            pos[item["name"]], item["asc"], item["nulls_first"]
        ))
    return out


class _ProjectedValue:
    def __init__(self, index, asc, nulls_first):
        self.index = index
        self.asc = asc
        self.nulls_first = nulls_first


def _projected_order_key(schema, declaration, values, row_id):
    """Sort key over plain Python values, implementing the declaration order."""
    key = []
    for item in declaration["columns"]:
        col = schema.column(item["name"])
        v = values[schema.columns.index(col)]
        key.append(_column_sort_key(item, v))
    key.append(row_id)  # final ascending row_id tie-break
    return tuple(key)


def _column_sort_key(item, value):
    nulls_first = item.get("nulls_first", True)
    asc = item.get("asc", True)
    # Encode placement then ascending/descending comparison without the
    # production encoder: (is_null_rank, value_rank, value_negated_for_desc).
    if value is None:
        return (0 if nulls_first else 2, 0)
    if isinstance(value, bool):
        vkey = 1 if value else 0
    elif isinstance(value, (int, float)):
        vkey = value
    elif isinstance(value, str):
        vkey = value
    elif isinstance(value, (bytes, bytearray)):
        vkey = bytes(value)
    else:
        raise AssertionError(type(value))
    return (1, vkey if asc else _Descending(vkey))


class _Descending:
    """Wrapper reversing comparison only (values never compared across kinds)."""

    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def __eq__(self, other):
        return isinstance(other, _Descending) and self.v == other.v

    def __lt__(self, other):
        return self.v > other.v

    def __gt__(self, other):
        return self.v < other.v

    def __le__(self, other):
        return self.v >= other.v

    def __ge__(self, other):
        return self.v <= other.v

    def __repr__(self):
        return f"D({self.v!r})"


def expected_index_order(schema, declaration, records):
    """Order full (values, rid) records by the declaration, independent code."""
    import functools

    def cmp(a, b):
        ka = _projected_order_key(schema, declaration, a[0], a[1])
        kb = _projected_order_key(schema, declaration, b[0], b[1])
        if ka < kb:
            return -1
        if ka > kb:
            return 1
        return 0

    return sorted(records, key=functools.cmp_to_key(cmp))


def matches_index_query(schema, declaration, values, row_id, query):
    """Reference filter: evaluate a query against the declaration's order.

    ``full_tuple`` bounds are positional in the declaration order and exclude
    row_id; ``prefix_range`` names columns explicitly.  Values outside the
    declared projection are irrelevant, but the returned record is full.
    """
    names = [item["name"] for item in declaration["columns"]]
    projected = [values[schema.columns.index(schema.column(n))] for n in names]
    flags = declaration["columns"]
    if query["shape"] == "full_tuple":
        lower = query.get("lower")
        upper = query.get("upper")
        if lower is not None:
            c = _cmp_projected(flags, projected, lower)
            if c < 0 or (c == 0 and not query.get("lower_inclusive", True)):
                return False
        if upper is not None:
            c = _cmp_projected(flags, projected, upper)
            if c > 0 or (c == 0 and not query.get("upper_inclusive", True)):
                return False
        return True
    if query["shape"] == "prefix_range":
        prefix = query.get("prefix", []) or []
        for i, (name, want) in enumerate(prefix):
            if name not in names or names[i] != name:
                # The planner requires prefix entries to line up with the
                # declaration order; the reference model enforces that too.
                return False
            if not _same_scalar(projected[i], want):
                return False
        k = len(prefix)
        if k >= len(names):
            return False
        range_name = query.get("range_column")
        if range_name is not None and range_name != names[k]:
            return False
        v = projected[k]
        if v is None:  # range column excludes null
            return False
        lo = query.get("lo")
        hi = query.get("hi")
        if lo is not None:
            if v < lo or (v == lo and not query.get("lo_inclusive", True)):
                return False
        if hi is not None:
            if v > hi or (v == hi and not query.get("hi_inclusive", True)):
                return False
        return True
    raise AssertionError(query["shape"])


def _cmp_projected(flags, values, bound):
    for flag, va, vb in zip(flags, values, bound):
        c = _cmp_one(flag, va, vb)
        if c != 0:
            return c
    return 0


def _cmp_one(flag, a, b):
    if a is None and b is None:
        return 0
    if a is None:
        return -1 if flag.get("nulls_first", True) else 1
    if b is None:
        return 1 if flag.get("nulls_first", True) else -1
    c = -1 if a < b else (1 if a > b else 0)
    return c if flag.get("asc", True) else -c


def _same_scalar(a, b):
    if a is None or b is None:
        return a is None and b is None
    return type(a) is type(b) and a == b


def query_index_at(ref, gen: int, declaration: dict, query: dict):
    """Full records at ``gen`` matching ``query`` in declaration order."""
    records = expected_index_order(ref.schema, declaration, ref.records_at(gen))
    return [
        rec for rec in records
        if matches_index_query(ref.schema, declaration, rec[0], rec[1], query)
    ]


def drain_index(coll, token: str, index_name: str, query: dict, page_size: int):
    """Page through a production ready index exactly as a client would."""
    out = []
    cursor = None
    pages = 0
    while True:
        page, stats, plan = coll.scan_index_page(
            token, index_name, query, page_size, cursor_token=cursor
        )
        out.extend(page.rows)
        pages += 1
        if page.next_cursor is None:
            return out, pages, plan
        cursor = page.next_cursor

