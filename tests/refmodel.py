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
