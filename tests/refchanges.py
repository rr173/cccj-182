"""Independent reference model for the persistent change stream.

This module never imports the production storage, paging or token code.  It
models a collection as a list of plain ``dict[row_id -> tuple]`` states (one
per generation) and derives the expected change feed directly from ordinary
dictionary semantics:

* a generation's event list is the state diff between ``states[g-1]`` and
  ``states[g]`` ordered by ascending ``row_id``;
* an insert has no before image, a delete has no after image, and a move /
  value replacement carries both;
* an empty or all-no-op generation has an empty event list but still exists
  as a generation (a batch boundary a consumer must be able to cross).

Subscription paging, acknowledgement validation and retention are recomputed
here with independent arithmetic (Python ``int``/``dict``), so the tests
compare two implementations rather than re-asserting the production one.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey.schema import Schema  # noqa: E402


class RefChangeError(Exception):
    pass


class RefChangeCollection:
    """Plain-dictionary mirror of generations plus an append-only event log."""

    def __init__(self, schema: Schema):
        self.schema = schema
        # states[g] = {rid: tuple} for generation g (g = 0 is the seed).
        self.states: list[dict[bytes, tuple]] = [{}]
        # events[g] = [(seq, rid, kind, before|None, after|None), ...]
        self.events: list[list[tuple]] = []
        # batch marker per generation: (batch_id, event_count)
        self.batches: dict[int, tuple] = {}
        self.receipts: dict[bytes, tuple] = {}

    @property
    def head(self) -> int:
        return len(self.states) - 1

    def seed(self, rows):
        assert self.head == 0
        self.states[0] = {rid: tuple(vals) for rid, vals in rows}

    def apply(self, batch_id: bytes, expected_head: int, operations) -> int:
        normalized = []
        seen = set()
        for item in operations:
            rid = item["row_id"]
            if rid in seen:
                raise RefChangeError("duplicate row_id in batch")
            seen.add(rid)
            if item["op"] == "delete":
                normalized.append((rid, "d", None))
            else:
                normalized.append(
                    (rid, "p", tuple(item[c.name] for c in self.schema.columns))
                )
        normalized.sort(key=lambda x: x[0])
        canon = tuple(normalized)
        if batch_id in self.receipts:
            old_eh, old_canon, old_head = self.receipts[batch_id]
            if old_eh != expected_head or old_canon != canon:
                raise RefChangeError("idempotency conflict")
            return old_head
        if expected_head != self.head:
            raise RefChangeError("expected_head mismatch")
        previous = dict(self.states[self.head])
        current = dict(previous)
        gen_events = []
        seq = 0
        for rid, kind, vals in canon:
            if kind == "d":
                if rid in current:
                    before = previous[rid]
                    del current[rid]
                    seq += 1
                    gen_events.append((seq, rid, "delete", before, None))
            else:
                if rid not in current:
                    current[rid] = vals
                    seq += 1
                    gen_events.append((seq, rid, "insert", None, vals))
                elif current[rid] != vals:
                    before = previous[rid]
                    current[rid] = vals
                    seq += 1
                    gen_events.append((seq, rid, "update", before, vals))
        self.states.append(current)
        self.events.append(gen_events)
        new_head = self.head
        self.batches[new_head] = (batch_id, seq)
        self.receipts[batch_id] = (expected_head, canon, new_head)
        return new_head

    def events_between(self, start, end):
        """Events at exclusive start (gen, seq) up to inclusive ``end``."""
        out = []
        sg, ss = start
        for gen in range(max(sg, 1), end[0] + 1):
            for ev in self.events[gen - 1]:
                if gen == sg and ev[0] <= ss:
                    continue
                out.append((gen, ev[0], ev[1], ev[2], ev[3], ev[4]))
        return out

    def event_count(self, gen: int) -> int:
        if gen < 1 or gen > self.head:
            return 0
        return len(self.events[gen - 1])


def drain_expected(ref: RefChangeCollection, start_gen: int, page_size: int):
    """Expected page sequence from ``(start_gen, 0)`` with a fixed page size.

    Independent re-derivation of the documented contract: positions are
    ``(gen, seq)``, ``seq == 0`` means nothing of that generation consumed,
    empty generations still appear as batch markers, and a page may end inside
    one generation.  The stream is complete only at ``(head + 1, 0)``.
    """
    pages = []
    pos = (start_gen, 0)
    while True:
        events = []
        markers = []
        gen, seq_after = pos
        stopped_mid = False
        while gen <= ref.head:
            if len(events) >= page_size or len(markers) >= page_size:
                break
            markers.append(gen)
            for ev in ref.events[gen - 1]:
                if ev[0] <= seq_after:
                    continue
                if len(events) >= page_size:
                    stopped_mid = True
                    break
                events.append((gen, ev))
            if stopped_mid:
                break
            gen += 1
            seq_after = 0
        if stopped_mid:
            pos = (gen, events[-1][1][0])
        else:
            pos = (gen, 0)
        more = pos < (ref.head + 1, 0)
        if not more:
            pos = (ref.head + 1, 0)
        pages.append({"events": events, "markers": markers,
                      "pos": pos, "has_more": more})
        if not more:
            return pages
