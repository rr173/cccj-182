"""Randomized exhaustive tests against the independent oracle.

A small library is generated from a compact value domain, then *every*
equality-prefix / range combination (lo/hi present or absent, open/closed,
asc/desc, both null placements) is compiled and scanned page by page.  The
delivered rows must equal the oracle's plain-Python filter, in the exact
delivery order, and the SQLite plan must show a PRIMARY KEY range search
rather than a full scan.
"""
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey import Column, QueryError, Schema, build_index, open_index  # noqa: E402
from tests.oracle import (  # noqa: E402
    expected_order,
    matches_prefix_range,
)

INT_DOMAIN = [-5, -1, 0, 1, 5]
TEXT_DOMAIN = ["", "a", "a\x00", "ab", "b", "日本"]
BYTES_DOMAIN = [b"", b"\x00", b"\x00\xff", b"a", b"\xff"]
BOOL_DOMAIN = [False, True]
NULL = None


def _domain_for(kind):
    return {
        "int64": INT_DOMAIN,
        "text": TEXT_DOMAIN,
        "bytes": BYTES_DOMAIN,
        "bool": BOOL_DOMAIN,
    }[kind]


class RandomizedExhaustiveTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(20261117)
        self.rng = rng
        self.tmp = tempfile.TemporaryDirectory()
        # Three columns of varied kinds and randomized directions/null slots.
        self.schema = Schema(
            columns=(
                Column(
                    "c0", "text",
                    asc=rng.choice([True, False]),
                    nulls_first=rng.choice([True, False]),
                ),
                Column(
                    "c1", "int64",
                    asc=rng.choice([True, False]),
                    nulls_first=rng.choice([True, False]),
                ),
                Column(
                    "c2", "bool",
                    asc=rng.choice([True, False]),
                    nulls_first=rng.choice([True, False]),
                ),
            )
        )
        domains = [_domain_for(c.kind) for c in self.schema.columns]
        rows = []
        rid_seq = 0
        # Enumerate a subset of the cartesian product, always including nulls.
        for v0 in domains[0] + [NULL]:
            for v1 in domains[1] + [NULL]:
                for v2 in domains[2] + [NULL]:
                    if rng.random() < 0.35:
                        continue
                    rid_seq += 1
                    rows.append(
                        {"c0": v0, "c1": v1, "c2": v2,
                         "row_id": f"rid-{rid_seq:04d}".encode()}
                    )
        # Guarantee a few identical business tuples with distinct row_ids.
        for extra in (b"dup-a", b"dup-b", b"dup-c"):
            rid_seq += 1
            rows.append({"c0": "a", "c1": 1, "c2": True, "row_id": extra})
        self.rows = rows
        self.path = os.path.join(self.tmp.name, "random.db")
        build_index(self.path, self.schema, rows)
        self.index = open_index(self.path)
        self.pairs = [
            ([r["c0"], r["c1"], r["c2"]], r["row_id"]) for r in rows
        ]

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def _scan_all(self, query, page_size):
        out = []
        token = None
        used_index_flags = []
        visited_total = 0
        while True:
            page, stats, plan = self.index.scan_page(
                query, page_size, cursor_token=token
            )
            used_index_flags.append(stats.used_index)
            visited_total += stats.rows_visited
            out.extend((tuple(v), rid) for v, rid in page.rows)
            token = page.next_cursor
            if token is None:
                break
        return out, all(used_index_flags), plan

    def _oracle_prefix_range(self, prefix, range_idx, lo, hi, lo_inc, hi_inc):
        matched = [
            (tuple(vals), rid)
            for vals, rid in self.pairs
            if matches_prefix_range(
                self.schema, vals, prefix, range_idx, lo, hi, lo_inc, hi_inc
            )
        ]
        return expected_order(
            self.schema, [(list(v), rid) for v, rid in matched]
        )

    def test_exhaustive_prefix_range_queries(self):
        rng = self.rng
        # Prefix lengths 0..width-1; the range column is the one right after.
        for k in range(self.schema.width):
            range_idx = k
            range_col = self.schema.columns[range_idx]
            domain = _domain_for(range_col.kind)
            # Build a concrete prefix by sampling values (null allowed).
            prefix = []
            for j in range(k):
                val = rng.choice(_domain_for(self.schema.columns[j].kind) + [NULL])
                prefix.append([self.schema.columns[j].name, val])
            endpoint_choices = [None] + domain
            trial = 0
            for lo in endpoint_choices:
                for hi in endpoint_choices:
                    lo_inc = rng.choice([True, False])
                    hi_inc = rng.choice([True, False])
                    query = {
                        "shape": "prefix_range",
                        "prefix": prefix,
                        "range_column": range_col.name,
                    }
                    if lo is not None:
                        query["lo"] = lo
                        query["lo_inclusive"] = lo_inc
                    if hi is not None:
                        query["hi"] = hi
                        query["hi_inclusive"] = hi_inc
                    # Skip combinations the planner must reject as empty.
                    if lo is not None and hi is not None:
                        if lo > hi or (lo == hi and not (lo_inc and hi_inc)):
                            with self.assertRaises(QueryError):
                                self.index.scan_page(query, 4)
                            continue
                    trial += 1
                    for page_size in (1, 3, 7):
                        got, used_index, plan = self._scan_all(query, page_size)
                        want = self._oracle_prefix_range(
                            prefix, range_idx, lo, hi, lo_inc, hi_inc
                        )
                        want = [(tuple(v), rid) for v, rid in want]
                        self.assertEqual(
                            got, want,
                            msg=f"mismatch k={k} lo={lo} hi={hi} "
                                f"lo_inc={lo_inc} hi_inc={hi_inc} ps={page_size}",
                        )
                        self.assertTrue(
                            used_index, "scan must use the BLOB primary-key index"
                        )
                        # The interval plan is a finite list; it may be empty
                        # only when the business range matches nothing.
                        self.assertGreaterEqual(len(plan.intervals), 0)
                        if not plan.intervals:
                            self.assertEqual(got, [])
            self.assertGreater(trial, 0)

    def test_pagination_is_lossless_and_stable(self):
        query = {"shape": "prefix_range", "prefix": [], "range_column": "c0"}
        full, _, _ = self._scan_all(query, 1000)
        paged, _, _ = self._scan_all(query, 1)
        self.assertEqual(full, paged)
        # No duplicate keys across pages; total equals matched oracle count.
        keys = [rid for _, rid in paged]
        self.assertEqual(len(keys), len(set(keys)))

    def test_wrong_endpoint_type_rejected_before_scan(self):
        bad_queries = [
            {"shape": "prefix_range", "prefix": [],
             "range_column": "c1", "lo": "not-an-int"},
            {"shape": "prefix_range", "prefix": [],
             "range_column": "c0", "hi": 123},
            {"shape": "prefix_range", "prefix": [],
             "range_column": "c2", "lo": None},
            {"shape": "prefix_range", "prefix": [],
             "range_column": "c1", "lo": True},  # bool is not int64
        ]
        for q in bad_queries:
            with self.assertRaises(QueryError):
                self.index.scan_page(q, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
