"""Targeted edge cases called out by the specification.

* DESC column with prefix-style text values and truncation boundaries;
* open/closed endpoints on a DESC range column (lo/hi in natural values);
* fully unbounded ranges;
* int64 extremes, NUL/0xFF raw bytes, non-ASCII text, empty string vs null;
* full-tuple open/closed bounds against an independent filter.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey import (  # noqa: E402
    INT64_MAX,
    INT64_MIN,
    Column,
    QueryError,
    Schema,
    build_index,
    open_index,
)
from tests.oracle import (  # noqa: E402
    expected_order,
    matches_full_tuple,
    matches_prefix_range,
)


class DescTextTruncationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Text column is DESC; region values include prefix chains.
        self.schema = Schema(
            columns=(Column("name", "text", asc=False, nulls_first=True),)
        )
        values = ["a", "a\x00", "ab", "abc", "b", "", None, "日本", "a"]
        self.rows = [
            {"name": v, "row_id": f"r{i:02d}".encode()}
            for i, v in enumerate(values)
        ]
        # Distinct row_ids keep the duplicate "a" legal.
        self.path = os.path.join(self.tmp.name, "desc.db")
        build_index(self.path, self.schema, self.rows)
        self.index = open_index(self.path)

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def test_desc_text_order_matches_oracle(self):
        pairs = [([r["name"]], r["row_id"]) for r in self.rows]
        # The range domain excludes null by construction; compare against the
        # oracle order restricted to non-null rows.
        non_null = [p for p in pairs if p[0][0] is not None]
        want = expected_order(self.schema, non_null)
        query = {"shape": "prefix_range", "prefix": [], "range_column": "name"}
        got = [(list(v), rid) for v, rid in self.index.iter_all(query, 2)]
        self.assertEqual(got, want)
        self.assertNotIn(None, [v[0] for v, _ in got])

    def test_null_delivered_first_via_full_tuple_domain(self):
        # An exact full-tuple bound on the null group reaches it directly.
        query = {
            "shape": "full_tuple",
            "lower": [None],
            "upper": [None],
            "lower_inclusive": True,
            "upper_inclusive": True,
        }
        got = [v[0] for v, _ in self.index.iter_all(query, 2)]
        self.assertEqual(got, [None])

    def test_desc_text_open_ended_prefix_boundaries(self):
        # Natural-value range over "a".. strictly below "ab"; null excluded.
        query = {
            "shape": "prefix_range",
            "prefix": [],
            "range_column": "name",
            "lo": "a",
            "hi": "ab",
            "hi_inclusive": False,
        }
        got = [v[0] for v, _ in self.index.iter_all(query, 2)]
        # Natural domain matches: "a", "a" (dup), and "a\x00" (prefix of ab
        # chain but distinct), excluding "ab" itself and null.
        oracle = [
            r["name"]
            for r in self.rows
            if matches_prefix_range(
                self.schema, [r["name"]], [], 0, "a", "ab", True, False
            )
        ]
        want_ordered = [
            p[0][0]
            for p in expected_order(
                self.schema, [([r["name"]], r["row_id"]) for r in self.rows]
            )
            if matches_prefix_range(
                self.schema, p[0], [], 0, "a", "ab", True, False
            )
        ]
        self.assertEqual(got, want_ordered)
        self.assertIn("a\x00", got)
        self.assertNotIn("ab", got)
        self.assertNotIn(None, got)
        self.assertEqual(sorted(oracle), sorted(got))


class ExtremeValueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.schema = Schema(
            columns=(
                Column("i", "int64"),
                Column("t", "text"),
                Column("b", "bytes"),
                Column("f", "bool"),
            )
        )
        self.rows = [
            {"i": INT64_MIN, "t": "日本語", "b": b"\x00\xff\x00", "f": True,
             "row_id": b"r1"},
            {"i": INT64_MAX, "t": "", "b": b"", "f": False, "row_id": b"r2"},
            {"i": 0, "t": None, "b": b"\xff" * 8, "f": None, "row_id": b"r3"},
            {"i": -1, "t": "", "b": b"\x00", "f": True, "row_id": b"r4"},
        ]
        self.path = os.path.join(self.tmp.name, "ext.db")
        build_index(self.path, self.schema, self.rows)
        self.index = open_index(self.path)

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def test_int64_extremes_range(self):
        query = {
            "shape": "prefix_range",
            "prefix": [],
            "range_column": "i",
            "lo": INT64_MIN,
            "hi": INT64_MAX,
        }
        got = list(self.index.iter_all(query, 2))
        # All four rows have non-null i; order is ascending int by encoding.
        ints = [v[0] for v, _ in got]
        self.assertEqual(ints, sorted(ints))
        self.assertEqual(set(ints), {INT64_MIN, INT64_MAX, 0, -1})

    def _all_decoded(self):
        # Test-only helper: read every key in storage order and decode.
        raw = [r[0] for r in self.index.conn.execute(
            "SELECT key FROM bytekey_index ORDER BY key"
        )]
        from bytekey import key_decode
        return [key_decode(self.schema, k) for k in raw]

    def test_empty_string_distinct_from_null_and_unbounded(self):
        # Empty string and null must both survive and never collide.
        decoded = self._all_decoded()
        texts = [v[1] for v, _ in decoded]
        self.assertIn(None, texts)
        self.assertIn("", texts)
        self.assertIn("日本語", texts)
        rids = [rid for _, rid in decoded]
        self.assertEqual(len(rids), len(set(rids)))

    def test_nul_and_0xff_bytes_round_trip_through_scan(self):
        got = {v[2] for v, _ in self._all_decoded()}
        self.assertIn(b"\x00\xff\x00", got)
        self.assertIn(b"\xff" * 8, got)
        self.assertIn(b"\x00", got)
        self.assertIn(b"", got)


class FullTupleBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.schema = Schema(
            columns=(
                Column("g", "text"),
                Column("n", "int64", asc=False, nulls_first=False),
            )
        )
        self.rows = [
            {"g": "x", "n": 1, "row_id": b"a"},
            {"g": "x", "n": 1, "row_id": b"b"},
            {"g": "x", "n": 2, "row_id": b"c"},
            {"g": "y", "n": 0, "row_id": b"d"},
        ]
        self.path = os.path.join(self.tmp.name, "fb.db")
        build_index(self.path, self.schema, self.rows)
        self.index = open_index(self.path)
        self.pairs = [([r["g"], r["n"]], r["row_id"]) for r in self.rows]

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def _oracle(self, lower, upper, li, ui):
        return [
            (tuple(p[0]), p[1])
            for p in expected_order(self.schema, self.pairs)
            if matches_full_tuple(self.schema, p[0], lower, upper, li, ui)
        ]

    def test_all_four_open_closed_combinations(self):
        lower = ["x", 1]
        upper = ["x", 2]
        for li in (True, False):
            for ui in (True, False):
                query = {
                    "shape": "full_tuple",
                    "lower": lower,
                    "upper": upper,
                    "lower_inclusive": li,
                    "upper_inclusive": ui,
                }
                got = [(tuple(v), rid) for v, rid in
                       self.index.iter_all(query, 1)]
                want = self._oracle(lower, upper, li, ui)
                self.assertEqual(got, want, msg=f"li={li} ui={ui}")

    def test_open_lower_excludes_whole_row_id_group(self):
        query = {
            "shape": "full_tuple",
            "lower": ["x", 1],
            "upper": ["y", 0],
            "lower_inclusive": False,
            "upper_inclusive": True,
        }
        got = [(tuple(v), rid) for v, rid in self.index.iter_all(query, 1)]
        # The whole (x,1) group with both row_ids is excluded by open lower.
        self.assertNotIn((("x", 1), b"a"), got)
        self.assertNotIn((("x", 1), b"b"), got)
        self.assertEqual(
            got, self._oracle(["x", 1], ["y", 0], False, True)
        )


class UnboundedRangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.schema = Schema(columns=(Column("i", "int64"),))
        self.rows = [
            {"i": v, "row_id": f"{n}".encode()}
            for n, v in enumerate([None, -100, 0, 100, None])
        ]
        self.path = os.path.join(self.tmp.name, "u.db")
        build_index(self.path, self.schema, self.rows)
        self.index = open_index(self.path)

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def test_no_endpoints_is_full_non_null_domain(self):
        query = {"shape": "prefix_range", "prefix": [], "range_column": "i"}
        got = [v[0] for v, _ in self.index.iter_all(query, 2)]
        self.assertEqual(got, [-100, 0, 100])

    def test_only_lo_and_only_hi(self):
        q_lo = {"shape": "prefix_range", "prefix": [],
                "range_column": "i", "lo": 0}
        self.assertEqual(
            [v[0] for v, _ in self.index.iter_all(q_lo, 2)], [0, 100]
        )
        q_hi = {"shape": "prefix_range", "prefix": [],
                "range_column": "i", "hi": 0, "hi_inclusive": False}
        self.assertEqual(
            [v[0] for v, _ in self.index.iter_all(q_hi, 2)], [-100]
        )

    def test_empty_open_range_rejected(self):
        with self.assertRaises(QueryError):
            self.index.scan_page(
                {"shape": "prefix_range", "prefix": [],
                 "range_column": "i", "lo": 5, "hi": 5,
                 "lo_inclusive": False, "hi_inclusive": False},
                5,
            )


class AllFFSuccessorTests(unittest.TestCase):
    def test_prefix_successor_all_ff_means_unbounded(self):
        from bytekey.planner import Interval, _extend, prefix_successor

        self.assertIsNone(prefix_successor(b"\xff\xff"))
        self.assertIsNone(_extend(b"\x05", prefix_successor(b"\xff")))
        # Unbounded sides must produce valid SQL rather than crashing.
        self.assertEqual(Interval(None, None).sql_clause(), ("1=1", []))
        clause, params = Interval(b"\x01", None).sql_clause()
        self.assertEqual(clause, "key >= ?")
        self.assertEqual(params, [b"\x01"])

    def test_desc_open_bound_near_inverted_ff_does_not_crash(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            schema = Schema(
                columns=(Column("b", "bytes", asc=False, nulls_first=True),)
            )
            rows = [
                {"b": b"\x00", "row_id": b"r1"},
                {"b": b"\x01", "row_id": b"r2"},
                {"b": None, "row_id": b"r3"},
            ]
            path = os.path.join(tmp.name, "ff.db")
            build_index(path, schema, rows)
            index = open_index(path)
            query = {
                "shape": "prefix_range",
                "prefix": [],
                "range_column": "b",
                "lo": b"\x00",
                "lo_inclusive": False,
            }
            got = [v[0] for v, _ in index.iter_all(query, 2)]
            # Only b"\x01" is strictly greater than b"\x00"; null excluded.
            self.assertEqual(got, [b"\x01"])
            index.close()
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
