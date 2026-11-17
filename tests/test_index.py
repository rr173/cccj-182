"""End-to-end index tests including the mandated hand-written fixture.

Fixture rows (region asc, time desc nulls_first):
    (a, 2), (a, 1), (a, null), region "", region "a\\x00", prefix "ab",
    with three different row_ids sharing one business tuple.
Every full-range scan is drained with page_size=1 and checked line by line
against the comparator that never uses the encoder.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey import (  # noqa: E402
    BuildError,
    Column,
    CursorError,
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


def fixture_schema():
    return Schema(
        columns=(
            Column("region", "text", asc=True, nulls_first=True),
            Column("ts", "int64", asc=False, nulls_first=True),
        )
    )


def fixture_records():
    """Return (row dicts, list of (values, row_id)) with known contents."""
    rows = [
        {"region": "a", "ts": 2, "row_id": b"r-01"},
        {"region": "a", "ts": 1, "row_id": b"r-02"},
        {"region": "a", "ts": None, "row_id": b"r-03"},
        {"region": "", "ts": 1, "row_id": b"r-04"},
        {"region": "a\x00", "ts": 1, "row_id": b"r-05"},
        {"region": "ab", "ts": 1, "row_id": b"r-06"},
        # Same business tuple, three row_ids: tie-break must be ascending.
        {"region": "tie", "ts": 7, "row_id": b"z"},
        {"region": "tie", "ts": 7, "row_id": b"m"},
        {"region": "tie", "ts": 7, "row_id": b"a"},
    ]
    pairs = [([r["region"], r["ts"]], r["row_id"]) for r in rows]
    return rows, pairs


class HandWrittenFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "fixture.db")
        self.schema = fixture_schema()
        rows, self.pairs = fixture_records()
        build_index(self.path, self.schema, rows)
        self.index = open_index(self.path)

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def test_hand_written_full_order_page_size_one(self):
        want = expected_order(self.schema, self.pairs)
        query = {"shape": "prefix_range", "prefix": [], "range_column": "region"}
        got = []
        token = None
        page_no = 0
        while True:
            page, stats, plan = self.index.scan_page(query, 1, cursor_token=token)
            page_no += 1
            self.assertEqual(stats.rows_returned, len(page.rows))
            self.assertTrue(stats.used_index)
            got.extend((list(v), rid) for v, rid in page.rows)
            token = page.next_cursor
            if token is None:
                break
        self.assertEqual(page_no, len(want))  # exactly one row per page
        self.assertEqual(got, want)

        # Explicit hand-written sequence, region asc then (null, 2, 1) on ts.
        hand = [
            (["", 1], b"r-04"),
            (["a", None], b"r-03"),
            (["a", 2], b"r-01"),
            (["a", 1], b"r-02"),
            (["a\x00", 1], b"r-05"),
            (["ab", 1], b"r-06"),
            (["tie", 7], b"a"),
            (["tie", 7], b"m"),
            (["tie", 7], b"z"),
        ]
        self.assertEqual(got, hand)

    def test_equal_business_tuple_row_ids_not_skipped_or_duplicated(self):
        query = {
            "shape": "prefix_range",
            "prefix": [["region", "tie"]],
            "range_column": "ts",
            "lo": 7,
            "hi": 7,
        }
        seen = []
        token = None
        cursors = []
        while True:
            page, _stats, _plan = self.index.scan_page(
                query, 1, cursor_token=token
            )
            seen.extend(page.rows)
            cursors.append(page.next_cursor)
            token = page.next_cursor
            if token is None:
                break
        self.assertEqual([rid for _, rid in seen], [b"a", b"m", b"z"])
        self.assertIsNone(cursors[-1])
    def test_cursor_does_not_repeat_at_range_endpoint(self):
        # Upper endpoint equals an existing business key; inclusive bound.
        query = {
            "shape": "prefix_range",
            "prefix": [["region", "a"]],
            "range_column": "ts",
            "lo": 1,
            "hi": 2,
        }
        seen = []
        token = None
        while True:
            page, _stats, _plan = self.index.scan_page(
                query, 1, cursor_token=token
            )
            seen.extend((tuple(v), rid) for v, rid in page.rows)
            token = page.next_cursor
            if token is None:
                break
        self.assertEqual(
            seen,
            [
                (("a", 2), b"r-01"),
                (("a", 1), b"r-02"),
            ],
        )

    def test_open_bounds_exclude_whole_business_group(self):
        query = {
            "shape": "full_tuple",
            "lower": ["a", 1],
            "upper": ["tie", 7],
            "lower_inclusive": False,
            "upper_inclusive": False,
        }
        got = [(tuple(v), rid) for v, rid in self.index.iter_all(query, 2)]
        want_pairs = [
            (tuple(p[0]), p[1])
            for p in expected_order(self.schema, self.pairs)
            if matches_full_tuple(
                self.schema, p[0], ["a", 1], ["tie", 7], False, False
            )
        ]
        self.assertEqual(got, want_pairs)
        # Explicitly: (a,1) group and the whole (tie,7) group are excluded.
        self.assertNotIn((("a", 1), b"r-02"), got)
        self.assertFalse(any(tuple(v) == ("tie", 7) for v, _ in got))

    def test_full_tuple_closed_bounds_include_groups(self):
        query = {
            "shape": "full_tuple",
            "lower": ["a", 1],
            "upper": ["tie", 7],
            "lower_inclusive": True,
            "upper_inclusive": True,
        }
        got = list(self.index.iter_all(query, 3))
        self.assertIn((("a", 1), b"r-02"), got)
        rids = [rid for v, rid in got if tuple(v) == ("tie", 7)]
        self.assertEqual(rids, [b"a", b"m", b"z"])

    def test_null_equality_prefix_vs_null_excluded_range(self):
        # An explicit null range endpoint is rejected before any scan.
        with self.assertRaises(QueryError):
            self.index.scan_page(
                {"shape": "prefix_range", "prefix": [["region", "a"]],
                 "range_column": "ts", "lo": None},
                5,
            )
        # A range with no endpoints is unbounded but still excludes null.
        q_eq = {
            "shape": "prefix_range",
            "prefix": [["region", "a"]],
            "range_column": "ts",
        }
        got_eq = list(self.index.iter_all(q_eq, 5))
        self.assertNotIn(None, [v[1] for v, _ in got_eq])
        self.assertEqual(
            [tuple(v) for v, _ in got_eq],
            [("a", 2), ("a", 1)],
        )

        q_null = {
            "shape": "full_tuple",
            "lower": ["a", None],
            "upper": ["a", None],
            "lower_inclusive": True,
            "upper_inclusive": True,
        }
        got_null = list(self.index.iter_all(q_null, 5))
        self.assertEqual(got_null, [(("a", None), b"r-03")])

    def test_unbounded_and_unconstrained_later_columns(self):
        query = {
            "shape": "prefix_range",
            "prefix": [["region", "a"]],
            "range_column": "ts",
        }
        got = list(self.index.iter_all(query, 2))
        # ts is DESC nulls_first, but nulls are excluded from the range
        # domain; non-null values deliver high to low.
        self.assertEqual(
            [tuple(v) for v, _ in got],
            [("a", 2), ("a", 1)],
        )

    def test_query_validation_happens_before_scan(self):
        with self.assertRaises(QueryError):
            self.index.scan_page(
                {"shape": "prefix_range", "prefix": [],
                 "range_column": "region", "lo": 123},  # wrong type
                5,
            )
        with self.assertRaises(QueryError):
            self.index.scan_page(
                {"shape": "prefix_range", "prefix": [],
                 "range_column": "region", "lo": None},  # null endpoint
                5,
            )
        with self.assertRaises(QueryError):
            self.index.scan_page({"shape": "nope"}, 5)
        with self.assertRaises(QueryError):
            self.index.scan_page(
                {"shape": "full_tuple", "lower": ["a"], "upper": ["a", 1]}, 5
            )
        with self.assertRaises(QueryError):
            self.index.scan_page(
                {"shape": "prefix_range", "prefix": [],
                 "range_column": "nope"}, 5
            )
        with self.assertRaises(QueryError):
            self.index.scan_page(
                {"shape": "prefix_range", "prefix": [], "range_column": "region",
                 "bogus": 1}, 5
            )


class BuildSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.schema = fixture_schema()

    def tearDown(self):
        self.tmp.cleanup()

    def test_completed_index_is_never_overwritten(self):
        path = os.path.join(self.tmp.name, "x.db")
        rows, _ = fixture_records()
        build_index(path, self.schema, rows)
        with self.assertRaises(BuildError):
            build_index(path, self.schema, rows)
        # Original is intact and readable.
        with open_index(path) as idx:
            self.assertEqual(idx.row_count, len(rows))

    def test_bad_source_aborts_and_leaves_no_index(self):
        path = os.path.join(self.tmp.name, "bad.db")
        good = [{"region": "a", "ts": 1, "row_id": b"r1"}]
        bad = [
            {"region": "a", "ts": 1, "row_id": b"r1"},
            {"region": "a\x00", "ts": "not-an-int", "row_id": b"r2"},
        ]
        with self.assertRaises(BuildError):
            build_index(path, self.schema, iter(bad))
        self.assertFalse(os.path.exists(path))
        leftovers = [f for f in os.listdir(self.tmp.name) if f.startswith(".bytekey")]
        self.assertEqual(leftovers, [])
        # The same target path can be used once a clean source is supplied.
        build_index(path, self.schema, iter(good))
        self.assertTrue(os.path.exists(path))

    def test_duplicate_row_id_fails_build(self):
        path = os.path.join(self.tmp.name, "dup.db")
        rows = [
            {"region": "a", "ts": 1, "row_id": b"same"},
            {"region": "b", "ts": 2, "row_id": b"same"},
        ]
        with self.assertRaises(BuildError):
            build_index(path, self.schema, iter(rows))
        self.assertFalse(os.path.exists(path))

    def test_empty_row_id_fails_build(self):
        path = os.path.join(self.tmp.name, "e.db")
        with self.assertRaises(BuildError):
            build_index(
                path,
                self.schema,
                iter([{"region": "a", "ts": 1, "row_id": b""}]),
            )


class CursorSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.schema = fixture_schema()
        rows, _ = fixture_records()
        self.path1 = os.path.join(self.tmp.name, "a.db")
        self.path2 = os.path.join(self.tmp.name, "b.db")
        build_index(self.path1, self.schema, rows)
        build_index(self.path2, self.schema, rows)
        self.query = {
            "shape": "prefix_range",
            "prefix": [],
            "range_column": "region",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _cursor_from(self, path, page_size=2):
        idx = open_index(path)
        page, _stats, _ = idx.scan_page(self.query, page_size)
        idx.close()
        return page.next_cursor

    def test_cursor_rejected_on_other_index(self):
        token = self._cursor_from(self.path1)
        other = open_index(self.path2)
        with self.assertRaises(CursorError):
            other.scan_page(self.query, 2, cursor_token=token)
        other.close()

    def test_cursor_rejected_on_other_query(self):
        token = self._cursor_from(self.path1)
        idx = open_index(self.path1)
        other_query = dict(self.query, hi="z")
        with self.assertRaises(CursorError):
            idx.scan_page(other_query, 2, cursor_token=token)
        idx.close()

    def test_cursor_rejected_on_page_size_change(self):
        token = self._cursor_from(self.path1, page_size=2)
        idx = open_index(self.path1)
        with self.assertRaises(CursorError):
            idx.scan_page(self.query, 3, cursor_token=token)
        idx.close()

    def test_tampered_cursor_rejected(self):
        token = self._cursor_from(self.path1)
        idx = open_index(self.path1)
        body, sig = token.split(".")
        with self.assertRaises(CursorError):
            idx.scan_page(self.query, 2, cursor_token=body + "." + sig[::-1])
        with self.assertRaises(CursorError):
            idx.scan_page(self.query, 2, cursor_token="garbage")
        idx.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
