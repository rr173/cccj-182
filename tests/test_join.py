"""Two-source equi-join: mandated fixture, modes, null rules, lifecycle.

Covers the hand-written mandated fixture (page size 1, reopened from real
files between pages), inner/left/full modes, the default and nulls_equal
rules, self-joins, empty sides, duplicate key groups spanning many pages,
zero-length bytes, credential misuse, source-eligibility failures (released
token, deleted/replaced index, replaced collection file), real process
kills during prepare, and close/cleanup semantics.
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey import (  # noqa: E402
    Column,
    IndexColumn,
    Schema,
    SecondaryDeclaration,
    create_collection,
    open_collection,
)
from bytekey.join import (  # noqa: E402
    JoinSource,
    JoinSourceError,
    JoinStateError,
    JoinCredentialError,
    JoinDeclarationError,
    join_cleanup,
    join_status,
    open_join,
    prepare_join,
    terminate_join,
)
from bytekey.secondary import CoverageError, IndexNotReadyError  # noqa: E402

ALLQ = {"shape": "full_tuple", "lower": None, "upper": None}

# ---------------------------------------------------------------------------
# Mandated fixture schemas (left: region/sku/qty; right: product/area/label)
# ---------------------------------------------------------------------------

LSCHEMA = Schema(columns=(
    Column("region", "text", asc=True, nulls_first=True),
    Column("sku", "text", asc=False, nulls_first=True),
    Column("qty", "int64", asc=True, nulls_first=True),
))
RSCHEMA = Schema(columns=(
    Column("product", "text", asc=False, nulls_first=True),
    Column("area", "text", asc=True, nulls_first=True),
    Column("label", "text", asc=True, nulls_first=True),
))

LROWS = [
    (b"l1", ("a", "x", 2)),
    (b"l2", ("a", "x", 3)),
    (b"l3", ("b", "y", 1)),
    (b"l4", (None, "z", 9)),
]
RROWS = [
    (b"r1", ("x", "a", "old")),
    (b"r2", ("x", "a", "new")),
    (b"r3", ("q", "c", "only")),
    (b"r4", ("z", None, "nil")),
]
MAPPING = [("region", "area"), ("sku", "product")]


def _seed_collection(path, schema, rows):
    create_collection(path, schema)
    with open_collection(path) as coll:
        coll.apply(b"seed", 0, iter([
            {"op": "put", "row_id": rid,
             **{c.name: v for c, v in zip(schema.columns, values)}}
            for rid, values in rows
        ]))


def _snapshot(path, gen=None):
    with open_collection(path) as coll:
        return coll.begin_snapshot(gen)


def _row(kind, left=None, right=None):
    return (kind, left, right)


def _v(side):
    return None if side is None else (side.values, side.row_id)


def drain(work, handle, reopen_each_page=True):
    """Drain a join read page by page; optionally reopen the handle per page."""
    rows = []
    pages = 0
    cursor = None
    join = None
    while True:
        if reopen_each_page or join is None:
            join = open_join(work, handle)
        page, stats = join.page(cursor)
        pages += 1
        rows.extend(_row(r.kind, _v(r.left), _v(r.right)) for r in page.rows)
        cursor = page.next_cursor
        if reopen_each_page:
            join.close()
            join = None
        if cursor is None:
            if join is not None:
                join.close()
            return rows, pages


class MandatedFixtureTests(unittest.TestCase):
    """The exact fixture from the task statement, checked against a
    hand-written expected stream, paged one row at a time from real files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lpath = os.path.join(self.tmp.name, "left.db")
        self.rpath = os.path.join(self.tmp.name, "right.db")
        _seed_collection(self.lpath, LSCHEMA, LROWS)
        _seed_collection(self.rpath, RSCHEMA, RROWS)
        self.ltok = _snapshot(self.lpath)
        self.rtok = _snapshot(self.rpath)
        self.work = os.path.join(self.tmp.name, "join.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _prepare(self, mode, nulls_equal=False, page_size=1, work=None):
        return prepare_join(
            work or self.work,
            left=JoinSource(self.lpath, self.ltok, ALLQ),
            right=JoinSource(self.rpath, self.rtok, ALLQ),
            mapping=MAPPING, mode=mode, nulls_equal=nulls_equal,
            page_size=page_size,
        )

    # Hand-written expected streams.  Left path order (region ASC nulls
    # first, sku DESC, qty ASC, row_id): l4, l1, l2, l3.  Right path order
    # (product DESC, area ASC nulls first, label ASC, row_id): r4, r2, r1, r3.
    L = {rid: vals for rid, vals in LROWS}
    R = {rid: vals for rid, vals in RROWS}

    def _exp_matched(self, lrid, rrid):
        return ("matched", (self.L[lrid], lrid), (self.R[rrid], rrid))

    def _exp_left_only(self, lrid):
        return ("left_only", (self.L[lrid], lrid), None)

    def _exp_right_only(self, rrid):
        return ("right_only", None, (self.R[rrid], rrid))

    def test_full_default_nulls_page_size_1(self):
        result = self._prepare("full", nulls_equal=False, page_size=1)
        expected = [
            self._exp_left_only(b"l4"),       # null region: no match (default)
            self._exp_matched(b"l1", b"r2"),  # l1's matches in right path order
            self._exp_matched(b"l1", b"r1"),
            self._exp_matched(b"l2", b"r2"),
            self._exp_matched(b"l2", b"r1"),
            self._exp_left_only(b"l3"),
            self._exp_right_only(b"r4"),      # null area: unmatched right row
            self._exp_right_only(b"r3"),      # key q/c present only on right
        ]
        rows, pages = drain(self.work, result["handle_token"])
        self.assertEqual(rows, expected)
        self.assertEqual(pages, 8)            # exactly 8 results, 8 pages
        # Prepare evidence: both sides fully visible, groups reported.
        stats = result["stats"]
        self.assertEqual(stats["left"]["rows_written"], 4)
        self.assertEqual(stats["right"]["rows_written"], 4)
        self.assertEqual(stats["left"]["candidates_examined"], 4)
        self.assertEqual(stats["matched_right_rows"], 2)
        self.assertTrue(stats["groups"]["groups_disk_resident"])
        self.assertGreater(stats["work_file_bytes"], 0)

    def test_full_nulls_equal(self):
        result = self._prepare("full", nulls_equal=True, page_size=1)
        expected = [
            self._exp_matched(b"l4", b"r4"),  # null==null region/area+sku z
            self._exp_matched(b"l1", b"r2"),
            self._exp_matched(b"l1", b"r1"),
            self._exp_matched(b"l2", b"r2"),
            self._exp_matched(b"l2", b"r1"),
            self._exp_left_only(b"l3"),
            self._exp_right_only(b"r3"),
        ]
        rows, pages = drain(self.work, result["handle_token"])
        self.assertEqual(rows, expected)
        self.assertEqual(pages, 7)

    def test_inner_and_left_modes(self):
        inner = self._prepare("inner", page_size=2,
                              work=os.path.join(self.tmp.name, "w-inner.db"))
        rows, _ = drain(os.path.join(self.tmp.name, "w-inner.db"),
                        inner["handle_token"])
        self.assertEqual(rows, [
            self._exp_matched(b"l1", b"r2"),
            self._exp_matched(b"l1", b"r1"),
            self._exp_matched(b"l2", b"r2"),
            self._exp_matched(b"l2", b"r1"),
        ])
        left = self._prepare("left", page_size=3,
                             work=os.path.join(self.tmp.name, "w-left.db"))
        rows, _ = drain(os.path.join(self.tmp.name, "w-left.db"),
                        left["handle_token"])
        self.assertEqual(rows, [
            self._exp_left_only(b"l4"),
            self._exp_matched(b"l1", b"r2"),
            self._exp_matched(b"l1", b"r1"),
            self._exp_matched(b"l2", b"r2"),
            self._exp_matched(b"l2", b"r1"),
            self._exp_left_only(b"l3"),
        ])

    def test_page_size_1_vs_bulk_identical_stream(self):
        one = self._prepare("full", page_size=1,
                            work=os.path.join(self.tmp.name, "p1.db"))
        bulk = self._prepare("full", page_size=100,
                             work=os.path.join(self.tmp.name, "p100.db"))
        rows1, _ = drain(os.path.join(self.tmp.name, "p1.db"),
                         one["handle_token"])
        rows100, pages = drain(os.path.join(self.tmp.name, "p100.db"),
                               bulk["handle_token"])
        self.assertEqual(rows1, rows100)
        self.assertEqual(pages, 1)

    def test_cursor_replay_is_deterministic(self):
        result = self._prepare("full", page_size=3)
        handle = result["handle_token"]
        join = open_join(self.work, handle)
        page1, _ = join.page()
        join.close()
        # The same cursor replayed from two independent handles yields the
        # same page content and the same next cursor.
        a = open_join(self.work, handle)
        pa, _ = a.page(page1.next_cursor)
        a.close()
        b = open_join(self.work, handle)
        pb, _ = b.page(page1.next_cursor)
        b.close()
        self.assertEqual(
            [(r.kind, _v(r.left), _v(r.right)) for r in pa.rows],
            [(r.kind, _v(r.left), _v(r.right)) for r in pb.rows],
        )
        self.assertEqual(pa.next_cursor, pb.next_cursor)

    def test_mutations_and_history_collection_do_not_leak(self):
        result = self._prepare("full", page_size=1)
        handle = result["handle_token"]
        join = open_join(self.work, handle)
        seen = []
        cursor = None
        for _ in range(3):                    # read a part of the stream
            page, _ = join.page(cursor)
            seen.extend(_row(r.kind, _v(r.left), _v(r.right)) for r in page.rows)
            cursor = page.next_cursor
        join.close()
        # Corrections on both collections: moved sort positions, delete +
        # re-create of one identity, join-key and non-join-field changes.
        with open_collection(self.lpath) as coll:
            head = coll.status().head
            coll.apply(b"fix-1", head, iter([
                {"op": "put", "row_id": b"l1", "region": "a", "sku": "x",
                 "qty": 20},                       # moves in (region,sku,qty)
                {"op": "delete", "row_id": b"l3"},
            ]))
            coll.apply(b"fix-2", coll.status().head, iter([
                {"op": "put", "row_id": b"l3", "region": "b", "sku": "y",
                 "qty": 99},                       # same identity re-created
                {"op": "put", "row_id": b"l4", "region": "zzz", "sku": "z",
                 "qty": 9},                        # join key changed later
            ]))
            hist_l = coll.collect_history(coll.status().head)
        with open_collection(self.rpath) as coll:
            head = coll.status().head
            coll.apply(b"rfix-1", head, iter([
                {"op": "put", "row_id": b"r1", "product": "x", "area": "a",
                 "label": "oldest"},               # non-join field change
                {"op": "delete", "row_id": b"r3"},
            ]))
            coll.apply(b"rfix-2", coll.status().head, iter([
                {"op": "put", "row_id": b"r4", "product": "z", "area": "a",
                 "label": "nil"},                  # null -> 'a' later
            ]))
            hist_r = coll.collect_history(coll.status().head)
        # The retention boundary advanced to head, but the join's pinned
        # snapshots protect every entry their generation can see, so the
        # old generations are NOT reclaimed underneath the join read.
        self.assertEqual(hist_l.entries_removed, 0)
        self.assertEqual(hist_r.entries_removed, 0)
        self.assertEqual(hist_l.retain_from, hist_l.head)
        self.assertEqual(hist_r.retain_from, hist_r.head)
        # The pinned stream is unaffected: continue from the saved cursor.
        expected_rest = [
            self._exp_matched(b"l2", b"r2"),
            self._exp_matched(b"l2", b"r1"),
            self._exp_left_only(b"l3"),
            self._exp_right_only(b"r4"),
            self._exp_right_only(b"r3"),
        ]
        join = open_join(self.work, handle)
        rest = []
        while cursor is not None:
            page, _ = join.page(cursor)
            rest.extend(_row(r.kind, _v(r.left), _v(r.right)) for r in page.rows)
            cursor = page.next_cursor
        join.close()
        self.assertEqual(rest, expected_rest)
        self.assertEqual(
            seen,
            [self._exp_left_only(b"l4"), self._exp_matched(b"l1", b"r2"),
             self._exp_matched(b"l1", b"r1")],
        )

    def test_releasing_one_side_fails_next_page_naming_side(self):
        result = self._prepare("full", page_size=1)
        handle = result["handle_token"]
        join = open_join(self.work, handle)
        page, _ = join.page()
        join.close()
        with open_collection(self.lpath) as coll:
            coll.release_snapshot(self.ltok)
        join = open_join(self.work, handle)
        with self.assertRaises(JoinSourceError) as ctx:
            join.page(page.next_cursor)
        self.assertIn("left", str(ctx.exception))
        self.assertIn("released", str(ctx.exception))
        join.close()
        # The right side alone is still verified fine: a join whose LEFT
        # token is alive but RIGHT token is released names 'right'.
        self.ltok = _snapshot(self.lpath)   # fresh left token
        result2 = self._prepare("inner", page_size=1,
                                work=os.path.join(self.tmp.name, "w-j2.db"))
        with open_collection(self.rpath) as coll:
            coll.release_snapshot(self.rtok)
        join = open_join(os.path.join(self.tmp.name, "w-j2.db"),
                         result2["handle_token"])
        with self.assertRaises(JoinSourceError) as ctx:
            join.page()
        self.assertIn("right", str(ctx.exception))
        join.close()


class NullAndEmptyValueTests(unittest.TestCase):
    """null, empty text and empty bytes never share a group; nulls_equal
    only changes the null==null rule, never empty==null."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lpath = os.path.join(self.tmp.name, "l.db")
        self.rpath = os.path.join(self.tmp.name, "r.db")
        lschema = Schema(columns=(
            Column("t", "text", asc=True, nulls_first=True),
            Column("b", "bytes", asc=True, nulls_first=True),
        ))
        rschema = Schema(columns=(
            Column("t", "text", asc=True, nulls_first=True),
            Column("b", "bytes", asc=True, nulls_first=True),
        ))
        _seed_collection(self.lpath, lschema, [
            (b"l-null", (None, None)),
            (b"l-empty", ("", b"")),
            (b"l-zero", ("\x00", b"\x00")),
        ])
        _seed_collection(self.rpath, rschema, [
            (b"r-null", (None, None)),
            (b"r-empty", ("", b"")),
            (b"r-zero", ("\x00", b"\x00")),
        ])
        self.ltok = _snapshot(self.lpath)
        self.rtok = _snapshot(self.rpath)

    def tearDown(self):
        self.tmp.cleanup()

    def _drain(self, mapping, nulls_equal, mode="full"):
        self._n = getattr(self, "_n", 0) + 1
        work = os.path.join(self.tmp.name, f"j{self._n}.db")
        result = prepare_join(
            work,
            left=JoinSource(self.lpath, self.ltok, ALLQ),
            right=JoinSource(self.rpath, self.rtok, ALLQ),
            mapping=mapping, mode=mode, nulls_equal=nulls_equal, page_size=1,
        )
        rows, _ = drain(work, result["handle_token"])
        return rows

    def test_empty_and_zero_match_but_null_does_not(self):
        rows = self._drain([("t", "t"), ("b", "b")], nulls_equal=False)
        # matched pairs by row_id: empty<->empty, zero<->zero; null rows
        # are delivered as left_only / right_only (default null rule).
        matched_pairs = sorted((l[1], r[1]) for k, l, r in rows
                               if k == "matched")
        self.assertEqual(matched_pairs,
                         [(b"l-empty", b"r-empty"), (b"l-zero", b"r-zero")])
        left_only = [l[1] for k, l, r in rows if k == "left_only"]
        right_only = [r[1] for k, l, r in rows if k == "right_only"]
        self.assertEqual(left_only, [b"l-null"])
        self.assertEqual(right_only, [b"r-null"])

    def test_nulls_equal_matches_null_keys(self):
        rows = self._drain([("t", "t"), ("b", "b")], nulls_equal=True)
        matched_pairs = sorted((l[1], r[1]) for k, l, r in rows
                               if k == "matched")
        self.assertEqual(matched_pairs, [
            (b"l-empty", b"r-empty"), (b"l-null", b"r-null"),
            (b"l-zero", b"r-zero"),
        ])

    def test_empty_bytes_never_equal_null_bytes(self):
        # Mapping on the bytes column alone: b"" matches b"", null matches
        # null only under nulls_equal; empty and null never share a group.
        rows = self._drain([("b", "b")], nulls_equal=True)
        matched = [(l[1], r[1]) for k, l, r in rows if k == "matched"]
        self.assertIn((b"l-empty", b"r-empty"), matched)
        self.assertIn((b"l-null", b"r-null"), matched)
        self.assertNotIn((b"l-empty", b"r-null"), matched)
        self.assertNotIn((b"l-null", b"r-empty"), matched)


class SelfJoinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        schema = Schema(columns=(
            Column("k", "text", asc=True, nulls_first=True),
            Column("v", "int64", asc=True, nulls_first=True),
        ))
        _seed_collection(self.path, schema, [
            (b"a", ("x", 1)), (b"b", ("x", 2)), (b"c", ("y", 3)),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_self_join_same_token(self):
        tok = _snapshot(self.path)
        work = os.path.join(self.tmp.name, "self.db")
        result = prepare_join(
            work,
            left=JoinSource(self.path, tok, ALLQ),
            right=JoinSource(self.path, tok, ALLQ),
            mapping=[("k", "k")], mode="inner", page_size=2,
        )
        rows, _ = drain(work, result["handle_token"])
        # k='x' group: a,b x a,b (4 pairs, right path order a then b per
        # left row); k='y': c x c (1 pair).
        pairs = [(l[0], r[0]) for k, l, r in rows if k == "matched"]
        self.assertEqual(pairs, [
            (("x", 1), ("x", 1)), (("x", 1), ("x", 2)),
            (("x", 2), ("x", 1)), (("x", 2), ("x", 2)),
            (("y", 3), ("y", 3)),
        ])

    def test_self_join_different_generations(self):
        with open_collection(self.path) as coll:
            tok_old = coll.begin_snapshot()          # gen 1 (3 rows)
            coll.apply(b"more", coll.status().head, iter([
                {"op": "put", "row_id": b"d", "k": "x", "v": 9},
                {"op": "put", "row_id": b"e", "k": "w", "v": 5},
                {"op": "delete", "row_id": b"c"},
            ]))
            tok_new = coll.begin_snapshot()          # gen 2
        work = os.path.join(self.tmp.name, "self2.db")
        result = prepare_join(
            work,
            left=JoinSource(self.path, tok_old, ALLQ),
            right=JoinSource(self.path, tok_new, ALLQ),
            mapping=[("k", "k")], mode="full", page_size=1,
        )
        rows, _ = drain(work, result["handle_token"])
        # Old left rows a,b match new x-rows a,b,d; old c (k='y') matches
        # nothing; new row e (k='w') is matched by no old row.
        matched = [(l[1], r[1]) for k, l, r in rows if k == "matched"]
        self.assertEqual(len(matched), 6)            # 2 left x 3 right
        left_only_rids = [l[1] for k, l, r in rows if k == "left_only"]
        right_only_rids = [r[1] for k, l, r in rows if k == "right_only"]
        self.assertEqual(left_only_rids, [b"c"])
        self.assertEqual(right_only_rids, [b"e"])


class EmptySideTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lpath = os.path.join(self.tmp.name, "l.db")
        self.rpath = os.path.join(self.tmp.name, "r.db")
        _seed_collection(self.lpath, LSCHEMA, LROWS)
        _seed_collection(self.rpath, RSCHEMA, RROWS)
        self.ltok = _snapshot(self.lpath)
        self.rtok = _snapshot(self.rpath)
        # Queries that match nothing on each side.
        self.empty_left_q = {"shape": "prefix_range",
                             "prefix": [["region", "zzz-absent"]],
                             "range_column": "sku"}
        self.empty_right_q = {"shape": "prefix_range",
                              "prefix": [["product", "zzz-absent"]],
                              "range_column": "area"}

    def tearDown(self):
        self.tmp.cleanup()

    def _drain(self, lq, rq, mode):
        work = os.path.join(self.tmp.name, f"j-{mode}.db")
        result = prepare_join(
            work,
            left=JoinSource(self.lpath, self.ltok, lq),
            right=JoinSource(self.rpath, self.rtok, rq),
            mapping=MAPPING, mode=mode, page_size=2,
        )
        rows, _ = drain(work, result["handle_token"])
        return rows

    def test_empty_left(self):
        self.assertEqual(self._drain(self.empty_left_q, ALLQ, "inner"), [])
        self.assertEqual(self._drain(self.empty_left_q, ALLQ, "left"), [])
        rows = self._drain(self.empty_left_q, ALLQ, "full")
        self.assertEqual([k for k, _, _ in rows], ["right_only"] * 4)
        # right_only rows arrive in right path order: r4, r2, r1, r3.
        self.assertEqual([r[1] for _, _, r in rows],
                         [b"r4", b"r2", b"r1", b"r3"])

    def test_empty_right(self):
        self.assertEqual(self._drain(ALLQ, self.empty_right_q, "inner"), [])
        rows = self._drain(ALLQ, self.empty_right_q, "left")
        self.assertEqual([k for k, _, _ in rows], ["left_only"] * 4)
        rows = self._drain(ALLQ, self.empty_right_q, "full")
        self.assertEqual([k for k, _, _ in rows], ["left_only"] * 4)

    def test_both_empty(self):
        for mode in ("inner", "left", "full"):
            self.assertEqual(
                self._drain(self.empty_left_q, self.empty_right_q, mode), [])


class DuplicateGroupTests(unittest.TestCase):
    """A duplicate key group larger than any sane memory buffer, paged one
    row at a time: the group is seekable, never re-read, never materialized."""

    NL = 25
    NR = 30

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lpath = os.path.join(self.tmp.name, "l.db")
        self.rpath = os.path.join(self.tmp.name, "r.db")
        lrows = [(f"l{i:03d}".encode(), ("hot", i)) for i in range(self.NL)]
        lrows += [(b"l-distinct", ("cold", 999))]
        rrows = [(f"r{j:03d}".encode(), ("hot", f"tag{j:03d}"))
                 for j in range(self.NR)]
        rrows += [(b"r-distinct", ("lukewarm", "x"))]
        schema_l = Schema(columns=(
            Column("k", "text", asc=True, nulls_first=True),
            Column("n", "int64", asc=True, nulls_first=True),
        ))
        schema_r = Schema(columns=(
            Column("k", "text", asc=True, nulls_first=True),
            Column("tag", "text", asc=True, nulls_first=True),
        ))
        _seed_collection(self.lpath, schema_l, lrows)
        _seed_collection(self.rpath, schema_r, rrows)
        self.ltok = _snapshot(self.lpath)
        self.rtok = _snapshot(self.rpath)
        self.work = os.path.join(self.tmp.name, "j.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_huge_group_page_size_1(self):
        result = prepare_join(
            self.work,
            left=JoinSource(self.lpath, self.ltok, ALLQ),
            right=JoinSource(self.rpath, self.rtok, ALLQ),
            mapping=[("k", "k")], mode="inner", page_size=1,
        )
        stats = result["stats"]
        self.assertEqual(stats["groups"]["max_left_group"], self.NL)
        self.assertEqual(stats["groups"]["max_right_group"], self.NR)
        self.assertTrue(stats["groups"]["groups_disk_resident"])
        # The work file stores only the inputs (25+1 + 30+1 rows), while the
        # pairing result is 25*30 = 750 rows and is never materialized.
        self.assertLess(stats["work_file_bytes"], 1024 * 1024)

        handle = result["handle_token"]
        expected = []
        for i in range(self.NL):
            for j in range(self.NR):
                expected.append((("hot", i), f"l{i:03d}".encode(),
                                 ("hot", f"tag{j:03d}"), f"r{j:03d}".encode()))
        # inner mode: the distinct 'cold'/'lukewarm' rows produce nothing.
        got = []
        cursor = None
        pages = 0
        max_right_candidates = 0
        max_left_examined = 0
        while True:
            join = open_join(self.work, handle)
            page, st = join.page(cursor)
            join.close()
            pages += 1
            max_right_candidates = max(max_right_candidates,
                                       st.right_candidates_examined)
            max_left_examined = max(max_left_examined, st.left_rows_examined)
            for r in page.rows:
                got.append((r.left.values, r.left.row_id,
                            r.right.values, r.right.row_id))
            cursor = page.next_cursor
            if cursor is None:
                break
        self.assertEqual(got, expected)
        self.assertEqual(pages, self.NL * self.NR)
        # Mid-group resume never re-reads the group head: each page pulls at
        # most page_size + 1 look-ahead right candidates, and later pages
        # never re-examine earlier left rows.
        self.assertLessEqual(max_right_candidates, 2)
        self.assertLessEqual(max_left_examined, 3)


class IndexPathTests(unittest.TestCase):
    """Joins over secondary indexes with different column orders, filtered
    ranges, and pinned old generations whose non-join fields later moved."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lpath = os.path.join(self.tmp.name, "l.db")
        self.rpath = os.path.join(self.tmp.name, "r.db")
        self.lschema = Schema(columns=(
            Column("region", "text", asc=True, nulls_first=True),
            Column("sku", "text", asc=False, nulls_first=True),
            Column("qty", "int64", asc=True, nulls_first=True),
            Column("note", "text", asc=True, nulls_first=True),
        ))
        self.rschema = Schema(columns=(
            Column("product", "text", asc=False, nulls_first=True),
            Column("area", "text", asc=True, nulls_first=True),
            Column("label", "text", asc=True, nulls_first=True),
            Column("flag", "bool", asc=True, nulls_first=True),
        ))
        _seed_collection(self.lpath, self.lschema, [
            (b"l1", ("a", "x", 2, "v0-l1")),
            (b"l2", ("a", "x", 3, "v0-l2")),
            (b"l3", ("b", "y", 1, "v0-l3")),
            (b"l5", ("c", "w", 50, "v0-l5")),   # qty 50: outside the filter
        ])
        _seed_collection(self.rpath, self.rschema, [
            (b"r1", ("x", "a", "keep", True)),
            (b"r2", ("x", "a", "drop", False)),  # label 'drop': filtered out
            (b"r3", ("q", "c", "keep", True)),
        ])
        # Secondary indexes with column orders different from the mains and
        # from each other; neither leads with a join column pair.
        with open_collection(self.lpath) as coll:
            coll.register_index(SecondaryDeclaration("qty_region", (
                IndexColumn("qty", asc=True), IndexColumn("region", asc=True))))
            while coll.advance_index("qty_region", 100).status != "ready":
                pass
            coll.register_index(SecondaryDeclaration("slow_idx", (
                IndexColumn("note", asc=True),)))
        with open_collection(self.rpath) as coll:
            coll.register_index(SecondaryDeclaration("label_product", (
                IndexColumn("label", asc=False),
                IndexColumn("product", asc=True))))
            while coll.advance_index("label_product", 100).status != "ready":
                pass
        # Pinned old generation BEFORE non-join-field corrections.
        self.ltok_old = _snapshot(self.lpath)
        self.rtok_old = _snapshot(self.rpath)
        with open_collection(self.lpath) as coll:
            coll.apply(b"notes", coll.status().head, iter([
                {"op": "put", "row_id": b"l1", "region": "a", "sku": "x",
                 "qty": 2, "note": "v1-changed"},
                {"op": "put", "row_id": b"l2", "region": "a", "sku": "x",
                 "qty": 3, "note": "v1-changed"},
            ]))
        with open_collection(self.rpath) as coll:
            coll.apply(b"flags", coll.status().head, iter([
                {"op": "put", "row_id": b"r1", "product": "x", "area": "a",
                 "label": "keep", "flag": False},
            ]))
        self.ltok_new = _snapshot(self.lpath)

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_paths_filters_and_old_generation_values(self):
        # Left: via qty_region index, qty in [1, 5] (excludes l5 qty=50).
        left_q = {"shape": "prefix_range", "prefix": [], "range_column": "qty",
                  "lo": 1, "hi": 5}
        # Right: via label_product index, label == 'keep' (excludes r2).
        right_q = {"shape": "prefix_range", "prefix": [["label", "keep"]],
                   "range_column": "product"}
        work = os.path.join(self.tmp.name, "j.db")
        result = prepare_join(
            work,
            left=JoinSource(self.lpath, self.ltok_old, left_q,
                            index="qty_region"),
            right=JoinSource(self.rpath, self.rtok_old, right_q,
                             index="label_product"),
            mapping=MAPPING, mode="full", page_size=1,
        )
        stats = result["stats"]
        # Filter-then-join: only 3 left and 2 right rows entered the join.
        self.assertEqual(stats["left"]["rows_written"], 3)
        self.assertEqual(stats["right"]["rows_written"], 2)
        self.assertEqual(stats["left"]["access"], "index:qty_region")
        self.assertEqual(stats["right"]["access"], "index:label_product")
        rows, _ = drain(work, result["handle_token"])
        # Left path order = index order (qty ASC, region ASC, row_id):
        # l3(qty1), l1(qty2), l2(qty3).  Right path order = (label DESC,
        # product ASC): r1, r3 both 'keep'; product q < x -> r3 first.
        seq = [(k, None if l is None else l[1], None if r is None else r[1])
               for k, l, r in rows]
        self.assertEqual(seq, [
            ("left_only", b"l3", None),
            ("matched", b"l1", b"r1"),
            ("matched", b"l2", b"r1"),
            ("right_only", None, b"r3"),
        ])
        # Full values come from the pinned OLD generation: note 'v0-*' and
        # flag True, not the later corrections.
        matched_l1 = next(r for r in rows if r[1] and r[1][1] == b"l1")
        self.assertEqual(matched_l1[1][0][3], "v0-l1")
        matched_r1 = next(r for r in rows if r[2] and r[2][1] == b"r1")
        self.assertEqual(matched_r1[2][0][3], True)
        # Non-join fields are all present (note/flag carried through).
        self.assertEqual(len(matched_l1[1][0]), 4)
        self.assertEqual(len(matched_r1[2][0]), 4)

    def test_building_index_path_is_refused(self):
        with open_collection(self.lpath) as coll:
            coll.advance_index("slow_idx", 1)   # still building
        tok = _snapshot(self.lpath)
        with self.assertRaises(IndexNotReadyError):
            prepare_join(
                os.path.join(self.tmp.name, "j2.db"),
                left=JoinSource(self.lpath, tok, ALLQ, index="slow_idx"),
                right=JoinSource(self.rpath, self.rtok_old, ALLQ),
                mapping=MAPPING, mode="inner", page_size=1,
            )

    def test_snapshot_older_than_index_coverage_is_refused(self):
        # A token pinned before the index's coverage start cannot use it.
        with open_collection(self.lpath) as coll:
            coll.register_index(SecondaryDeclaration("late_idx", (
                IndexColumn("qty", asc=True),)))
            while coll.advance_index("late_idx", 100).status != "ready":
                pass
            tok_gen0 = coll.begin_snapshot(0)
        with self.assertRaises(CoverageError):
            prepare_join(
                os.path.join(self.tmp.name, "j3.db"),
                left=JoinSource(self.lpath, tok_gen0, ALLQ, index="late_idx"),
                right=JoinSource(self.rpath, self.rtok_old, ALLQ),
                mapping=MAPPING, mode="inner", page_size=1,
            )


class SourceEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lpath = os.path.join(self.tmp.name, "l.db")
        self.rpath = os.path.join(self.tmp.name, "r.db")
        _seed_collection(self.lpath, LSCHEMA, LROWS)
        _seed_collection(self.rpath, RSCHEMA, RROWS)
        with open_collection(self.lpath) as coll:
            coll.register_index(SecondaryDeclaration("by_sku", (
                IndexColumn("sku", asc=True),)))
            while coll.advance_index("by_sku", 100).status != "ready":
                pass
        self.ltok = _snapshot(self.lpath)
        self.rtok = _snapshot(self.rpath)
        self.work = os.path.join(self.tmp.name, "j.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _prepare(self, left_index=None):
        return prepare_join(
            self.work,
            left=JoinSource(self.lpath, self.ltok, ALLQ, index=left_index),
            right=JoinSource(self.rpath, self.rtok, ALLQ),
            mapping=MAPPING, mode="full", page_size=1,
        )

    def test_deleted_index_instance_fails_paging(self):
        result = self._prepare(left_index="by_sku")
        handle = result["handle_token"]
        join = open_join(self.work, handle)
        page, _ = join.page()
        join.close()
        with open_collection(self.lpath) as coll:
            coll.delete_index("by_sku")
        join = open_join(self.work, handle)
        with self.assertRaises(JoinSourceError) as ctx:
            join.page(page.next_cursor)
        self.assertIn("left", str(ctx.exception))
        self.assertIn("by_sku", str(ctx.exception))
        join.close()

    def test_same_name_replacement_fails_paging(self):
        result = self._prepare(left_index="by_sku")
        handle = result["handle_token"]
        with open_collection(self.lpath) as coll:
            coll.delete_index("by_sku")
            coll.register_index(SecondaryDeclaration("by_sku", (
                IndexColumn("sku", asc=True),)))
            while coll.advance_index("by_sku", 100).status != "ready":
                pass
        join = open_join(self.work, handle)
        with self.assertRaises(JoinSourceError) as ctx:
            join.page()
        self.assertIn("left", str(ctx.exception))
        join.close()

    def test_replaced_collection_file_fails_paging(self):
        result = self._prepare()
        handle = result["handle_token"]
        # Replace the right collection file with a different collection
        # (main file plus its WAL sidecars, as a clean replacement).
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(self.rpath + suffix):
                os.remove(self.rpath + suffix)
        _seed_collection(self.rpath, RSCHEMA, RROWS)
        join = open_join(self.work, handle)
        with self.assertRaises(JoinSourceError) as ctx:
            join.page()
        self.assertIn("right", str(ctx.exception))
        self.assertIn("replaced", str(ctx.exception))
        join.close()

    def test_prepare_fails_fast_on_released_token(self):
        with open_collection(self.lpath) as coll:
            coll.release_snapshot(self.ltok)
        with self.assertRaises(JoinSourceError) as ctx:
            self._prepare()
        self.assertIn("left", str(ctx.exception))


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lpath = os.path.join(self.tmp.name, "l.db")
        self.rpath = os.path.join(self.tmp.name, "r.db")
        _seed_collection(self.lpath, LSCHEMA, LROWS)
        _seed_collection(self.rpath, RSCHEMA, RROWS)
        self.ltok = _snapshot(self.lpath)
        self.rtok = _snapshot(self.rpath)

    def tearDown(self):
        self.tmp.cleanup()

    def _prepare(self, work, page_size=1):
        return prepare_join(
            work,
            left=JoinSource(self.lpath, self.ltok, ALLQ),
            right=JoinSource(self.rpath, self.rtok, ALLQ),
            mapping=MAPPING, mode="full", page_size=page_size,
        )

    def test_foreign_handle_and_cursor_rejected(self):
        w1 = os.path.join(self.tmp.name, "j1.db")
        w2 = os.path.join(self.tmp.name, "j2.db")
        r1 = self._prepare(w1)
        r2 = self._prepare(w2)   # identical declaration, different identity
        # Handle of join 1 on join 2.
        with self.assertRaises(JoinCredentialError):
            open_join(w2, r1["handle_token"])
        # Cursor of join 1 on join 2.
        join1 = open_join(w1, r1["handle_token"])
        page1, _ = join1.page()
        join1.close()
        join2 = open_join(w2, r2["handle_token"])
        with self.assertRaises(JoinCredentialError):
            join2.page(page1.next_cursor)
        join2.close()

    def test_page_size_binding(self):
        w1 = os.path.join(self.tmp.name, "p1.db")
        w3 = os.path.join(self.tmp.name, "p3.db")
        r1 = self._prepare(w1, page_size=1)
        r3 = self._prepare(w3, page_size=3)
        join1 = open_join(w1, r1["handle_token"])
        page1, _ = join1.page()
        join1.close()
        join3 = open_join(w3, r3["handle_token"])
        with self.assertRaises(JoinCredentialError):
            join3.page(page1.next_cursor)
        join3.close()

    def test_garbage_tokens(self):
        w1 = os.path.join(self.tmp.name, "j1.db")
        r1 = self._prepare(w1)
        with self.assertRaises(JoinCredentialError):
            open_join(w1, "not-a-token")
        join = open_join(w1, r1["handle_token"])
        with self.assertRaises(JoinCredentialError):
            join.page("garbage.cursor")
        join.close()

    def test_terminated_join_rejects_handle_and_cursor(self):
        w1 = os.path.join(self.tmp.name, "j1.db")
        r1 = self._prepare(w1)
        join = open_join(w1, r1["handle_token"])
        page1, _ = join.page()
        join.close()
        terminate_join(w1, r1["handle_token"])
        with self.assertRaises(JoinStateError):
            open_join(w1, r1["handle_token"])
        # join_status still describes the closed read.
        self.assertEqual(join_status(w1)["state"], "closed")
        # Cleanup removes the closed work file; the path is reusable.
        join_cleanup(w1)
        r2 = self._prepare(w1)
        self.assertEqual(join_status(w1)["state"], "ready")
        terminate_join(w1, r2["handle_token"], reclaim=True)
        self.assertFalse(os.path.exists(w1))

    def test_prepare_never_overwrites(self):
        w1 = os.path.join(self.tmp.name, "j1.db")
        r1 = self._prepare(w1)
        with self.assertRaises(JoinStateError) as ctx:
            self._prepare(w1)
        self.assertIn("completed", str(ctx.exception))
        terminate_join(w1, r1["handle_token"])
        with self.assertRaises(JoinStateError) as ctx:
            self._prepare(w1)
        self.assertIn("closed", str(ctx.exception))
        # A foreign file at the path is also protected.
        foreign = os.path.join(self.tmp.name, "foreign.db")
        with open(foreign, "w", encoding="ascii") as fh:
            fh.write("not a database")
        with self.assertRaises(JoinStateError):
            self._prepare(foreign)

    def test_cleanup_refuses_active_join(self):
        w1 = os.path.join(self.tmp.name, "j1.db")
        self._prepare(w1)
        with self.assertRaises(JoinStateError):
            join_cleanup(w1)
        self.assertEqual(join_status(w1)["state"], "ready")

    def test_close_object_does_not_end_the_join_read(self):
        w1 = os.path.join(self.tmp.name, "j1.db")
        r1 = self._prepare(w1)
        join = open_join(w1, r1["handle_token"])
        page1, _ = join.page()
        join.close()                      # object closed...
        with self.assertRaises(JoinStateError):
            join.page()                   # ...the object is spent...
        join2 = open_join(w1, r1["handle_token"])
        page2, _ = join2.page(page1.next_cursor)  # ...but the read continues
        self.assertEqual(len(page2.rows), 1)
        join2.close()


class CrashRecoveryTests(unittest.TestCase):
    """Real process kills during prepare leave a recognizable building file
    that cannot serve pages and can be cleaned up or re-prepared."""

    HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "join_prepare_helper.py")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lpath = os.path.join(self.tmp.name, "l.db")
        self.rpath = os.path.join(self.tmp.name, "r.db")
        _seed_collection(self.lpath, LSCHEMA, LROWS)
        _seed_collection(self.rpath, RSCHEMA, RROWS)
        self.ltok = _snapshot(self.lpath)
        self.rtok = _snapshot(self.rpath)
        self.work = os.path.join(self.tmp.name, "j.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _job(self, crash):
        job = {
            "work": self.work,
            "left": {"collection": self.lpath, "token": self.ltok,
                     "query": ALLQ},
            "right": {"collection": self.rpath, "token": self.rtok,
                      "query": ALLQ},
            "mapping": [[l, r] for l, r in MAPPING],
            "mode": "full",
            "nulls_equal": False,
            "page_size": 1,
        }
        if crash:
            job["crash"] = crash
        job_path = os.path.join(self.tmp.name, "job.json")
        with open(job_path, "w", encoding="utf-8") as fh:
            json.dump(job, fh)
        return job_path

    def _run_helper(self, crash):
        return subprocess.run(
            [sys.executable, "-I", self.HELPER, "--job", self._job(crash)],
            capture_output=True, text=True,
        )

    def test_kill_mid_prepare_then_cleanup_and_reprepare(self):
        for phase in ("after_meta", "after_left", "after_right",
                      "after_matched", "pre_commit"):
            proc = self._run_helper(phase)
            self.assertNotEqual(proc.returncode, 0, phase)
            # The unfinished file is recognized, not served.
            self.assertEqual(join_status(self.work)["state"], "building")
            with self.assertRaises(JoinStateError) as ctx:
                open_join(self.work, "whatever")
            self.assertIn("did not complete", str(ctx.exception))
            # Cleanup removes it; a fresh prepare succeeds.
            outcome = join_cleanup(self.work)
            self.assertEqual(outcome["was_state"], "building")
            proc2 = self._run_helper(None)
            self.assertEqual(proc2.returncode, 0, proc2.stderr)
            result = json.loads(proc2.stdout)
            rows, pages = drain(self.work, result["handle_token"])
            self.assertEqual(len(rows), 8)
            terminate_join(self.work, result["handle_token"], reclaim=True)

    def test_kill_after_commit_keeps_ready_work_file(self):
        proc = self._run_helper("post_commit")
        self.assertNotEqual(proc.returncode, 0)
        # The commit landed before the kill: the join read is usable.
        self.assertEqual(join_status(self.work)["state"], "ready")

    def test_interrupted_before_declaration_commit(self):
        # A kill in the window between work-file creation and the
        # declaration commit leaves only the application id on disk; the
        # file is still recognized as an unfinished join, never served, and
        # can be cleaned up.
        conn = sqlite3.connect(self.work)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA application_id = 0x626A6F69")
        conn.close()
        status = join_status(self.work)
        self.assertEqual(status["state"], "building")
        self.assertTrue(status["incomplete"])
        with self.assertRaises(JoinStateError) as ctx:
            open_join(self.work, "whatever")
        self.assertIn("interrupted", str(ctx.exception))
        outcome = join_cleanup(self.work)
        self.assertEqual(outcome["was_state"], "building")
        self.assertFalse(os.path.exists(self.work))

    def test_prepare_failure_leaves_no_usable_state(self):
        # A prepare that fails because a token was released mid-way must not
        # leave a servable file behind.
        with open_collection(self.lpath) as coll:
            coll.release_snapshot(self.ltok)
        with self.assertRaises(JoinSourceError):
            prepare_join(
                self.work,
                left=JoinSource(self.lpath, self.ltok, ALLQ),
                right=JoinSource(self.rpath, self.rtok, ALLQ),
                mapping=MAPPING, mode="inner", page_size=1,
            )
        self.assertFalse(os.path.exists(self.work))

    def test_token_released_mid_prepare_aborts_ready_switch(self):
        # The source-eligibility re-check at prepare commit catches a token
        # released while the inputs were being copied: the work file stays
        # 'building' and can never serve a page.
        def release_hook(phase, _conn):
            if phase == "after_left":
                with open_collection(self.lpath) as coll:
                    coll.release_snapshot(self.ltok)

        with self.assertRaises(JoinSourceError) as ctx:
            prepare_join(
                self.work,
                left=JoinSource(self.lpath, self.ltok, ALLQ),
                right=JoinSource(self.rpath, self.rtok, ALLQ),
                mapping=MAPPING, mode="inner", page_size=1,
                crash_hook=release_hook,
            )
        self.assertIn("left", str(ctx.exception))
        self.assertEqual(join_status(self.work)["state"], "building")
        with self.assertRaises(JoinStateError):
            open_join(self.work, "whatever")
        outcome = join_cleanup(self.work)
        self.assertEqual(outcome["was_state"], "building")
        self.assertFalse(os.path.exists(self.work))

    def test_unwritable_work_file_location_is_locatable(self):
        bad = os.path.join(self.lpath, "sub", "join.db")  # parent is a file
        with self.assertRaises(JoinStateError) as ctx:
            prepare_join(
                bad,
                left=JoinSource(self.lpath, self.ltok, ALLQ),
                right=JoinSource(self.rpath, self.rtok, ALLQ),
                mapping=MAPPING, mode="inner", page_size=1,
            )
        self.assertIn("work file", str(ctx.exception))


class SameFileTwoIndexesTests(unittest.TestCase):
    """One collection may supply both sides: different generations and
    different ready indexes on the same file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        schema = Schema(columns=(
            Column("k", "text", asc=True, nulls_first=True),
            Column("v", "int64", asc=True, nulls_first=True),
            Column("note", "text", asc=True, nulls_first=True),
        ))
        _seed_collection(self.path, schema, [
            (b"a", ("x", 1, "n1")), (b"b", ("x", 2, "n2")),
            (b"c", ("y", 3, "n3")),
        ])
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration("by_v", (
                IndexColumn("v", asc=False),)))
            coll.register_index(SecondaryDeclaration("by_note", (
                IndexColumn("note", asc=True),)))
            while coll.advance_index("by_v", 100).status != "ready":
                pass
            while coll.advance_index("by_note", 100).status != "ready":
                pass
            self.tok = coll.begin_snapshot()

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_file_two_index_paths(self):
        work = os.path.join(self.tmp.name, "j.db")
        # Left reads via by_v (v DESC), right via by_note (note ASC); join on
        # the business column k which is neither index's leading column.
        result = prepare_join(
            work,
            left=JoinSource(self.path, self.tok, ALLQ, index="by_v"),
            right=JoinSource(self.path, self.tok, ALLQ, index="by_note"),
            mapping=[("k", "k")], mode="inner", page_size=1,
        )
        rows, _ = drain(work, result["handle_token"])
        # Left path order is v DESC: c(v=3), b(v=2), a(v=1).  Right path
        # order is note ASC: a(n1), b(n2), c(n3).  k='x' rows: a,b; k='y': c.
        seq = [(k, l[1], r[1]) for k, l, r in rows]
        self.assertEqual(seq, [
            ("matched", b"c", b"c"),   # left c (k=y) x right c (k=y)
            ("matched", b"b", b"a"),   # left b (k=x) x right a,b (k=x)
            ("matched", b"b", b"b"),
            ("matched", b"a", b"a"),   # left a (k=x) x right a,b
            ("matched", b"a", b"b"),
        ])


class DeclarationValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lpath = os.path.join(self.tmp.name, "l.db")
        self.rpath = os.path.join(self.tmp.name, "r.db")
        _seed_collection(self.lpath, LSCHEMA, LROWS)
        _seed_collection(self.rpath, RSCHEMA, RROWS)
        self.ltok = _snapshot(self.lpath)
        self.rtok = _snapshot(self.rpath)
        self.work = os.path.join(self.tmp.name, "j.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _prepare(self, **overrides):
        args = dict(
            left=JoinSource(self.lpath, self.ltok, ALLQ),
            right=JoinSource(self.rpath, self.rtok, ALLQ),
            mapping=MAPPING, mode="inner", page_size=1,
        )
        args.update(overrides)
        return prepare_join(self.work, **args)

    def test_mapping_validation(self):
        with self.assertRaises(JoinDeclarationError):
            self._prepare(mapping=[])
        with self.assertRaises(JoinDeclarationError):
            self._prepare(mapping={"left": ["region"], "right": []})
        with self.assertRaises(JoinDeclarationError):
            self._prepare(mapping=[("region", "area"), ("region", "label")])
        with self.assertRaises(JoinDeclarationError):
            self._prepare(mapping=[("region", "area"), ("sku", "area")])
        with self.assertRaises(JoinDeclarationError):
            self._prepare(mapping=[("nope", "area")])
        with self.assertRaises(JoinDeclarationError):
            self._prepare(mapping=[("region", "nope")])
        with self.assertRaises(JoinDeclarationError):
            self._prepare(mapping=[("region", "qty")])   # text vs int64
        with self.assertRaises(JoinDeclarationError):
            self._prepare(mapping="region")

    def test_mode_and_page_size_validation(self):
        with self.assertRaises(JoinDeclarationError):
            self._prepare(mode="outer")
        with self.assertRaises(JoinDeclarationError):
            self._prepare(page_size=0)
        with self.assertRaises(JoinDeclarationError):
            self._prepare(page_size="1")
        with self.assertRaises(JoinDeclarationError):
            self._prepare(nulls_equal="yes")

    def test_query_validation(self):
        with self.assertRaises(ValueError):
            self._prepare(left=JoinSource(self.lpath, self.ltok,
                                          {"shape": "nope"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
