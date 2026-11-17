"""Mandated hand-written fixture acceptance against the independent oracle.

Scenario (region ASC, ts DESC with nulls first):

* generation 0: r1=(a,10), r2=(a,10), r3=(b,null)
* generation 1: r1 moves to (c,10); r2 is deleted; r4=(a,9) is added
* generation 2: r1 moves back to (a,8); r2 is re-created at (a,10)

Generation 0 is seeded by building a *static index* and importing it -- which
also exercises the migration path.  The first page of generation 0 is read
with page size 1 *before* the two batches are applied.  Every later check is
compared page by page with the plain-dictionary oracle in tests/refmodel.py.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bytekey import Column, Schema, build_index  # noqa: E402
from bytekey.collection import open_collection  # noqa: E402
from bytekey.snapshot import TokenError  # noqa: E402
from tests.refmodel import RefCollection, drain_snapshot  # noqa: E402

SCHEMA = Schema(
    columns=(
        Column("region", "text", asc=True, nulls_first=True),
        Column("ts", "int64", asc=False, nulls_first=True),
    )
)
GEN0 = [
    {"region": "a", "ts": 10, "row_id": b"r1"},
    {"region": "a", "ts": 10, "row_id": b"r2"},
    {"region": "b", "ts": None, "row_id": b"r3"},
]
BATCH1 = [
    {"op": "put", "region": "c", "ts": 10, "row_id": b"r1"},
    {"op": "delete", "row_id": b"r2"},
    {"op": "put", "region": "a", "ts": 9, "row_id": b"r4"},
]
BATCH2 = [
    {"op": "put", "region": "a", "ts": 8, "row_id": b"r1"},
    {"op": "put", "region": "a", "ts": 10, "row_id": b"r2"},
]
ALL_QUERY = {"shape": "prefix_range", "prefix": []}
REGION_A = {
    "shape": "prefix_range",
    "prefix": [["region", "a"]],
    "range_column": "ts",
}


def _seed_ref() -> RefCollection:
    ref = RefCollection(SCHEMA)
    ref.seed([(r["row_id"], (r["region"], r["ts"])) for r in GEN0])
    return ref


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.static_path = os.path.join(self.tmp.name, "static.db")
        self.coll_path = os.path.join(self.tmp.name, "coll.db")
        build_index(self.static_path, SCHEMA, iter(GEN0))
        from bytekey.collection import import_from_index

        self.cid, n = import_from_index(self.static_path, self.coll_path)
        self.assertEqual(n, 3)
        self.ref = _seed_ref()

    def tearDown(self):
        self.tmp.cleanup()

    # -- full fixture, page by page ----------------------------------------

    def test_fixture_generations_against_oracle(self):
        with open_collection(self.coll_path) as coll:
            self.assertEqual(coll.status().head, 0)

            # Read the first page of generation 0 BEFORE any apply, size 1.
            tok0 = coll.begin_snapshot(0)
            page, stats, _ = coll.scan_snapshot_page(tok0, ALL_QUERY, 1)
            self.assertEqual(page.rows, [(("a", 10), b"r1")])
            self.assertIsNotNone(page.next_cursor)
            self.assertEqual(stats.rows_delivered, 1)

            # Publish the two batches, mirroring into the oracle.
            r1 = coll.apply(b"batch-1", 0, iter(BATCH1))
            self.ref.apply(b"batch-1", 0, BATCH1)
            self.assertEqual((r1.head, r1.applied, r1.noop), (1, 3, 0))
            r2 = coll.apply(b"batch-2", 1, iter(BATCH2))
            self.ref.apply(b"batch-2", 1, BATCH2)
            self.assertEqual((r2.head, r2.applied, r2.noop), (2, 2, 0))
            self.assertEqual(coll.status().head, 2)

            tok1 = coll.begin_snapshot(1)
            tok2 = coll.begin_snapshot(2)
            for gen, tok in ((0, tok0), (1, tok1), (2, tok2)):
                # Drain with page size 1 to exercise every cursor transition.
                got, pages, _plan = drain_snapshot(coll, tok, ALL_QUERY, 1)
                want = self.ref.records_at(gen)
                self.assertEqual(got, want, f"full gen {gen} mismatch")
                self.assertEqual(pages, len(want))

            # Expected hand-written orderings.
            got0, _, _ = drain_snapshot(coll, tok0, ALL_QUERY, 1)
            self.assertEqual(
                got0,
                [(("a", 10), b"r1"), (("a", 10), b"r2"), (("b", None), b"r3")],
            )
            got1, _, _ = drain_snapshot(coll, tok1, ALL_QUERY, 2)
            self.assertEqual(
                got1,
                [(("a", 9), b"r4"), (("b", None), b"r3"), (("c", 10), b"r1")],
            )
            got2, _, _ = drain_snapshot(coll, tok2, ALL_QUERY, 2)
            # r1 moved back under 'a', so its gen-1 'c' key is dead at gen 2:
            # the oracle dictionary contains no (c,10) row anymore.
            self.assertEqual(got2, self.ref.records_at(2))
            self.assertEqual(
                got2,
                [
                    (("a", 10), b"r2"),
                    (("a", 9), b"r4"),
                    (("a", 8), b"r1"),
                    (("b", None), b"r3"),
                ],
            )

            # region-a filter per generation, compared with the oracle.
            for gen, tok in ((0, tok0), (1, tok1), (2, tok2)):
                got, _, _ = drain_snapshot(coll, tok, REGION_A, 1)
                want = self.ref.query_at(gen, REGION_A)
                self.assertEqual(got, want, f"region-a gen {gen} mismatch")
            got1a, _, _ = drain_snapshot(coll, tok1, REGION_A, 1)
            self.assertEqual(got1a, [(("a", 9), b"r4")])
            got2a, _, _ = drain_snapshot(coll, tok2, REGION_A, 1)
            self.assertEqual(
                got2a,
                [(("a", 10), b"r2"), (("a", 9), b"r4"), (("a", 8), b"r1")],
            )

    # -- same-key identity tie-break ---------------------------------------

    def test_same_business_tuple_identity_tiebreak(self):
        with open_collection(self.coll_path) as coll:
            tok = coll.begin_snapshot(0)
            got, pages, _ = drain_snapshot(coll, tok, REGION_A, 1)
            # r1 and r2 share (a,10); row_id bytes are the tie-break and
            # both siblings must be delivered, r1 before r2.
            self.assertEqual(got, [(("a", 10), b"r1"), (("a", 10), b"r2")])
            self.assertEqual(pages, 2)
            # full_tuple bound on (a,10) includes the whole equal group.
            group_q = {
                "shape": "full_tuple",
                "lower": ["a", 10],
                "upper": ["a", 10],
                "lower_inclusive": True,
                "upper_inclusive": True,
            }
            got_group, _, _ = drain_snapshot(coll, tok, group_q, 1)
            self.assertEqual(
                got_group, [(("a", 10), b"r1"), (("a", 10), b"r2")]
            )

    # -- cross-process resume with only path/token/query/cursor ------------

    def test_cross_process_cursor_continue(self):
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "snapshot_page_helper.py")
        with open_collection(self.coll_path) as coll:
            tok = coll.begin_snapshot(0)
            page, _, _ = coll.scan_snapshot_page(tok, ALL_QUERY, 1)
            first_cursor = page.next_cursor
            self.assertEqual(page.rows, [(("a", 10), b"r1")])

        # New process: nothing but the collection path, token, query and cursor.
        query_path = os.path.join(self.tmp.name, "q.json")
        with open(query_path, "w", encoding="utf-8") as fh:
            json.dump(ALL_QUERY, fh)
        proc = subprocess.run(
            [sys.executable, "-I", helper,
             "--collection", self.coll_path,
             "--token", tok,
             "--query", query_path,
             "--page-size", "1",
             "--cursor", first_cursor],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["rows"], [
            {"values": ["a", 10], "row_id": {"__hex__": b"r2".hex()}}
        ])
        cursor2 = payload["next_cursor"]
        self.assertIsNotNone(cursor2)
        # And one more hop, still a fresh process.
        proc = subprocess.run(
            [sys.executable, "-I", helper,
             "--collection", self.coll_path,
             "--token", tok,
             "--query", query_path,
             "--page-size", "1",
             "--cursor", cursor2],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["rows"], [
            {"values": ["b", None], "row_id": {"__hex__": b"r3".hex()}}
        ])
        self.assertIsNone(payload["next_cursor"])

        # A cursor from snapshot tok cannot be replayed through a *different*
        # snapshot of the same generation.
        with open_collection(self.coll_path) as coll:
            other = coll.begin_snapshot(0)
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(other, ALL_QUERY, 1,
                                        cursor_token=first_cursor)

    # -- snapshot pins even while writes continue --------------------------

    def test_snapshot_is_immune_to_later_publishes(self):
        with open_collection(self.coll_path) as coll:
            tok0 = coll.begin_snapshot(0)
            coll.apply(b"batch-1", 0, iter(BATCH1))
            # mid-stream: deleted r2 must still be readable on the old token,
            # moved r1 must still appear at its OLD position only.
            got, _, _ = drain_snapshot(coll, tok0, ALL_QUERY, 1)
            self.assertEqual(
                got,
                [(("a", 10), b"r1"), (("a", 10), b"r2"), (("b", None), b"r3")],
            )
            coll.apply(b"batch-2", 1, iter(BATCH2))
            got, _, _ = drain_snapshot(coll, tok0, ALL_QUERY, 1)
            self.assertEqual(
                got,
                [(("a", 10), b"r1"), (("a", 10), b"r2"), (("b", None), b"r3")],
            )
            # Re-created r2 (gen 2) must not be confused with the deleted one:
            # a current snapshot shows exactly one r2 at (a,10).
            tok2 = coll.begin_snapshot(2)
            got2, _, _ = drain_snapshot(coll, tok2, ALL_QUERY, 1)
            self.assertEqual([rid for _, rid in got2].count(b"r2"), 1)

    # -- empty / all-noop batches still consume a generation ---------------

    def test_empty_and_noop_batches_consume_generations(self):
        with open_collection(self.coll_path) as coll:
            r_empty = coll.apply(b"empty", 0, iter(()))
            self.assertEqual((r_empty.head, r_empty.applied, r_empty.noop), (1, 0, 0))
            r_noop = coll.apply(b"noops", 1, iter([
                {"op": "delete", "row_id": b"does-not-exist"},
                {"op": "put", "region": "a", "ts": 10, "row_id": b"r1"},
            ]))
            self.assertEqual((r_noop.head, r_noop.applied, r_noop.noop), (2, 0, 2))
            # Deleting a missing id is a success no-op, then re-creating works.
            coll.apply(b"delete-r3", 2, iter([
                {"op": "delete", "row_id": b"r3"},
            ]))
            again = coll.apply(b"recreate", 3, iter([
                {"op": "put", "region": "z", "ts": 1, "row_id": b"r3"},
            ]))
            self.assertEqual((again.applied, again.noop), (1, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
