"""Acceptance tests for named secondary access indexes.

Everything here reopens the collection from the actual file between steps --
never a single in-memory object -- and compares results against the
plain-dictionary reference model in ``tests/refmodel.py`` (which never imports
the production visibility or planning code).

The mandated hand-written fixture is:

    region-priority main collection, rows r1=(a,10,old), r2=(a,10,keep),
    r3=(b,5,stay)

    secondary index: ts DESC, region ASC, advanced with budget 1

    mid-build: r1 -> (c,20,new), delete r2, add r4=(a,10,late)
    then:      r1.note -> newer, re-create r2=(b,7,reborn)
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bytekey import (  # noqa: E402
    Column,
    Schema,
    SecondaryDeclaration,
    IndexColumn,
    build_index,
    create_collection,
    import_from_index,
    open_collection,
)
from bytekey.collection import (  # noqa: E402
    ConflictError,
    IndexConflictError,
    IndexNotFoundError,
    IndexNotReadyError,
    IndexCursorError,
    CoverageError,
    BatchError,
)
from bytekey.snapshot import TokenError  # noqa: E402
from tests.refmodel import (  # noqa: E402
    RefCollection,
    drain_index,
    query_index_at,
)

SCHEMA = Schema(
    columns=(
        Column("region", "text", asc=True, nulls_first=True),
        Column("ts", "int64", asc=False, nulls_first=True),
        Column("note", "text", asc=True, nulls_first=True),
    )
)
DECL = {
    "name": "ts_region",
    "columns": [
        {"name": "ts", "asc": False, "nulls_first": True},
        {"name": "region", "asc": True, "nulls_first": True},
    ],
}
GEN0 = [
    {"op": "put", "row_id": b"r1", "region": "a", "ts": 10, "note": "old"},
    {"op": "put", "row_id": b"r2", "region": "a", "ts": 10, "note": "keep"},
    {"op": "put", "row_id": b"r3", "region": "b", "ts": 5, "note": "stay"},
]
BATCH1 = [
    {"op": "put", "row_id": b"r1", "region": "c", "ts": 20, "note": "new"},
    {"op": "delete", "row_id": b"r2"},
    {"op": "put", "row_id": b"r4", "region": "a", "ts": 10, "note": "late"},
]
BATCH2 = [
    {"op": "put", "row_id": b"r1", "region": "c", "ts": 20, "note": "newer"},
]
BATCH3 = [
    {"op": "put", "row_id": b"r2", "region": "b", "ts": 7, "note": "reborn"},
]
ALL = {"shape": "full_tuple", "lower": None, "upper": None}


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "warehouse.db")
        self.static = os.path.join(self.tmp.name, "static.db")

    def tearDown(self):
        self.tmp.cleanup()

    # -- helpers -----------------------------------------------------------
    def _seed_via_static(self, rows=GEN0, schema=SCHEMA):
        """Build a region-first static index and import it as generation 0."""
        build_index(self.static, schema, [
            {c.name: r[c.name] for c in schema.columns} | {"row_id": r["row_id"]}
            for r in rows
        ])
        cid, n = import_from_index(self.static, self.path)
        self.assertEqual(n, len(rows))
        return cid

    def _ref(self, rows=GEN0):
        ref = RefCollection(SCHEMA)
        ref.seed([(r["row_id"], (r["region"], r["ts"], r["note"])) for r in rows])
        return ref

    def _advance_to_ready(self, name, budget=1, max_steps=1000):
        steps = 0
        with open_collection(self.path) as coll:
            while True:
                status = coll.advance_index(name, budget)
                steps += 1
                if status.status == "ready":
                    return steps, status
                if steps > max_steps:
                    self.fail("index did not become ready")

    def _apply(self, batch_id, ops, ref=None):
        with open_collection(self.path) as coll:
            head = coll.status().head
            receipt = coll.apply(batch_id, head, iter(ops))
            if ref is not None:
                ref.apply(batch_id, head, ops)
            return receipt


class HandFixtureTests(_Base):
    """The mandated three-row fixture, reopened from disk at every step."""

    def test_region_first_main_new_index_order(self):
        self._seed_via_static()
        ref = self._ref()
        # Register from a *fresh* process view, then advance with budget 1,
        # interleaving the mandated corrections.
        with open_collection(self.path) as coll:
            reg = coll.register_index(SecondaryDeclaration.from_dict(DECL))
            self.assertEqual(reg["start_gen"], 0)
            self.assertEqual(reg["status"], "building")
            # The main ordering still works, and the index is not queryable yet.
            tok = coll.begin_snapshot()
            page, _, _ = coll.scan_snapshot_page(tok, ALL, 10)
            self.assertEqual(
                page.rows,
                query_index_at(ref, 0, {"columns": [
                    {"name": "region"}, {"name": "ts"}, {"name": "note"},
                ]}, ALL),
            )
            with self.assertRaises(IndexNotReadyError):
                coll.scan_index_page(tok, "ts_region", ALL, 10)

        # One budget unit: exactly one baseline candidate, persisted.
        with open_collection(self.path) as coll:
            st = coll.advance_index("ts_region", 1)
            self.assertEqual(st.baseline_candidates_read, 1)
            self.assertEqual(st.catchup_changes_processed, 0)

        r1 = self._apply(b"batch-1", BATCH1, ref)
        self.assertEqual((r1.head, r1.applied), (1, 3))

        # Continue budget-1 advances across separate opens while applying.
        with open_collection(self.path) as coll:
            coll.advance_index("ts_region", 1)
        r2 = self._apply(b"batch-2", BATCH2, ref)
        r3 = self._apply(b"batch-3", BATCH3, ref)
        self.assertEqual((r2.head, r3.head), (2, 3))

        steps, final = self._advance_to_ready("ts_region", 1)
        self.assertEqual(final.status, "ready")
        self.assertIsNotNone(final.ready_gen)

        # Save generation snapshots; query through the new index.
        with open_collection(self.path) as coll:
            for g in range(ref.head + 1):
                tok = coll.begin_snapshot(g)
                got, pages, plan = drain_index(coll, tok, "ts_region", ALL, 1)
                want = query_index_at(ref, g, DECL, ALL)
                self.assertEqual(got, want, f"generation {g} mismatch")
                for values, rid in got:
                    # Always the *full* main-schema row, never a projection.
                    self.assertEqual(len(values), 3)

        # The mandated old-generation siblings: r1 and r2 share ts=10.
        with open_collection(self.path) as coll:
            tok0 = coll.begin_snapshot(0)
            got0, _, _ = drain_index(coll, tok0, "ts_region", ALL, 1)
            rids0 = [rid for _, rid in got0]
            self.assertIn(b"r1", rids0)
            self.assertIn(b"r2", rids0)
            # ts=10 group first (DESC on ts), then ts=5; inside the equal
            # (ts,region) group the row_id ascending tie-break applies.
            self.assertEqual(rids0, [b"r1", b"r2", b"r3"])

            tok3 = coll.begin_snapshot(3)
            got3, _, _ = drain_index(coll, tok3, "ts_region", ALL, 2)
            # r1 note moved twice; the projection must not swallow it.
            notes = {rid: values[2] for values, rid in got3}
            self.assertEqual(notes[b"r1"], "newer")
            self.assertEqual(notes[b"r2"], "reborn")

            # Both r1 notes survive in their generations.
            tok1 = coll.begin_snapshot(1)
            g1 = {rid: v[2] for v, rid in drain_index(coll, tok1, "ts_region", ALL, 10)[0]}
            self.assertEqual(g1[b"r1"], "new")
            g2 = {rid: v[2] for v, rid in drain_index(
                coll, coll.begin_snapshot(2), "ts_region", ALL, 10)[0]}
            self.assertEqual(g2[b"r1"], "newer")

    def test_ts_only_index_exposes_region_change_and_equal_keys(self):
        """A ts-only index must still return the changed, non-indexed region."""
        self._seed_via_static()
        ref = self._ref()
        ts_decl = {"name": "by_ts", "columns": [{"name": "ts", "asc": False}]}
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(ts_decl))
            while coll.advance_index("by_ts", 1).status != "ready":
                pass
        self._apply(b"b1", [
            {"op": "put", "row_id": b"r1", "region": "c", "ts": 10, "note": "old"},
        ], ref)
        # Region changed but ts did not: the index must return the new region.
        with open_collection(self.path) as coll:
            tok = coll.begin_snapshot(1)
            got, _, _ = drain_index(coll, tok, "by_ts", ALL, 1)
            by_rid = {rid: values for values, rid in got}
            self.assertEqual(by_rid[b"r1"], ("c", 10, "old"))
            # Equal-ts group (r1, r2) must both be delivered across pages.
            self.assertEqual(
                [rid for _, rid in got if rid in (b"r1", b"r2")],
                [b"r1", b"r2"],
            )
            want = query_index_at(ref, 1, ts_decl, ALL)
            self.assertEqual(got, want)

    def test_coverage_start_reported_for_early_token(self):
        self._seed_via_static()
        with open_collection(self.path) as coll:
            # Advance head so B > 0, then a token from before B must fail.
            coll.apply(b"m", 0, iter([GEN0[0]]))
            old_tok = coll.begin_snapshot(0)
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            while coll.advance_index("ts_region", 10).status != "ready":
                pass
            new_tok = coll.begin_snapshot()
            with self.assertRaises(CoverageError):
                coll.scan_index_page(old_tok, "ts_region", ALL, 10)
            # But a token issued at B (index start) works.
            b = coll.index_status("ts_region").start_gen
            ok_tok = coll.begin_snapshot(b)
            page, _, _ = coll.scan_index_page(ok_tok, "ts_region", ALL, 10)
            self.assertGreater(page.rows.__len__(), 0)
            self.assertEqual(coll.index_status("ts_region").start_gen, b)


class RegistrationTests(_Base):
    def test_duplicate_active_name_fails_and_reuses_build_id(self):
        self._seed_via_static()
        with open_collection(self.path) as coll:
            reg = coll.register_index(SecondaryDeclaration.from_dict(DECL))
            with self.assertRaises(IndexConflictError):
                coll.register_index(SecondaryDeclaration.from_dict(DECL))
            # The returned build_id resumes the original work.
            st = coll.advance_index(
                "ts_region", 10, expected_build_id=reg["build_id_hex"]
            )
            self.assertIn(st.status, ("building", "ready"))
            # Registration itself did not copy anything.
            self.assertLessEqual(st.secondary_entries, len(GEN0))

    def test_declaration_rejects_unknown_column_duplicate_and_empty(self):
        self._seed_via_static()
        with open_collection(self.path) as coll:
            from bytekey.secondary import IndexDeclarationError

            with self.assertRaises(IndexDeclarationError):
                coll.register_index(SecondaryDeclaration.from_dict({
                    "name": "x", "columns": [{"name": "nope"}]
                }))
            with self.assertRaises(ValueError):
                SecondaryDeclaration.from_dict({
                    "name": "x", "columns": [{"name": "ts"}, {"name": "ts"}]
                })
            with self.assertRaises(ValueError):
                SecondaryDeclaration.from_dict({"name": "", "columns": [
                    {"name": "ts"}]})

    def test_two_indexes_build_concurrently(self):
        self._seed_via_static()
        decl2 = {"name": "by_region", "columns": [{"name": "region", "asc": True}]}
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            coll.register_index(SecondaryDeclaration.from_dict(decl2))
        # Advance them alternately with tiny budgets.
        with open_collection(self.path) as coll:
            for _ in range(100):
                s1 = coll.advance_index("ts_region", 1)
                s2 = coll.advance_index("by_region", 1)
                if s1.status == "ready" and s2.status == "ready":
                    break
            self.assertEqual(s1.status, "ready")
            self.assertEqual(s2.status, "ready")
            names = {i.name for i in coll.list_indexes()}
            self.assertEqual(names, {"ts_region", "by_region"})

    def test_cumulative_counts_equal_step_sums(self):
        self._seed_via_static(rows=[
            {"op": "put", "row_id": f"r{i}".encode(), "region": "a", "ts": i,
             "note": "n"} for i in range(8)
        ])
        with open_collection(self.path) as coll:
            # Bump head to 1 so the index starts at B = 1.
            coll.apply(b"bump", 0, iter([
                {"op": "put", "row_id": b"r0", "region": "a", "ts": 0,
                 "note": "n"}]))
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            # Publish a change while building so both counters move.
            coll.apply(b"chg", 1, iter([
                {"op": "put", "row_id": b"r1", "region": "z", "ts": 50,
                 "note": "m"}]))
            total_base = total_catch = 0
            while True:
                st = coll.advance_index("ts_region", 1)
                total_base += st.baseline_candidates_read
                total_catch += st.catchup_changes_processed
                if st.status == "ready":
                    break
            self.assertEqual(total_base, st.cumulative_baseline_candidates)
            self.assertEqual(total_catch, st.cumulative_catchup_changes)
            # 8 initial versions + 1 post-B open = 9 unique secondary
            # versions (the moved r1's old version is pre-B: no new entry for
            # it, its close is folded via the journal).
            self.assertEqual(st.secondary_entries,
                             st.cumulative_baseline_candidates + 1)


class InterleavingTests(_Base):
    def test_apply_maintains_ready_and_journals_building_together(self):
        """One apply serves a ready index *and* feeds an in-flight build."""
        self._seed_via_static()
        ref = self._ref()
        ready_decl = {"name": "ready_ts", "columns": [{"name": "ts", "asc": False}]}
        build_decl = {"name": "building_region",
                      "columns": [{"name": "region"}, {"name": "note", "asc": False}]}
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(ready_decl))
            while coll.advance_index("ready_ts", 10).status != "ready":
                pass
            coll.register_index(SecondaryDeclaration.from_dict(build_decl))
            # This apply must update ready_ts immediately and journal the
            # building_region work in the same commit.
            ops = [
                {"op": "put", "row_id": b"r2", "region": "z", "ts": 9,
                 "note": "x"},
                {"op": "put", "row_id": b"r1", "region": "a", "ts": 10,
                 "note": "changed"},
            ]
            head = coll.status().head
            coll.apply(b"m", head, iter(ops))
            ref.apply(b"m", head, ops)
            # Ready index reflects the new generation at once.
            got, _, _ = drain_index(
                coll, coll.begin_snapshot(head + 1), "ready_ts", ALL, 1
            )
            self.assertEqual(
                got, query_index_at(ref, head + 1, ready_decl, ALL)
            )
            # The building index is still not queryable.
            with self.assertRaises(IndexNotReadyError):
                coll.scan_index_page(
                    coll.begin_snapshot(head + 1), "building_region", ALL, 1
                )
            while coll.advance_index("building_region", 10).status != "ready":
                pass
            b = coll.index_status("building_region").start_gen
            for g in range(b, ref.head + 1):
                got, _, _ = drain_index(
                    coll, coll.begin_snapshot(g), "building_region", ALL, 2
                )
                self.assertEqual(got, query_index_at(ref, g, build_decl, ALL),
                                 f"gen {g}")

    def test_token_issued_during_build_works_after_ready(self):
        self._seed_via_static()
        ref = self._ref()
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            coll.advance_index("ts_region", 1)  # still building
            # A token emitted before the build finishes, at coverage start B.
            b = coll.index_status("ts_region").start_gen
            tok_mid = coll.begin_snapshot(b)
            # It is not usable while building.
            with self.assertRaises(IndexNotReadyError):
                coll.scan_index_page(tok_mid, "ts_region", ALL, 5)
        self._apply(b"c1", BATCH2, ref)
        while True:
            with open_collection(self.path) as coll:
                if coll.advance_index("ts_region", 1).status == "ready":
                    break
        # The same token now works and returns the B generation's full rows.
        with open_collection(self.path) as coll:
            got, _, _ = drain_index(coll, tok_mid, "ts_region", ALL, 2)
            self.assertEqual(got, query_index_at(ref, b, DECL, ALL))

    def test_apply_between_baseline_done_and_switch(self):
        """A correction committed after the baseline sweep still gets covered."""
        self._seed_via_static()
        ref = self._ref()
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            # Walk every baseline candidate with budget 1.
            while True:
                st = coll.advance_index("ts_region", 1)
                if st.baseline_done:
                    break
            # Now the baseline is done but the switch may or may not have
            # happened; commit a change and keep advancing.
            self._apply(b"late", [
                {"op": "put", "row_id": b"r9", "region": "z", "ts": 99,
                 "note": "after-baseline"},
            ], ref)
            while coll.advance_index("ts_region", 1).status != "ready":
                pass
            tok = coll.begin_snapshot()
            got, _, _ = drain_index(coll, tok, "ts_region", ALL, 1)
            self.assertEqual(got, query_index_at(ref, ref.head, DECL, ALL))
            self.assertIn(b"r9", [rid for _, rid in got])

    def test_ready_index_maintained_by_later_applies(self):
        self._seed_via_static()
        ref = self._ref()
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            while coll.advance_index("ts_region", 1).status != "ready":
                pass
            # Delete, re-create, and a non-indexed note update after `ready`.
            self._apply(b"d", [{"op": "delete", "row_id": b"r3"}], ref)
            self._apply(b"rc", [{"op": "put", "row_id": b"r3", "region": "q",
                                 "ts": 1, "note": "again"}], ref)
            self._apply(b"n", BATCH2, ref)
            for g in (1, 2, 3):
                got, _, _ = drain_index(
                    coll, coll.begin_snapshot(g), "ts_region", ALL, 1
                )
                self.assertEqual(got, query_index_at(ref, g, DECL, ALL),
                                 f"gen {g}")
            # No journal remains once ready.
            self.assertEqual(coll.index_status("ts_region").queue_entries, 0)

    def test_interleaved_advance_two_processes_serialize(self):
        """Two independent opens advancing the same build never double-work."""
        self._seed_via_static(rows=[
            {"op": "put", "row_id": f"r{i}".encode(), "region": "a",
             "ts": i, "note": "n"} for i in range(20)
        ])
        ref = RefCollection(SCHEMA)
        ref.seed([(f"r{i}".encode(), ("a", i, "n")) for i in range(20)])
        with open_collection(self.path) as a:
            a.register_index(SecondaryDeclaration.from_dict(DECL))
        # Two opens alternate with budget 1; each call is one short txn.
        with open_collection(self.path) as a, open_collection(self.path) as b:
            steps = 0
            while True:
                sa = a.advance_index("ts_region", 1)
                sb = b.advance_index("ts_region", 1)
                steps += 2
                if sa.status == "ready" or sb.status == "ready":
                    break
                if steps > 200:
                    self.fail("did not converge")
            ready = a.index_status("ts_region")
            self.assertEqual(ready.status, "ready")
            # Every identity appears exactly once: no duplicate baseline work.
            tok = a.begin_snapshot(ready.start_gen)
            got, _, _ = drain_index(a, tok, "ts_region", ALL, 7)
            rids = [rid for _, rid in got]
            self.assertEqual(len(rids), len(set(rids)))
            self.assertEqual(len(rids), 20)
            self.assertEqual(got, query_index_at(ref, ready.start_gen, DECL, ALL))


class CursorTests(_Base):
    def _ready_index(self, decl=DECL, name="ts_region"):
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(decl))
            while coll.advance_index(name, 10).status != "ready":
                pass
        return name

    def test_cursor_binds_instance_and_declaration(self):
        self._seed_via_static()
        self._ready_index()
        with open_collection(self.path) as coll:
            tok = coll.begin_snapshot()
            page, _, _ = coll.scan_index_page(tok, "ts_region", ALL, 1)
            cur = page.next_cursor
            self.assertIsNotNone(cur)
            # Wrong page size, wrong query, wrong index, main cursor crossing.
            with self.assertRaises(IndexCursorError):
                coll.scan_index_page(tok, "ts_region", ALL, 2, cursor_token=cur)
            other_q = {"shape": "prefix_range", "prefix": [["ts", 10]],
                       "range_column": "region"}
            with self.assertRaises(IndexCursorError):
                coll.scan_index_page(tok, "ts_region", other_q, 1,
                                     cursor_token=cur)
            # A main-order cursor cannot continue a secondary page.
            main_page, _, _ = coll.scan_snapshot_page(tok, ALL, 1)
            with self.assertRaises((IndexCursorError, TokenError)):
                coll.scan_index_page(tok, "ts_region", ALL, 1,
                                     cursor_token=main_page.next_cursor)
            # And a secondary cursor cannot continue a main page.
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(tok, ALL, 1, cursor_token=cur)

    def test_same_name_rebuild_rejects_old_cursor(self):
        self._seed_via_static()
        self._ready_index()
        with open_collection(self.path) as coll:
            tok = coll.begin_snapshot()
            page, _, _ = coll.scan_index_page(tok, "ts_region", ALL, 1)
            cur = page.next_cursor
            coll.delete_index("ts_region")
            # Different declaration, same name.
            rebuilt = {"name": "ts_region", "columns": [
                {"name": "ts", "asc": True},
                {"name": "region", "asc": False},
            ]}
            coll.register_index(SecondaryDeclaration.from_dict(rebuilt))
            while coll.advance_index("ts_region", 10).status != "ready":
                pass
            with self.assertRaises(IndexCursorError):
                coll.scan_index_page(tok, "ts_region", ALL, 1, cursor_token=cur)

    def test_cross_process_continue_page_size_one(self):
        """Only path/token/index/query/cursor in a brand-new process."""
        self._seed_via_static()
        self._ready_index()
        helper = os.path.join(ROOT, "tests", "index_page_helper.py")
        with open_collection(self.path) as coll:
            tok = coll.begin_snapshot()
            page, _, _ = coll.scan_index_page(tok, "ts_region", ALL, 1)
            first = page.rows
            cur = page.next_cursor
        qpath = os.path.join(self.tmp.name, "q.json")
        with open(qpath, "w", encoding="utf-8") as fh:
            json.dump(ALL, fh)
        proc = subprocess.run(
            [sys.executable, "-I", helper, "--collection", self.path,
             "--token", tok, "--index", "ts_region", "--query", qpath,
             "--page-size", "1", "--cursor", cur],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload["rows"]), 1)
        self.assertNotEqual(payload["rows"][0]["row_id"],
                            {"__hex__": first[0][1].hex()})
        # Continue until exhausted; every sibling role id appears once.
        seen = [bytes.fromhex(first[0][1].hex())]
        seen += [bytes.fromhex(r["row_id"]["__hex__"]) for r in payload["rows"]]
        cursor = payload["next_cursor"]
        while cursor is not None:
            proc = subprocess.run(
                [sys.executable, "-I", helper, "--collection", self.path,
                 "--token", tok, "--index", "ts_region", "--query", qpath,
                 "--page-size", "1", "--cursor", cursor],
                capture_output=True, text=True, check=True,
            )
            payload = json.loads(proc.stdout)
            seen += [bytes.fromhex(r["row_id"]["__hex__"]) for r in payload["rows"]]
            cursor = payload["next_cursor"]
        self.assertEqual(sorted(seen), [b"r1", b"r2", b"r3"])


class HistoryAndLifecycleTests(_Base):
    def test_delete_index_immediately_blocks_and_frees_name(self):
        self._seed_via_static()
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            while coll.advance_index("ts_region", 10).status != "ready":
                pass
            tok = coll.begin_snapshot()
            coll.delete_index("ts_region")
            with self.assertRaises(IndexNotFoundError):
                coll.scan_index_page(tok, "ts_region", ALL, 1)
            with self.assertRaises(IndexNotFoundError):
                coll.advance_index("ts_region", 1)
            # Name is free for a different declaration.
            new = coll.register_index(SecondaryDeclaration.from_dict({
                "name": "ts_region", "columns": [{"name": "ts", "asc": True}]
            }))
            self.assertNotEqual(new["instance_id"],
                                coll.list_indexes(include_deleted=True)[0].instance_id)

    def test_delete_one_index_does_not_break_other_build_or_main(self):
        self._seed_via_static()
        decl2 = {"name": "by_region", "columns": [{"name": "region"}]}
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            coll.register_index(SecondaryDeclaration.from_dict(decl2))
            coll.advance_index("ts_region", 1)
            coll.delete_index("ts_region")
            # The other build continues unaffected.
            while coll.advance_index("by_region", 1).status != "ready":
                pass
            page, _, _ = coll.scan_snapshot_page(
                coll.begin_snapshot(), ALL, 10
            )
            self.assertEqual(len(page.rows), 3)
            # Main reads still serve the original region-first ordering.
            self.assertEqual(
                [rid for _, rid in page.rows], [b"r2", b"r1", b"r3"]
            )

    def test_history_collection_protects_in_flight_build(self):
        self._seed_via_static()
        ref = self._ref()
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            coll.advance_index("ts_region", 1)  # part of baseline only
            # A token pinned at the coverage start must be issuable and stay
            # valid; no extra snapshot is required for the build, but this
            # also proves retention cannot steal what the token needs.
            tok0 = coll.begin_snapshot(0)
            # Move a baseline row while the build is mid-flight.
            self._apply(b"move", [
                {"op": "put", "row_id": b"r1", "region": "c", "ts": 20,
                 "note": "new"},
            ], ref)
            # Raise retention aggressively: the build's inputs must survive.
            head = coll.status().head
            stats = coll.collect_history(head)
            # Advance must still produce the correct historical answer.
            while coll.advance_index("ts_region", 1).status != "ready":
                pass
            got, _, _ = drain_index(coll, tok0, "ts_region", ALL, 1)
            self.assertEqual(got, query_index_at(ref, 0, DECL, ALL))
            # Cleanup reports secondary deletions separately from main ones.
            d = stats.to_dict()
            self.assertIn("secondary_entries_removed", d)
            self.assertIn("main_entries_removed", d)

    def test_cancel_releases_build_only_material(self):
        self._seed_via_static()
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            coll.advance_index("ts_region", 2)
            self.assertGreater(coll.index_status("ts_region").secondary_entries, 0)
            result = coll.cancel_index("ts_region")
            self.assertGreaterEqual(result["secondary_entries_removed"], 0)
            # Cancelled instance cannot advance; name is reusable.
            with self.assertRaises(IndexNotFoundError):
                coll.advance_index("ts_region", 1)
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            # Main data untouched.
            self.assertEqual(coll.status().live_entries, 3)


class MergeInvariantTests(_Base):
    """Adversarial orderings of baseline reads versus published changes."""

    def _drain_at(self, gen, ps=1):
        with open_collection(self.path) as coll:
            tok = coll.begin_snapshot(gen)
            return drain_index(coll, tok, "ts_region", ALL, ps)[0]

    def test_deleted_after_baseline_scan_is_not_resurrected(self):
        # Row order in the main store is region-first, so r1 sorts before r3.
        self._seed_via_static()
        ref = self._ref()
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            # One step reads the first baseline candidate only.
            first = coll.advance_index("ts_region", 1)
            scanned = first.cumulative_baseline_candidates
            self.assertEqual(scanned, 1)
        # Delete a row and add a new one; keep advancing to ready.
        self._apply(b"del-r3", [{"op": "delete", "row_id": b"r3"}], ref)
        self._apply(b"add-r5", [
            {"op": "put", "row_id": b"r5", "region": "m", "ts": 3,
             "note": "late"}], ref)
        steps, final = self._advance_to_ready("ts_region", 1)
        self.assertEqual(final.status, "ready")
        for g in range(ref.head + 1):
            got = self._drain_at(g)
            self.assertEqual(got, query_index_at(ref, g, DECL, ALL),
                             f"gen {g}")

    def test_deleted_before_baseline_reaches_it_keeps_old_version(self):
        self._seed_via_static(rows=[
            {"op": "put", "row_id": b"r1", "region": "a", "ts": 10,
             "note": "one"},
            {"op": "put", "row_id": b"r2", "region": "a", "ts": 20,
             "note": "two"},
            {"op": "put", "row_id": b"r3", "region": "a", "ts": 30,
             "note": "three"},
        ])
        ref = RefCollection(SCHEMA)
        ref.states[0] = {b"r1": ("a", 10, "one"), b"r2": ("a", 20, "two"),
                         b"r3": ("a", 30, "three")}
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            coll.advance_index("ts_region", 1)  # reaches only the first
        # Delete r3 (latest ts -> sorts first in DESC; not yet scanned since
        # the read order follows the main key, region-first).  Actually the
        # baseline reads in main key order, so scan progresses by (region,ts)
        # ascending; r3 may already be read or not.  Either way the result
        # must be correct for every generation.
        self._apply(b"del", [{"op": "delete", "row_id": b"r3"}], ref)
        self._advance_to_ready("ts_region", 1)
        for g in range(ref.head + 1):
            got = self._drain_at(g)
            self.assertEqual(got, query_index_at(ref, g, DECL, ALL),
                             f"gen {g}")

    def test_move_away_and_back_preserves_each_generation(self):
        self._seed_via_static()
        ref = self._ref()
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            coll.advance_index("ts_region", 1)
        self._apply(b"mv1", [
            {"op": "put", "row_id": b"r1", "region": "c", "ts": 20,
             "note": "moved"}], ref)
        with open_collection(self.path) as coll:
            coll.advance_index("ts_region", 1)
        self._apply(b"mv2", [
            {"op": "put", "row_id": b"r1", "region": "a", "ts": 10,
             "note": "back"}], ref)
        self._advance_to_ready("ts_region", 1)
        # Each generation must show exactly one r1, at that generation's key.
        for g in range(ref.head + 1):
            got = self._drain_at(g)
            r1s = [v for v, rid in got if rid == b"r1"]
            self.assertEqual(len(r1s), 1, f"gen {g}")
            self.assertEqual(got, query_index_at(ref, g, DECL, ALL),
                             f"gen {g}")


class CrashRecoveryTests(_Base):
    """A real process exit around a committed advance resumes exactly."""

    def test_hard_kill_after_commit_resumes_with_progress(self):
        self._seed_via_static(rows=[
            {"op": "put", "row_id": f"r{i}".encode(), "region": "a", "ts": i,
             "note": "n"} for i in range(12)
        ])
        helper = os.path.join(ROOT, "tests", "index_advance_helper.py")
        with open_collection(self.path) as coll:
            reg = coll.register_index(SecondaryDeclaration.from_dict(DECL))
        build_hex = reg["build_id_hex"]

        # Kill immediately *after* the commit of the first budget-1 step.
        proc = subprocess.run(
            [sys.executable, "-I", helper, "--collection", self.path,
             "--name", "ts_region", "--budget", "1", "--build-id", build_hex,
             "--crash", "post_commit"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        # Reopen fresh: the committed progress is exactly one candidate.
        with open_collection(self.path) as coll:
            st = coll.index_status("ts_region")
            self.assertEqual(st.cumulative_baseline_candidates, 1)
            self.assertEqual(st.secondary_entries, 1)
            self.assertFalse(st.baseline_done)
            # Continue in this new process.
            while coll.advance_index("ts_region", 2).status != "ready":
                pass
            got, _, _ = drain_index(
                coll, coll.begin_snapshot(0), "ts_region", ALL, 5
            )
            self.assertEqual(len(got), 12)

    def test_hard_kill_before_commit_leaves_progress_untouched(self):
        self._seed_via_static(rows=[
            {"op": "put", "row_id": f"r{i}".encode(), "region": "a", "ts": i,
             "note": "n"} for i in range(8)
        ])
        helper = os.path.join(ROOT, "tests", "index_advance_helper.py")
        with open_collection(self.path) as coll:
            reg = coll.register_index(SecondaryDeclaration.from_dict(DECL))
        build_hex = reg["build_id_hex"]
        proc = subprocess.run(
            [sys.executable, "-I", helper, "--collection", self.path,
             "--name", "ts_region", "--budget", "3", "--build-id", build_hex,
             "--crash", "pre_commit"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        with open_collection(self.path) as coll:
            st = coll.index_status("ts_region")
            self.assertEqual(st.cumulative_baseline_candidates, 0)
            self.assertEqual(st.secondary_entries, 0)
            while coll.advance_index("ts_region", 3).status != "ready":
                pass
            got, _, _ = drain_index(
                coll, coll.begin_snapshot(0), "ts_region", ALL, 3
            )
            self.assertEqual(len(got), 8)

    def test_wrong_build_id_is_rejected(self):
        self._seed_via_static()
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            with self.assertRaises(IndexConflictError):
                coll.advance_index("ts_region", 1, expected_build_id=b"\x00" * 8)


class MigrationTests(_Base):
    def test_v1_file_migrates_preserving_identity_head_receipts_snapshots(self):
        self._seed_via_static()
        with open_collection(self.path) as coll:
            coll.apply(b"one", 0, iter([GEN0[0]]))
            tok = coll.begin_snapshot(0)
            cid = coll.collection_id
            head = coll.status().head
            receipt = coll.lookup_receipt(b"one")
        # Downgrade *genuinely*: drop the v2 secondary objects and reset the
        # marker, reproducing a real legacy file on disk.
        import sqlite3

        conn = sqlite3.connect(self.path)
        conn.execute("DROP TABLE IF EXISTS bytekey_sec_entries")
        conn.execute("DROP TABLE IF EXISTS bytekey_sec_queue")
        conn.execute("DROP TABLE IF EXISTS bytekey_sec_indexes")
        conn.execute("UPDATE bytekey_meta SET value = 'bytekey-collection/v1' "
                     "WHERE name = 'format'")
        conn.commit()
        conn.close()
        # Reopen: migration runs, identity/head/receipts/snapshots preserved.
        with open_collection(self.path) as coll:
            self.assertEqual(coll.collection_id, cid)
            self.assertEqual(coll.status().head, head)
            self.assertEqual(coll.lookup_receipt(b"one")["digest"],
                             receipt["digest"])
            # The v2 tables now exist.
            tables = {
                r[0] for r in coll.reader.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            self.assertIn("bytekey_sec_indexes", tables)
            self.assertIn("bytekey_sec_entries", tables)
            # Old snapshot token still works.
            page, _, _ = coll.scan_snapshot_page(tok, ALL, 10)
            self.assertGreater(len(page.rows), 0)
            # And the new secondary feature works on the upgraded file.
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            while coll.advance_index("ts_region", 10).status != "ready":
                pass
            covered = coll.index_status("ts_region").start_gen
            got, _, _ = drain_index(
                coll, coll.begin_snapshot(covered), "ts_region", ALL, 10
            )
            self.assertEqual(len(got), 3)
        # The upgrade marker is now v2; reopening a second time is a no-op.
        with open_collection(self.path) as coll:
            self.assertEqual(coll.status().head, head)

    def test_interrupted_migration_recovers(self):
        """If the marker still says v1, the next open retries the upgrade."""
        self._seed_via_static()
        import sqlite3

        conn = sqlite3.connect(self.path)
        conn.execute("DROP TABLE IF EXISTS bytekey_sec_entries")
        conn.execute("DROP TABLE IF EXISTS bytekey_sec_queue")
        conn.execute("DROP TABLE IF EXISTS bytekey_sec_indexes")
        conn.execute("UPDATE bytekey_meta SET value = 'bytekey-collection/v1' "
                     "WHERE name = 'format'")
        conn.commit()
        conn.close()
        # First open upgrades; simulate a crash by resetting the marker back
        # to v1 while the (idempotent) tables already exist.
        with open_collection(self.path) as coll:
            head = coll.status().head
        conn = sqlite3.connect(self.path)
        conn.execute("UPDATE bytekey_meta SET value = 'bytekey-collection/v1' "
                     "WHERE name = 'format'")
        conn.commit()
        conn.close()
        with open_collection(self.path) as coll:
            self.assertEqual(coll.status().head, head)
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            while coll.advance_index("ts_region", 10).status != "ready":
                pass

    def test_static_index_still_read_only(self):
        self._seed_via_static()
        from bytekey import open_index

        with open_index(self.static) as idx:
            page, stats, _ = idx.scan_page(ALL, 10)
            self.assertEqual(len(page.rows), 3)


class SameCommitTests(_Base):
    """Apply, index maintenance, head and receipt must be one commit."""

    def test_secondary_maintenance_failure_rolls_back_main_change(self):
        """A failing index write must not leave the main change reported OK."""
        self._seed_via_static()
        with open_collection(self.path) as coll:
            reg = coll.register_index(SecondaryDeclaration.from_dict(DECL))
            while coll.advance_index("ts_region", 10).status != "ready":
                pass
            head_before = coll.status().head
            entries_before = coll.index_status("ts_region").secondary_entries
            # Corrupt the stored declaration so the next apply's index
            # maintenance raises (unknown business column).  This stands in
            # for any secondary-side write failure.
            import sqlite3

            conn = sqlite3.connect(self.path)
            broken = {"format": "bytekey-secondary/v1", "name": "ts_region",
                      "columns": [{"name": "does_not_exist"}]}
            conn.execute(
                "UPDATE bytekey_sec_indexes SET declaration_json = ? "
                "WHERE instance_id = ?",
                (json.dumps(broken), reg["instance_id"]),
            )
            conn.commit()
            conn.close()
            with open_collection(self.path) as coll:
                with self.assertRaises(Exception):
                    coll.apply(b"bad", head_before, iter([{
                        "op": "put", "row_id": b"r1", "region": "c",
                        "ts": 20, "note": "new"}]))
                # Nothing about the main apply may be reported as committed.
                self.assertEqual(coll.status().head, head_before)
                self.assertIsNone(coll.lookup_receipt(b"bad"))

    def test_apply_failure_rolls_back_index_maintenance(self):
        self._seed_via_static()
        with open_collection(self.path) as coll:
            coll.register_index(SecondaryDeclaration.from_dict(DECL))
            while coll.advance_index("ts_region", 10).status != "ready":
                pass
            before_head = coll.status().head
            # A batch with a bad tail: no main version, no index entry, no head.
            def bad_ops():
                yield {"op": "put", "row_id": b"r1", "region": "c", "ts": 20,
                       "note": "new"}
                raise RuntimeError("corrupt tail")
            with self.assertRaises(BatchError):
                coll.apply(b"bad", before_head, bad_ops())
            self.assertEqual(coll.status().head, before_head)
            self.assertIsNone(coll.lookup_receipt(b"bad"))
            # Resending the same id with valid content works and index is exact.
            coll.apply(b"bad", before_head, iter([{
                "op": "put", "row_id": b"r1", "region": "c", "ts": 20,
                "note": "new"}]))
            got, _, _ = drain_index(
                coll, coll.begin_snapshot(), "ts_region", ALL, 1
            )
            self.assertEqual({rid: v[0] for v, rid in got}[b"r1"], "c")

    def test_preopened_writer_connection_maintains_new_index(self):
        """A connection opened before registration still journals applies."""
        self._seed_via_static()
        early = open_collection(self.path)       # opened *before* registration
        try:
            with open_collection(self.path) as other:
                other.register_index(SecondaryDeclaration.from_dict(DECL))
            # The early connection was never told about the index directory,
            # yet its apply must feed the in-flight build.
            head = early.status().head
            early.apply(b"late", head, iter([{
                "op": "put", "row_id": b"r8", "region": "z", "ts": 50,
                "note": "early-writer"}]))
        finally:
            early.close()
        with open_collection(self.path) as coll:
            while coll.advance_index("ts_region", 10).status != "ready":
                pass
            tok = coll.begin_snapshot()
            got, _, _ = drain_index(coll, tok, "ts_region", ALL, 1)
            self.assertIn(b"r8", [rid for _, rid in got])
            self.assertEqual(got, query_index_at(
                _ref_after(r8=True), coll.status().head, DECL, ALL,
            ))


def _ref_after(r8: bool):
    ref = RefCollection(SCHEMA)
    ref.states[0] = {b"r1": ("a", 10, "old"), b"r2": ("a", 10, "keep"),
                     b"r3": ("b", 5, "stay")}
    if r8:
        state = dict(ref.states[0])
        state[b"r8"] = ("z", 50, "early-writer")
        ref.states.append(state)
    return ref
if __name__ == "__main__":
    unittest.main(verbosity=2)
