"""Snapshot lifecycle, cursor binding and history collection invariants.

Covers: pinning current/older generations, forged and released credentials,
cursor identity binding (collection, snapshot sid, query, page size, last
key), bytes predicates across pages, releasing one of two same-generation
tokens, monotonic retention boundaries, snapshots below the boundary,
non-contiguous old snapshots, and real physical reclamation (including the
"release last protecting token, then collect" two-step rule).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bytekey import Column, Schema  # noqa: E402
from bytekey.collection import (  # noqa: E402
    HistoryError,
    TokenError,
    create_collection,
    open_collection,
)
from bytekey.snapshot import (  # noqa: E402
    decode_collection_cursor,
    decode_snapshot_token,
    encode_collection_cursor,
    encode_snapshot_token,
)

SCHEMA = Schema(
    columns=(
        Column("region", "text", asc=True, nulls_first=True),
        Column("ts", "int64", asc=False, nulls_first=True),
        Column("blob", "bytes", asc=True, nulls_first=True),
    )
)


def put(rid, region, ts, blob=b"x"):
    return {"op": "put", "row_id": rid, "region": region, "ts": ts, "blob": blob}


def build_history(path):
    """gen0 seed + 4 apply generations with moves, deletes and re-creates."""
    create_collection(path, SCHEMA)
    with open_collection(path) as coll:
        coll.apply(b"g1", 0, iter([
            put(b"r1", "a", 10), put(b"r2", "a", 20), put(b"r3", "b", None),
        ]))
        coll.apply(b"g2", 1, iter([
            {"op": "delete", "row_id": b"r2"},
            put(b"r1", "c", 10),
            put(b"r4", "a", 9, b"\x00\x01"),
        ]))
        coll.apply(b"g3", 2, iter([
            put(b"r1", "a", 8), put(b"r2", "a", 20),  # r2 re-created
        ]))
        coll.apply(b"g4", 3, iter([
            put(b"r5", "d", 1, b"\x00\x00\xff"),
        ]))
        assert coll.status().head == 4


class SnapshotTokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        build_history(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pin_current_and_older_generation(self):
        with open_collection(self.path) as coll:
            latest = coll.begin_snapshot()
            older = coll.begin_snapshot(2)
            t_latest = decode_snapshot_token(latest, coll.secret)
            t_older = decode_snapshot_token(older, coll.secret)
            self.assertEqual((t_latest.gen, t_older.gen), (4, 2))
            q = {"shape": "prefix_range", "prefix": []}
            page_l, _, _ = coll.scan_snapshot_page(latest, q, 100)
            page_o, _, _ = coll.scan_snapshot_page(older, q, 100)
            rids_latest = sorted(rid for _, rid in page_l.rows)
            rids_older = sorted(rid for _, rid in page_o.rows)
            self.assertEqual(rids_latest, [b"r1", b"r2", b"r3", b"r4", b"r5"])
            # gen2: r2 deleted, r5 not yet born.
            self.assertEqual(rids_older, [b"r1", b"r3", b"r4"])

    def test_unknown_generation_refused(self):
        with open_collection(self.path) as coll:
            with self.assertRaises(HistoryError):
                coll.begin_snapshot(99)
            with self.assertRaises(HistoryError):
                coll.begin_snapshot(-1)

    def test_forged_token_refused_no_silent_latest_switch(self):
        with open_collection(self.path) as coll:
            good = coll.begin_snapshot(2)
            # Tamper with the signature portion.
            body, sig = good.split(".")
            forged = body + "." + ("A" * len(sig))
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(forged,
                                        {"shape": "prefix_range", "prefix": []}, 10)
            # Token signed for a different collection id.
            foreign = encode_snapshot_token(
                secret=coll.secret, collection_id="deadbeef", gen=2,
                sid="nope", db_path=coll.path,
            )
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(foreign,
                                        {"shape": "prefix_range", "prefix": []}, 10)
            # A well-formed token with an unknown sid.
            ticket = decode_snapshot_token(good, coll.secret)
            ghost = encode_snapshot_token(
                secret=coll.secret, collection_id=ticket.collection_id, gen=2,
                sid="0" * 32, db_path=ticket.db_path,
            )
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(ghost,
                                        {"shape": "prefix_range", "prefix": []}, 10)

    def test_released_token_and_its_cursor_are_dead(self):
        with open_collection(self.path) as coll:
            q = {"shape": "prefix_range", "prefix": []}
            tok = coll.begin_snapshot(2)
            page, _, _ = coll.scan_snapshot_page(tok, q, 1)
            cursor = page.next_cursor
            coll.release_snapshot(tok)
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(tok, q, 1)
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(tok, q, 1, cursor_token=cursor)
            # Releasing twice is an explicit failure.
            with self.assertRaises(TokenError):
                coll.release_snapshot(tok)

    def test_releasing_one_token_keeps_its_same_generation_sibling(self):
        with open_collection(self.path) as coll:
            q = {"shape": "prefix_range", "prefix": []}
            tok_a = coll.begin_snapshot(2)
            tok_b = coll.begin_snapshot(2)
            coll.release_snapshot(tok_a)
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(tok_a, q, 10)
            page, _, _ = coll.scan_snapshot_page(tok_b, q, 100)
            self.assertEqual(
                sorted(rid for _, rid in page.rows), [b"r1", b"r3", b"r4"]
            )

    def test_cursor_binds_query_page_size_and_snapshot(self):
        with open_collection(self.path) as coll:
            q1 = {"shape": "prefix_range", "prefix": [["region", "a"]],
                  "range_column": "ts"}
            q2 = {"shape": "prefix_range", "prefix": []}
            tok = coll.begin_snapshot(4)
            page, _, _ = coll.scan_snapshot_page(tok, q1, 1)
            cursor = page.next_cursor
            self.assertIsNotNone(cursor)
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(tok, q2, 1, cursor_token=cursor)
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(tok, q1, 2, cursor_token=cursor)
            other_tok = coll.begin_snapshot(4)
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(other_tok, q1, 1, cursor_token=cursor)
            # Tampered cursor signature.
            body, sig = cursor.split(".")
            with self.assertRaises(TokenError):
                coll.scan_snapshot_page(tok, q1, 1,
                                        cursor_token=body + "." + ("B" * len(sig)))
            # The cursor itself carries the full binding.
            decoded = decode_collection_cursor(cursor, coll.secret)
            self.assertEqual(decoded.sid, decode_snapshot_token(tok, coll.secret).sid)
            self.assertEqual(decoded.page_size, 1)

    def test_bytes_predicate_across_pages(self):
        with open_collection(self.path) as coll:
            # r4 and r5 both carry distinctive blobs; r4 = 00 01, r5 = 00 00 ff.
            tok = coll.begin_snapshot(4)
            q = {
                "shape": "prefix_range",
                "prefix": [],
                "range_column": "region",
                "lo": "a",
            }
            # Page size 1 forces the bytes-bearing rows across page boundaries.
            out, pages = [], 0
            cursor = None
            while True:
                page, stats, _ = coll.scan_snapshot_page(
                    tok, q, 1, cursor_token=cursor
                )
                out.extend(page.rows)
                pages += 1
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
            blobs = {rid: vals[2] for vals, rid in out}
            self.assertEqual(blobs[b"r4"], b"\x00\x01")
            self.assertEqual(blobs[b"r5"], b"\x00\x00\xff")
            self.assertGreater(pages, 1)
            # Exact bytes prefix query (full tuple) survives paging too.
            qt = {
                "shape": "full_tuple",
                "lower": ["d", 1, b"\x00\x00\xff"],
                "upper": ["d", 1, b"\x00\x00\xff"],
                "lower_inclusive": True,
                "upper_inclusive": True,
            }
            page, _, _ = coll.scan_snapshot_page(tok, qt, 1)
            self.assertEqual(page.rows, [(("d", 1, b"\x00\x00\xff"), b"r5")])


class HistoryCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        build_history(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _kv_rows(self):
        with open_collection(self.path) as coll:
            return coll.reader.execute(
                "SELECT rid, cgen, dgen FROM bytekey_kv ORDER BY key, rid, cgen"
            ).fetchall(), coll.status()

    def test_boundary_monotonic_and_bounded_by_head(self):
        with open_collection(self.path) as coll:
            r1 = coll.collect_history(2)
            self.assertEqual((r1.previous_retain_from, r1.retain_from), (0, 2))
            with self.assertRaises(HistoryError):
                coll.collect_history(1)  # cannot move backwards
            with self.assertRaises(HistoryError):
                coll.collect_history(99)  # above head
            # A no-op advance at the same value is legal.
            r2 = coll.collect_history(2)
            self.assertEqual(r2.entries_removed, 0)

    def test_old_generation_snapshot_refused_after_collection(self):
        with open_collection(self.path) as coll:
            before = coll.begin_snapshot(1)  # active before collection: kept
            coll.collect_history(3)
            # gen 1 below the new boundary: no NEW snapshot allowed.
            with self.assertRaises(HistoryError):
                coll.begin_snapshot(1)
            # ... but the pre-existing token still reads its pinned data.
            q = {"shape": "prefix_range", "prefix": []}
            page, _, _ = coll.scan_snapshot_page(before, q, 100)
            self.assertEqual(
                sorted(rid for _, rid in page.rows), [b"r1", b"r2", b"r3"]
            )

    def test_non_contiguous_old_snapshot_pin(self):
        # Active snapshots on non-adjacent generations (1 and 4) must each
        # retain every entry their window can see during collection.
        with open_collection(self.path) as coll:
            tok1 = coll.begin_snapshot(1)
            tok4 = coll.begin_snapshot(4)
            coll.collect_history(3)
            q = {"shape": "prefix_range", "prefix": []}
            p1, _, _ = coll.scan_snapshot_page(tok1, q, 100)
            p4, _, _ = coll.scan_snapshot_page(tok4, q, 100)
            self.assertEqual(
                sorted(rid for _, rid in p1.rows), [b"r1", b"r2", b"r3"]
            )
            self.assertEqual(
                sorted(rid for _, rid in p4.rows),
                [b"r1", b"r2", b"r3", b"r4", b"r5"],
            )

    def test_physical_reclamation_two_step_release_then_collect(self):
        rows0, status0 = self._kv_rows()
        total_before = status0.total_version_entries
        with open_collection(self.path) as coll:
            tok = coll.begin_snapshot(1)  # protects gen-1-visible dead entries
            # First collection while the token is active: nothing it needs dies.
            r_active = coll.collect_history(2)
        rows1, status1 = self._kv_rows()
        with open_collection(self.path) as coll:
            coll.release_snapshot(tok)
            # Releasing alone never deletes history.
        rows2, status2 = self._kv_rows()
        self.assertEqual(status2.total_version_entries,
                         status1.total_version_entries)
        with open_collection(self.path) as coll:
            r_after = coll.collect_history(2)
        rows3, status3 = self._kv_rows()
        # Real physical removal happened, receipts and head survived.
        self.assertLess(status3.total_version_entries, total_before)
        self.assertGreater(r_after.entries_removed, 0)
        self.assertEqual(status3.head, 4)
        self.assertEqual(status3.receipt_count, 4)
        # A re-sent collected batch still returns its stored receipt.
        with open_collection(self.path) as coll:
            receipt = coll.lookup_receipt(b"g2")
            self.assertIsNotNone(receipt)
            self.assertEqual(receipt["head"], 2)

    def test_collect_keeps_live_and_retained_history(self):
        with open_collection(self.path) as coll:
            tok3 = coll.begin_snapshot(3)
            coll.collect_history(3)
            # gen-3 and gen-4 snapshots remain constructible.
            tok4 = coll.begin_snapshot(4)
            q = {"shape": "prefix_range", "prefix": []}
            p3, _, _ = coll.scan_snapshot_page(tok3, q, 100)
            p4, _, _ = coll.scan_snapshot_page(tok4, q, 100)
            # r5 born at gen4 is absent at gen3 and present at gen4.
            self.assertNotIn(b"r5", [rid for _, rid in p3.rows])
            self.assertIn(b"r5", [rid for _, rid in p4.rows])
            # Every still-live row is readable.
            self.assertEqual(
                sorted(rid for _, rid in p4.rows),
                [b"r1", b"r2", b"r3", b"r4", b"r5"],
            )


class SnapshotPersistsAcrossProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        build_history(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_token_survives_process_exit_and_new_writes(self):
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "snapshot_page_helper.py")
        q_path = os.path.join(self.tmp.name, "q.json")
        with open(q_path, "w", encoding="utf-8") as fh:
            json.dump({"shape": "prefix_range", "prefix": []}, fh)
        with open_collection(self.path) as coll:
            token = coll.begin_snapshot(2)
        # Fresh process reads the pinned generation without any shared state.
        proc = subprocess.run(
            [sys.executable, "-I", helper, "--collection", self.path,
             "--token", token, "--query", q_path, "--page-size", "2"],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["gen"], 2)
        self.assertEqual(len(payload["rows"]), 2)
        self.assertIsNotNone(payload["next_cursor"])
        # Meanwhile another process publishes more data.
        with open_collection(self.path) as coll:
            coll.apply(b"g5", 4, iter([put(b"r6", "z", 0)]))
        proc2 = subprocess.run(
            [sys.executable, "-I", helper, "--collection", self.path,
             "--token", token, "--query", q_path, "--page-size", "2",
             "--cursor", payload["next_cursor"]],
            capture_output=True, text=True, check=True,
        )
        payload2 = json.loads(proc2.stdout)
        # Still gen 2: r6 (gen 5) is invisible on the old token.
        self.assertEqual(payload2["gen"], 2)
        all_rids = {r["row_id"]["__hex__"] for r in payload["rows"]} | {
            r["row_id"]["__hex__"] for r in payload2["rows"]
        }
        self.assertNotIn(b"r6".hex(), all_rids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
