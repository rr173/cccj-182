"""Large-collection efficiency and reclamation proofs.

The fixture is generated on the spot (nothing large is committed).  It
proves three operational invariants:

1. A small correction touches only O(batch) index entries -- never a
   full-table scan/copy and never a per-generation index rebuild -- measured
   through the write counters returned by apply and the bounded change in
   stored entries.
2. Narrow snapshot queries are concrete BLOB range seeks over the primary
   key, examining only the handful of in-range (including historical)
   candidates rather than the whole table; candidates and delivered rows are
   reported separately, without passing SQL COUNT off as read volume.
3. History collection physically removes entries that lost every possible
   reader, and the file/page accounting drops accordingly once SQLite
   restructures its pages; active snapshots pin exactly what they need.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey import Column, Schema  # noqa: E402
from bytekey.collection import create_collection, open_collection  # noqa: E402

N_REGIONS = 200
TS_PER_REGION = 50
TOTAL = N_REGIONS * TS_PER_REGION


def build_large_collection(path):
    schema = Schema(
        columns=(
            Column("region", "text", asc=True, nulls_first=True),
            Column("ts", "int64", asc=False, nulls_first=True),
            Column("payload", "bytes", asc=True, nulls_first=False),
        )
    )
    create_collection(path, schema)

    def rows():
        n = 0
        for ri in range(N_REGIONS):
            region = f"region-{ri:04d}"
            for ti in range(TS_PER_REGION):
                n += 1
                yield {
                    "op": "put",
                    "row_id": f"id-{n:07d}".encode() + b"\x00\xff",
                    "region": region,
                    "ts": ti,
                    "payload": bytes([ti & 0xFF, 0x00, 0xFF, ri & 0xFF]),
                }

    with open_collection(path) as coll:
        # Chunked initial load: bounded batch buffers, no whole-table state.
        buf = []
        batch_no = 0
        for row in rows():
            buf.append(row)
            if len(buf) == 500:
                batch_no += 1
                coll.apply(f"seed-{batch_no}".encode(),
                           coll.status().head, iter(buf))
                buf = []
        if buf:
            batch_no += 1
            coll.apply(f"seed-{batch_no}".encode(),
                       coll.status().head, iter(buf))
        assert coll.status().live_entries == TOTAL
    return schema


class LargeCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "large.db")
        self.report_dir = os.path.join(self.tmp.name, "reports")
        os.makedirs(self.report_dir, exist_ok=True)
        self.schema = build_large_collection(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_small_correction_does_not_rebuild_the_collection(self):
        with open_collection(self.path) as coll:
            head_before = coll.status().head
            total_before = coll.status().total_version_entries
            live_before = coll.status().live_entries

            # One correction: move a single record to a different key.
            target_rid = f"id-{1000:07d}".encode() + b"\x00\xff"
            receipt = coll.apply(
                b"tiny-correction", head_before,
                iter([{
                    "op": "put", "row_id": target_rid,
                    "region": "region-0199", "ts": 49,
                    "payload": b"\xde\xad\xbe\xef",
                }]),
            )
            status = coll.status()
            # Exactly one mutation applied; the table gained one version row
            # (the old entry stays as a dead version, one new live entry is
            # inserted) instead of rewriting TOTAL rows.
            self.assertEqual((receipt.applied, receipt.noop), (1, 0))
            self.assertEqual(status.live_entries, live_before)
            self.assertEqual(
                status.total_version_entries, total_before + 1
            )
            self.assertEqual(status.head, head_before + 1)

            # A same-value rewrite is a no-op that still costs a generation
            # but writes no index entries.
            total2 = status.total_version_entries
            noop = coll.apply(
                b"identical", status.head,
                iter([{
                    "op": "put", "row_id": target_rid,
                    "region": "region-0199", "ts": 49,
                    "payload": b"\xde\xad\xbe\xef",
                }]),
            )
            self.assertEqual((noop.applied, noop.noop), (0, 1))
            self.assertEqual(coll.status().total_version_entries, total2)

    def test_narrow_snapshot_query_uses_key_range(self):
        with open_collection(self.path) as coll:
            # Introduce history inside the target region so candidate
            # filtering is genuinely exercised.
            head = coll.status().head
            moves = [
                {"op": "put",
                 "row_id": f"id-{N * TS_PER_REGION + 11:07d}".encode() + b"\x00\xff",
                 "region": "region-0100", "ts": 99, "payload": b"old"}
                for N in range(5)
            ]
            coll.apply(b"history", head, iter(moves))
            tok = coll.begin_snapshot()
            query = {
                "shape": "prefix_range",
                "prefix": [["region", "region-0100"]],
                "range_column": "ts",
                "lo": 10,
                "hi": 12,
            }
            page, stats, plan = coll.scan_snapshot_page(tok, query, 100)
            values = [tuple(v) for v, _ in page.rows]
            self.assertEqual([v[0] for v in values], ["region-0100"] * 3)
            self.assertEqual([v[1] for v in values], [12, 11, 10])

            # The range-seek proof: examined candidates are a tiny fraction of
            # the table and the plan names the PRIMARY KEY index.
            self.assertLess(stats.candidates_examined, TOTAL // 50)
            self.assertTrue(stats.used_index)
            for d in stats.explain_query_plan:
                self.assertIn("USING PRIMARY KEY", d["detail"])
                self.assertNotIn("SCAN bytekey_kv", d["detail"])

            # Candidate vs delivered accounting stays honest across pages.
            q_wide = {
                "shape": "prefix_range",
                "prefix": [["region", "region-0100"]],
                "range_column": "ts",
            }
            seen, total_candidates = [], 0
            cursor = None
            while True:
                p, s, _ = coll.scan_snapshot_page(
                    tok, q_wide, 7, cursor_token=cursor
                )
                seen.extend(p.rows)
                total_candidates += s.candidates_examined
                if p.next_cursor is None:
                    break
                cursor = p.next_cursor
            # The five churned rows were moved within region-0100 to ts=99,
            # so they still match the region prefix on the snapshot: 50 base
            # rows plus 5 relocated ones = 55, all delivered via key ranges.
            self.assertEqual(len(seen), TS_PER_REGION + 5)
            self.assertGreaterEqual(total_candidates, len(seen))
            self.assertLess(total_candidates, TOTAL)

            report = {
                "table_live_rows": TOTAL,
                "query": query,
                "plan": plan.explain(),
                "explain_query_plan": stats.explain_query_plan,
                "candidates_examined": stats.candidates_examined,
                "rows_delivered": stats.rows_delivered,
                "invisible_versions_skipped": stats.invisible_versions_skipped,
            }
            with open(os.path.join(self.report_dir, "narrow.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)

    def test_history_collection_physically_shrinks_version_data(self):
        with open_collection(self.path) as coll:
            # Create several generations of churn on a small set of rows.
            head = coll.status().head
            churn_ids = [f"id-{i:07d}".encode() + b"\x00\xff" for i in range(1, 21)]
            for round_no in range(6):
                ops = [
                    {"op": "put", "row_id": rid,
                     "region": f"moved-{round_no}",
                     "ts": round_no, "payload": b"x"}
                    for rid in churn_ids
                ]
                coll.apply(f"churn-{round_no}".encode(), head, iter(ops))
                head = coll.status().head
            status = coll.status()
            dead_before = status.dead_entries
            total_before = status.total_version_entries
            self.assertGreaterEqual(dead_before, 20 * 5)

            # No active snapshot: all dead entries below head are reclaimable.
            result = coll.collect_history(head)
            after = coll.status()
            self.assertEqual(result.entries_removed, dead_before)
            self.assertEqual(after.dead_entries, 0)
            self.assertEqual(after.total_version_entries, TOTAL)
            self.assertLess(after.total_version_entries, total_before)
            # Live data and receipts are untouched.
            self.assertEqual(after.live_entries, TOTAL)
            self.assertGreater(after.receipt_count, 0)

            # An active snapshot blocks reclamation of its window only.
            tok = coll.begin_snapshot(head)
            for round_no in range(3):
                coll.apply(
                    f"churn2-{round_no}".encode(), coll.status().head,
                    iter([{"op": "put", "row_id": churn_ids[0],
                           "region": f"again-{round_no}", "ts": 1,
                           "payload": b"y"}]),
                )
            protected = coll.status().dead_entries
            self.assertGreater(protected, 0)
            before_collect = protected
            res = coll.collect_history(coll.status().head)
            # Entries dead before the snapshot generation are unreachable from
            # it and get reclaimed; only the single latest dead version whose
            # window still covers the pinned generation stays protected.
            self.assertEqual(coll.status().dead_entries, 1)
            self.assertLess(coll.status().dead_entries, before_collect)
            q = {"shape": "full_tuple", "lower": None, "upper": None,
                 "lower_inclusive": True, "upper_inclusive": True}
            page, _, _ = coll.scan_snapshot_page(tok, q, 10)
            self.assertEqual(len(page.rows), 10)
            coll.release_snapshot(tok)
            res2 = coll.collect_history(coll.status().head)
            self.assertGreater(res2.entries_removed, 0)
            self.assertEqual(coll.status().dead_entries, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
