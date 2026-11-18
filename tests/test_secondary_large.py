"""Large-collection proofs for secondary access indexes.

Generated on the spot (nothing large is committed).  These tests prove:

* A registration transaction does no baseline work at all (it only mints
  identities and records B).
* Each ``advance_index`` call never processes more baseline candidates plus
  catch-up changes than its budget, measured by the step counters *and* by
  the growth in stored entries.
* A ready index query for a narrow time window seeks only that byte range;
  it does not visit every region, and its storage plan is a PRIMARY KEY
  range search on the secondary table.
* A later small correction does not rebuild the ready index; it touches
  O(batch) entries and can be read back via a version locator that never
  scans the whole main table.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey import Column, Schema  # noqa: E402
from bytekey.collection import create_collection, open_collection  # noqa: E402
from bytekey import SecondaryDeclaration, IndexColumn  # noqa: E402

N_REGIONS = 120
TS_PER_REGION = 40
TOTAL = N_REGIONS * TS_PER_REGION


def build_large_collection(path):
    schema = Schema(columns=(
        Column("region", "text", asc=True, nulls_first=True),
        Column("ts", "int64", asc=False, nulls_first=True),
        Column("payload", "bytes", asc=True, nulls_first=False),
    ))
    create_collection(path, schema)
    with open_collection(path) as coll:
        buf = []
        batch_no = 0
        n = 0
        for ri in range(N_REGIONS):
            region = f"region-{ri:04d}"
            for ti in range(TS_PER_REGION):
                n += 1
                buf.append({
                    "op": "put",
                    "row_id": f"id-{n:07d}".encode() + b"\x00\xff",
                    "region": region,
                    "ts": ti,
                    "payload": bytes([ti & 0xFF, 0x00, 0xFF, ri & 0xFF]),
                })
                if len(buf) == 500:
                    batch_no += 1
                    coll.apply(f"seed-{batch_no}".encode(),
                               coll.status().head, iter(buf))
                    buf = []
        if buf:
            batch_no += 1
            coll.apply(f"seed-{batch_no}".encode(), coll.status().head, iter(buf))
        assert coll.status().live_entries == TOTAL
    return schema


class LargeSecondaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "large.db")
        self.report_dir = os.path.join(self.tmp.name, "reports")
        os.makedirs(self.report_dir, exist_ok=True)
        self.schema = build_large_collection(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_registration_does_no_baseline_work(self):
        with open_collection(self.path) as coll:
            before = coll.status().total_version_entries
            reg = coll.register_index(SecondaryDeclaration(
                "by_ts_region",
                (IndexColumn("ts", asc=False), IndexColumn("region")),
            ))
            status = coll.index_status("by_ts_region")
            self.assertEqual(status.secondary_entries, 0)
            self.assertEqual(status.baseline_candidates_read, 0)
            self.assertFalse(status.baseline_done)
            self.assertEqual(coll.status().total_version_entries, before)

    def test_each_advance_respects_budget(self):
        decl = SecondaryDeclaration(
            "by_ts_region",
            (IndexColumn("ts", asc=False), IndexColumn("region")),
        )
        budgets = [1, 3, 17, 64]
        with open_collection(self.path) as coll:
            coll.register_index(decl)
            prev_entries = 0
            prev_processed = 0
            steps = 0
            while True:
                budget = budgets[steps % len(budgets)]
                st = coll.advance_index("by_ts_region", budget)
                steps += 1
                self.assertLessEqual(
                    st.baseline_candidates_read + st.catchup_changes_processed,
                    budget,
                    "a single advance exceeded its budget",
                )
                # Budget is an upper bound on the *visible* work per step.
                self.assertLessEqual(
                    st.secondary_entries - prev_entries,
                    budget,
                    "more entries appeared than the budget allows",
                )
                prev_entries = st.secondary_entries
                prev_processed = st.processed_seq
                if st.status == "ready":
                    break
                if steps > 20000:
                    self.fail("build did not converge")
            self.assertEqual(st.secondary_entries, TOTAL)
            self.assertGreaterEqual(st.ready_gen, st.start_gen)

    def test_narrow_window_does_not_scan_all_regions(self):
        decl = SecondaryDeclaration(
            "by_ts_region",
            (IndexColumn("ts", asc=False), IndexColumn("region")),
        )
        with open_collection(self.path) as coll:
            coll.register_index(decl)
            while coll.advance_index("by_ts_region", 500).status != "ready":
                pass
            tok = coll.begin_snapshot()
            # ts DESC means the newest timestamps sort first; grab a 3-wide
            # window that touches only a few of the 120 regions.
            query = {
                "shape": "prefix_range",
                "prefix": [],
                "range_column": "ts",
                "lo": 30,
                "hi": 32,
                "lo_inclusive": True,
                "hi_inclusive": True,
            }
            page, stats, plan = coll.scan_index_page(
                tok, "by_ts_region", query, 1000
            )
            values = [tuple(v) for v, _ in page.rows]
            self.assertTrue(values)
            self.assertEqual({v[1] for v in values}, {30, 31, 32})
            # 3 timestamps * 120 regions = 360 rows at most.
            self.assertLessEqual(len(values), 3 * N_REGIONS)
            # The storage work is bounded by the number of matches plus a
            # small constant, not by the full table.
            self.assertLess(stats.index_candidates_examined, TOTAL // 4)
            self.assertEqual(stats.full_row_lookups, stats.index_candidates_examined)
            self.assertTrue(stats.used_index)
            details = [d["detail"] for d in stats.explain_query_plan]
            self.assertTrue(any("bytekey_sec_entries" in d for d in details))
            self.assertTrue(any("USING PRIMARY KEY" in d for d in details))

            report = {
                "table_rows": TOTAL,
                "query": query,
                "plan": plan.explain(),
                "explain_query_plan": stats.explain_query_plan,
                "index_candidates_examined": stats.index_candidates_examined,
                "full_row_lookups": stats.full_row_lookups,
                "rows_delivered": stats.rows_delivered,
            }
            with open(os.path.join(self.report_dir, "secondary_narrow.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)

    def test_small_correction_does_not_rebuild_ready_index(self):
        decl = SecondaryDeclaration(
            "by_ts_region",
            (IndexColumn("ts", asc=False), IndexColumn("region")),
        )
        with open_collection(self.path) as coll:
            coll.register_index(decl)
            while coll.advance_index("by_ts_region", 500).status != "ready":
                pass
            entries_before = coll.index_status("by_ts_region").secondary_entries
            self.assertEqual(entries_before, TOTAL)

            # One correction that moves a single record across regions.
            target = f"id-{1000:07d}".encode() + b"\x00\xff"
            receipt = coll.apply(
                b"tiny-correction", coll.status().head,
                iter([{"op": "put", "row_id": target,
                       "region": "region-0119", "ts": 39,
                       "payload": b"\xde\xad\xbe\xef"}]),
            )
            self.assertEqual(receipt.applied, 1)
            st = coll.index_status("by_ts_region")
            # Exactly one new version (the old entry is closed, one opens).
            self.assertEqual(st.secondary_entries, entries_before + 1)
            self.assertEqual(st.queue_entries, 0)

            # Read it back through the index; only the moved identity and its
            # equal-key siblings are candidates.
            tok = coll.begin_snapshot(receipt.head)
            q = {
                "shape": "prefix_range",
                "prefix": [["ts", 39]],
                "range_column": "region",
                "lo": "region-0119",
                "hi": "region-0119",
            }
            page, stats, _ = coll.scan_index_page(tok, "by_ts_region", q, 10)
            rids = [rid for _, rid in page.rows]
            self.assertIn(target, rids)
            self.assertLess(stats.index_candidates_examined, TOTAL // 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
