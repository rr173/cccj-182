"""Large-library narrow-query proof.

Builds a sizable index (streaming), then issues very narrow byte-range
queries and records EXPLAIN QUERY PLAN plus the number of entries actually
visited.  A correct implementation visits only the handful of rows inside
the byte interval rather than the whole table, proving the BLOB range index
-- not a decode-and-filter pass -- does the work.  Artifacts are written to
a report directory for inspection.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey import Column, Schema, build_index, open_index  # noqa: E402

N_REGIONS = 200
TS_PER_REGION = 50
TOTAL = N_REGIONS * TS_PER_REGION


def _build_large(path):
    schema = Schema(
        columns=(
            Column("region", "text", asc=True, nulls_first=True),
            Column("ts", "int64", asc=False, nulls_first=True),
            Column("payload", "bytes", asc=True, nulls_first=False),
        )
    )

    def rows():
        n = 0
        for ri in range(N_REGIONS):
            region = f"region-{ri:04d}"
            for ti in range(TS_PER_REGION):
                n += 1
                yield {
                    "region": region,
                    "ts": ti,
                    # Include embedded NUL and 0xFF bytes in payload/row_id.
                    "payload": bytes([ti & 0xFF, 0x00, 0xFF, ri & 0xFF]),
                    "row_id": f"id-{n:07d}".encode() + b"\x00\xff",
                }

    stats = build_index(path, schema, rows())
    return schema, stats


class LargeLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "large.db")
        self.report_dir = os.path.join(self.tmp.name, "reports")
        os.makedirs(self.report_dir, exist_ok=True)
        self.schema, self.build_stats = _build_large(self.path)
        self.index = open_index(self.path)

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def test_build_size_and_row_count(self):
        self.assertEqual(self.index.row_count, TOTAL)
        self.assertEqual(self.build_stats.rows, TOTAL)

    def test_narrow_query_visits_only_matching_entries(self):
        target_region = "region-0100"
        query = {
            "shape": "prefix_range",
            "prefix": [["region", target_region]],
            "range_column": "ts",
            "lo": 10,
            "hi": 12,
        }
        page, stats, plan = self.index.scan_page(query, 100)
        values = [tuple(v) for v, _ in page.rows]

        # Business expectation: ts 12, 11, 10 (DESC delivery), same region.
        self.assertEqual([v[0] for v in values], [target_region] * 3)
        self.assertEqual([v[1] for v in values], [12, 11, 10])

        # The decisive proof: visited entries must be a tiny fraction of the
        # table and exactly equal the matched set (no full-table decode).
        self.assertEqual(stats.rows_visited, 3)
        self.assertLess(stats.rows_visited, TOTAL // 100)
        self.assertTrue(stats.used_index)
        for d in stats.explain_query_plan:
            self.assertIn("USING PRIMARY KEY", d["detail"])
            self.assertNotIn("SCAN bytekey_index", d["detail"])

        # Bytecode must contain an index Seek, never a table Rewind scan.
        opcodes = {op["opcode"] for op in stats.explain_bytecode}
        self.assertIn("SeekGE", opcodes)
        self.assertNotIn("Rewind", opcodes)

        # Persist the evidence.
        report = {
            "table_rows": TOTAL,
            "query": query,
            "plan": plan.explain(),
            "explain_query_plan": stats.explain_query_plan,
            "seek_bytecode": [
                op for op in stats.explain_bytecode
                if op["opcode"] in ("SeekGE", "SeekGT", "SeekLE", "SeekLT", "Last")
            ],
            "rows_visited": stats.rows_visited,
            "rows_returned": stats.rows_returned,
            "visited_divisor_of_table": TOTAL // max(stats.rows_visited, 1),
        }
        with open(os.path.join(self.report_dir, "narrow_query.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

    def test_point_lookup_via_full_tuple(self):
        target = ["region-0050", 25, b"\x19\x00\xff\x32"]
        query = {
            "shape": "full_tuple",
            "lower": target,
            "upper": target,
            "lower_inclusive": True,
            "upper_inclusive": True,
        }
        page, stats, _ = self.index.scan_page(query, 50)
        # All row_ids sharing this exact business tuple (just one here).
        self.assertEqual(len(page.rows), 1)
        self.assertEqual(tuple(page.rows[0][0]), tuple(target))
        self.assertEqual(stats.rows_visited, 1)
        self.assertTrue(stats.used_index)

    def test_paging_does_not_skip_or_repeat_on_large_index(self):
        query = {
            "shape": "prefix_range",
            "prefix": [["region", "region-0007"]],
            "range_column": "ts",
        }
        seen = []
        token = None
        pages = 0
        while True:
            page, stats, _ = self.index.scan_page(query, 7, cursor_token=token)
            seen.extend((tuple(v), rid) for v, rid in page.rows)
            pages += 1
            token = page.next_cursor
            if token is None:
                break
        # All non-null ts values for one region, in DESC delivery order.
        self.assertEqual(len(seen), TS_PER_REGION)
        self.assertEqual([v[1] for v, _ in seen], list(range(49, -1, -1)))
        rids = [rid for _, rid in seen]
        self.assertEqual(len(rids), len(set(rids)))
        self.assertGreater(pages, 1)

    def test_explain_command_data_is_captured(self):
        query = {
            "shape": "prefix_range",
            "prefix": [["region", "region-0001"]],
            "range_column": "ts",
            "lo": 5,
        }
        explained = self.index.explain(query)
        self.assertEqual(explained["plan"]["interval_count"], 1)
        self.assertTrue(
            all("USING PRIMARY KEY" in d["detail"]
                for d in explained["query_plan_detail"])
        )
        with open(os.path.join(self.report_dir, "explain.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(explained, fh, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
