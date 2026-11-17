"""End-to-end demonstration and evidence generator.

Builds the mandated hand-written fixture (region ASC, time DESC nulls_first,
three row_ids sharing one business tuple, empty-string region, region with
embedded NUL and a prefix text), drains a full-range scan with page size 1
line by line, and writes EXPLAIN QUERY PLAN plus actual access counts to a
report directory.

Run (isolated):  python3 -I demo/demo_e2e.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bytekey import (  # noqa: E402
    Column,
    Schema,
    build_index,
    open_index,
)
from tests.oracle import expected_order  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "demo", "report")


def main() -> int:
    os.makedirs(REPORT_DIR, exist_ok=True)
    schema = Schema(
        columns=(
            Column("region", "text", asc=True, nulls_first=True),
            Column("ts", "int64", asc=False, nulls_first=True),
        )
    )
    rows = [
        {"region": "a", "ts": 2, "row_id": b"r-01"},
        {"region": "a", "ts": 1, "row_id": b"r-02"},
        {"region": "a", "ts": None, "row_id": b"r-03"},
        {"region": "", "ts": 1, "row_id": b"r-04"},
        {"region": "a\x00", "ts": 1, "row_id": b"r-05"},
        {"region": "ab", "ts": 1, "row_id": b"r-06"},
        {"region": "tie", "ts": 7, "row_id": b"z"},
        {"region": "tie", "ts": 7, "row_id": b"m"},
        {"region": "tie", "ts": 7, "row_id": b"a"},
    ]
    db_path = os.path.join(REPORT_DIR, "fixture.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    stats = build_index(db_path, schema, rows)

    index = open_index(db_path)
    query = {"shape": "prefix_range", "prefix": [], "range_column": "region"}

    # Page size 1, drain to the end, recording every page's plan evidence.
    delivered = []
    pages_evidence = []
    token = None
    while True:
        page, scan_stats, plan = index.scan_page(query, 1, cursor_token=token)
        for values, rid in page.rows:
            delivered.append((list(values), rid.hex()))
        pages_evidence.append(
            {
                "eqp": scan_stats.explain_query_plan,
                "used_index": scan_stats.used_index,
                "rows_visited_inside_interval": scan_stats.rows_visited,
                "seek_bytecode": [
                    op for op in scan_stats.explain_bytecode
                    if op["opcode"] in ("SeekGE", "SeekGT", "SeekLE", "SeekLT")
                ],
            }
        )
        token = page.next_cursor
        if token is None:
            break

    pairs = [([r["region"], r["ts"]], r["row_id"]) for r in rows]
    want = [(list(v), rid.hex()) for v, rid in expected_order(schema, pairs)]

    # A narrow query proving only matching entries are physically visited.
    narrow = {
        "shape": "prefix_range",
        "prefix": [["region", "tie"]],
        "range_column": "ts",
        "lo": 7,
        "hi": 7,
    }
    narrow_page, narrow_stats, narrow_plan = index.scan_page(narrow, 10)
    index.close()

    report = {
        "schema": schema.to_dict(),
        "build": {"rows": stats.rows, "bytes_written": stats.bytes_written},
        "full_range_page_size_1": {
            "delivered": delivered,
            "oracle_expected": want,
            "matches_oracle": delivered == want,
            "page_count": len(pages_evidence),
            "first_page_evidence": pages_evidence[0],
        },
        "narrow_query": {
            "query": narrow,
            "plan": narrow_plan.explain(),
            "explain_query_plan": narrow_stats.explain_query_plan,
            "rows_visited": narrow_stats.rows_visited,
            "rows_returned": narrow_stats.rows_returned,
            "returned_row_ids": [rid.hex() for _, rid in narrow_page.rows],
        },
    }
    out_path = os.path.join(REPORT_DIR, "e2e_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("page-size-1 delivery vs independent oracle:")
    for got, exp in zip(delivered, want):
        flag = "ok " if got == exp else "BAD"
        print(f"  [{flag}] {got}")
    assert delivered == want, "delivery does not match independent oracle"
    print(f"\nnarrow query visited {narrow_stats.rows_visited} entries, "
          f"returned {narrow_stats.rows_returned}")
    print("EQP:", narrow_stats.explain_query_plan[0]["detail"])
    print(f"\nreport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
