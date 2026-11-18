"""Runnable end-to-end demonstration of named secondary access indexes.

Reproduces the mandated fixture from real files on disk (every step reopens
the collection), advances the index with budget 1, interleaves the mandated
corrections, then pages the *full* original records in the new order for a
build-start snapshot and a newer snapshot.  Finally it adds a ts-only index,
shows the region change still surfaces, and prints evidence that the query is
a narrow key-range seek (backfill candidates, catch-up changes, index
candidates, full-row locators and delivered counts are reported separately).

Run (isolated):  python3 -I demo/demo_secondary.py [--dir OUTDIR]
"""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bytekey import (  # noqa: E402
    Column,
    Schema,
    SecondaryDeclaration,
    IndexColumn,
    build_index,
    import_from_index,
    open_collection,
)

SCHEMA = Schema(columns=(
    Column("region", "text", asc=True, nulls_first=True),
    Column("ts", "int64", asc=False, nulls_first=True),
    Column("note", "text", asc=True, nulls_first=True),
))

GEN0 = [
    {"op": "put", "row_id": b"r1", "region": "a", "ts": 10, "note": "old"},
    {"op": "put", "row_id": b"r2", "region": "a", "ts": 10, "note": "keep"},
    {"op": "put", "row_id": b"r3", "region": "b", "ts": 5, "note": "stay"},
]
MID = [
    {"op": "put", "row_id": b"r1", "region": "c", "ts": 20, "note": "new"},
    {"op": "delete", "row_id": b"r2"},
    {"op": "put", "row_id": b"r4", "region": "a", "ts": 10, "note": "late"},
]
AFTER = [
    {"op": "put", "row_id": b"r1", "region": "c", "ts": 20, "note": "newer"},
]
REBIRTH = [
    {"op": "put", "row_id": b"r2", "region": "b", "ts": 7, "note": "reborn"},
]

DECL = SecondaryDeclaration("ts_region", (
    IndexColumn("ts", asc=False, nulls_first=True),
    IndexColumn("region", asc=True, nulls_first=True),
))


def drain(coll, token, name, query, page_size):
    rows, cursor, pages, stats_total = [], None, 0, 0
    while True:
        page, stats, _plan = coll.scan_index_page(
            token, name, query, page_size, cursor_token=cursor
        )
        rows.extend(page.rows)
        pages += 1
        stats_total += stats.index_candidates_examined
        cursor = page.next_cursor
        if cursor is None:
            return rows, pages, stats_total


def show(tag, rows):
    print(f"  {tag}:")
    for values, rid in rows:
        print(f"    {rid.decode():>3} -> {values}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "demo", "secondary_report"))
    args = ap.parse_args()
    outdir = args.dir
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "warehouse.db")
    static = os.path.join(outdir, "region_first_static.db")

    # Prepare the region-priority main collection as generation 0 by
    # building a static index in the main (region ASC, ts DESC) order and
    # importing it.  The static file stays read-only and untouched.
    build_index(static, SCHEMA, [
        {c.name: r[c.name] for c in SCHEMA.columns} | {"row_id": r["row_id"]}
        for r in GEN0
    ])
    import_from_index(static, path)
    with open_collection(path) as coll:
        print(f"imported region-priority generation 0 (head {coll.status().head})")

    # --- Register (short transaction, no copying) -----------------------
    with open_collection(path) as coll:
        reg = coll.register_index(DECL)
        print(f"registered {reg['name']} build={reg['build_id_hex']} "
              f"B={reg['start_gen']} status={reg['status']}")
        # Not queryable while building.
        tok = coll.begin_snapshot()
        try:
            coll.scan_index_page(tok, "ts_region", {"shape": "full_tuple"}, 10)
        except Exception as exc:
            print(f"  building index refused: {type(exc).__name__}")

    # --- Advance budget 1, interleaving the mandated corrections --------
    with open_collection(path) as coll:
        st = coll.advance_index("ts_region", 1)
        print(f"step1 baseline={st.baseline_candidates_read} "
              f"catchup={st.catchup_changes_processed} "
              f"done={st.baseline_done}")
    with open_collection(path) as coll:
        coll.apply(b"mid", coll.status().head, iter(MID))
        print(f"applied mid-batch -> head {coll.status().head}")
    with open_collection(path) as coll:
        st = coll.advance_index("ts_region", 1)
        print(f"step2 baseline={st.baseline_candidates_read} "
              f"catchup={st.catchup_changes_processed}")
    with open_collection(path) as coll:
        coll.apply(b"after", coll.status().head, iter(AFTER))
        coll.apply(b"reborn", coll.status().head, iter(REBIRTH))

    with open_collection(path) as coll:
        steps = 2
        while True:
            st = coll.advance_index("ts_region", 1)
            steps += 1
            if st.status == "ready":
                break
        print(f"ready after {steps} budget-1 steps; ready_gen={st.ready_gen} "
              f"head={st.head}")
        print(f"  cumulative baseline={st.cumulative_baseline_candidates} "
              f"catchup={st.cumulative_catchup_changes} "
              f"entries={st.secondary_entries}")

    # --- Query the build-start and later snapshots ----------------------
    allq = {"shape": "full_tuple", "lower": None, "upper": None}
    with open_collection(path) as coll:
        head = coll.status().head
        b = coll.index_status("ts_region").start_gen
        print("\nfull records, new (ts DESC, region ASC) order:")
        for g in range(b, head + 1):
            tok = coll.begin_snapshot(g)
            rows, pages, cand = drain(coll, tok, "ts_region", allq, 1)
            print(f"  generation {g} ({pages} pages, {cand} index candidates):")
            show(f"gen {g}", rows)

        # The old-vs-new full row for r2: deleted then re-created.
        tok_b = coll.begin_snapshot(b)
        rows_b, _, _ = drain(coll, tok_b, "ts_region", allq, 10)
        print("\n  r2 at build-start generation:", dict(
            (rid, v) for v, rid in rows_b)[b"r2"])
        tok_latest = coll.begin_snapshot()
        rows_l, _, _ = drain(coll, tok_latest, "ts_region", allq, 10)
        print("  r2 at latest generation:      ", dict(
            (rid, v) for v, rid in rows_l)[b"r2"])

    # --- A ts-only index still returns the changed, non-indexed region --
    ts_decl = SecondaryDeclaration("ts_only", (IndexColumn("ts", asc=False),))
    with open_collection(path) as coll:
        coll.register_index(ts_decl)
        while coll.advance_index("ts_only", 5).status != "ready":
            pass
        tok = coll.begin_snapshot()
        rows, _, _ = drain(coll, tok, "ts_only", allq, 1)
        print("\nts-only index (region not indexed) still returns full rows:")
        show("ts_only", rows)

    # --- Narrow window proof -------------------------------------------
    with open_collection(path) as coll:
        tok = coll.begin_snapshot()
        narrow = {
            "shape": "prefix_range",
            "prefix": [["ts", 20]],
            "range_column": "region",
            "lo": "c", "hi": "c",
        }
        page, stats, plan = coll.scan_index_page(
            tok, "ts_region", narrow, 100
        )
        report = {
            "query": narrow,
            "plan": plan.explain(),
            "explain_query_plan": stats.explain_query_plan,
            "index_candidates_examined": stats.index_candidates_examined,
            "full_row_lookups": stats.full_row_lookups,
            "rows_delivered": stats.rows_delivered,
            "rows": [(list(v), rid.hex()) for v, rid in page.rows],
        }
        print(f"\nnarrow (ts=20, region='c'): delivered={stats.rows_delivered} "
              f"candidates={stats.index_candidates_examined} "
              f"locators={stats.full_row_lookups}")
        print("  plan:", [d["detail"] for d in stats.explain_query_plan])

    # --- Operational tour: pre-opened writer, retention, cancel, rebuild -
    print("\noperational tour:")
    other_decl = SecondaryDeclaration("ts_other", (IndexColumn("ts", asc=False),))
    early = open_collection(path)          # opened BEFORE the new registration
    try:
        with open_collection(path) as coll:
            coll.register_index(other_decl)
            coll.register_index(SecondaryDeclaration("doomed", (
                IndexColumn("region", asc=True),)))
            coll.advance_index("doomed", 1)
        # The early connection was never told about the new indexes; its apply
        # must still feed the in-flight builds in the same commit.
        head = early.status().head
        early.apply(b"early-writer", head, iter([
            {"op": "put", "row_id": b"r9", "region": "q", "ts": 50,
             "note": "from-early-connection"},
        ]))
        print(f"  early connection published head={head + 1}")
    finally:
        early.close()
    with open_collection(path) as coll:
        st = coll.advance_index("ts_other", 1)
        # The early write was journaled in the same commit as its main
        # version/head/receipt and is picked up by the build.
        print(f"  ts_other caught up the early write via journal "
              f"(pending step work counted as catch-up later)")
        # Raise retention while a build is mid-flight; its inputs survive.
        h = coll.status().head
        hist = coll.collect_history(h)
        print(f"  collect_history -> main removed={hist.entries_removed} "
              f"secondary removed={hist.secondary_entries_removed} "
              f"queue removed={hist.queue_entries_removed}")
        # Cancel one build and confirm its material is released.
        cancel = coll.cancel_index("doomed")
        print(f"  cancelled 'doomed': entries removed="
              f"{cancel['secondary_entries_removed']}")
        # Finish ts_other and show the early write is in it.
        while coll.advance_index("ts_other", 5).status != "ready":
            pass
        rows, _, _ = drain(coll, coll.begin_snapshot(), "ts_other", allq, 10)
        print(f"  ts_other latest generation contains r9: "
              f"{b'r9' in [rid for _, rid in rows]}")
        # Capture a cursor from the current ts_region instance, then delete
        # that instance and rebuild under the same name with a new order.
        tok_old = coll.begin_snapshot()
        page_old, _, _ = coll.scan_index_page(
            tok_old, "ts_region", allq, 2
        )
        old_cursor = page_old.next_cursor
        coll.delete_index("ts_region")
        coll.register_index(SecondaryDeclaration("ts_region", (
            IndexColumn("ts", asc=True),)))
        while coll.advance_index("ts_region", 5).status != "ready":
            pass
        print("  rebuilt 'ts_region' under the same name with a new order")
        # The old instance's cursor must not continue the new instance.
        try:
            coll.scan_index_page(tok_old, "ts_region", allq, 2,
                                 cursor_token=old_cursor)
            print("  BAD: old cursor was accepted")
        except Exception as exc:
            print(f"  old cursor from previous instance refused: "
                  f"{type(exc).__name__}")

    with open(os.path.join(outdir, "report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    # Keep only the small bounded JSON evidence in the repository; the
    # generated database files are runtime fixtures.
    for name in ("warehouse.db", "warehouse.db-wal", "warehouse.db-shm",
                 "region_first_static.db"):
        stray = os.path.join(outdir, name)
        if os.path.exists(stray):
            os.remove(stray)
    print(f"\nreport written to {os.path.join(outdir, 'report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
