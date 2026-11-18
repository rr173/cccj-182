"""Runnable end-to-end demonstration of the disk-backed two-source equi-join.

Reproduces the mandated fixture from real files on disk (every page read
reopens the work file, like separate CLI invocations would):

* left  (region, sku, qty) stored region ASC / sku DESC with
  l1=(a,x,2) l2=(a,x,3) l3=(b,y,1) l4=(null,z,9)
* right (product, area, label) stored product DESC / area ASC with
  r1=(x,a,old) r2=(x,a,new) r3=(q,c,only) r4=(z,null,nil)
* join region/area + sku/product, full mode, page size 1

then shows nulls_equal, snapshot stability under corrections + history
collection, token-release refusal, a second join over two differently
ordered secondary indexes with filtered ranges and pinned old-generation
values, and a generated large duplicate-key group consumed segment by
segment with bounded per-page work.  Evidence (prepare/page access counts,
group spill report, query plans) is written to demo/join_report/report.json.

Run (isolated):  python3 -I demo/demo_join.py [--dir OUTDIR]
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
    IndexColumn,
    Schema,
    SecondaryDeclaration,
    create_collection,
    open_collection,
)
from bytekey.join import (  # noqa: E402
    JoinSource,
    JoinSourceError,
    join_cleanup,
    join_status,
    open_join,
    prepare_join,
    terminate_join,
)

ALLQ = {"shape": "full_tuple", "lower": None, "upper": None}

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


def seed(path, schema, rows):
    create_collection(path, schema)
    with open_collection(path) as coll:
        coll.apply(b"seed", 0, iter([
            {"op": "put", "row_id": rid,
             **{c.name: v for c, v in zip(schema.columns, vals)}}
            for rid, vals in rows
        ]))


def snapshot(path, gen=None):
    with open_collection(path) as coll:
        return coll.begin_snapshot(gen)


def fmt_side(side):
    if side is None:
        return None
    return {"row_id": side.row_id.decode("ascii", "backslashreplace"),
            "values": list(side.values)}


def drain_reopening(work, handle):
    """Drain page by page, reopening the join handle for every page."""
    rows = []
    cursor = None
    pages = 0
    while True:
        join = open_join(work, handle)
        page, stats = join.page(cursor)
        join.close()
        pages += 1
        for r in page.rows:
            rows.append((r.kind, fmt_side(r.left), fmt_side(r.right)))
        cursor = page.next_cursor
        if cursor is None:
            return rows, pages


def show(rows):
    for kind, left, right in rows:
        lv = "-" if left is None else f"{left['row_id']}={left['values']}"
        rv = "-" if right is None else f"{right['row_id']}={right['values']}"
        print(f"    {kind:10} left={lv:28} right={rv}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "demo", "join_report"))
    args = ap.parse_args()
    outdir = args.dir
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    lpath = os.path.join(outdir, "left.db")
    rpath = os.path.join(outdir, "right.db")
    report = {"scenarios": {}}

    print("=== 1. mandated fixture: full join, default null rule, page size 1")
    seed(lpath, LSCHEMA, LROWS)
    seed(rpath, RSCHEMA, RROWS)
    ltok = snapshot(lpath)
    rtok = snapshot(rpath)
    work = os.path.join(outdir, "join-full.db")
    result = prepare_join(
        work,
        left=JoinSource(lpath, ltok, ALLQ),
        right=JoinSource(rpath, rtok, ALLQ),
        mapping=MAPPING, mode="full", nulls_equal=False, page_size=1,
    )
    handle = result["handle_token"]
    rows, pages = drain_reopening(work, handle)
    got = [(k, tuple(l["values"]) if l else None,
            l and l["row_id"].encode() if l else None,
            tuple(r["values"]) if r else None,
            r and r["row_id"].encode() if r else None)
           for k, l, r in rows]
    # Hand-written expected stream (left path order: l4, l1, l2, l3; each
    # left row's matches in right path order r2, r1; right_only r4, r3).
    hand = [
        ("left_only", (None, "z", 9), b"l4", None, None),
        ("matched", ("a", "x", 2), b"l1", ("x", "a", "new"), b"r2"),
        ("matched", ("a", "x", 2), b"l1", ("x", "a", "old"), b"r1"),
        ("matched", ("a", "x", 3), b"l2", ("x", "a", "new"), b"r2"),
        ("matched", ("a", "x", 3), b"l2", ("x", "a", "old"), b"r1"),
        ("left_only", ("b", "y", 1), b"l3", None, None),
        ("right_only", None, None, ("z", None, "nil"), b"r4"),
        ("right_only", None, None, ("q", "c", "only"), b"r3"),
    ]
    assert got == hand, f"fixture mismatch:\n{got}\n!=\n{hand}"
    print(f"  {pages} pages of size 1, each from a fresh handle:")
    show(rows)
    report["scenarios"]["mandated_full_default"] = {
        "pages": pages,
        "rows": [(k, l, r) for k, l, r in rows],
        "prepare_stats": result["stats"],
    }

    print("=== 2. same fixture with nulls_equal=true")
    work2 = os.path.join(outdir, "join-nulleq.db")
    result2 = prepare_join(
        work2,
        left=JoinSource(lpath, ltok, ALLQ),
        right=JoinSource(rpath, rtok, ALLQ),
        mapping=MAPPING, mode="full", nulls_equal=True, page_size=1,
    )
    rows2, pages2 = drain_reopening(work2, result2["handle_token"])
    hand2 = [
        ("matched", (None, "z", 9), b"l4", ("z", None, "nil"), b"r4"),
        ("matched", ("a", "x", 2), b"l1", ("x", "a", "new"), b"r2"),
        ("matched", ("a", "x", 2), b"l1", ("x", "a", "old"), b"r1"),
        ("matched", ("a", "x", 3), b"l2", ("x", "a", "new"), b"r2"),
        ("matched", ("a", "x", 3), b"l2", ("x", "a", "old"), b"r1"),
        ("left_only", ("b", "y", 1), b"l3", None, None),
        ("right_only", None, None, ("q", "c", "only"), b"r3"),
    ]
    got2 = [(k, tuple(l["values"]) if l else None,
             l and l["row_id"].encode() if l else None,
             tuple(r["values"]) if r else None,
             r and r["row_id"].encode() if r else None)
            for k, l, r in rows2]
    assert got2 == hand2, f"nulls_equal mismatch:\n{got2}"
    print(f"  {pages2} pages: l4 now pairs with r4, r3 stays right_only")
    show(rows2)
    terminate_join(work2, result2["handle_token"], reclaim=True)

    print("=== 3. corrections + history collection keep the pinned stream")
    # Read the first 3 results, then correct both collections and reclaim
    # history; the pinned join keeps delivering generation-1 results.
    join = open_join(work, handle)
    cursor = None
    prefix_rows = []
    for _ in range(3):
        page, _ = join.page(cursor)
        prefix_rows.extend((r.kind, fmt_side(r.left), fmt_side(r.right))
                           for r in page.rows)
        cursor = page.next_cursor
    join.close()
    with open_collection(lpath) as coll:
        coll.apply(b"fix1", coll.status().head, iter([
            {"op": "put", "row_id": b"l1", "region": "a", "sku": "x",
             "qty": 20},
            {"op": "delete", "row_id": b"l3"},
        ]))
        coll.apply(b"fix2", coll.status().head, iter([
            {"op": "put", "row_id": b"l3", "region": "b", "sku": "y",
             "qty": 99},
            {"op": "put", "row_id": b"l4", "region": "zzz", "sku": "z",
             "qty": 9},
        ]))
        hist_l = coll.collect_history(coll.status().head)
    with open_collection(rpath) as coll:
        coll.apply(b"rfix1", coll.status().head, iter([
            {"op": "put", "row_id": b"r1", "product": "x", "area": "a",
             "label": "oldest"},
            {"op": "delete", "row_id": b"r3"},
        ]))
        hist_r = coll.collect_history(coll.status().head)
    rest = []
    join = open_join(work, handle)
    while cursor is not None:
        page, _ = join.page(cursor)
        rest.extend((r.kind, fmt_side(r.left), fmt_side(r.right))
                    for r in page.rows)
        cursor = page.next_cursor
    join.close()
    assert prefix_rows + rest == rows, "pinned stream changed after corrections"
    print(f"  corrections published (left head=3, right head=2), "
          f"history collected (retain moved; pinned entries protected: "
          f"left removed={hist_l.entries_removed}, "
          f"right removed={hist_r.entries_removed})")
    print(f"  remaining {len(rest)} pages identical to the original stream")
    report["scenarios"]["pinned_stream_survives_corrections"] = {
        "history": {"left": hist_l.to_dict(), "right": hist_r.to_dict()},
        "stream_unchanged": True,
    }

    print("=== 4. releasing one side's token fails the next page")
    join = open_join(work, handle)
    page1, _ = join.page()
    join.close()
    with open_collection(lpath) as coll:
        coll.release_snapshot(ltok)
    try:
        join = open_join(work, handle)
        join.page(page1.next_cursor)
        print("  BAD: page succeeded after token release")
        report["scenarios"]["token_release"] = {"refused": False}
    except JoinSourceError as exc:
        print(f"  refused as required: {exc}")
        report["scenarios"]["token_release"] = {"refused": True,
                                                "error": str(exc)}
    # A fresh left token lets the same work file finish (right side intact).
    ltok2 = snapshot(lpath)
    terminate_join(work, handle, reclaim=True)

    print("=== 5. second join: two differently ordered indexes, filters, "
          "old-generation values")
    l2path = os.path.join(outdir, "left2.db")
    r2path = os.path.join(outdir, "right2.db")
    lschema2 = Schema(columns=(
        Column("region", "text", asc=True, nulls_first=True),
        Column("sku", "text", asc=False, nulls_first=True),
        Column("qty", "int64", asc=True, nulls_first=True),
        Column("note", "text", asc=True, nulls_first=True),
    ))
    rschema2 = Schema(columns=(
        Column("product", "text", asc=False, nulls_first=True),
        Column("area", "text", asc=True, nulls_first=True),
        Column("label", "text", asc=True, nulls_first=True),
        Column("flag", "bool", asc=True, nulls_first=True),
    ))
    seed(l2path, lschema2, [
        (b"l1", ("a", "x", 2, "v0-l1")),
        (b"l2", ("a", "x", 3, "v0-l2")),
        (b"l3", ("b", "y", 1, "v0-l3")),
        (b"l5", ("c", "w", 50, "v0-l5")),
    ])
    seed(r2path, rschema2, [
        (b"r1", ("x", "a", "keep", True)),
        (b"r2", ("x", "a", "drop", False)),
        (b"r3", ("q", "c", "keep", True)),
    ])
    with open_collection(l2path) as coll:
        coll.register_index(SecondaryDeclaration("qty_region", (
            IndexColumn("qty", asc=True), IndexColumn("region", asc=True))))
        while coll.advance_index("qty_region", 100).status != "ready":
            pass
    with open_collection(r2path) as coll:
        coll.register_index(SecondaryDeclaration("label_product", (
            IndexColumn("label", asc=False),
            IndexColumn("product", asc=True))))
        while coll.advance_index("label_product", 100).status != "ready":
            pass
    ltok_old = snapshot(l2path)
    rtok_old = snapshot(r2path)
    # Later generations change only NON-join fields.
    with open_collection(l2path) as coll:
        coll.apply(b"notes", coll.status().head, iter([
            {"op": "put", "row_id": b"l1", "region": "a", "sku": "x",
             "qty": 2, "note": "v1-changed"},
        ]))
    with open_collection(r2path) as coll:
        coll.apply(b"flags", coll.status().head, iter([
            {"op": "put", "row_id": b"r1", "product": "x", "area": "a",
             "label": "keep", "flag": False},
        ]))
    work5 = os.path.join(outdir, "join-indexed.db")
    left_q = {"shape": "prefix_range", "prefix": [], "range_column": "qty",
              "lo": 1, "hi": 5}
    right_q = {"shape": "prefix_range", "prefix": [["label", "keep"]],
               "range_column": "product"}
    result5 = prepare_join(
        work5,
        left=JoinSource(l2path, ltok_old, left_q, index="qty_region"),
        right=JoinSource(r2path, rtok_old, right_q, index="label_product"),
        mapping=MAPPING, mode="full", page_size=1,
    )
    rows5, pages5 = drain_reopening(work5, result5["handle_token"])
    hand5 = [
        ("left_only", ("b", "y", 1, "v0-l3"), b"l3", None, None),
        ("matched", ("a", "x", 2, "v0-l1"), b"l1", ("x", "a", "keep", True),
         b"r1"),
        ("matched", ("a", "x", 3, "v0-l2"), b"l2", ("x", "a", "keep", True),
         b"r1"),
        ("right_only", None, None, ("q", "c", "keep", True), b"r3"),
    ]
    got5 = [(k, tuple(l["values"]) if l else None,
             l and l["row_id"].encode() if l else None,
             tuple(r["values"]) if r else None,
             r and r["row_id"].encode() if r else None)
            for k, l, r in rows5]
    assert got5 == hand5, f"indexed join mismatch:\n{got5}"
    print(f"  {pages5} pages: filtered (qty 1..5 / label='keep') then joined;")
    print("  values are the pinned generation's (note=v0-*, flag=True):")
    show(rows5)
    report["scenarios"]["indexed_filtered_join"] = {
        "pages": pages5,
        "rows": [(k, l, r) for k, l, r in rows5],
        "prepare_stats": result5["stats"],
    }
    terminate_join(work5, result5["handle_token"], reclaim=True)

    print("=== 6. generated large duplicate group, consumed segment by segment")
    gpath = os.path.join(outdir, "groups-l.db")
    hpath = os.path.join(outdir, "groups-r.db")
    gschema = Schema(columns=(
        Column("k", "text", asc=True, nulls_first=True),
        Column("n", "int64", asc=True, nulls_first=True),
    ))
    hschema = Schema(columns=(
        Column("k", "text", asc=True, nulls_first=True),
        Column("tag", "text", asc=True, nulls_first=True),
    ))
    NL, NR, DISTINCT = 40, 60, 300
    seed(gpath, gschema,
         [(f"l{i:03d}".encode(), ("hot", i)) for i in range(NL)] +
         [(f"ld{i:03d}".encode(), (f"distinct-{i}", i)) for i in range(DISTINCT)])
    seed(hpath, hschema,
         [(f"r{j:03d}".encode(), ("hot", f"t{j:03d}")) for j in range(NR)] +
         [(b"rd-only", ("lukewarm", "z"))])
    gtok = snapshot(gpath)
    htok = snapshot(hpath)
    work6 = os.path.join(outdir, "join-groups.db")
    result6 = prepare_join(
        work6,
        left=JoinSource(gpath, gtok, ALLQ),
        right=JoinSource(hpath, htok, ALLQ),
        mapping=[("k", "k")], mode="inner", page_size=1,
    )
    st6 = result6["stats"]
    print(f"  group stats: max_left_group={st6['groups']['max_left_group']} "
          f"max_right_group={st6['groups']['max_right_group']} "
          f"disk_resident={st6['groups']['groups_disk_resident']}")
    print(f"  work file bytes={st6['work_file_bytes']} for "
          f"{NL * NR} hot pairs (result never materialized)")
    # Consume a segment: the last distinct-key page, then deep into the hot
    # group, proving per-page work stays bounded inside the giant group.
    handle6 = result6["handle_token"]
    cursor = None
    seen = 0
    segment_evidence = []
    join = open_join(work6, handle6)
    while True:
        page, stats = join.page(cursor)
        seen += len(page.rows)
        if seen >= NL * NR // 2 and len(segment_evidence) < 5:
            segment_evidence.append({
                "page_index": seen,
                "left_rows_examined": stats.left_rows_examined,
                "right_candidates_examined": stats.right_candidates_examined,
                "full_row_reads": stats.full_row_reads,
                "eqp_probe": [d["detail"] for d in
                              stats.explain_query_plan["right_probe"]],
            })
        cursor = page.next_cursor
        if cursor is None or seen >= NL * NR // 2 + 400:
            break
    join.close()
    max_probe = max(e["right_candidates_examined"] for e in segment_evidence)
    print(f"  consumed {seen} pairs one page at a time; mid-group pages "
          f"examined at most {max_probe} right candidates each")
    report["scenarios"]["large_duplicate_group"] = {
        "left_rows": NL + DISTINCT,
        "right_rows": NR + 1,
        "hot_pairs": NL * NR,
        "prepare_stats": st6,
        "segment_evidence": segment_evidence,
        "consumed_pairs": seen,
    }
    terminate_join(work6, handle6, reclaim=True)

    report["note"] = (
        "All database files are runtime fixtures and are removed after the "
        "run; only this bounded JSON report remains."
    )
    out_path = os.path.join(outdir, "report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    # Clean up runtime fixtures; keep only the bounded report.
    for name in os.listdir(outdir):
        if name != "report.json":
            os.remove(os.path.join(outdir, name))
    print(f"\nreport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
