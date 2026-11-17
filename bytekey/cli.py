"""Command line interface: build, range, explain, encode, decode.

Plus the updatable-collection operations: ``collection-create``,
``collection-import``, ``apply``, ``snapshot-begin``, ``snapshot-release``,
``snapshot-range``, ``collect-history`` and ``status``.

All JSON ``bytes`` values use the single-field object form ``{"__hex__": "..."}``
on input and output (row_id, batch_id and every bytes-typed column alike).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from typing import Any

from .db import build_index, open_index
from .encoding import key_decode, key_encode
from .planner import plan_query
from .schema import Schema
from .collection import (
    BatchError,
    ConflictError,
    HistoryError,
    TokenError,
    create_collection,
    import_from_index,
    open_collection,
)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _coerce_value(col_kind: str, value: Any) -> Any:
    if value is None:
        return None
    if col_kind == "bytes":
        if isinstance(value, dict) and set(value) == {"__hex__"} and isinstance(
            value["__hex__"], str
        ):
            try:
                return bytes.fromhex(value["__hex__"])
            except ValueError as exc:
                raise SystemExit(f"bytes __hex__ must be hexadecimal: {exc}") from None
        if isinstance(value, str):
            try:
                return bytes.fromhex(value)
            except ValueError as exc:
                raise SystemExit(f"bytes must be hex strings: {exc}") from None
        if isinstance(value, list) and all(isinstance(x, int) for x in value):
            return bytes(value)
        raise SystemExit(
            'bytes values must use {"__hex__": "..."} (or hex strings / int arrays)'
        )
    return value


def _as_hex_object(value: Any, label: str) -> bytes:
    """Parse the mandatory ``{"__hex__": hex}`` single-field object."""
    if not isinstance(value, dict) or set(value) != {"__hex__"}:
        raise SystemExit(f"{label} must be a {{\"__hex__\": \"...\"}} object")
    try:
        raw = bytes.fromhex(value["__hex__"])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} hex is invalid: {exc}") from None
    if not raw:
        raise SystemExit(f"{label} must be non-empty")
    return raw


def _coerce_row(schema: Schema, row: dict) -> dict:
    out = {}
    for col in schema.columns:
        if col.name not in row:
            raise SystemExit(f"row missing column {col.name!r}")
        out[col.name] = _coerce_value(col.kind, row[col.name])
    rid = row.get(schema.row_id_name)
    if rid is None:
        raise SystemExit(f"row missing {schema.row_id_name!r}")
    if isinstance(rid, dict) and set(rid) == {"__hex__"}:
        try:
            rid = bytes.fromhex(rid["__hex__"])
        except ValueError as exc:
            raise SystemExit(f"{schema.row_id_name} hex is invalid: {exc}") from None
    elif isinstance(rid, str):
        try:
            rid = bytes.fromhex(rid)
        except ValueError as exc:
            raise SystemExit(f"{schema.row_id_name} must be hex bytes: {exc}") from None
    out[schema.row_id_name] = rid
    return out


def _iter_jsonl(path: str, schema: Schema):
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"source line {line_no}: invalid JSON: {exc}") from None
            yield _coerce_row(schema, row)


def _load_rows(path: str, schema: Schema):
    if path.endswith(".jsonl"):
        yield from _iter_jsonl(path, schema)
        return
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise SystemExit("source JSON must be a list of row objects (or use .jsonl)")
    for row in payload:
        yield _coerce_row(schema, row)


def _coerce_query(schema: Schema, query: dict) -> dict:
    out = dict(query)
    shape = query.get("shape")
    if shape == "full_tuple":
        for side in ("lower", "upper"):
            if out.get(side) is not None:
                out[side] = [
                    _coerce_value(col.kind, v)
                    for col, v in zip(schema.columns, out[side])
                ]
    elif shape == "prefix_range":
        prefix = []
        for item in out.get("prefix", []):
            name, value = item
            col = schema.column(name)
            prefix.append([name, _coerce_value(col.kind, value)])
        out["prefix"] = prefix
        rcol_name = out.get("range_column", schema.columns[len(prefix)].name)
        rcol = schema.column(rcol_name)
        for side in ("lo", "hi"):
            if out.get(side) is not None:
                out[side] = _coerce_value(rcol.kind, out[side])
    return out


def _emit(obj: Any) -> None:
    json.dump(obj, sys.stdout, ensure_ascii=False, indent=2, default=_json_default)
    sys.stdout.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"__hex__": bytes(value).hex()}
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def cmd_build(args: argparse.Namespace) -> int:
    schema = Schema.from_dict(_load_json(args.schema))
    stats = build_index(
        args.index, schema, _load_rows(args.source, schema),
        chunk_size=args.chunk_size,
    )
    _emit({"status": "built", "path": args.index, "rows": stats.rows,
           "bytes_written": stats.bytes_written})
    return 0


def cmd_range(args: argparse.Namespace) -> int:
    with open_index(args.index) as index:
        query = _coerce_query(index.schema, _load_json(args.query))
        token = args.cursor
        pages = args.pages
        page_count = 0
        result_pages: list[dict] = []
        while True:
            page, stats, plan = index.scan_page(
                query, args.page_size, cursor_token=token
            )
            page_count += 1
            result_pages.append(
                {
                    "page": page_count,
                    "rows": [
                        {"values": list(values),
                         index.row_id_name: {"__hex__": rid.hex()}}
                        for values, rid in page.rows
                    ],
                    "next_cursor": page.next_cursor,
                    "visited": stats.rows_visited,
                    "used_index": stats.used_index,
                    "eqp": stats.explain_query_plan,
                }
            )
            token = page.next_cursor
            if token is None or (pages is not None and page_count >= pages):
                break
        _emit({"plan": plan.explain(), "pages": result_pages})
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    with open_index(args.index) as index:
        query = _coerce_query(index.schema, _load_json(args.query))
        plan = plan_query(index.schema, query)
        page, stats, _ = index.scan_page(query, args.page_size)
        _emit(
            {
                "plan": plan.explain(),
                "explain_query_plan": stats.explain_query_plan,
                "explain_bytecode_seek_ops": [
                    op for op in stats.explain_bytecode
                    if op["opcode"] in ("SeekGE", "SeekGT", "SeekLE", "SeekLT",
                                        "Rewind", "Last")
                ],
                "sample_page_rows_returned": stats.rows_returned,
                "rows_visited_inside_interval": stats.rows_visited,
                "index_row_count": index.row_count,
                "next_cursor_present": page.next_cursor is not None,
            }
        )
    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    schema = Schema.from_dict(_load_json(args.schema))
    payload = _load_json(args.input)
    row = _coerce_row(schema, payload)
    values = [row[c.name] for c in schema.columns]
    key = key_encode(schema, values, row[schema.row_id_name])
    _emit({"key_hex": key.hex(), "length": len(key)})
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    schema = Schema.from_dict(_load_json(args.schema))
    key = bytes.fromhex(args.key_hex)
    values, row_id = key_decode(schema, key)
    _emit({"values": list(values), schema.row_id_name: {"__hex__": row_id.hex()}})
    return 0


# ---------------------------------------------------------------------------
# Updatable collection commands
# ---------------------------------------------------------------------------

def _iter_operations(path: str, schema: Schema):
    """Yield validated operation dicts from a JSON array or JSONL stream."""
    rid_name = schema.row_id_name
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise BatchError(f"source line {line_no}: invalid JSON: {exc}") from None
                yield _coerce_operation(item, schema, line_no)
        return
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise BatchError("batch source must be a JSON array of operations (or use .jsonl)")
    for line_no, item in enumerate(payload, start=1):
        yield _coerce_operation(item, schema, line_no)


def _coerce_operation(item: Any, schema: Schema, line_no: int) -> dict:
    if not isinstance(item, dict):
        raise BatchError(f"line {line_no}: operation must be an object")
    op = item.get("op")
    if op not in ("put", "delete"):
        raise BatchError(f"line {line_no}: op must be 'put' or 'delete'")
    rid_name = schema.row_id_name
    if rid_name not in item:
        raise BatchError(f"line {line_no}: missing {rid_name!r}")
    rid = _as_hex_object(item[rid_name], f"line {line_no}: {rid_name}")
    out: dict[str, Any] = {"op": op, rid_name: rid}
    if op == "put":
        for col in schema.columns:
            if col.name not in item:
                raise BatchError(f"line {line_no}: put missing column {col.name!r}")
            out[col.name] = _coerce_value(col.kind, item[col.name])
    else:
        extra = set(item) - {"op", rid_name}
        if extra:
            raise BatchError(
                f"line {line_no}: delete carries only {rid_name}, "
                f"found {sorted(extra)}"
            )
    return out


def cmd_collection_create(args: argparse.Namespace) -> int:
    schema = Schema.from_dict(_load_json(args.schema))
    cid = create_collection(args.collection, schema)
    with open_collection(args.collection) as coll:
        status = coll.status().to_dict()
    _emit({"status": "created", "collection_id": cid, **status})
    return 0


def cmd_collection_import(args: argparse.Namespace) -> int:
    cid, rows = import_from_index(
        args.index, args.collection, chunk_size=args.chunk_size
    )
    with open_collection(args.collection) as coll:
        status = coll.status().to_dict()
    _emit({
        "status": "imported",
        "collection_id": cid,
        "imported_rows": rows,
        **status,
    })
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        batch_id = _as_hex_object(_load_json(args.batch_id), "batch_id")
        receipt = coll.apply(
            batch_id,
            args.expected_head,
            _iter_operations(args.source, coll.schema),
        )
        _emit({"status": "applied", **receipt.to_dict()})
    return 0


def cmd_snapshot_begin(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        token = coll.begin_snapshot(None if args.gen is None else args.gen)
        status = coll.status()
    _emit({
        "status": "snapshot_open",
        "snapshot_token": token,
        "gen": args.gen if args.gen is not None else status.head,
        "head": status.head,
        "retain_from": status.retain_from,
    })
    return 0


def cmd_snapshot_release(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        result = coll.release_snapshot(args.token)
    _emit({"status": "snapshot_released", **result})
    return 0


def cmd_snapshot_range(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        query = _coerce_query(coll.schema, _load_json(args.query))
        page, stats, plan = coll.scan_snapshot_page(
            args.token, query, args.page_size, cursor_token=args.cursor
        )
        _emit({
            "status": "page",
            "gen": page.gen,
            "sid": page.sid,
            "plan": plan.explain(),
            "rows": [
                {"values": list(values),
                 coll.row_id_name: {"__hex__": rid.hex()}}
                for values, rid in page.rows
            ],
            "next_cursor": page.next_cursor,
            "last_key_hex": page.last_key.hex() if page.last_key else None,
            "stats": {
                "candidates_examined": stats.candidates_examined,
                "invisible_versions_skipped": stats.invisible_versions_skipped,
                "rows_delivered": stats.rows_delivered,
                "has_more": stats.has_more,
                "used_index": stats.used_index,
                "explain_query_plan": stats.explain_query_plan,
                "intervals": stats.intervals,
            },
        })
    return 0


def cmd_collect_history(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        result = coll.collect_history(args.retain_from)
        status = coll.status().to_dict()
    _emit({
        "status": "history_collected",
        "retain_from": result.retain_from,
        "previous_retain_from": result.previous_retain_from,
        "entries_removed": result.entries_removed,
        "active_snapshots": result.active_snapshots,
        "head": result.head,
        **{k: v for k, v in status.items() if k not in ("head", "retain_from")},
    })
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        _emit(coll.status().to_dict())
    return 0


def cmd_explain_snapshot(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        query = _coerce_query(coll.schema, _load_json(args.query))
        _emit(coll.explain_snapshot_scan(query, args.page_size))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bytekey", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="build a read-only byte-key index")
    p.add_argument("--schema", required=True)
    p.add_argument("--source", required=True, help="JSON list or .jsonl rows")
    p.add_argument("--index", required=True)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("range", help="run a paged range query")
    p.add_argument("--index", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--cursor", default=None)
    p.add_argument("--pages", type=int, default=None,
                   help="stop after N pages (default: drain to the end)")
    p.set_defaults(func=cmd_range)

    p = sub.add_parser("explain", help="show byte intervals and SQLite plan")
    p.add_argument("--index", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--page-size", type=int, default=100)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("encode", help="encode one row to a hex key")
    p.add_argument("--schema", required=True)
    p.add_argument("--input", required=True)
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("decode", help="decode a hex key")
    p.add_argument("--schema", required=True)
    p.add_argument("key_hex")
    p.set_defaults(func=cmd_decode)

    # -- updatable collections --------------------------------------------
    p = sub.add_parser("collection-create", help="create an empty updatable collection")
    p.add_argument("--schema", required=True)
    p.add_argument("--collection", required=True)
    p.set_defaults(func=cmd_collection_create)

    p = sub.add_parser("collection-import",
                       help="import a finished static index as generation 0")
    p.add_argument("--index", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.set_defaults(func=cmd_collection_import)

    p = sub.add_parser("apply", help="atomically apply one idempotent batch")
    p.add_argument("--collection", required=True)
    p.add_argument("--batch-id", required=True,
                   help='JSON file containing {"__hex__": "..."}')
    p.add_argument("--expected-head", type=int, required=True)
    p.add_argument("--source", required=True,
                   help="JSON array or .jsonl stream of operations")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("snapshot-begin", help="pin a generation, get a durable token")
    p.add_argument("--collection", required=True)
    p.add_argument("--gen", type=int, default=None,
                   help="generation to pin (default: current head)")
    p.set_defaults(func=cmd_snapshot_begin)

    p = sub.add_parser("snapshot-release", help="revoke a snapshot token and cursors")
    p.add_argument("--collection", required=True)
    p.add_argument("--token", required=True)
    p.set_defaults(func=cmd_snapshot_release)

    p = sub.add_parser("snapshot-range", help="read exactly one page from a snapshot")
    p.add_argument("--collection", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--cursor", default=None)
    p.set_defaults(func=cmd_snapshot_range)

    p = sub.add_parser("snapshot-explain", help="show intervals and plan of a snapshot scan")
    p.add_argument("--collection", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--page-size", type=int, default=100)
    p.set_defaults(func=cmd_explain_snapshot)

    p = sub.add_parser("collect-history", help="move retention boundary and reclaim history")
    p.add_argument("--collection", required=True)
    p.add_argument("--retain-from", type=int, required=True)
    p.set_defaults(func=cmd_collect_history)

    p = sub.add_parser("status", help="show collection generations, entries and snapshots")
    p.add_argument("--collection", required=True)
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except sqlite3.Error as exc:
        # Storage-level failures (unwritable path, locked/corrupt file) are
        # presented as ordinary domain errors rather than raw tracebacks.
        print(f"error: storage failure: {exc}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, OSError, ValueError, RuntimeError) as exc:
        # All domain errors derive from ValueError (Encoding/Query/Cursor/
        # Token/Batch/History/Schema errors) or RuntimeError (Build/
        # Collection/Conflict); present them as a clean failure with exit 2.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
