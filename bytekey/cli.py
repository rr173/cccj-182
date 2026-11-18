"""Command line interface: build, range, explain, encode, decode.

Plus the updatable-collection operations: ``collection-create``,
``collection-import``, ``apply``, ``snapshot-begin``, ``snapshot-release``,
``snapshot-range``, ``collect-history`` and ``status``, the named
secondary access index operations ``index-register``, ``index-advance``,
``index-status``, ``index-explain``, ``index-range``, ``index-cancel`` and
``index-delete``, and the persistent change-subscription operations
``subscription-create``, ``subscription-list``, ``subscription-read``,
``subscription-ack``, ``subscription-release`` and ``change-retention``.

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
from .changes import (
    ChangeError,
    ChangeLogError,
    ChangeRetentionError,
    SubscriptionConflictError,
    SubscriptionCursorError,
    SubscriptionError,
    SubscriptionNameConflict,
    SubscriptionNotFoundError,
    SubscriptionSkippedError,
)
from .collection import (
    BatchError,
    ConflictError,
    HistoryError,
    TokenError,
    create_collection,
    import_from_index,
    open_collection,
)
from .join import (
    JoinSource,
    join_cleanup,
    join_status,
    open_join,
    prepare_join,
    terminate_join,
)
from .secondary import (
    SecondaryDeclaration,
    SecondaryError,
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


def _coerce_query(
    schema: Schema, query: dict, *, columns: list[str] | None = None,
    declaration: bool = False,
) -> dict:
    """Coerce JSON query values to their declared column types.

    ``columns`` (the index declaration's ordered business column names) is
    used when the query is interpreted against a secondary index; default is
    the main schema order.
    """
    if declaration:
        raise TypeError("declaration is selected via `columns`")
    names = list(columns) if columns is not None else [c.name for c in schema.columns]
    cols = [schema.column(n) for n in names]
    out = dict(query)
    shape = query.get("shape")
    if shape == "full_tuple":
        for side in ("lower", "upper"):
            if out.get(side) is not None:
                out[side] = [
                    _coerce_value(col.kind, v) for col, v in zip(cols, out[side])
                ]
    elif shape == "prefix_range":
        prefix = []
        for name, value in out.get("prefix", []):
            col = schema.column(name)
            prefix.append([name, _coerce_value(col.kind, value)])
        out["prefix"] = prefix
        rcol_name = out.get("range_column")
        if rcol_name is None:
            rcol_name = names[len(prefix)]
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
        **result.to_dict(),
        **{k: v for k, v in status.items()
           if k not in ("head", "retain_from", "path")},
    })
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        _emit(coll.status().to_dict())
    return 0


# ---------------------------------------------------------------------------
# Persistent change subscriptions
# ---------------------------------------------------------------------------

def cmd_subscription_create(args: argparse.Namespace) -> int:
    if args.start_gen is not None and args.from_head:
        raise SubscriptionError(
            "give either --start-gen or --from-head, not both"
        )
    with open_collection(args.collection) as coll:
        info = coll.begin_subscription(
            args.name,
            start_gen=args.start_gen,
            from_head=bool(args.from_head),
        )
        head = coll.status().head
    _emit({"status": "subscription_created", "head": head, **info.to_dict()})
    return 0


def cmd_subscription_list(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        if args.name is None:
            subs = coll.list_subscriptions()
            _emit({
                "status": "subscription_list",
                "subscriptions": [s.to_dict() for s in subs],
                "change_retain_from": coll.change_retention()["retain_from"],
            })
        else:
            _emit({"status": "subscription_status",
                   **coll.subscription_info(args.name).to_dict()})
    return 0


def cmd_subscription_read(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        page = coll.read_changes_page(
            args.name, args.page_size, cursor_token=args.cursor
        )
        payload = page.to_dict()
        payload["collection_id"] = page.collection_id
        payload["row_id_name"] = coll.row_id_name
    _emit({"status": "page", **payload})
    return 0


def cmd_subscription_ack(args: argparse.Namespace) -> int:
    if args.cursor is not None and (
        args.gen is not None or args.seq is not None
    ):
        raise SystemExit("give either --gen/--seq or --cursor, not both")
    if args.cursor is None and (args.gen is None or args.seq is None):
        raise SystemExit("acknowledgement requires --gen and --seq, or --cursor")
    with open_collection(args.collection) as coll:
        result = coll.acknowledge_changes(
            args.name,
            gen=args.gen,
            seq=args.seq,
            expected_gen=args.expected_gen,
            expected_seq=args.expected_seq,
            instance_id=args.instance_id,
            cursor_token=args.cursor,
        )
    _emit({"status": "acknowledged", **result})
    return 0


def cmd_subscription_release(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        result = coll.release_subscription(
            args.name, instance_id=args.instance_id
        )
    _emit({"status": "subscription_released", **result})
    return 0


def cmd_change_retention(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        _emit({"status": "change_retention", **coll.change_retention()})
    return 0


def cmd_explain_snapshot(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        query = _coerce_query(coll.schema, _load_json(args.query))
        _emit(coll.explain_snapshot_scan(query, args.page_size))
    return 0


# ---------------------------------------------------------------------------
# Secondary access index commands
# ---------------------------------------------------------------------------

def _load_declaration(path: str) -> SecondaryDeclaration:
    return SecondaryDeclaration.from_dict(_load_json(path))


def cmd_index_register(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        declaration = _load_declaration(args.declaration)
        result = coll.register_index(declaration)
    _emit({"status": "index_registered", **result})
    return 0


def cmd_index_advance(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        expected = None if args.build_id is None else bytes.fromhex(args.build_id)
        status = coll.advance_index(args.name, args.budget,
                                    expected_build_id=expected)
        payload = status.to_dict()
    _emit({"status": "index_advanced", **payload})
    return 0


def cmd_index_status(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        if args.name is None:
            _emit({
                "status": "index_list",
                "indexes": [
                    {**i.to_dict(), "progress": coll.index_status(i.name).to_dict()}
                    for i in coll.list_indexes()
                ],
            })
        else:
            _emit({"status": "index_status",
                   **coll.index_status(args.name).to_dict()})
    return 0


def cmd_index_cancel(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        _emit({"status": "index_cancelled", **coll.cancel_index(args.name)})
    return 0


def cmd_index_delete(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        result = coll.delete_index(args.name, instance_id=args.instance_id)
    _emit({"status": "index_deleted", **result})
    return 0


def cmd_index_explain(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        index_info = next(
            (i for i in coll.list_indexes(include_deleted=False)
             if i.name == args.name),
            None,
        )
        if index_info is None:
            raise SecondaryError(f"no live secondary index named {args.name!r}")
        columns = [c["name"] for c in index_info.declaration.get("columns", [])]
        query = _coerce_query(coll.schema, _load_json(args.query), columns=columns)
        _emit(coll.explain_index_scan(args.name, query, args.page_size))
    return 0


def cmd_index_range(args: argparse.Namespace) -> int:
    with open_collection(args.collection) as coll:
        info = coll.list_indexes()
        match = next((i for i in info if i.name == args.name), None)
        columns = None
        if match is not None:
            columns = [c["name"] for c in match.declaration.get("columns", [])]
        query = _coerce_query(
            coll.schema, _load_json(args.query), columns=columns,
        )
        page, stats, plan = coll.scan_index_page(
            args.token, args.name, query, args.page_size,
            cursor_token=args.cursor,
        )
        _emit({
            "status": "page",
            "gen": page.gen,
            "sid": page.sid,
            "index": page.index_name,
            "instance_id": page.instance_id,
            "plan": plan.explain(),
            "rows": [
                {"values": list(values),
                 coll.row_id_name: {"__hex__": rid.hex()}}
                for values, rid in page.rows
            ],
            "next_cursor": page.next_cursor,
            "last_index_key_hex": (
                page.last_index_key.hex() if page.last_index_key else None
            ),
            "stats": stats.to_dict(),
        })
    return 0


# ---------------------------------------------------------------------------
# Disk-backed two-source equi-join commands
# ---------------------------------------------------------------------------

def _coerce_join_query(collection: str, index_name: str | None, query_path: str) -> dict:
    """Load a JSON query and coerce its values against the collection schema."""
    with open_collection(collection) as coll:
        raw = _load_json(query_path)
        if not isinstance(raw, dict):
            raise SystemExit("query JSON must be an object")
        columns = None
        if index_name is not None:
            match = next((i for i in coll.list_indexes() if i.name == index_name),
                         None)
            if match is not None:
                columns = [c["name"] for c in match.declaration.get("columns", [])]
        return _coerce_query(coll.schema, raw, columns=columns)


def _join_source_args(args: argparse.Namespace, side: str) -> JoinSource:
    collection = getattr(args, f"{side}_collection")
    token = getattr(args, f"{side}_token")
    index_name = getattr(args, f"{side}_index")
    query = _coerce_join_query(
        collection, index_name, getattr(args, f"{side}_query")
    )
    return JoinSource(
        collection=collection, token=token, query=query, index=index_name,
    )


def _join_row_json(row) -> dict:
    def side_json(side):
        if side is None:
            return None
        return {
            "row_id": {"__hex__": side.row_id.hex()},
            "values": list(side.values),
        }

    return {"kind": row.kind, "left": side_json(row.left),
            "right": side_json(row.right)}


def cmd_join_prepare(args: argparse.Namespace) -> int:
    mapping = _load_json(args.mapping)
    result = prepare_join(
        args.work,
        left=_join_source_args(args, "left"),
        right=_join_source_args(args, "right"),
        mapping=mapping,
        mode=args.mode,
        nulls_equal=bool(args.nulls_equal),
        page_size=args.page_size,
    )
    _emit(result)
    return 0


def cmd_join_page(args: argparse.Namespace) -> int:
    with open_join(args.work, args.handle) as join:
        page, stats = join.page(args.cursor)
        _emit({
            "status": "page",
            "join_id": page.join_id,
            "declaration_hash": page.declaration_hash,
            "rows": [_join_row_json(r) for r in page.rows],
            "next_cursor": page.next_cursor,
            "stats": stats.to_dict(),
        })
    return 0


def cmd_join_close(args: argparse.Namespace) -> int:
    result = terminate_join(args.work, args.handle, reclaim=bool(args.reclaim))
    _emit({"status": "join_closed", **result})
    return 0


def cmd_join_cleanup(args: argparse.Namespace) -> int:
    _emit({"status": "join_cleaned", **join_cleanup(args.work)})
    return 0


def cmd_join_status(args: argparse.Namespace) -> int:
    _emit(join_status(args.work))
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

    # -- persistent change subscriptions ------------------------------------
    p = sub.add_parser("subscription-create",
                       help="create a durable named change subscription")
    p.add_argument("--collection", required=True)
    p.add_argument("--name", required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--start-gen", type=int, default=None,
                       help="first retained generation to deliver")
    group.add_argument("--from-head", action="store_true",
                       help="start at the current head generation")
    p.set_defaults(func=cmd_subscription_create)

    p = sub.add_parser("subscription-list",
                       help="list subscriptions, or show one by name")
    p.add_argument("--collection", required=True)
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_subscription_list)

    p = sub.add_parser("subscription-read",
                       help="read exactly one bounded page of change events")
    p.add_argument("--collection", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--page-size", type=int, required=True)
    p.add_argument("--cursor", default=None,
                   help="next_cursor from the previous page")
    p.set_defaults(func=cmd_subscription_read)

    p = sub.add_parser("subscription-ack",
                       help="advance the durable acknowledged position")
    p.add_argument("--collection", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--gen", type=int, default=None,
                   help="acknowledged generation (with --seq)")
    p.add_argument("--seq", type=int, default=None,
                   help="acknowledged sequence (with --gen)")
    p.add_argument("--cursor", default=None,
                   help="acknowledge the page end named by this credential")
    p.add_argument("--expected-gen", type=int, required=True)
    p.add_argument("--expected-seq", type=int, required=True)
    p.add_argument("--instance-id", default=None,
                   help="refuse if the name now refers to another instance")
    p.set_defaults(func=cmd_subscription_ack)

    p = sub.add_parser("subscription-release",
                       help="release a subscription and invalidate its cursors")
    p.add_argument("--collection", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--instance-id", default=None)
    p.set_defaults(func=cmd_subscription_release)

    p = sub.add_parser("change-retention",
                       help="show the retained change horizon and subscriptions")
    p.add_argument("--collection", required=True)
    p.set_defaults(func=cmd_change_retention)

    # -- named secondary access indexes -----------------------------------
    p = sub.add_parser("index-register",
                       help="register a named secondary index (coverage start = head)")
    p.add_argument("--collection", required=True)
    p.add_argument("--declaration", required=True,
                   help="JSON index declaration (name + ordered columns)")
    p.set_defaults(func=cmd_index_register)

    p = sub.add_parser("index-advance",
                       help="do at most N units of baseline/catch-up work and commit")
    p.add_argument("--collection", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--build-id", default=None, help="hex build handle (resume)")
    p.set_defaults(func=cmd_index_advance)

    p = sub.add_parser("index-status", help="show one index (or list all) progress")
    p.add_argument("--collection", required=True)
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_index_status)

    p = sub.add_parser("index-cancel", help="cancel a building secondary index")
    p.add_argument("--collection", required=True)
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_index_cancel)

    p = sub.add_parser("index-delete", help="retire a named index instance")
    p.add_argument("--collection", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--instance-id", default=None,
                   help="refuse if the name now refers to a different instance")
    p.set_defaults(func=cmd_index_delete)

    p = sub.add_parser("index-explain",
                       help="show intervals and SQLite plan of an index scan")
    p.add_argument("--collection", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--page-size", type=int, default=100)
    p.set_defaults(func=cmd_index_explain)

    p = sub.add_parser("index-range",
                       help="read one page through a ready secondary index")
    p.add_argument("--collection", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--cursor", default=None)
    p.set_defaults(func=cmd_index_range)

    # -- disk-backed two-source equi-join -----------------------------------
    p = sub.add_parser("join-prepare",
                       help="build the work file for one join read")
    p.add_argument("--work", required=True, help="join work file to create")
    for side in ("left", "right"):
        p.add_argument(f"--{side}-collection", required=True)
        p.add_argument(f"--{side}-token", required=True,
                       help="existing snapshot token for this side")
        p.add_argument(f"--{side}-index", default=None,
                       help="ready secondary index name (default: main order)")
        p.add_argument(f"--{side}-query", required=True,
                       help="JSON bytekey range query filtering this side")
    p.add_argument("--mapping", required=True,
                   help='JSON column mapping: {"left": [...], "right": [...]} '
                        "or [[left_col, right_col], ...]")
    p.add_argument("--mode", choices=["inner", "left", "full"], required=True)
    p.add_argument("--nulls-equal", action="store_true",
                   help="treat null join-key components as equal")
    p.add_argument("--page-size", type=int, required=True,
                   help="fixed page size bound into this join read")
    p.set_defaults(func=cmd_join_prepare)

    p = sub.add_parser("join-page", help="deliver exactly one join page")
    p.add_argument("--work", required=True)
    p.add_argument("--handle", required=True)
    p.add_argument("--cursor", default=None)
    p.set_defaults(func=cmd_join_page)

    p = sub.add_parser("join-close",
                       help="end the join read (handle and cursors expire)")
    p.add_argument("--work", required=True)
    p.add_argument("--handle", required=True)
    p.add_argument("--reclaim", action="store_true",
                   help="also delete the work file")
    p.set_defaults(func=cmd_join_close)

    p = sub.add_parser("join-cleanup",
                       help="delete an unfinished or closed join work file")
    p.add_argument("--work", required=True)
    p.set_defaults(func=cmd_join_cleanup)

    p = sub.add_parser("join-status", help="show a join work file's state")
    p.add_argument("--work", required=True)
    p.set_defaults(func=cmd_join_status)
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
    except KeyError as exc:
        # Unknown column/key references (e.g. in a query or mapping).
        print(f"error: unknown column or key: {exc}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, OSError, ValueError, RuntimeError) as exc:
        # All domain errors derive from ValueError (Encoding/Query/Cursor/
        # Token/Batch/History/Schema errors) or RuntimeError (Build/
        # Collection/Conflict); present them as a clean failure with exit 2.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
