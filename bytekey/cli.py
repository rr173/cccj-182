"""Command line interface: build, range, explain, encode, decode."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .db import build_index, open_index
from .encoding import key_decode, key_encode
from .planner import plan_query
from .schema import Schema


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _coerce_value(col_kind: str, value: Any) -> Any:
    if value is None:
        return None
    if col_kind == "bytes":
        if isinstance(value, str):
            try:
                return bytes.fromhex(value)
            except ValueError as exc:
                raise SystemExit(f"bytes must be hex strings: {exc}") from None
        if isinstance(value, list) and all(isinstance(x, int) for x in value):
            return bytes(value)
        raise SystemExit("bytes values must be hex strings or integer arrays")
    return value


def _coerce_row(schema: Schema, row: dict) -> dict:
    out = {}
    for col in schema.columns:
        if col.name not in row:
            raise SystemExit(f"row missing column {col.name!r}")
        out[col.name] = _coerce_value(col.kind, row[col.name])
    rid = row.get(schema.row_id_name)
    if rid is None:
        raise SystemExit(f"row missing {schema.row_id_name!r}")
    if isinstance(rid, str):
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (json.JSONDecodeError, OSError, ValueError, RuntimeError) as exc:
        # All domain errors derive from ValueError (Encoding/Query/Cursor/
        # Build/Schema errors); present them as a clean failure with exit 2.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
