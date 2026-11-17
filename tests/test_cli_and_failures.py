"""End-to-end CLI tests plus explicit failure-mode coverage.

Covers the build/range/explain/encode/decode commands, a corrupt last source
line aborting an unfinished build, a write failure to an unwritable path,
foreign/tampered cursors and bad endpoint types.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bytekey import (  # noqa: E402
    BuildError,
    Column,
    CursorError,
    QueryError,
    Schema,
    build_index,
    open_index,
)


SCHEMA = {
    "columns": [
        {"name": "region", "kind": "text", "asc": True, "nulls_first": True},
        {"name": "ts", "kind": "int64", "asc": False, "nulls_first": True},
    ],
    "row_id_name": "row_id",
}

ROWS = [
    {"region": "", "ts": 1, "row_id": "722d3034"},
    {"region": "a", "ts": 2, "row_id": "722d3031"},
    {"region": "a", "ts": 1, "row_id": "722d3032"},
    {"region": "a", "ts": None, "row_id": "722d3033"},
    {"region": "a\u0000", "ts": 1, "row_id": "722d3035"},
    {"region": "ab", "ts": 1, "row_id": "722d3036"},
]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.schema_path = os.path.join(self.tmp.name, "schema.json")
        self.source_path = os.path.join(self.tmp.name, "rows.json")
        self.index_path = os.path.join(self.tmp.name, "index.db")
        with open(self.schema_path, "w", encoding="utf-8") as fh:
            json.dump(SCHEMA, fh)
        with open(self.source_path, "w", encoding="utf-8") as fh:
            json.dump(ROWS, fh)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *args):
        # Under -I neither cwd nor PYTHONPATH is searchable, and running the
        # package's __main__ by path makes its own directory sys.path[0]
        # (breaking relative imports).  Use a tiny generated launcher whose
        # own directory is still imported under -I and which adds ROOT.
        launcher = os.path.join(self.tmp.name, "bytekey_launcher.py")
        with open(launcher, "w", encoding="utf-8") as fh:
            fh.write(
                "import sys\n"
                f"sys.path.insert(0, {ROOT!r})\n"
                "from bytekey.cli import main\n"
                "raise SystemExit(main())\n"
            )
        return subprocess.run(
            [sys.executable, "-I", launcher, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=dict(os.environ),
        )

    def test_build_range_explain_encode_decode(self):
        r = self._run("build", "--schema", self.schema_path,
                      "--source", self.source_path, "--index", self.index_path)
        self.assertEqual(r.returncode, 0, r.stderr)
        built = json.loads(r.stdout)
        self.assertEqual(built["rows"], len(ROWS))

        query = {
            "shape": "prefix_range",
            "prefix": [],
            "range_column": "region",
        }
        qpath = os.path.join(self.tmp.name, "query.json")
        with open(qpath, "w", encoding="utf-8") as fh:
            json.dump(query, fh)

        r = self._run("explain", "--index", self.index_path,
                      "--query", qpath, "--page-size", "2")
        self.assertEqual(r.returncode, 0, r.stderr)
        explained = json.loads(r.stdout)
        self.assertEqual(explained["plan"]["interval_count"], 1)
        self.assertTrue(
            all("USING PRIMARY KEY" in d["detail"]
                for d in explained["explain_query_plan"])
        )

        # Page through with page size 1 and drain every page.
        r = self._run("range", "--index", self.index_path,
                      "--query", qpath, "--page-size", "1")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        delivered = [p["rows"][0]["values"] for p in out["pages"]]
        self.assertEqual(len(out["pages"]), len(ROWS))
        # region ASC; within region 'a', ts DESC nulls_first.
        self.assertEqual(
            delivered,
            [
                ["", 1],
                ["a", None],
                ["a", 2],
                ["a", 1],
                ["a\x00", 1],
                ["ab", 1],
            ],
        )
        # Every page reports primary-key index usage.
        self.assertTrue(all(p["used_index"] for p in out["pages"]))

        # encode / decode round trip through the CLI.
        one_row = {"region": "a", "ts": 2, "row_id": "722d3031"}
        in_path = os.path.join(self.tmp.name, "one.json")
        with open(in_path, "w", encoding="utf-8") as fh:
            json.dump(one_row, fh)
        r = self._run("encode", "--schema", self.schema_path, "--input", in_path)
        self.assertEqual(r.returncode, 0, r.stderr)
        key_hex = json.loads(r.stdout)["key_hex"]
        r = self._run("decode", "--schema", self.schema_path, key_hex)
        self.assertEqual(r.returncode, 0, r.stderr)
        decoded = json.loads(r.stdout)
        self.assertEqual(decoded["values"], ["a", 2])
        self.assertEqual(decoded["row_id"]["__hex__"], "722d3031")

    def test_jsonl_bad_last_line_aborts_build(self):
        jsonl_path = os.path.join(self.tmp.name, "rows.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for row in ROWS[:-1]:
                fh.write(json.dumps(row) + "\n")
            fh.write("{not valid json}\n")  # corrupt final line
        target = os.path.join(self.tmp.name, "should_not_exist.db")
        r = self._run("build", "--schema", self.schema_path,
                      "--source", jsonl_path, "--index", target)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(os.path.exists(target))

    def test_write_failure_unwritable_path(self):
        # A path whose parent directory does not exist and cannot be created
        # here: point at a file (not a directory) as the parent.
        blocker = os.path.join(self.tmp.name, "blocker")
        with open(blocker, "wb") as fh:
            fh.write(b"x")
        target = os.path.join(blocker, "index.db")
        schema = Schema.from_dict(SCHEMA)
        with self.assertRaises((BuildError, OSError)):
            build_index(
                target, schema,
                ({"region": "a", "ts": 1, "row_id": b"r1"} for _ in range(1)),
            )
        self.assertFalse(os.path.exists(target))


class ApiFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.schema = Schema(
            columns=(
                Column("region", "text"),
                Column("ts", "int64", asc=False, nulls_first=True),
            )
        )
        self.path = os.path.join(self.tmp.name, "api.db")
        rows = [
            {"region": "a", "ts": 1, "row_id": b"r1"},
            {"region": "b", "ts": 2, "row_id": b"r2"},
        ]
        build_index(self.path, self.schema, rows)
        self.index = open_index(self.path)

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def test_bad_endpoint_type_rejected(self):
        with self.assertRaises(QueryError):
            self.index.scan_page(
                {"shape": "prefix_range", "prefix": [],
                 "range_column": "ts", "lo": "x"},
                10,
            )

    def test_foreign_and_garbled_cursor_rejected(self):
        page, _, _ = self.index.scan_page(
            {"shape": "prefix_range", "prefix": [], "range_column": "region"},
            1,
        )
        token = page.next_cursor
        self.assertIsNotNone(token)
        # Tamper with the signature.
        body, sig = token.split(".")
        with self.assertRaises(CursorError):
            self.index.scan_page(
                {"shape": "prefix_range", "prefix": [], "range_column": "region"},
                1, cursor_token=body + "." + sig[:-2] + "aa",
            )
        # Garbage token.
        with self.assertRaises(CursorError):
            self.index.scan_page(
                {"shape": "prefix_range", "prefix": [], "range_column": "region"},
                1, cursor_token="@@@not-a-token@@@",
            )

    def test_reopen_completed_index_is_readonly(self):
        # The read-only URI connection rejects writes.
        with self.assertRaises(sqlite3_readonly_error()):
            self.index.conn.execute(
                "INSERT INTO bytekey_index(key) VALUES (x'00')"
            )


def sqlite3_readonly_error():
    import sqlite3

    return sqlite3.OperationalError


if __name__ == "__main__":
    unittest.main(verbosity=2)
