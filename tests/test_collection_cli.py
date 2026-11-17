"""End-to-end CLI coverage for updatable collections.

Exercises collection-create / collection-import / apply / snapshot-begin /
snapshot-range (one page per invocation) / snapshot-release /
collect-history / status, the __hex__ bytes convention, nonzero exits with
locatable messages (line numbers, conflicts, invalid tokens, write errors),
and cross-process cursor continuation driven purely through the CLI.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "bytekey_cli.py")

SCHEMA = {
    "columns": [
        {"name": "region", "kind": "text", "asc": True, "nulls_first": True},
        {"name": "ts", "kind": "int64", "asc": False, "nulls_first": True},
        {"name": "blob", "kind": "bytes", "asc": True, "nulls_first": True},
    ],
    "row_id_name": "row_id",
}


class CollectionCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.schema_path = os.path.join(self.tmp.name, "schema.json")
        self.coll = os.path.join(self.tmp.name, "coll.db")
        self.static = os.path.join(self.tmp.name, "static.db")
        with open(self.schema_path, "w", encoding="utf-8") as fh:
            json.dump(SCHEMA, fh)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args, expect=0):
        proc = subprocess.run(
            [sys.executable, "-I", CLI, *args],
            capture_output=True, text=True,
        )
        if proc.returncode != expect:
            self.fail(
                f"cli {args[:2]} exit {proc.returncode}\n"
                f"stdout={proc.stdout}\nstderr={proc.stderr}"
            )
        return proc

    def test_full_collection_lifecycle_and_migration(self):
        # 1. Static index -> collection migration (generation 0).
        seed = [
            {"region": "a", "ts": 1, "blob": {"__hex__": "00ff"},
             "row_id": {"__hex__": "7231"}},
            {"region": "b", "ts": None, "blob": {"__hex__": "01"},
             "row_id": {"__hex__": "7232"}},
        ]
        seed_path = os.path.join(self.tmp.name, "seed.json")
        with open(seed_path, "w", encoding="utf-8") as fh:
            json.dump(seed, fh)
        self.run_cli("build", "--schema", self.schema_path,
                     "--source", seed_path, "--index", self.static)
        out = self.run_cli("collection-import", "--index", self.static,
                           "--collection", self.coll)
        payload = json.loads(out.stdout)
        self.assertEqual(payload["head"], 0)
        self.assertEqual(payload["live_entries"], 2)

        # 2. Empty-collection creation path also works head=0.
        empty = os.path.join(self.tmp.name, "empty.db")
        out = self.run_cli("collection-create", "--schema", self.schema_path,
                           "--collection", empty)
        self.assertEqual(json.loads(out.stdout)["head"], 0)

        # 3. Apply a batch.
        batch_path = os.path.join(self.tmp.name, "b1.json")
        with open(batch_path, "w", encoding="utf-8") as fh:
            json.dump({"__hex__": "6231"}, fh)
        ops_path = os.path.join(self.tmp.name, "ops.jsonl")
        with open(ops_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "op": "put", "row_id": {"__hex__": "7233"},
                "region": "a", "ts": 9, "blob": {"__hex__": ""},
            }) + "\n")
            fh.write(json.dumps({"op": "delete",
                                 "row_id": {"__hex__": "7232"}}) + "\n")
        out = self.run_cli("apply", "--collection", self.coll,
                           "--batch-id", batch_path,
                           "--expected-head", "0", "--source", ops_path)
        receipt = json.loads(out.stdout)
        self.assertEqual((receipt["head"], receipt["applied"]), (1, 2))

        # 4. Idempotent re-send through the CLI returns replayed receipt.
        out2 = self.run_cli("apply", "--collection", self.coll,
                            "--batch-id", batch_path,
                            "--expected-head", "0", "--source", ops_path)
        self.assertTrue(json.loads(out2.stdout)["replayed"])

        # 5. Snapshot on generation 0 survives later writes.
        out = self.run_cli("snapshot-begin", "--collection", self.coll,
                           "--gen", "0")
        old_token = json.loads(out.stdout)["snapshot_token"]
        out = self.run_cli("snapshot-begin", "--collection", self.coll)
        latest = json.loads(out.stdout)
        new_token = latest["snapshot_token"]
        self.assertEqual(latest["gen"], 1)

        query_path = os.path.join(self.tmp.name, "q.json")
        with open(query_path, "w", encoding="utf-8") as fh:
            json.dump({"shape": "full_tuple",
                       "lower": None, "upper": None,
                       "lower_inclusive": True, "upper_inclusive": True}, fh)

        # 6. One page per process; continue via returned cursor.
        out = self.run_cli("snapshot-range", "--collection", self.coll,
                           "--token", old_token, "--query", query_path,
                           "--page-size", "1")
        page1 = json.loads(out.stdout)
        self.assertEqual(page1["gen"], 0)
        self.assertEqual(len(page1["rows"]), 1)
        self.assertIsNotNone(page1["next_cursor"])
        self.assertTrue(page1["stats"]["used_index"] is False)  # unbounded
        out = self.run_cli("snapshot-range", "--collection", self.coll,
                           "--token", old_token, "--query", query_path,
                           "--page-size", "1", "--cursor",
                           page1["next_cursor"])
        page2 = json.loads(out.stdout)
        self.assertEqual(len(page2["rows"]), 1)
        self.assertIsNone(page2["next_cursor"])
        rids = {page1["rows"][0]["row_id"]["__hex__"],
                page2["rows"][0]["row_id"]["__hex__"]}
        self.assertEqual(rids, {"7231", "7232"})  # gen-0 view, r2 still there

        # 7. Latest snapshot reflects the delete/add.
        out = self.run_cli("snapshot-range", "--collection", self.coll,
                           "--token", new_token, "--query", query_path,
                           "--page-size", "10")
        latest_rows = json.loads(out.stdout)["rows"]
        self.assertEqual(
            sorted(r["row_id"]["__hex__"] for r in latest_rows),
            ["7231", "7233"],
        )
        # bytes round-trip uses the __hex__ single-field object.
        r33 = next(r for r in latest_rows
                   if r["row_id"]["__hex__"] == "7233")
        self.assertEqual(r33["values"][2], {"__hex__": ""})

        # 8. Release invalidates the token and its cursors.
        self.run_cli("snapshot-release", "--collection", self.coll,
                     "--token", old_token)
        proc = self.run_cli("snapshot-range", "--collection", self.coll,
                            "--token", old_token, "--query", query_path,
                            "--page-size", "1", expect=2)
        self.assertIn("released", proc.stderr.lower())
        # Forged token refused, never silently retargeted.
        body, sig = new_token.split(".")
        proc = self.run_cli("snapshot-range", "--collection", self.coll,
                            "--token", body + "." + ("z" * len(sig)),
                            "--query", query_path, "--page-size", "1",
                            expect=2)
        self.assertIn("signature", proc.stderr.lower())

        # 9. History collection and status.
        out = self.run_cli("collect-history", "--collection", self.coll,
                           "--retain-from", "1")
        hist = json.loads(out.stdout)
        self.assertEqual(hist["retain_from"], 1)
        self.assertGreaterEqual(hist["entries_removed"], 1)
        out = self.run_cli("status", "--collection", self.coll)
        status = json.loads(out.stdout)
        self.assertEqual(status["head"], 1)
        self.assertEqual(status["retain_from"], 1)
        # Backwards boundary move is an explicit failure.
        proc = self.run_cli("collect-history", "--collection", self.coll,
                            "--retain-from", "0", expect=2)
        self.assertIn("backwards", proc.stderr.lower())

    def test_error_locations_and_nonzero_exits(self):
        self.run_cli("collection-create", "--schema", self.schema_path,
                     "--collection", self.coll)
        bid = os.path.join(self.tmp.name, "b.json")
        with open(bid, "w", encoding="utf-8") as fh:
            json.dump({"__hex__": "01"}, fh)

        # Corrupt source tail: line number must be reported, exit nonzero.
        bad = os.path.join(self.tmp.name, "bad.jsonl")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "op": "put", "row_id": {"__hex__": "01"},
                "region": "a", "ts": 1, "blob": {"__hex__": ""},
            }) + "\n")
            fh.write("{broken json\n")
        proc = self.run_cli("apply", "--collection", self.coll,
                            "--batch-id", bid, "--expected-head", "0",
                            "--source", bad, expect=2)
        self.assertIn("line 2", proc.stderr)

        # Duplicate identity reports both line numbers.
        dup = os.path.join(self.tmp.name, "dup.jsonl")
        with open(dup, "w", encoding="utf-8") as fh:
            for _ in range(2):
                fh.write(json.dumps({
                    "op": "put", "row_id": {"__hex__": "01"},
                    "region": "a", "ts": 1, "blob": {"__hex__": ""},
                }) + "\n")
        proc = self.run_cli("apply", "--collection", self.coll,
                            "--batch-id", bid, "--expected-head", "0",
                            "--source", dup, expect=2)
        self.assertIn("line 2", proc.stderr)

        # expected_head conflict after a successful write.
        good = os.path.join(self.tmp.name, "good.jsonl")
        with open(good, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "op": "put", "row_id": {"__hex__": "02"},
                "region": "a", "ts": 1, "blob": {"__hex__": ""},
            }) + "\n")
        bid2 = os.path.join(self.tmp.name, "b2.json")
        with open(bid2, "w", encoding="utf-8") as fh:
            json.dump({"__hex__": "02"}, fh)
        self.run_cli("apply", "--collection", self.coll,
                     "--batch-id", bid2, "--expected-head", "0",
                     "--source", good)
        # A stale head with a fresh batch_id is an explicit conflict.
        bid3 = os.path.join(self.tmp.name, "b3.json")
        with open(bid3, "w", encoding="utf-8") as fh:
            json.dump({"__hex__": "03"}, fh)
        proc = self.run_cli("apply", "--collection", self.coll,
                            "--batch-id", bid3, "--expected-head", "0",
                            "--source", good, expect=2)
        self.assertIn("head", proc.stderr.lower())
        # Reusing the old batch id with different content conflicts too.
        changed = os.path.join(self.tmp.name, "changed.jsonl")
        with open(changed, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "op": "put", "row_id": {"__hex__": "02"},
                "region": "zzz", "ts": 1, "blob": {"__hex__": ""},
            }) + "\n")
        proc = self.run_cli("apply", "--collection", self.coll,
                            "--batch-id", bid2, "--expected-head", "0",
                            "--source", changed, expect=2)
        self.assertIn("already applied", proc.stderr.lower())

        # Write error: an unwritable target directory fails with exit 2.
        ro_dir = os.path.join(self.tmp.name, "ro")
        os.makedirs(ro_dir)
        os.chmod(ro_dir, 0o500)
        try:
            proc = self.run_cli(
                "collection-create", "--schema", self.schema_path,
                "--collection", os.path.join(ro_dir, "x.db"), expect=2,
            )
            self.assertNotEqual(proc.returncode, 0)
        finally:
            os.chmod(ro_dir, 0o700)

        # A nonexistent collection path is a clean nonzero failure.
        proc = self.run_cli("status", "--collection",
                            os.path.join(self.tmp.name, "nope.db"), expect=2)
        self.assertIn("does not exist", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
