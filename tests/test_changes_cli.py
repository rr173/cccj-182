"""End-to-end CLI coverage for change subscriptions.

Drives subscription-create / subscription-list / subscription-read /
subscription-ack / subscription-release / change-retention across separate
*processes* (one page per invocation), including restart continuation,
replayed reads and acknowledgements, cursor cross-use, a deleted/re-created
name, stale expected positions and the __hex__ bytes convention.  Every
failure mode must exit nonzero with a locatable message.
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


class ChangeSubscriptionCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.coll = os.path.join(self.dir, "coll.db")
        self.schema = os.path.join(self.dir, "schema.json")
        with open(self.schema, "w", encoding="utf-8") as fh:
            json.dump(SCHEMA, fh)
        self._create()

    def tearDown(self):
        self.tmp.cleanup()

    # -- helpers -----------------------------------------------------------
    def run_cli(self, *args, expect=0):
        proc = subprocess.run(
            [sys.executable, "-I", CLI, *args],
            capture_output=True, text=True,
        )
        if proc.returncode != expect:
            self.fail(
                f"cli {args[:2]} exit {proc.returncode} (wanted {expect})\n"
                f"stdout={proc.stdout}\nstderr={proc.stderr}"
            )
        return json.loads(proc.stdout) if expect == 0 else proc

    def _create(self):
        self.run_cli("collection-create", "--schema", self.schema,
                     "--collection", self.coll)

    def apply(self, batch_hex, expected_head, ops):
        bpath = os.path.join(self.dir, "b.json")
        with open(bpath, "w", encoding="utf-8") as fh:
            json.dump({"__hex__": batch_hex}, fh)
        spath = os.path.join(self.dir, "ops.jsonl")
        with open(spath, "w", encoding="utf-8") as fh:
            for item in ops:
                fh.write(json.dumps(item) + "\n")
        return self.run_cli(
            "apply", "--collection", self.coll, "--batch-id", bpath,
            "--expected-head", str(expected_head), "--source", spath,
        )

    def seed(self):
        self.apply("01", 0, [
            {"op": "put", "row_id": {"__hex__": "7231"},
             "region": "a", "ts": 10, "blob": {"__hex__": "6f6c64"}},
            {"op": "put", "row_id": {"__hex__": "7232"},
             "region": "a", "ts": 10, "blob": {"__hex__": "6b656570"}},
        ])
        self.apply("02", 1, [
            {"op": "put", "row_id": {"__hex__": "7231"},
             "region": "c", "ts": 20, "blob": {"__hex__": "6e6577"}},
            {"op": "delete", "row_id": {"__hex__": "7232"}},
        ])
        self.apply("03", 2, [])
        self.apply("04", 3, [
            {"op": "put", "row_id": {"__hex__": "7232"},
             "region": "b", "ts": 8, "blob": {"__hex__": "7265626f726e"}},
        ])

    # -- tests -------------------------------------------------------------
    def test_read_ack_restart_and_reclaim(self):
        self.seed()
        sub = self.run_cli("subscription-create", "--collection", self.coll,
                           "--name", "down", "--start-gen", "1")
        self.assertEqual(sub["start_gen"], 1)

        # One page per process, page size 1, collecting cursors between runs.
        events, markers, cursor, pos = [], [], None, (1, 0)
        for _ in range(50):
            cmd = ["subscription-read", "--collection", self.coll,
                   "--name", "down", "--page-size", "1"]
            if cursor is not None:
                cmd += ["--cursor", cursor]
            page = self.run_cli(*cmd)
            events.extend(page["events"])
            markers.extend(m["gen"] for m in page["batches"])
            # Re-running the identical read is deterministic and repeats the
            # successor credential.
            again = self.run_cli(*cmd)
            self.assertEqual(again["events"], page["events"])
            self.assertEqual(again["next_cursor"], page["next_cursor"])
            ack = self.run_cli(
                "subscription-ack", "--collection", self.coll,
                "--name", "down", "--gen", str(page["gen"]),
                "--seq", str(page["seq"]),
                "--expected-gen", str(pos[0]), "--expected-seq", str(pos[1]),
            )
            self.assertFalse(ack["replayed"])
            # Lost-response retry of the same ack is idempotent.
            ack2 = self.run_cli(
                "subscription-ack", "--collection", self.coll,
                "--name", "down", "--gen", str(page["gen"]),
                "--seq", str(page["seq"]),
                "--expected-gen", str(pos[0]), "--expected-seq", str(pos[1]),
            )
            self.assertTrue(ack2["replayed"])
            pos = (page["gen"], page["seq"])
            if not page["has_more"]:
                break
            cursor = page["next_cursor"]

        self.assertEqual(
            [(e["gen"], e["seq"], e["row_id"]["__hex__"], e["op"])
             for e in events],
            [(1, 1, "7231", "insert"), (1, 2, "7232", "insert"),
             (2, 1, "7231", "update"), (2, 2, "7232", "delete"),
             (4, 1, "7232", "insert")],
        )
        # Empty generation 3 still appeared as a batch boundary.
        self.assertIn(3, markers)
        # Full before/after values, with absent markers.
        update = events[2]
        self.assertEqual(
            update["before"],
            {"present": True,
             "values": ["a", 10, {"__hex__": "6f6c64"}]},
        )
        self.assertEqual(
            update["after"],
            {"present": True,
             "values": ["c", 20, {"__hex__": "6e6577"}]},
        )
        self.assertEqual(events[3]["after"], {"present": False})
        self.assertEqual(events[0]["before"], {"present": False})

        # Acknowledged everything: retention advances and records are gone.
        hist = self.run_cli("collect-history", "--collection", self.coll,
                            "--retain-from", "4")
        self.assertGreaterEqual(hist["change_events_removed"], 1)
        self.assertGreaterEqual(hist["change_batches_removed"], 1)
        ret = self.run_cli("change-retention", "--collection", self.coll)
        self.assertEqual(ret["retain_from"], 5)
        status = self.run_cli("status", "--collection", self.coll)
        self.assertEqual(status["change_events_retained"], 0)
        # Subscribing below the boundary fails, naming it.
        bad = self.run_cli("subscription-create", "--collection", self.coll,
                           "--name", "late", "--start-gen", "1", expect=2)
        self.assertIn("boundary", bad.stderr)
        self.assertNotEqual(bad.returncode, 0)

    def test_cursor_cross_use_conflicts_and_recreated_name(self):
        self.seed()
        self.run_cli("subscription-create", "--collection", self.coll,
                     "--name", "a", "--start-gen", "1")
        self.run_cli("subscription-create", "--collection", self.coll,
                     "--name", "b", "--start-gen", "1")
        page = self.run_cli("subscription-read", "--collection", self.coll,
                            "--name", "b", "--page-size", "1")
        cursor = page["next_cursor"]
        # Cursor from b rejected by a.
        bad = self.run_cli("subscription-read", "--collection", self.coll,
                           "--name", "a", "--page-size", "1",
                           "--cursor", cursor, expect=2)
        self.assertIn("subscription", bad.stderr.lower())
        # Read a second page with the cursor, acknowledge *past* the first
        # page, then try to reuse the first page's cursor: stale.
        page2 = self.run_cli("subscription-read", "--collection", self.coll,
                             "--name", "b", "--page-size", "1",
                             "--cursor", cursor)
        self.run_cli("subscription-ack", "--collection", self.coll,
                     "--name", "b", "--gen", str(page2["gen"]),
                     "--seq", str(page2["seq"]), "--expected-gen", "1",
                     "--expected-seq", "0")
        bad = self.run_cli("subscription-read", "--collection", self.coll,
                           "--name", "b", "--page-size", "1",
                           "--cursor", cursor, expect=2)
        self.assertIn("acknowledged", bad.stderr.lower())
        # Release, re-create the same name, reuse the old cursor: rejected.
        self.run_cli("subscription-release", "--collection", self.coll,
                     "--name", "b")
        self.run_cli("subscription-create", "--collection", self.coll,
                     "--name", "b", "--start-gen", "1")
        bad = self.run_cli("subscription-read", "--collection", self.coll,
                           "--name", "b", "--page-size", "1",
                           "--cursor", cursor, expect=2)
        self.assertIn("instance", bad.stderr.lower())
        # Stale expected ack position after a concurrent advance.
        info = self.run_cli("subscription-list", "--collection", self.coll,
                            "--name", "b")
        self.assertEqual(info["ack_gen"], 1)
        # A duplicate active name is refused.
        bad = self.run_cli("subscription-create", "--collection", self.coll,
                           "--name", "a", "--start-gen", "2", expect=2)
        self.assertIn("already exists", bad.stderr)
        # Reading an unknown subscription is a locatable error.
        bad = self.run_cli("subscription-read", "--collection", self.coll,
                           "--name", "nope", "--page-size", "1", expect=2)
        self.assertIn("nope", bad.stderr)

    def test_ack_skipping_unread_and_conflicts(self):
        self.seed()
        self.run_cli("subscription-create", "--collection", self.coll,
                     "--name", "s", "--start-gen", "1")
        page = self.run_cli("subscription-read", "--collection", self.coll,
                            "--name", "s", "--page-size", "1")
        # Ack a later position without reading it.
        bad = self.run_cli(
            "subscription-ack", "--collection", self.coll, "--name", "s",
            "--gen", "4", "--seq", "1", "--expected-gen", "1",
            "--expected-seq", "0", expect=2,
        )
        self.assertTrue(bad.stderr)
        # Stale expected old position.
        bad = self.run_cli(
            "subscription-ack", "--collection", self.coll, "--name", "s",
            "--gen", str(page["gen"]), "--seq", str(page["seq"]),
            "--expected-gen", "9", "--expected-seq", "9", expect=2,
        )
        self.assertIn("conflict", bad.stderr.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
