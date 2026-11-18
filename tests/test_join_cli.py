"""End-to-end CLI coverage for the two-source equi-join.

Drives join-prepare / join-page (one page per process) / join-status /
join-close / join-cleanup through real subprocesses, checks the __hex__
bytes convention, cross-process cursor continuation, and nonzero exits
with locatable messages for credential, state and source failures.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "bytekey_cli.py")

LSCHEMA = {
    "columns": [
        {"name": "region", "kind": "text", "asc": True, "nulls_first": True},
        {"name": "sku", "kind": "text", "asc": False, "nulls_first": True},
        {"name": "qty", "kind": "int64", "asc": True, "nulls_first": True},
        {"name": "blob", "kind": "bytes", "asc": True, "nulls_first": True},
    ],
}
RSCHEMA = {
    "columns": [
        {"name": "product", "kind": "text", "asc": False, "nulls_first": True},
        {"name": "area", "kind": "text", "asc": True, "nulls_first": True},
        {"name": "label", "kind": "text", "asc": True, "nulls_first": True},
        {"name": "tag", "kind": "bytes", "asc": True, "nulls_first": True},
    ],
}


class JoinCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.lpath = os.path.join(t, "left.db")
        self.rpath = os.path.join(t, "right.db")
        self.work = os.path.join(t, "join.db")
        for name, schema in (("lschema.json", LSCHEMA), ("rschema.json", RSCHEMA)):
            with open(os.path.join(t, name), "w", encoding="utf-8") as fh:
                json.dump(schema, fh)
        with open(os.path.join(t, "bid.json"), "w", encoding="utf-8") as fh:
            json.dump({"__hex__": "6231"}, fh)
        with open(os.path.join(t, "lops.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"op": "put", "row_id": {"__hex__": "6c31"},
                                 "region": "a", "sku": "x", "qty": 2,
                                 "blob": {"__hex__": ""}}) + "\n")
            fh.write(json.dumps({"op": "put", "row_id": {"__hex__": "6c32"},
                                 "region": "a", "sku": "x", "qty": 3,
                                 "blob": {"__hex__": "00ff"}}) + "\n")
            fh.write(json.dumps({"op": "put", "row_id": {"__hex__": "6c33"},
                                 "region": "b", "sku": "y", "qty": 1,
                                 "blob": {"__hex__": "01"}}) + "\n")
        with open(os.path.join(t, "rops.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"op": "put", "row_id": {"__hex__": "7231"},
                                 "product": "x", "area": "a", "label": "old",
                                 "tag": {"__hex__": ""}}) + "\n")
            fh.write(json.dumps({"op": "put", "row_id": {"__hex__": "7232"},
                                 "product": "x", "area": "a", "label": "new",
                                 "tag": {"__hex__": "00ff"}}) + "\n")
            fh.write(json.dumps({"op": "put", "row_id": {"__hex__": "7233"},
                                 "product": "q", "area": "c", "label": "only",
                                 "tag": {"__hex__": "dead"}}) + "\n")
        with open(os.path.join(t, "allq.json"), "w", encoding="utf-8") as fh:
            json.dump({"shape": "full_tuple", "lower": None, "upper": None}, fh)
        with open(os.path.join(t, "mapping.json"), "w", encoding="utf-8") as fh:
            json.dump([["region", "area"], ["sku", "product"]], fh)

        self.run_cli("collection-create", "--schema",
                     os.path.join(t, "lschema.json"), "--collection", self.lpath)
        self.run_cli("collection-create", "--schema",
                     os.path.join(t, "rschema.json"), "--collection", self.rpath)
        self.run_cli("apply", "--collection", self.lpath, "--batch-id",
                     os.path.join(t, "bid.json"), "--expected-head", "0",
                     "--source", os.path.join(t, "lops.jsonl"))
        self.run_cli("apply", "--collection", self.rpath, "--batch-id",
                     os.path.join(t, "bid.json"), "--expected-head", "0",
                     "--source", os.path.join(t, "rops.jsonl"))
        out = self.run_cli("snapshot-begin", "--collection", self.lpath)
        self.ltok = json.loads(out.stdout)["snapshot_token"]
        out = self.run_cli("snapshot-begin", "--collection", self.rpath)
        self.rtok = json.loads(out.stdout)["snapshot_token"]

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

    def _prepare_args(self, work, mode="full", page_size="1", extra=()):
        t = self.tmp.name
        return [
            "join-prepare", "--work", work,
            "--left-collection", self.lpath, "--left-token", self.ltok,
            "--left-query", os.path.join(t, "allq.json"),
            "--right-collection", self.rpath, "--right-token", self.rtok,
            "--right-query", os.path.join(t, "allq.json"),
            "--mapping", os.path.join(t, "mapping.json"),
            "--mode", mode, "--page-size", page_size, *extra,
        ]

    def _prepare(self, work=None, mode="full", page_size="1", extra=(),
                 expect=0):
        work = work or self.work
        return self.run_cli(*self._prepare_args(work, mode, page_size, extra),
                            expect=expect)

    def _page(self, handle, cursor=None, expect=0):
        args = ["join-page", "--work", self.work, "--handle", handle]
        if cursor:
            args += ["--cursor", cursor]
        return self.run_cli(*args, expect=expect)

    def test_full_cli_lifecycle(self):
        out = self._prepare()
        prep = json.loads(out.stdout)
        self.assertEqual(prep["status"], "ready")
        handle = prep["handle_token"]
        # Prepare evidence: per-side candidates and visible rows, disk work.
        self.assertEqual(prep["stats"]["left"]["rows_written"], 3)
        self.assertEqual(prep["stats"]["right"]["rows_written"], 3)
        self.assertEqual(prep["stats"]["left"]["candidates_examined"], 3)
        self.assertGreater(prep["stats"]["work_file_bytes"], 0)
        self.assertTrue(prep["stats"]["groups"]["groups_disk_resident"])

        # One page per process; cursor saved between invocations.
        kinds = []
        blobs = []
        cursor = None
        for _ in range(20):
            out = self._page(handle, cursor)
            page = json.loads(out.stdout)
            self.assertEqual(page["status"], "page")
            for row in page["rows"]:
                kinds.append(row["kind"])
                if row["left"] is not None:
                    blobs.append(row["left"]["values"][3])
                # Absent side is an explicit null, never a null-filled row.
                if row["kind"] == "left_only":
                    self.assertIsNone(row["right"])
                if row["kind"] == "right_only":
                    self.assertIsNone(row["left"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        else:
            self.fail("join stream did not terminate")
        # full mode: l1,l2 match r1,r2 (4 pairs), l3 left_only, r3 right_only.
        self.assertEqual(
            kinds,
            ["matched", "matched", "matched", "matched",
             "left_only", "right_only"],
        )
        # bytes columns round-trip as __hex__ single-field objects.
        self.assertIn({"__hex__": ""}, blobs)
        self.assertIn({"__hex__": "00ff"}, blobs)

        out = self.run_cli("join-status", "--work", self.work)
        status = json.loads(out.stdout)
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["mode"], "full")
        self.assertEqual(status["left"]["gen"], 1)

        # Cleanup of an active join is refused (nonzero exit).
        self.run_cli("join-cleanup", "--work", self.work, expect=2)
        # Close the join read; handle and cursors expire.
        self.run_cli("join-close", "--work", self.work, "--handle", handle)
        out = self.run_cli("join-status", "--work", self.work)
        self.assertEqual(json.loads(out.stdout)["state"], "closed")
        self._page(handle, expect=2)
        # Now cleanup works and the path is reusable.
        self.run_cli("join-cleanup", "--work", self.work)
        self.assertFalse(os.path.exists(self.work))
        out = self._prepare()
        handle2 = json.loads(out.stdout)["handle_token"]
        self.run_cli("join-close", "--work", self.work,
                     "--handle", handle2, "--reclaim")
        self.assertFalse(os.path.exists(self.work))

    def test_close_with_reclaim_via_cli(self):
        out = self._prepare()
        handle = json.loads(out.stdout)["handle_token"]
        self.run_cli("join-close", "--work", self.work, "--handle", handle,
                     "--reclaim")
        self.assertFalse(os.path.exists(self.work))

    def test_cli_failures_are_nonzero_and_locatable(self):
        out = self._prepare()
        handle = json.loads(out.stdout)["handle_token"]
        # Prepare over the completed work file is refused.
        proc = self._prepare(expect=2)
        self.assertIn("refusing to overwrite", proc.stderr)
        # Garbage handle.
        proc = self._page("garbage", expect=2)
        self.assertIn("token", proc.stderr)
        # Garbage cursor.
        proc = self._page(handle, "junk.cursor", expect=2)
        self.assertIn("jcur", proc.stderr)
        # Released left token: prepare fails naming the side.
        self.run_cli("snapshot-release", "--collection", self.lpath,
                     "--token", self.ltok)
        work2 = os.path.join(self.tmp.name, "join2.db")
        proc = self._prepare(work=work2, expect=2)
        self.assertIn("left source", proc.stderr)
        self.assertIn("released", proc.stderr)
        # A query naming an unknown column exits nonzero with a clean message.
        badq = os.path.join(self.tmp.name, "badq.json")
        with open(badq, "w", encoding="utf-8") as fh:
            json.dump({"shape": "prefix_range",
                       "prefix": [["nosuchcol", "a"]],
                       "range_column": "sku"}, fh)
        proc = self.run_cli(
            "join-prepare", "--work", work2,
            "--left-collection", self.lpath, "--left-token", self.ltok,
            "--left-query", badq,
            "--right-collection", self.rpath, "--right-token", self.rtok,
            "--right-query", os.path.join(self.tmp.name, "allq.json"),
            "--mapping", os.path.join(self.tmp.name, "mapping.json"),
            "--mode", "inner", "--page-size", "1", expect=2)
        self.assertIn("error:", proc.stderr)

    def test_nulls_equal_flag_and_mapping_object_form(self):
        t = self.tmp.name
        with open(os.path.join(t, "mapping_obj.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"left": ["region", "sku"],
                       "right": ["area", "product"]}, fh)
        out = self.run_cli(
            "join-prepare", "--work", self.work,
            "--left-collection", self.lpath, "--left-token", self.ltok,
            "--left-query", os.path.join(t, "allq.json"),
            "--right-collection", self.rpath, "--right-token", self.rtok,
            "--right-query", os.path.join(t, "allq.json"),
            "--mapping", os.path.join(t, "mapping_obj.json"),
            "--mode", "inner", "--nulls-equal", "--page-size", "10",
        )
        prep = json.loads(out.stdout)
        self.assertEqual(prep["status"], "ready")
        handle = prep["handle_token"]
        out = self._page(handle)
        page = json.loads(out.stdout)
        self.assertEqual(len(page["rows"]), 4)   # the two x-groups pair up
        self.assertIsNone(page["next_cursor"])

    def test_same_cursor_replayed_by_two_processes(self):
        out = self._prepare(page_size="2")
        handle = json.loads(out.stdout)["handle_token"]
        page1 = json.loads(self._page(handle).stdout)
        cursor = page1["next_cursor"]
        self.assertIsNotNone(cursor)
        # Two independent processes replaying the same cursor see the same
        # page content and the same next cursor.
        pa = json.loads(self._page(handle, cursor).stdout)
        pb = json.loads(self._page(handle, cursor).stdout)
        self.assertEqual(pa["rows"], pb["rows"])
        self.assertEqual(pa["next_cursor"], pb["next_cursor"])

    def test_work_file_write_failure_exits_nonzero(self):
        ro_dir = os.path.join(self.tmp.name, "ro")
        os.makedirs(ro_dir)
        os.chmod(ro_dir, 0o500)
        try:
            proc = self._prepare(work=os.path.join(ro_dir, "j.db"), expect=2)
            self.assertIn("work file", proc.stderr)
        finally:
            os.chmod(ro_dir, 0o700)

    def test_join_with_bytes_join_keys_via_cli(self):        # Join on the bytes columns (blob <-> tag): empty bytes match empty
        # bytes, 00ff matches 00ff; 01/dead and null-ish sides stay apart.
        t = self.tmp.name
        with open(os.path.join(t, "bmap.json"), "w", encoding="utf-8") as fh:
            json.dump([["blob", "tag"]], fh)
        out = self.run_cli(
            "join-prepare", "--work", self.work,
            "--left-collection", self.lpath, "--left-token", self.ltok,
            "--left-query", os.path.join(t, "allq.json"),
            "--right-collection", self.rpath, "--right-token", self.rtok,
            "--right-query", os.path.join(t, "allq.json"),
            "--mapping", os.path.join(t, "bmap.json"),
            "--mode", "full", "--page-size", "1",
        )
        handle = json.loads(out.stdout)["handle_token"]
        kinds = []
        cursor = None
        while True:
            out = self._page(handle, cursor)
            page = json.loads(out.stdout)
            kinds.extend(r["kind"] for r in page["rows"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        # l1<->r1 (empty), l2<->r2 (00ff): 2 matched; l3 left_only; r3
        # right_only.
        self.assertEqual(kinds, ["matched", "matched", "left_only",
                                 "right_only"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
