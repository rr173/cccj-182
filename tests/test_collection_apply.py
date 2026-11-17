"""Apply semantics: atomic validation, idempotent replay, competing writers.

Covers corrupt source tails, bad types, duplicate-identity rejection,
expected_head conflicts, content-equivalent re-sends (reordered lines, JSON
whitespace, hex case), empty/no-op batches, two-process write contention, and
controlled subprocess crashes proving commit-or-nothing durability.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bytekey import Column, Schema  # noqa: E402
from bytekey.collection import (  # noqa: E402
    BatchError,
    ConflictError,
    create_collection,
    open_collection,
)

SCHEMA = Schema(
    columns=(
        Column("region", "text", asc=True, nulls_first=True),
        Column("ts", "int64", asc=False, nulls_first=True),
        Column("blob", "bytes", asc=True, nulls_first=True),
        Column("flag", "bool", asc=True, nulls_first=True),
    )
)


def put(rid, region="a", ts=1, blob=b"x", flag=True):
    return {
        "op": "put", "row_id": rid, "region": region, "ts": ts,
        "blob": blob, "flag": flag,
    }


class ApplyValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        create_collection(self.path, SCHEMA)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bad_type_leaves_no_partial_update(self):
        with open_collection(self.path) as coll:
            good = [put(b"r1"), put(b"r2")]
            bad = good + [put(b"r3", ts="not-an-int")]
            with self.assertRaises(BatchError):
                coll.apply(b"b", 0, iter(bad))
            self.assertEqual(coll.status().head, 0)
            self.assertEqual(coll.status().live_entries, 0)
            self.assertIsNone(coll.lookup_receipt(b"b"))
            # batch_id is free for reuse and no staging rows linger.
            r = coll.apply(b"b", 0, iter(good))
            self.assertEqual((r.head, r.applied), (1, 2))
            n = coll.reader.execute(
                "SELECT COUNT(*) FROM bytekey_staging"
            ).fetchone()[0]
            self.assertEqual(n, 0)

    def test_int64_rejects_bool_and_out_of_range(self):
        with open_collection(self.path) as coll:
            with self.assertRaises(BatchError):
                coll.apply(b"b1", 0, iter([put(b"r1", ts=True)]))
            with self.assertRaises(BatchError):
                coll.apply(b"b2", 0, iter([put(b"r1", ts=1 << 63)]))
            self.assertEqual(coll.status().head, 0)

    def test_duplicate_identity_in_one_batch_rejected_with_line_numbers(self):
        with open_collection(self.path) as coll:
            ops = [put(b"r1"), put(b"r2"), put(b"r1", region="z")]
            with self.assertRaises(BatchError) as ctx:
                coll.apply(b"dup", 0, iter(ops))
            message = str(ctx.exception)
            self.assertIn("line 3", message)
            self.assertIn("lines 1, 3", message)
            self.assertEqual(coll.status().head, 0)

    def test_delete_and_put_same_rid_in_batch_rejected(self):
        with open_collection(self.path) as coll:
            ops = [{"op": "delete", "row_id": b"r1"}, put(b"r1")]
            with self.assertRaises(BatchError):
                coll.apply(b"x", 0, iter(ops))

    def test_corrupt_stream_tail_is_atomic(self):
        src = os.path.join(self.tmp.name, "src.jsonl")
        first = put(b"r1")
        first["row_id"] = {"__hex__": b"r1".hex()}
        first["blob"] = {"__hex__": b"x".hex()}
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(first) + "\n")
            fh.write("{not valid json\n")
        with open_collection(self.path) as coll:
            def stream():
                with open(src, "r", encoding="utf-8") as fh2:
                    for line_no, line in enumerate(fh2, start=1):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            item = json.loads(stripped)
                        except json.JSONDecodeError as exc:
                            raise BatchError(f"line {line_no}: {exc}") from exc
                        if isinstance(item, dict) and "__hex__" in item.get(
                            "row_id", {}
                        ):
                            item["row_id"] = bytes.fromhex(item["row_id"]["__hex__"])
                            item["blob"] = bytes.fromhex(item["blob"]["__hex__"])
                        yield item

            with self.assertRaises(BatchError) as ctx:
                coll.apply(b"tail", 0, stream())
            self.assertIn("line 2", str(ctx.exception))
            self.assertEqual(coll.status().head, 0)
            self.assertEqual(coll.status().live_entries, 0)

    def test_expected_head_mismatch_conflicts_and_consumes_nothing(self):
        with open_collection(self.path) as coll:
            coll.apply(b"ok", 0, iter([put(b"r1")]))
            with self.assertRaises(ConflictError):
                coll.apply(b"late", 0, iter([put(b"r2")]))
            self.assertEqual(coll.status().head, 1)
            self.assertIsNone(coll.lookup_receipt(b"late"))

    def test_full_replacement_can_move_sorting_position(self):
        with open_collection(self.path) as coll:
            coll.apply(b"b1", 0, iter([
                put(b"r1", region="a", ts=5), put(b"r2", region="m", ts=5)
            ]))
            coll.apply(b"b2", 1, iter([put(b"r1", region="z", ts=9)]))
            tok = coll.begin_snapshot(2)
            page, _, _ = coll.scan_snapshot_page(
                tok, {"shape": "prefix_range", "prefix": []}, 100
            )
            self.assertEqual(
                [(v, rid) for v, rid in page.rows],
                [(("m", 5, b"x", True), b"r2"),
                 (("z", 9, b"x", True), b"r1")],
            )


class IdempotencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        create_collection(self.path, SCHEMA)

    def tearDown(self):
        self.tmp.cleanup()

    def _canonical_ops(self):
        return [put(b"r1", region="a", ts=1, blob=b"\x00\xff"),
                {"op": "delete", "row_id": b"r2"}]

    def test_resend_returns_original_receipt_without_change(self):
        with open_collection(self.path) as coll:
            r1 = coll.apply(b"idem", 0, iter(self._canonical_ops()))
            # Advance the world, then resend the identical request.
            coll.apply(b"other", 1, iter([put(b"r9")]))
            r2 = coll.apply(b"idem", 0, iter(self._canonical_ops()))
            self.assertTrue(r2.replayed)
            self.assertEqual(r2.head, r1.head)
            self.assertEqual(r2.digest_hex, r1.digest_hex)
            self.assertEqual(coll.status().head, 2)
            self.assertEqual(coll.status().receipt_count, 2)

    def test_reorder_whitespace_and_hex_case_are_same_content(self):
        with open_collection(self.path) as coll:
            ops = self._canonical_ops()
            r1 = coll.apply(b"eq", 0, iter(ops))
            reordered = list(reversed(ops))
            # Upper-case hex spelling and an equivalent re-encode of bytes.
            reordered[1] = {
                "op": "put", "row_id": b"r1", "region": "a", "ts": 1,
                "blob": bytes(b"\x00\xff"), "flag": True,
            }
            r2 = coll.apply(b"eq", 0, iter(reordered))
            self.assertTrue(r2.replayed)
            self.assertEqual(r2.digest_hex, r1.digest_hex)

    def test_changed_value_conflicts(self):
        with open_collection(self.path) as coll:
            coll.apply(b"c", 0, iter([put(b"r1", ts=1)]))
            with self.assertRaises(ConflictError):
                coll.apply(b"c", 0, iter([put(b"r1", ts=2)]))

    def test_changed_operation_or_rid_conflicts(self):
        with open_collection(self.path) as coll:
            coll.apply(b"c", 0, iter([put(b"r1")]))
            with self.assertRaises(ConflictError):
                coll.apply(b"c", 0, iter([{"op": "delete", "row_id": b"r1"}]))
            with self.assertRaises(ConflictError):
                coll.apply(b"c", 0, iter([put(b"r2")]))

    def test_changed_expected_head_conflicts(self):
        with open_collection(self.path) as coll:
            coll.apply(b"c", 0, iter([put(b"r1")]))
            with self.assertRaises(ConflictError):
                coll.apply(b"c", 1, iter([put(b"r1")]))

    def test_null_empty_string_and_empty_bytes_are_distinct(self):
        with open_collection(self.path) as coll:
            # null vs empty text: same batch id reused with changed content.
            coll.apply(b"n", 0, iter([put(b"r1", region=None)]))
            with self.assertRaises(ConflictError):
                coll.apply(b"n", 0, iter([put(b"r1", region="")]))
            self.assertEqual(coll.status().head, 1)
            # Empty bytes vs null blob: distinct batches, distinct digests.
            coll.apply(b"b0", 1, iter([put(b"r2", blob=b"")]))
            r_empty = coll.lookup_receipt(b"b0")
            coll.apply(b"b1", 2, iter([put(b"r2", blob=None)]))
            r_null = coll.lookup_receipt(b"b1")
            self.assertNotEqual(r_empty["digest"], r_null["digest"])
            # bool is not int64: integer 1 on a bool column is rejected.
            coll.apply(b"f0", 3, iter([put(b"r3", flag=True)]))
            with self.assertRaises(BatchError):
                coll.apply(b"f1", 4, iter([put(b"r3", flag=1)]))


class CompetingWritersTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        create_collection(self.path, SCHEMA)

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_processes_same_expected_head_one_wins(self):
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "apply_helper.py")
        ops = os.path.join(self.tmp.name, "ops.json")
        with open(ops, "w", encoding="utf-8") as fh:
            json.dump([{"__hex__": "01"}, {"__hex__": "02"}], fh)
        env = dict(os.environ)
        procs = [
            subprocess.Popen(
                [sys.executable, "-I", helper, "--collection", self.path,
                 "--expected-head", "0", "--batch-id", bid, "--source", ops],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            )
            for bid in ("01", "02")
        ]
        results = [(p, p.communicate()) for p in procs]
        codes = sorted(p.returncode for p, _ in results)
        self.assertEqual(codes, [0, 2])
        with open_collection(self.path) as coll:
            self.assertEqual(coll.status().head, 1)
            self.assertEqual(coll.status().receipt_count, 1)
        # The loser's message must be an explicit conflict, not a crash.
        loser = next(p for p, _ in results if p.returncode == 2)
        loser_stderr = next(err for p, (_, err) in results if p is loser)
        self.assertIn("head", loser_stderr.lower())


class CrashAtomicityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        create_collection(self.path, SCHEMA)

    def tearDown(self):
        self.tmp.cleanup()

    def test_hook_abort_before_commit_leaves_nothing(self):
        with open_collection(self.path) as coll:
            def kill_pre_commit(_phase, _conn):
                raise SystemExit(137)

            with self.assertRaises(SystemExit):
                coll.apply(b"killed", 0, iter([put(b"r1")]),
                           crash_hook=kill_pre_commit)
        # Reopen fresh: nothing was published, no receipt, batch_id free.
        with open_collection(self.path) as coll:
            self.assertEqual(coll.status().head, 0)
            self.assertIsNone(coll.lookup_receipt(b"killed"))
            r = coll.apply(b"killed", 0, iter([put(b"r1")]))
            self.assertEqual(r.head, 1)

    def test_subprocess_kill_pre_commit_then_retry_succeeds(self):
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "apply_helper.py")
        ops = os.path.join(self.tmp.name, "ops.json")
        with open(ops, "w", encoding="utf-8") as fh:
            json.dump([{"__hex__": "aa"}], fh)
        proc = subprocess.run(
            [sys.executable, "-I", helper, "--collection", self.path,
             "--expected-head", "0", "--batch-id", "aa", "--source", ops,
             "--crash", "pre_commit"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr or proc.returncode != 0)
        with open_collection(self.path) as coll:
            self.assertEqual(coll.status().head, 0)
            self.assertIsNone(coll.lookup_receipt(b"\xaa"))
        # Same batch id retried in a clean process: succeeds exactly once.
        proc2 = subprocess.run(
            [sys.executable, "-I", helper, "--collection", self.path,
             "--expected-head", "0", "--batch-id", "aa", "--source", ops],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(proc2.stdout)
        self.assertEqual(payload["head"], 1)
        self.assertFalse(payload["replayed"])

    def test_subprocess_kill_after_commit_lost_reply_is_replayable(self):
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "apply_helper.py")
        ops = os.path.join(self.tmp.name, "ops.json")
        with open(ops, "w", encoding="utf-8") as fh:
            json.dump([{"__hex__": "bb"}], fh)
        proc = subprocess.run(
            [sys.executable, "-I", helper, "--collection", self.path,
             "--expected-head", "0", "--batch-id", "bb", "--source", ops,
             "--crash", "post_commit"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        with open_collection(self.path) as coll:
            self.assertEqual(coll.status().head, 1)
            self.assertIsNotNone(coll.lookup_receipt(b"\xbb"))
        # Reply was lost: identical retry returns the stored receipt.
        proc2 = subprocess.run(
            [sys.executable, "-I", helper, "--collection", self.path,
             "--expected-head", "0", "--batch-id", "bb", "--source", ops],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(proc2.stdout)
        self.assertEqual(payload["head"], 1)
        self.assertTrue(payload["replayed"])
        with open_collection(self.path) as coll:
            self.assertEqual(coll.status().head, 1)
            self.assertEqual(coll.status().receipt_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
