"""Change-stream reclamation, retention boundaries and crash safety.

Proves, with real row counts and a real ``mtime``/reopen cycle:

* a slow subscriber pins exactly the events and before-images it still needs;
  acknowledged generations no other subscriber needs are physically removed;
* deleting a subscriber invalidates its cursors immediately and only the
  *next* collection may remove what only it was holding;
* the earliest start generation advances over the reclaimed prefix and a
  subscribe/read attempt below it fails naming the boundary;
* a pre-commit kill leaves neither a generation nor a change record, while a
  post-commit lost reply is replayable with the same batch_id and no second
  change batch.
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bytekey import Column, Schema  # noqa: E402
from bytekey.changes import (  # noqa: E402
    ChangeRetentionError,
    SubscriptionConflictError,
    SubscriptionCursorError,
    SubscriptionNotFoundError,
)
from bytekey.collection import create_collection, open_collection  # noqa: E402

SCHEMA = Schema(columns=(
    Column("a", "text", asc=True, nulls_first=True),
    Column("b", "int64", asc=False, nulls_first=True),
))


def put(rid, a, b):
    return {"op": "put", "row_id": rid, "a": a, "b": b}


def build(path, upto=6):
    create_collection(path, SCHEMA)
    with open_collection(path) as coll:
        coll.apply(b"seed", 0, iter([put(b"r1", "a", 1), put(b"r2", "b", 2)]))
        for gen in range(1, upto):
            ops = [put(b"r1", "v" + str(gen), gen)]
            if gen % 2 == 0:
                ops.append({"op": "delete", "row_id": b"r2"})
            else:
                ops.append(put(b"r2", "w" + str(gen), gen))
            if gen == 3:
                ops = []  # an empty generation in the middle
            coll.apply(("b%d" % gen).encode(), gen, iter(ops))


def read_all(coll, name, page_size=2):
    out = []
    cursor = None
    while True:
        page = coll.read_changes_page(name, page_size, cursor_token=cursor)
        out.extend(page.events)
        if not page.has_more:
            return out
        cursor = page.next_cursor


def ack_page(coll, name, page):
    info = coll.subscription_info(name)
    return coll.acknowledge_changes(
        name, gen=page.gen, seq=page.seq,
        expected_gen=info.ack_gen, expected_seq=info.ack_seq,
    )


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        build(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _rows(self, table):
        conn = sqlite3.connect(self.path)
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            conn.close()

    def test_slow_subscriber_pins_only_what_it_needs(self):
        with open_collection(self.path) as coll:
            coll.begin_subscription("slow", start_gen=1)
            coll.begin_subscription("fast", start_gen=1)
        # slow reads and acknowledges only generation 1 (page size 1 lands
        # exactly on the generation boundary after its second event).
        with open_collection(self.path) as coll:
            page = coll.read_changes_page("slow", 2)
            self.assertEqual((page.gen, page.seq), (2, 0))
            ack_page(coll, "slow", page)      # acked at (2, 0)
            # fast consumes everything.
            read_all(coll, "fast", 100)
            fast_page = coll.read_changes_page("fast", 10)
            ack_page(coll, "fast", fast_page)
            head = coll.status().head
            before_events = self._rows("bytekey_changes")
            before_batches = self._rows("bytekey_change_batches")
            hist = coll.collect_history(head)
            after_events = self._rows("bytekey_changes")
            after_batches = self._rows("bytekey_change_batches")
            # Generation 1 was consumed by both, so its records were removed.
            # Generation 2 is still needed by slow: nothing else is.
            self.assertEqual(after_events, before_events - 2)
            self.assertEqual(after_batches, before_batches - 1)
            self.assertEqual(hist.change_events_removed, 2)
            self.assertEqual(hist.change_batches_removed, 1)
            self.assertEqual(coll.change_retention()["retain_from"], 2)
        # slow can still finish from its old position with complete values.
        with open_collection(self.path) as coll:
            events = read_all(coll, "slow", 2)
        self.assertEqual(events[0].gen, 2)
        self.assertEqual(events[0].before, ("a", 1))
        self.assertEqual(events[0].after, ("v1", 1))

    def test_release_then_collect_frees_the_rest_and_advances_boundary(self):
        with open_collection(self.path) as coll:
            coll.begin_subscription("only", start_gen=1)
            page = coll.read_changes_page("only", 2)   # exactly generation 1
            self.assertEqual((page.gen, page.seq), (2, 0))
            ack_page(coll, "only", page)
            head = coll.status().head
            hist1 = coll.collect_history(head)
            self.assertGreaterEqual(hist1.change_events_removed, 1)
            boundary = coll.change_retention()["retain_from"]
            self.assertGreater(boundary, 1)
            # still-retained events remain readable by the same subscription
            events = read_all(coll, "only", 3)
            self.assertTrue(events)
            self.assertEqual(events[0].gen, boundary)
            # Old cursors keep working until release.
            cursor = coll.read_changes_page("only", 1).next_cursor
            coll.read_changes_page("only", 1, cursor_token=cursor)
            coll.release_subscription("only")
            # Released: the cursor is immediately invalid.
            with self.assertRaises(Exception):
                coll.read_changes_page("only", 1, cursor_token=cursor)
        with open_collection(self.path) as coll:
            hist2 = coll.collect_history(coll.status().head)
            self.assertGreaterEqual(hist2.change_batches_removed, 1)
            self.assertEqual(self._rows("bytekey_changes"), 0)
            self.assertEqual(self._rows("bytekey_change_batches"), 0)
            self.assertEqual(coll.change_retention()["retain_from"],
                             coll.status().head + 1)
            # A new subscription below the boundary is refused, naming it.
            with self.assertRaises(ChangeRetentionError) as ctx:
                coll.begin_subscription("late", start_gen=1)
            self.assertIn("boundary", str(ctx.exception))

    def test_delete_subscription_does_not_release_other_snapshot_or_sub(self):
        with open_collection(self.path) as coll:
            coll.begin_subscription("keep", start_gen=1)
            coll.begin_subscription("drop", start_gen=1)
            token = coll.begin_snapshot(1)
            snapshot_rows_before = self._rows("bytekey_kv")
            keep_cursor = coll.read_changes_page("keep", 1).next_cursor
            drop_cursor = coll.read_changes_page("drop", 1).next_cursor
            coll.release_subscription("drop")
            # The other subscription's cursor and the snapshot are untouched.
            coll.read_changes_page("keep", 1, cursor_token=keep_cursor)
            from bytekey import plan_query
            page, _, _ = coll.scan_snapshot_page(
                token, {"shape": "prefix_range", "prefix": []}, 10
            )
            self.assertTrue(page.rows)
            self.assertEqual(self._rows("bytekey_kv"), snapshot_rows_before)
            with self.assertRaises(SubscriptionNotFoundError):
                coll.read_changes_page("drop", 1, cursor_token=drop_cursor)

    def test_main_version_reclamation_is_independent_of_change_retention(self):
        # A subscriber stalled on generation 1 still needs generation 1's
        # before-images, but those live in the change outbox: reclamation may
        # freely remove the corresponding dead main versions, and the
        # subscriber must still read complete before values afterwards.
        with open_collection(self.path) as coll:
            coll.begin_subscription("stalled", start_gen=1)
            kv_before = self._rows("bytekey_kv")
            head = coll.status().head
            hist = coll.collect_history(head)     # no snapshot pins anything
            kv_after = self._rows("bytekey_kv")
            self.assertLess(kv_after, kv_before)
            self.assertGreater(hist.entries_removed, 0)
            # Nothing was reclaimed from the change stream: the stalled
            # subscriber has acknowledged nothing.
            self.assertEqual(hist.change_events_removed, 0)
            events = read_all(coll, "stalled", 10)
        self.assertEqual(events[0].gen, 1)
        self.assertEqual(events[0].kind, "insert")
        self.assertEqual(events[0].after, ("a", 1))
        update = next(e for e in events if e.gen == 2 and e.kind == "update")
        self.assertEqual(update.before, ("a", 1))
        self.assertEqual(update.after, ("v1", 1))

    def test_acked_events_are_independently_reread_until_reclaimed(self):
        with open_collection(self.path) as coll:
            coll.begin_subscription("s", start_gen=1)
            page = coll.read_changes_page("s", 100)
            ack_page(coll, "s", page)
            # Re-reading from the acknowledged position yields nothing new
            # while the records are still retained; the subscription can
            # independently acknowledge and re-read later generations.
            again = coll.read_changes_page("s", 100)
            self.assertEqual(again.events, [])
            self.assertFalse(again.has_more)
            head = coll.status().head
            coll.apply(b"more", head, iter([put(b"r1", "later", head + 1)]))
            page = coll.read_changes_page("s", 100)
            self.assertEqual(len(page.events), 1)
            ack_page(coll, "s", page)
            # Acknowledged just past the new generation (whole-gen crossing).
            self.assertEqual(
                (coll.subscription_info("s").ack_gen,
                 coll.subscription_info("s").ack_seq),
                (head + 2, 0),
            )


class ConcurrencyTests(unittest.TestCase):
    """Two processes: one slow reader, one full drain, no interference."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        build(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_slow_reader_does_not_observe_fast_reader_progress(self):
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "changes_page_helper.py")
        with open_collection(self.path) as coll:
            coll.begin_subscription("slow", start_gen=1)
            coll.begin_subscription("fast", start_gen=1)
        # A fresh process reads one page of "fast".
        proc = subprocess.run(
            [sys.executable, "-I", helper, "--collection", self.path,
             "--name", "fast", "--page-size", "100"],
            capture_output=True, text=True, check=True,
        )
        fast_page = json.loads(proc.stdout)
        self.assertGreater(len(fast_page["events"]), 0)
        # Meanwhile, in-process, "slow" reads its first page only.
        with open_collection(self.path) as coll:
            slow_page = coll.read_changes_page("slow", 2)
            slow_first = [(e.gen, e.seq) for e in slow_page.events]
            # Acknowledge fast's whole page in a separate process.
            helper_ack = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "changes_ack_helper.py",
            )
            ack = subprocess.run(
                [sys.executable, "-I", helper_ack, "--collection", self.path,
                 "--name", "fast", "--gen", str(fast_page["gen"]),
                 "--seq", str(fast_page["seq"]),
                 "--expected-gen", "1", "--expected-seq", "0"],
                capture_output=True, text=True, check=True,
            )
            self.assertTrue(json.loads(ack.stdout)["ack_gen"] >= 1)
            # slow's page and cursor are unchanged by fast's progress.
            slow2 = coll.read_changes_page(
                "slow", 2, cursor_token=slow_page.next_cursor
            )
        self.assertEqual(slow_first, [(e.gen, e.seq)
                                      for e in slow_page.events])
        self.assertEqual(
            slow2.stats.start_gen, slow_page.gen
        )
        self.assertNotEqual(slow2.stats.start_gen, fast_page["gen"] + 1)


class CrashSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        build(self.path, upto=2)

    def tearDown(self):
        self.tmp.cleanup()

    def _helper(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "changes_apply_helper.py")

    def _rows(self, rows):
        p = os.path.join(self.tmp.name, "rows.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        return p

    def _counts(self):
        with open_collection(self.path) as coll:
            s = coll.status()
            return s.head, s.change_event_count, s.change_head_count

    def test_pre_commit_kill_leaves_no_generation_or_change(self):
        before = self._counts()
        rows = self._rows([["aa", "r-aa", 170], ["bb", "r-bb", 187]])
        proc = subprocess.run(
            [sys.executable, "-I", self._helper(), "--collection", self.path,
             "--expected-head", str(before[0]), "--batch-id", "cc",
             "--rows", rows, "--crash", "pre_commit"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(self._counts(), before)
        with open_collection(self.path) as coll:
            self.assertIsNone(coll.lookup_receipt(b"\xcc"))
            # The batch id is free and publishes exactly once with events.
            r = coll.apply(b"\xcc", before[0], iter([
                {"op": "put", "row_id": b"\xaa", "a": "r-aa", "b": 170},
                {"op": "put", "row_id": b"\xbb", "a": "r-bb", "b": 187},
            ]))
        after = self._counts()
        self.assertEqual(after[0], before[0] + 1)
        self.assertEqual(after[1], before[1] + 2)
        self.assertEqual(after[2], before[2] + 1)
        self.assertEqual(r.head, before[0] + 1)

    def test_post_commit_lost_reply_replays_without_second_change_batch(self):
        before = self._counts()
        rows = self._rows([["dd", "r-dd", 221]])
        proc = subprocess.run(
            [sys.executable, "-I", self._helper(), "--collection", self.path,
             "--expected-head", str(before[0]), "--batch-id", "ee",
             "--rows", rows, "--crash", "post_commit"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        after_crash = self._counts()
        self.assertEqual(after_crash[0], before[0] + 1)
        self.assertEqual(after_crash[2], before[2] + 1)
        # Identical retry returns the stored receipt: no second generation.
        proc2 = subprocess.run(
            [sys.executable, "-I", self._helper(), "--collection", self.path,
             "--expected-head", str(before[0]), "--batch-id", "ee",
             "--rows", rows],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(proc2.stdout)
        self.assertTrue(payload["replayed"])
        self.assertEqual(self._counts(), after_crash)
        with open_collection(self.path) as coll:
            sub = coll.begin_subscription("after", start_gen=before[0] + 1)
            events = read_all(coll, "after", 10)
        self.assertEqual(len(events), 1)  # exactly one event, not two
        self.assertEqual(events[0].kind, "insert")
        self.assertEqual(sub.start_gen, before[0] + 1)


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        build(self.path, upto=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_corrupt_event_image_is_rejected_locatably(self):
        conn = sqlite3.connect(self.path)
        conn.execute(
            "UPDATE bytekey_changes SET after_json = '{not json' "
            "WHERE gen = 1 AND seq = 1"
        )
        conn.commit()
        conn.close()
        with open_collection(self.path) as coll:
            coll.begin_subscription("s", start_gen=1)
            with self.assertRaises(Exception) as ctx:
                coll.read_changes_page("s", 10)
            message = str(ctx.exception)
            self.assertIn("(1, 1)", message)
            self.assertIn("image", message)

    def test_missing_batch_marker_is_reported_not_skipped(self):
        with open_collection(self.path) as coll:
            coll.begin_subscription("s", start_gen=1)
        conn = sqlite3.connect(self.path)
        conn.execute("DELETE FROM bytekey_change_batches WHERE gen = 2")
        conn.commit()
        conn.close()
        with open_collection(self.path) as coll:
            with self.assertRaises(Exception) as ctx:
                read_all(coll, "s", 1)
            self.assertIn("marker", str(ctx.exception))
            self.assertIn("2", str(ctx.exception))

    def test_bad_tail_does_not_append_change_rows(self):
        with open_collection(self.path) as coll:
            before = coll.status()
            with self.assertRaises(Exception):
                coll.apply(b"tail", 3, iter([
                    put(b"r9", "ok", 9),
                    put(b"r10", "bad", "not-an-int"),
                ]))
            after = coll.status()
            self.assertEqual(after.head, before.head)
            self.assertEqual(after.change_event_count,
                             before.change_event_count)
            self.assertEqual(after.change_head_count,
                             before.change_head_count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
