"""Hand-computed acceptance test for the persistent change stream.

Fixture (all before/after values written out by hand, not derived from the
implementation):

    seed   r1=(a,10,old), r2=(a,10,keep)
    gen 1  r1 -> (c,20,new); delete r2
    gen 2  insert r3=(a,7,late)
    gen 3  empty batch
    gen 4  delete missing (a no-op generation)
    gen 5  re-create r2=(b,8,reborn)

Every page is checked event by event with complete before/after images, the
batch boundaries (including empty ones) are checked, and a page size of one
is used across a *process* boundary.  Re-reading an unacknowledged page and
re-sending an acknowledgement both have to be exact.
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
from bytekey.changes import (  # noqa: E402
    ChangeLogError,
    ChangeRetentionError,
    SubscriptionConflictError,
    SubscriptionCursorError,
    SubscriptionNameConflict,
    SubscriptionNotFoundError,
)
from bytekey.collection import create_collection, open_collection  # noqa: E402

SCHEMA = Schema(columns=(
    Column("a", "text", asc=True, nulls_first=True),
    Column("b", "int64", asc=False, nulls_first=True),
    Column("c", "text", asc=True, nulls_first=False),
))

ABSENT = {"present": False}


def present(*values):
    return {"present": True, "values": list(values)}


def put(rid, a, b, c):
    return {"op": "put", "row_id": rid, "a": a, "b": b, "c": c}


# The expected feed, written by hand.
EXPECTED_EVENTS = [
    # (gen, seq, rid, op, before, after)
    (1, 1, b"r1", "insert", ABSENT, present("a", 10, "old")),
    (1, 2, b"r2", "insert", ABSENT, present("a", 10, "keep")),
    (2, 1, b"r1", "update", present("a", 10, "old"), present("c", 20, "new")),
    (2, 2, b"r2", "delete", present("a", 10, "keep"), ABSENT),
    (3, 1, b"r3", "insert", ABSENT, present("a", 7, "late")),
    (6, 1, b"r2", "insert", ABSENT, present("b", 8, "reborn")),
]
# One batch marker per generation 1..6, with the event count of each.
EXPECTED_MARKERS = {1: 2, 2: 2, 3: 1, 4: 0, 5: 0, 6: 1}


def build_fixture(path):
    create_collection(path, SCHEMA)
    with open_collection(path) as coll:
        coll.apply(b"seed", 0, iter([put(b"r1", "a", 10, "old"),
                                     put(b"r2", "a", 10, "keep")]))
        coll.apply(b"g1", 1, iter([put(b"r1", "c", 20, "new"),
                                   {"op": "delete", "row_id": b"r2"}]))
        coll.apply(b"g2", 2, iter([put(b"r3", "a", 7, "late")]))
        coll.apply(b"g3", 3, iter([]))
        coll.apply(b"g4", 4, iter([{"op": "delete", "row_id": b"missing"}]))
        coll.apply(b"g5", 5, iter([put(b"r2", "b", 8, "reborn")]))


def row_id_name(schema):
    return schema.row_id_name


class HandComputedFeedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        build_fixture(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _event_tuple(self, ev, row_id_name="row_id"):
        return (
            ev["gen"], ev["seq"], bytes.fromhex(ev[row_id_name]["__hex__"]),
            ev["op"], ev["before"], ev["after"],
        )

    def test_every_generation_event_and_batch_boundary(self):
        with open_collection(self.path) as coll:
            coll.begin_subscription("down", start_gen=1)
            pages = []
            cursor = None
            while True:
                page = coll.read_changes_page("down", 1, cursor_token=cursor)
                pages.append(page)
                if not page.has_more:
                    break
                cursor = page.next_cursor
            # Flatten exactly as a downstream consumer would.
            got_events = [self._event_tuple(e.to_dict())
                          for p in pages for e in p.events]
            got_markers = {}
            for p in pages:
                for b in p.batches:
                    got_markers.setdefault(b.gen, b.event_count)
            self.assertEqual(got_events, EXPECTED_EVENTS)
            self.assertEqual(got_markers, EXPECTED_MARKERS)
            # Every page held at most one event: bounded reading.
            for p in pages:
                self.assertLessEqual(len(p.events), 1)
            # The empty generation 4 and the no-op generation 5 still appear
            # as their own pages with no events.
            empty_pages = [p for p in pages if not p.events]
            self.assertEqual(
                [p.batches[0].gen for p in empty_pages if p.batches], [4, 5]
            )

    def test_absent_is_distinct_from_all_null_business_row(self):
        with open_collection(self.path) as coll:
            head = coll.status().head
            coll.apply(b"nulls", head, iter([
                put(b"rnull", None, None, None),
            ]))
            info = coll.begin_subscription("nulls", start_gen=head + 1)
            page = coll.read_changes_page("nulls", 10)
            self.assertEqual(len(page.events), 1)
            ev = page.events[0]
            self.assertIsNone(ev.before)             # insert: absent before
            self.assertEqual(ev.after, (None, None, None))  # present nulls
            d = ev.to_dict(info.name)
            self.assertEqual(d["before"], {"present": False})
            self.assertEqual(
                d["after"], {"present": True, "values": [None, None, None]}
            )

    def test_page_across_process_boundary_and_replay_consistency(self):
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "changes_read_helper.py")
        with open_collection(self.path) as coll:
            coll.begin_subscription("proc", start_gen=1)
        cursor = None
        collected = []
        pages = 0
        while True:
            cmd = [sys.executable, "-I", helper, "--collection", self.path,
                   "--name", "proc", "--page-size", "1"]
            if cursor is not None:
                cmd += ["--cursor", cursor]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  check=True)
            payload = json.loads(proc.stdout)
            pages += 1
            # Re-run the *same* invocation: identical events and successor.
            proc2 = subprocess.run(cmd, capture_output=True, text=True,
                                   check=True)
            payload2 = json.loads(proc2.stdout)
            self.assertEqual(payload["events"], payload2["events"])
            self.assertEqual(payload["batches"], payload2["batches"])
            self.assertEqual(payload["next_cursor"], payload2["next_cursor"])
            collected.extend(self._event_tuple(e) for e in payload["events"])
            if not payload["has_more"]:
                break
            cursor = payload["next_cursor"]
        self.assertEqual(collected, EXPECTED_EVENTS)
        self.assertGreater(pages, 1)

    def test_resending_same_batch_does_not_duplicate_change_records(self):
        with open_collection(self.path) as coll:
            before = coll.status()
            receipt = coll.apply(
                b"g5", 5, iter([put(b"r2", "b", 8, "reborn")])
            )
            self.assertTrue(receipt.replayed)
            after = coll.status()
            self.assertEqual(after.head, before.head)
            self.assertEqual(after.change_event_count,
                             before.change_event_count)
            self.assertEqual(after.change_head_count,
                             before.change_head_count)
            self.assertEqual(receipt.head, 6)

    def test_failed_validation_leaves_no_change_record(self):
        with open_collection(self.path) as coll:
            before = coll.status()
            with self.assertRaises(Exception):
                coll.apply(b"bad", 6, iter([put(b"r9", "x", "not-int", "y")]))
            after = coll.status()
            self.assertEqual(after.head, 6)
            self.assertEqual(after.change_event_count,
                             before.change_event_count)
            self.assertEqual(after.change_head_count,
                             before.change_head_count)
            self.assertIsNone(coll.lookup_receipt(b"bad"))


class AcknowledgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        build_fixture(self.path)
        with open_collection(self.path) as coll:
            self.sub = coll.begin_subscription("a", start_gen=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ack_retry_is_idempotent_and_conflicts_are_rejected(self):
        with open_collection(self.path) as coll:
            page = coll.read_changes_page("a", 2)
            res = coll.acknowledge_changes(
                "a", gen=page.gen, seq=page.seq,
                expected_gen=page.ack_gen, expected_seq=page.ack_seq,
            )
            self.assertFalse(res["replayed"])
            res2 = coll.acknowledge_changes(
                "a", gen=page.gen, seq=page.seq,
                expected_gen=page.ack_gen, expected_seq=page.ack_seq,
            )
            self.assertTrue(res2["replayed"])
            # Stale expected position (concurrent advance gone by).
            with self.assertRaises(SubscriptionConflictError):
                coll.acknowledge_changes(
                    "a", gen=page.gen + 1, seq=0,
                    expected_gen=page.ack_gen, expected_seq=page.ack_seq,
                )
            # Backwards.
            with self.assertRaises(SubscriptionConflictError):
                coll.acknowledge_changes(
                    "a", gen=1, seq=0,
                    expected_gen=page.gen, expected_seq=page.seq,
                )

    def test_ack_beyond_read_frontier_is_rejected(self):
        with open_collection(self.path) as coll:
            # Never read anything.
            with self.assertRaises(Exception):
                coll.acknowledge_changes(
                    "a", gen=6, seq=1, expected_gen=1, expected_seq=0,
                )
            page = coll.read_changes_page("a", 1)
            with self.assertRaises(Exception):
                coll.acknowledge_changes(
                    "a", gen=6, seq=1,
                    expected_gen=page.ack_gen, expected_seq=page.ack_seq,
                )

    def test_cursor_from_other_subscription_rejected(self):
        with open_collection(self.path) as coll:
            coll.begin_subscription("b", start_gen=1)
            page = coll.read_changes_page("b", 1)
            with self.assertRaises(Exception):
                coll.read_changes_page("a", 1, cursor_token=page.next_cursor)
            # Acknowledging through another subscription's credential is also
            # rejected (its instance id does not match "a").
            with self.assertRaises(SubscriptionConflictError):
                coll.acknowledge_changes(
                    "a", expected_gen=1, expected_seq=0,
                    cursor_token=page.next_cursor,
                )

    def test_acknowledge_through_own_cursor(self):
        with open_collection(self.path) as coll:
            page = coll.read_changes_page("a", 2)
            res = coll.acknowledge_changes(
                "a", expected_gen=1, expected_seq=0,
                cursor_token=page.next_cursor,
            )
            self.assertEqual((res["ack_gen"], res["ack_seq"]),
                             (page.gen, page.seq))
            # Lost-response replay through the same credential is idempotent.
            again = coll.acknowledge_changes(
                "a", expected_gen=1, expected_seq=0,
                cursor_token=page.next_cursor,
            )
            self.assertTrue(again["replayed"])

    def test_same_name_recreated_cannot_use_old_cursor(self):
        with open_collection(self.path) as coll:
            page = coll.read_changes_page("a", 1)
            cursor = page.next_cursor
            coll.release_subscription("a")
            coll.begin_subscription("a", start_gen=1)
            with self.assertRaises(SubscriptionCursorError):
                coll.read_changes_page("a", 1, cursor_token=cursor)
            # Old subscription is gone; its name is a fresh instance.
            self.assertNotEqual(coll.subscription_info("a").instance_id,
                                self.sub.instance_id)

    def test_name_conflict_and_release_visibility(self):
        with open_collection(self.path) as coll:
            with self.assertRaises(SubscriptionNameConflict):
                coll.begin_subscription("a", start_gen=2)
            coll.release_subscription("a")
            with self.assertRaises(SubscriptionNotFoundError):
                coll.subscription_info("a")
            # Release is idempotent enough to report "already gone" clearly.
            with self.assertRaises(SubscriptionNotFoundError):
                coll.release_subscription("a")
            coll.begin_subscription("a", start_gen=2)


class StartPositionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        build_fixture(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_from_head_delivers_the_published_generation(self):
        with open_collection(self.path) as coll:
            gen6 = [put(b"r2", "b", 8, "reborn")]
            sub = coll.begin_subscription("head", from_head=True)
            self.assertEqual((sub.start_gen, sub.ack_gen), (6, 6))
            page = coll.read_changes_page("head", 10)
            self.assertEqual(
                [(e.gen, e.seq, e.kind) for e in page.events], [(6, 1, "insert")]
            )

    def test_default_starts_at_the_next_publish(self):
        with open_collection(self.path) as coll:
            sub = coll.begin_subscription("future")
            self.assertEqual(sub.start_gen, 7)
            page = coll.read_changes_page("future", 10)
            self.assertEqual(page.events, [])
            self.assertFalse(page.has_more)
            coll.apply(b"g6", 6, iter([put(b"r4", "q", 1, "new")]))
            page = coll.read_changes_page("future", 10)
            self.assertEqual([(e.gen, e.row_id) for e in page.events],
                             [(7, b"r4")])

    def test_start_beyond_head_is_rejected(self):
        with open_collection(self.path) as coll:
            with self.assertRaises(ChangeRetentionError):
                coll.begin_subscription("too-far", start_gen=99)


class TwoSubscriberTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.db")
        build_fixture(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_independent_interleaved_readers(self):
        with open_collection(self.path) as coll:
            coll.begin_subscription("s1", start_gen=1)
            coll.begin_subscription("s2", start_gen=3)
            out1, out2 = [], []
            cur1 = cur2 = None
            ack1 = (1, 0)
            done1 = done2 = False
            while not (done1 and done2):
                if not done1:
                    p1 = coll.read_changes_page("s1", 1, cursor_token=cur1)
                    out1.extend(self._ev(e) for e in p1.events)
                    coll.acknowledge_changes(
                        "s1", gen=p1.gen, seq=p1.seq,
                        expected_gen=ack1[0], expected_seq=ack1[1],
                    )
                    ack1 = (p1.gen, p1.seq)
                    cur1 = p1.next_cursor
                    done1 = not p1.has_more
                if not done2:
                    p2 = coll.read_changes_page("s2", 2, cursor_token=cur2)
                    out2.extend(self._ev(e) for e in p2.events)
                    r = coll.acknowledge_changes(
                        "s2", gen=p2.gen, seq=p2.seq,
                        expected_gen=coll.subscription_info("s2").ack_gen,
                        expected_seq=coll.subscription_info("s2").ack_seq,
                    )
                    self.assertGreaterEqual(r["ack_gen"], 3)
                    cur2 = p2.next_cursor
                    done2 = not p2.has_more
            # s1 reads the whole feed; s2 starts at generation 3 and sees only
            # the later events.  Interleaving them changes neither sequence.
            self.assertEqual(
                [self._render(e) for e in out1],
                [self._render_expected(e) for e in EXPECTED_EVENTS],
            )
            self.assertEqual(
                [self._render(e) for e in out2],
                [self._render_expected(e) for e in EXPECTED_EVENTS
                 if e[0] >= 3],
            )

    @staticmethod
    def _ev(ev):
        return (ev.gen, ev.seq, ev.row_id, ev.kind, ev.before, ev.after)

    @staticmethod
    def _render(ev):
        return (ev[0], ev[1], ev[2], ev[3], _present(ev[4]), _present(ev[5]))

    @staticmethod
    def _render_expected(e):
        return (e[0], e[1], e[2], e[3],
                e[4].get("values"), e[5].get("values"))


def _present(values):
    return None if values is None else list(values)


class UpgradeTests(unittest.TestCase):
    def test_legacy_collection_upgrades_with_head_boundary(self):
        import sqlite3

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "legacy.db")
        build_fixture(path)
        with open_collection(path) as coll:
            cid = coll.collection_id
            head = coll.status().head
            token = coll.begin_snapshot(head)
            receipt = coll.lookup_receipt(b"g1")
        # Genuinely downgrade: drop the change tables and the newer marker.
        conn = sqlite3.connect(path)
        conn.execute("DROP TABLE bytekey_changes")
        conn.execute("DROP TABLE bytekey_change_batches")
        conn.execute("DROP TABLE bytekey_subscriptions")
        conn.execute("DELETE FROM bytekey_meta WHERE name = 'change_retain_from'")
        conn.execute("UPDATE bytekey_meta SET value = 'bytekey-collection/v2' "
                     "WHERE name = 'format'")
        conn.commit()
        conn.close()

        with open_collection(path) as coll:
            self.assertEqual(coll.collection_id, cid)
            self.assertEqual(coll.status().head, head)
            self.assertEqual(coll.lookup_receipt(b"g1")["digest"],
                             receipt["digest"])
            # Pre-upgrade snapshot token still reads its pinned generation.
            from bytekey import plan_query
            q = {"shape": "prefix_range", "prefix": []}
            page, _, _ = coll.scan_snapshot_page(token, q, 100)
            self.assertGreater(len(page.rows), 0)
            # Subscriptions begin at the upgrade head: no fabricated events.
            with self.assertRaises(ChangeRetentionError):
                coll.begin_subscription("old", start_gen=1)
            info = coll.begin_subscription("up", start_gen=head)
            with open_collection(path) as coll2:
                page = coll2.read_changes_page("up", 10)
            # The upgrade head crosses with zero events, then new publishes
            # are delivered normally.
            self.assertEqual(page.events, [])
            with open_collection(path) as coll3:
                coll3.apply(b"post-upgrade", head, iter([
                    put(b"r1", "z", 1, "post"),
                ]))
                page = coll3.read_changes_page("up", 10)
                self.assertEqual(len(page.events), 1)
                self.assertEqual(page.events[0].kind, "update")
                self.assertEqual(page.events[0].before, ("c", 20, "new"))
                self.assertEqual(page.events[0].after, ("z", 1, "post"))
            self.assertEqual(info.start_gen, head)

        # A second reopen is a no-op for the marker/format.
        with open_collection(path) as coll:
            self.assertEqual(coll.status().head, head + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
