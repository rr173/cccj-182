"""Randomized differential test of the change stream vs. a plain-dict model.

``tests/refchanges.py`` derives every generation's event list from ordinary
``dict`` state diffs (ordered by row_id) and its own paging arithmetic; it
never imports the production storage, paging or token code.  This suite
generates many generations containing moves (sort-position changes), deletes,
re-creations, empty batches, no-op batches, all-null business rows and bytes
values, then compares every page, acknowledgement and retention boundary.
"""
import os
import random
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bytekey import Column, Schema  # noqa: E402
from bytekey.collection import create_collection, open_collection  # noqa: E402
from tests.refchanges import RefChangeCollection, drain_expected  # noqa: E402

SCHEMA = Schema(columns=(
    Column("region", "text", asc=True, nulls_first=True),
    Column("n", "int64", asc=False, nulls_first=True),
    Column("blob", "bytes", asc=True, nulls_first=True),
))

REGIONS = ["a", "b", "c", None]
ROW_IDS = [b"r1", b"r2", b"r3", b"r4", b"r5", b"r6"]


def random_values(rng):
    return (
        rng.choice(REGIONS),
        rng.choice([None, -2, 0, 5, 17, 31]),
        rng.choice([None, b"", b"\x00", b"\x00\xff", b"\xff", b"data"]),
    )


def generate(rng, n_gens=14):
    """Yield (batch_id, operations) pairs covering the interesting shapes."""
    current = {}
    out = []
    for gen in range(1, n_gens + 1):
        kind = rng.random()
        ops = []
        if kind < 0.12:
            pass                                    # empty batch
        elif kind < 0.20 and current:
            rid = rng.choice(sorted(current))       # all-no-op batch
            vals = current[rid]
            ops = [{"op": "put", "row_id": rid,
                    "region": vals[0], "n": vals[1], "blob": vals[2]}]
        elif kind < 0.30:
            ops = [{"op": "delete", "row_id": rng.choice(ROW_IDS)}]
        else:
            chosen = rng.sample(ROW_IDS, rng.randint(1, 3))
            for rid in chosen:
                if rng.random() < 0.25:
                    ops.append({"op": "delete", "row_id": rid})
                    current.pop(rid, None)
                else:
                    vals = random_values(rng)
                    ops.append({"op": "put", "row_id": rid,
                                "region": vals[0], "n": vals[1],
                                "blob": vals[2]})
                    current[rid] = vals
        out.append((("batch-%d" % gen).encode(), ops))
    return out


def build(path, batches):
    create_collection(path, SCHEMA)
    with open_collection(path) as coll:
        head = 0
        for batch_id, ops in batches:
            coll.apply(batch_id, head, iter(ops))
            head += 1


class RandomChangeStreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_pages_ack_and_retention_match_reference(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                self._one(seed)

    def _one(self, seed):
        rng = random.Random(seed)
        batches = generate(rng)
        path = os.path.join(self.tmp.name, "c%d.db" % seed)
        build(path, batches)

        ref = RefChangeCollection(SCHEMA)
        ref.seed([])
        for batch_id, ops in batches:
            ref.apply(batch_id, ref.head, ops)
            self.assertLessEqual(ref.head, 14)

        page_size = rng.choice([1, 2, 3, 100])
        with open_collection(path) as coll:
            self.assertEqual(coll.status().head, ref.head)
            coll.begin_subscription("all", start_gen=1)
            expected_pages = drain_expected(ref, 1, page_size)
            got_pages = []
            cursor = None
            while True:
                page = coll.read_changes_page(
                    "all", page_size, cursor_token=cursor
                )
                got_pages.append(page)
                if not page.has_more:
                    break
                cursor = page.next_cursor
            self.assertEqual(len(got_pages), len(expected_pages))
            for got, want in zip(got_pages, expected_pages):
                got_events = [
                    (e.gen, e.seq, e.row_id, e.kind, e.before, e.after)
                    for e in got.events
                ]
                want_events = [(g, ev[0], ev[1], ev[2], ev[3], ev[4])
                               for g, ev in want["events"]]
                self.assertEqual(got_events, want_events)
                self.assertEqual([b.gen for b in got.batches],
                                 want["markers"])
                for b in got.batches:
                    self.assertEqual(b.event_count,
                                     ref.event_count(b.gen))
                self.assertEqual((got.gen, got.seq), want["pos"])
                self.assertEqual(got.has_more, want["has_more"])

            # Acknowledge page by page from the beginning; every step must be
            # accepted, and re-sending each acknowledgement must be idempotent.
            coll.release_subscription("all")
            coll.begin_subscription("all", start_gen=1)
            ack = (1, 0)
            cursor = None
            for want in expected_pages:
                page = coll.read_changes_page(
                    "all", page_size, cursor_token=cursor
                )
                info = coll.subscription_info("all")
                res = coll.acknowledge_changes(
                    "all", gen=page.gen, seq=page.seq,
                    expected_gen=info.ack_gen, expected_seq=info.ack_seq,
                )
                self.assertFalse(res["replayed"])
                replay = coll.acknowledge_changes(
                    "all", gen=page.gen, seq=page.seq,
                    expected_gen=info.ack_gen, expected_seq=info.ack_seq,
                )
                self.assertTrue(replay["replayed"])
                ack = (page.gen, page.seq)
                if not page.has_more:
                    break
                cursor = page.next_cursor
            # Fully acknowledged: nothing is retained after a collection.
            head = coll.status().head
            total = sum(len(g) for g in ref.events)
            hist = coll.collect_history(head)
            self.assertEqual(hist.change_events_removed, total)
            self.assertEqual(hist.change_batches_removed, ref.head)
            self.assertEqual(coll.status().change_event_count, 0)
            self.assertEqual(coll.change_retention()["retain_from"], head + 1)

    def test_retention_boundary_matches_oldest_unread_subscription(self):
        rng = random.Random(99)
        batches = generate(rng, n_gens=10)
        path = os.path.join(self.tmp.name, "ret.db")
        build(path, batches)
        ref = RefChangeCollection(SCHEMA)
        ref.seed([])
        for batch_id, ops in batches:
            ref.apply(batch_id, ref.head, ops)

        with open_collection(path) as coll:
            coll.begin_subscription("slow", start_gen=1)
            coll.begin_subscription("fast", start_gen=1)
            # fast drains and acknowledges everything.
            cursor = None
            while True:
                page = coll.read_changes_page("fast", 100, cursor_token=cursor)
                info = coll.subscription_info("fast")
                coll.acknowledge_changes(
                    "fast", gen=page.gen, seq=page.seq,
                    expected_gen=info.ack_gen, expected_seq=info.ack_seq,
                )
                if not page.has_more:
                    break
                cursor = page.next_cursor
            # slow acknowledges only its first page.
            page = coll.read_changes_page("slow", 1)
            info = coll.subscription_info("slow")
            coll.acknowledge_changes(
                "slow", gen=page.gen, seq=page.seq,
                expected_gen=info.ack_gen, expected_seq=info.ack_seq,
            )
            slow_pos = coll.subscription_info("slow")
            head = coll.status().head
            first_kept = min(ref.head, slow_pos.ack_gen)
            hist = coll.collect_history(head)
            floor = coll.change_retention()["retain_from"]
            self.assertLessEqual(floor, first_kept)
            # Every generation still needed by slow survives in full.
            for gen in range(floor, head + 1):
                self.assertEqual(
                    coll._event_count(gen), ref.event_count(gen)
                )
            # slow can continue and sees exactly the reference remainder.
            events = []
            cursor = None
            while True:
                page = coll.read_changes_page("slow", 3, cursor_token=cursor)
                events.extend(
                    (e.gen, e.seq, e.row_id, e.kind, e.before, e.after)
                    for e in page.events
                )
                if not page.has_more:
                    break
                cursor = page.next_cursor
            expected = ref.events_between(
                (floor, 0), (head, ref.event_count(head))
            )
            self.assertEqual(events, expected)
            self.assertGreaterEqual(hist.change_batches_removed, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
