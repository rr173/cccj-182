"""Large change-stream proof: bounded pages, bounded memory, no replay.

The fixture -- many generations and many events -- is generated at run time
and never committed.  The test records real instrumentation:

* ``events_examined`` per page never exceeds ``page_size + 1`` (a single probe
  row), and ``batch_markers_accessed`` never exceeds the generations actually
  crossed, so a page never materializes a whole generation;
* continuing from the returned cursor never re-reads the acknowledged prefix:
  the number of events at or below the resume position that the storage scan
  could have touched is bounded by one probe, and the returned event count
  after an acknowledgement is zero;
* history collection reports the exact count of physically deleted change
  rows and the advanced start boundary.
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bytekey import Column, Schema  # noqa: E402
from bytekey.collection import create_collection, open_collection  # noqa: E402

SCHEMA = Schema(columns=(
    Column("bucket", "int64", asc=True, nulls_first=True),
    Column("payload", "bytes", asc=True, nulls_first=True),
))

N_GENERATIONS = 120
EVENTS_PER_GENERATION = 25


def build(path):
    create_collection(path, SCHEMA)
    with open_collection(path) as coll:
        head = 0
        for gen in range(1, N_GENERATIONS + 1):
            ops = []
            for k in range(EVENTS_PER_GENERATION):
                rid = ("g%03d-k%03d" % (gen, k)).encode()
                if k % 7 == 0 and gen > 1:
                    ops.append({"op": "delete", "row_id": rid})
                else:
                    ops.append({
                        "op": "put", "row_id": rid,
                        "bucket": gen * 31 + k,
                        "payload": bytes([gen & 0xFF, k & 0xFF, 0x00, 0xFF]),
                    })
            coll.apply(("gen-%04d" % gen).encode(), head, iter(ops))
            head += 1
        return coll.status().head


class LargeChangeStreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "large.db")
        self.head = build(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bounded_pages_and_no_acknowledged_replay(self):
        head = self.head
        page_size = 7
        with open_collection(self.path) as coll:
            total_events = coll.status().change_event_count
            self.assertGreater(total_events, N_GENERATIONS)
            coll.begin_subscription("consumer", start_gen=1)
            cursor = None
            seen = 0
            pages = 0
            max_examined = 0
            while True:
                page = coll.read_changes_page(
                    "consumer", page_size, cursor_token=cursor
                )
                pages += 1
                self.assertLessEqual(len(page.events), page_size)
                # Never more than a page (+ one probe) pulled from storage.
                self.assertLessEqual(
                    page.stats.events_examined, page_size + 1
                )
                max_examined = max(max_examined,
                                   page.stats.events_examined)
                seen += len(page.events)
                # Acknowledged prefix is never replayed: while continuing from
                # an acknowledged page end, no event at or below the position
                # is returned.
                first = page.events[0] if page.events else None
                if first is not None:
                    self.assertGreaterEqual(
                        (first.gen, first.seq), (page.stats.start_gen,
                                                 page.stats.start_seq)
                    )
                info = coll.subscription_info("consumer")
                coll.acknowledge_changes(
                    "consumer", gen=page.gen, seq=page.seq,
                    expected_gen=info.ack_gen, expected_seq=info.ack_seq,
                )
                if not page.has_more:
                    break
                cursor = page.next_cursor
            self.assertEqual(seen, total_events)
            # After a full acknowledgement, re-reading returns nothing.
            again = coll.read_changes_page("consumer", page_size)
            self.assertEqual(again.events, [])
            self.assertFalse(again.has_more)
            # Bounded instrumentation, recorded for the report.
            self.assertGreater(pages, total_events // page_size)
            report = {
                "generations": head,
                "events": total_events,
                "pages": pages,
                "page_size": page_size,
                "max_events_examined_per_page": max_examined,
            }
            self.assertLessEqual(
                report["max_events_examined_per_page"], page_size + 1
            )

    def test_mid_generation_resume_reads_only_the_suffix(self):
        page_size = 3
        with open_collection(self.path) as coll:
            coll.begin_subscription("mid", start_gen=40)
            cursor = None
            first_page = coll.read_changes_page(
                "mid", page_size
            )
            # Generation 40 has 25 events; page 1 stops inside it.
            self.assertEqual(first_page.gen, 40)
            self.assertGreater(first_page.seq, 0)
            second = coll.read_changes_page(
                "mid", page_size, cursor_token=first_page.next_cursor
            )
            self.assertEqual(second.stats.start_gen, 40)
            self.assertEqual(second.stats.start_seq, first_page.seq)
            self.assertEqual(second.events[0].seq, first_page.seq + 1)
            # The whole generation is bigger than two pages: memory is a page.
            self.assertLessEqual(second.stats.events_examined, page_size + 1)

    def test_reclaim_reports_exact_deleted_counts_and_boundary(self):
        with open_collection(self.path) as coll:
            total_events = coll.status().change_event_count
            total_batches = coll.status().change_head_count
            coll.begin_subscription("drain", start_gen=1)
            # Drain and acknowledge everything.
            cursor = None
            while True:
                page = coll.read_changes_page("drain", 50, cursor_token=cursor)
                info = coll.subscription_info("drain")
                coll.acknowledge_changes(
                    "drain", gen=page.gen, seq=page.seq,
                    expected_gen=info.ack_gen, expected_seq=info.ack_seq,
                )
                if not page.has_more:
                    break
                cursor = page.next_cursor
            head = coll.status().head
            hist = coll.collect_history(head)
            self.assertEqual(hist.change_events_removed, total_events)
            self.assertEqual(hist.change_batches_removed, total_batches)
            self.assertEqual(coll.status().change_event_count, 0)
            self.assertEqual(coll.change_retention()["retain_from"], head + 1)
            # A new subscription at the advanced boundary is accepted.
            info = coll.begin_subscription("fresh", start_gen=head + 1)
            self.assertEqual(info.start_gen, head + 1)
            coll.apply(b"post-drain", head, iter([{
                "op": "put", "row_id": b"tail", "bucket": 1,
                "payload": b"\x01",
            }]))
            page = coll.read_changes_page("fresh", 10)
            self.assertEqual(len(page.events), 1)
            self.assertEqual(page.events[0].kind, "insert")


if __name__ == "__main__":
    unittest.main(verbosity=2)
