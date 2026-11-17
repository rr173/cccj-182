"""Randomized multi-generation differential testing against the oracle.

A fixed seed makes every run reproducible.  Random batches (puts that move
records around the key space, deletes, re-creates, no-ops and empty batches)
are applied to both the production collection and the plain-dictionary
reference model.  After every generation, randomized queries of both shapes
with randomized page sizes are drained and compared in full -- including
candidate/visibility counters, cursor continuation, and the bound that the
storage never decodes rows outside the planned byte intervals.
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey import Column, Schema  # noqa: E402
from bytekey.collection import ConflictError, create_collection, open_collection  # noqa: E402
from tests.refmodel import RefCollection, drain_snapshot, WHOLE_TABLE_QUERY  # noqa: E402

REGIONS = ["a", "a", "b", "c", None, "ab", "z"]
TS_VALUES = [None, 1, 2, 3, 5, 8, 13]
BLOBS = [b"", b"\x00", b"\x00\xff", b"abc", b"abc\x00", b"\xff"]
FLAGS = [True, False]


def make_schema():
    return Schema(
        columns=(
            Column("region", "text", asc=True, nulls_first=True),
            Column("ts", "int64", asc=False, nulls_first=True),
            Column("blob", "bytes", asc=True, nulls_first=False),
            Column("flag", "bool", asc=True, nulls_first=True),
        )
    )


def random_row(rng, rid):
    return {
        "op": "put",
        "row_id": rid,
        "region": rng.choice(REGIONS),
        "ts": rng.choice(TS_VALUES),
        "blob": rng.choice(BLOBS),
        "flag": rng.choice(FLAGS),
    }


def random_query(rng, schema):
    if rng.random() < 0.5:
        # prefix_range with a random prefix depth and optional range endpoints
        k = rng.randrange(schema.width)  # 0..width-1 prefix length
        prefix = []
        for i in range(k):
            col = schema.columns[i]
            if col.kind == "text":
                value = rng.choice([v for v in REGIONS if isinstance(v, str)] + [None])
            elif col.kind == "int64":
                value = rng.choice(TS_VALUES)
            elif col.kind == "bytes":
                value = rng.choice(BLOBS)
            else:
                value = rng.choice(FLAGS)
            prefix.append([col.name, value])
        q = {"shape": "prefix_range", "prefix": prefix}
        range_col = schema.columns[k]
        q["range_column"] = range_col.name
        if range_col.kind == "int64":
            pool = [v for v in TS_VALUES if v is not None]
            lo, hi = rng.sample(pool, 2) if rng.random() < 0.6 else (None, None)
            if lo is not None and lo > hi:
                lo, hi = hi, lo
            if lo is not None:
                q["lo"] = lo
            if hi is not None:
                q["hi"] = hi
        elif range_col.kind == "text":
            pool = [v for v in REGIONS if isinstance(v, str)]
            if rng.random() < 0.6:
                lo, hi = rng.sample(pool, 2)
                if lo > hi:
                    lo, hi = hi, lo
                q["lo"], q["hi"] = lo, hi
        elif range_col.kind == "bytes":
            if rng.random() < 0.6:
                lo, hi = rng.sample(BLOBS, 2)
                if lo > hi:
                    lo, hi = hi, lo
                q["lo"], q["hi"] = lo, hi
        else:
            if rng.random() < 0.6:
                q["lo"], q["hi"] = False, True
        return q
    # full_tuple: random bound pairs from actually plausible tuples
    width = schema.width
    lower = [rng.choice(REGIONS), rng.choice(TS_VALUES), rng.choice(BLOBS),
             rng.choice(FLAGS)]
    upper = [rng.choice(REGIONS), rng.choice(TS_VALUES), rng.choice(BLOBS),
             rng.choice(FLAGS)]
    return {
        "shape": "full_tuple",
        "lower": lower if rng.random() < 0.8 else None,
        "upper": upper if rng.random() < 0.8 else None,
        "lower_inclusive": rng.choice([True, False]),
        "upper_inclusive": rng.choice([True, False]),
    }


class RandomDifferentialTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "random.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_multi_generation_diff(self):
        rng = random.Random(20261117)
        schema = make_schema()
        create_collection(self.path, schema)
        ref = RefCollection(schema)
        rids = [f"rid-{i:03d}".encode() for i in range(40)]
        batch_seq = 0

        with open_collection(self.path) as coll:
            for gen_round in range(24):
                # Compose a random batch, sometimes empty, sometimes reusing
                # ids already live, sometimes deleting missing ids (no-ops).
                batch_seq += 1
                batch_id = f"batch-{batch_seq:03d}".encode()
                head = coll.status().head
                if rng.random() < 0.08:
                    operations = []  # empty batch still consumes a generation
                else:
                    n = rng.randrange(1, 12)
                    chosen = rng.sample(rids, n)
                    operations = []
                    for rid in chosen:
                        if rng.random() < 0.35:
                            operations.append({"op": "delete", "row_id": rid})
                        else:
                            operations.append(random_row(rng, rid))
                try:
                    receipt = coll.apply(batch_id, head, iter(operations))
                except ConflictError:
                    # Only possible if our bookkeeping diverged; fail loudly.
                    raise
                ref.apply(batch_id, head, operations)
                self.assertEqual(coll.status().head, ref.head)
                self.assertEqual(receipt.head, ref.head)

                # Snapshot the fresh generation and verify several queries.
                token = coll.begin_snapshot()
                ref_token = ref.begin_snapshot(ref.head)
                for _ in range(6):
                    query = random_query(rng, schema)
                    page_size = rng.choice([1, 2, 3, 7, 50])
                    got, pages, plan = drain_snapshot(coll, token, query, page_size)
                    want = ref.query_snapshot(ref_token, query)
                    self.assertEqual(
                        got, want,
                        msg=f"query mismatch gen={ref.head} q={query}",
                    )
                    # Every bounded scan must use the PRIMARY KEY range
                    # index; a fully unbounded query is legitimately a scan.
                    page0, stats0, _ = coll.scan_snapshot_page(
                        token, query, page_size
                    )
                    bounded = bool(
                        query.get("lower") is not None
                        or query.get("upper") is not None
                        or query.get("prefix")
                        or query.get("lo") is not None
                        or query.get("hi") is not None
                    )
                    if plan.intervals and bounded:
                        self.assertTrue(stats0.used_index)
                        for d0 in stats0.explain_query_plan:
                            self.assertIn("USING PRIMARY KEY", d0["detail"])
                    # Candidate accounting: delivered <= examined, no negatives.
                    self.assertGreaterEqual(
                        stats0.candidates_examined, stats0.rows_delivered
                    )
                    self.assertGreaterEqual(stats0.invisible_versions_skipped, 0)

            # Current head must match the oracle after every round.
            self.assertEqual(coll.status().head, ref.head)

            # Final full-table oracle comparison across every generation.
            for g in range(ref.head + 1):
                tok = coll.begin_snapshot(g)
                got, _, _ = drain_snapshot(
                    coll, tok, WHOLE_TABLE_QUERY, rng.choice([1, 4, 33]),
                )
                self.assertEqual(got, ref.records_at(g),
                                 msg=f"full generation {g} mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
