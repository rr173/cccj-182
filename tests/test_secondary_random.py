"""Randomized differential testing for secondary access indexes.

Multiple named indexes are registered at different coverage starts, advanced
in randomly interleaved small steps, while batches (puts that move records,
deletes, re-creates, non-indexed-only updates, no-ops and empty batches) are
published at random points.  After every generation *every* ready index is
drained with a random page size and compared with the plain-dictionary
reference model.

The generated data deliberately includes ``None``, empty strings, NUL bytes
and 0xFF bytes in a bytes column, so null placement, empty-vs-null and
embedded-zero encoding are all exercised through the secondary path.
"""
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey import (  # noqa: E402
    Column,
    Schema,
    SecondaryDeclaration,
    create_collection,
    open_collection,
)
from bytekey.collection import ConflictError  # noqa: E402
from tests.refmodel import (  # noqa: E402
    RefCollection,
    drain_index,
    drain_snapshot,
    query_index_at,
    WHOLE_TABLE_QUERY,
)

REGIONS = [None, "", "a", "a\x00", "ab", "b", "z"]
TS = [None, -5, 0, 1, 2, 7, 10, 99]
BLOBS = [None, b"", b"\x00", b"\x00\x00", b"\x00\xff", b"\xff", b"abc",
         b"abc\x00"]
FLAGS = [None, True, False]


def make_schema():
    return Schema(columns=(
        Column("region", "text", asc=True, nulls_first=True),
        Column("ts", "int64", asc=False, nulls_first=True),
        Column("blob", "bytes", asc=True, nulls_first=False),
        Column("flag", "bool", asc=True, nulls_first=True),
    ))


DECLARATIONS = [
    {"name": "by_ts_region", "columns": [
        {"name": "ts", "asc": False, "nulls_first": True},
        {"name": "region", "asc": True, "nulls_first": True},
    ]},
    {"name": "by_blob", "columns": [
        {"name": "blob", "asc": True, "nulls_first": False},
    ]},
    {"name": "by_flag_ts_region", "columns": [
        {"name": "flag", "asc": True, "nulls_first": False},
        {"name": "ts", "asc": True, "nulls_first": False},
        {"name": "region", "asc": False, "nulls_first": True},
    ]},
    {"name": "by_region_nullslast", "columns": [
        {"name": "region", "asc": False, "nulls_first": False},
        {"name": "blob", "asc": False, "nulls_first": True},
    ]},
]


def random_values(rng):
    return {
        "region": rng.choice(REGIONS),
        "ts": rng.choice(TS),
        "blob": rng.choice(BLOBS),
        "flag": rng.choice(FLAGS),
    }


def random_query(rng, schema, declaration):
    names = [c["name"] for c in declaration["columns"]]
    kinds = {c.name: c.kind for c in schema.columns}
    pools = {
        "region": REGIONS,
        "ts": TS,
        "blob": BLOBS,
        "flag": FLAGS,
    }
    if rng.random() < 0.5:
        # prefix_range in the declaration order
        k = rng.randrange(len(names))
        prefix = [[names[i], rng.choice(pools[names[i]])] for i in range(k)]
        q = {"shape": "prefix_range", "prefix": prefix,
             "range_column": names[k]}
        pool = [v for v in pools[names[k]] if v is not None]
        if pool and rng.random() < 0.7:
            lo, hi = sorted(rng.sample(pool, 2))
            if rng.random() < 0.8:
                q["lo"] = lo
            if rng.random() < 0.8:
                q["hi"] = hi
        return q
    lower = [rng.choice(pools[n]) for n in names]
    upper = [rng.choice(pools[n]) for n in names]
    return {
        "shape": "full_tuple",
        "lower": lower if rng.random() < 0.7 else None,
        "upper": upper if rng.random() < 0.7 else None,
        "lower_inclusive": rng.choice([True, False]),
        "upper_inclusive": rng.choice([True, False]),
    }


class SecondaryRandomTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "random.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_multi_index_multi_generation_diff(self):
        rng = random.Random(987654321)
        schema = make_schema()
        create_collection(self.path, schema)
        ref = RefCollection(schema)
        rids = [f"rid-{i:03d}".encode() + bytes([i & 0xFF]) for i in range(30)]
        batch_seq = 0
        # Coverage start B per name, recorded on registration.
        start = {}
        ready = set()

        with open_collection(self.path) as coll:
            # Seed generation 0 with a first batch so head >= 0.
            ops = [{"op": "put", "row_id": rid, **random_values(rng)}
                   for rid in rng.sample(rids, 15)]
            coll.apply(b"seed", 0, iter(ops))
            ref.apply(b"seed", 0, ops)
            # Register two indexes at B=1 (all later generations covered).
            for decl in DECLARATIONS[:2]:
                reg = coll.register_index(SecondaryDeclaration.from_dict(decl))
                start[decl["name"]] = reg["start_gen"]
            registered_late = DECLARATIONS[2]
            start[registered_late["name"]] = None  # registered later

            step_budget = 1
            for round_no in range(30):
                # Advance some indexes.
                for decl in DECLARATIONS:
                    name = decl["name"]
                    if start.get(name) is None:
                        continue
                    if name not in ready:
                        st = coll.advance_index(name, step_budget)
                        if st.status == "ready":
                            ready.add(name)
                # Maybe register the late index mid-flight.
                if (registered_late["name"] not in start or
                        start[registered_late["name"]] is None):
                    if rng.random() < 0.2:
                        reg = coll.register_index(
                            SecondaryDeclaration.from_dict(registered_late)
                        )
                        start[registered_late["name"]] = reg["start_gen"]
                # Publish a random batch.
                batch_seq += 1
                head = coll.status().head
                bid = f"batch-{batch_seq:03d}".encode()
                if rng.random() < 0.1:
                    operations = []
                else:
                    n = rng.randrange(1, 9)
                    operations = []
                    for rid in rng.sample(rids, n):
                        if rng.random() < 0.3:
                            operations.append({"op": "delete", "row_id": rid})
                        else:
                            operations.append(
                                {"op": "put", "row_id": rid, **random_values(rng)}
                            )
                coll.apply(bid, head, iter(operations))
                ref.apply(bid, head, operations)
                self.assertEqual(coll.status().head, ref.head)

                # Every ready index must agree at a couple of generations,
                # including already-issued historical tokens.
                for name, b in start.items():
                    if b is None or name not in ready:
                        continue
                    gens = [b]
                    if ref.head > b:
                        gens.append(rng.randrange(b, ref.head + 1))
                    decl = next(d for d in DECLARATIONS if d["name"] == name)
                    for g in gens:
                        tok = coll.begin_snapshot(g)
                        query = random_query(rng, schema, decl)
                        page_size = rng.choice([1, 2, 3, 5])
                        got, _, _ = drain_index(
                            coll, tok, name, query, page_size
                        )
                        want = query_index_at(ref, g, decl, query)
                        self.assertEqual(
                            got, want,
                            msg=f"index {name} gen {g} q={query} "
                                f"ps={page_size}",
                        )

            # Finish all builds, including the late one, and full-drain every
            # generation on every index for the strongest check.
            for decl in DECLARATIONS:
                name = decl["name"]
                if start.get(name) is None:
                    reg = coll.register_index(
                        SecondaryDeclaration.from_dict(decl)
                    )
                    start[name] = reg["start_gen"]
                while coll.advance_index(name, 3).status != "ready":
                    pass
                ready.add(name)

            for name in [d["name"] for d in DECLARATIONS]:
                decl = next(d for d in DECLARATIONS if d["name"] == name)
                b = start[name]
                for g in range(b, ref.head + 1):
                    tok = coll.begin_snapshot(g)
                    got, _, _ = drain_index(
                        coll, tok, name, WHOLE_TABLE_QUERY, rng.choice([1, 4, 9])
                    )
                    self.assertEqual(
                        got, query_index_at(ref, g, decl, WHOLE_TABLE_QUERY),
                        msg=f"final index {name} gen {g}",
                    )

            # Main-order snapshot reads are unchanged.
            for g in range(ref.head + 1):
                got, _, _ = drain_snapshot(
                    coll, coll.begin_snapshot(g), WHOLE_TABLE_QUERY, 3
                )
                self.assertEqual(got, ref.records_at(g))


if __name__ == "__main__":
    unittest.main(verbosity=2)
