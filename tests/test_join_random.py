"""Randomized differential tests for the two-source equi-join.

Expected results are computed independently: source rows come from the
plain-dictionary reference model (``tests/refmodel.py``, which never imports
the production visibility code), and the join itself is a plain double loop
with strict type-aware value equality -- the production joiner is never
used to produce answers.  Production results are drained from real work
files, reopening the join handle between pages like separate processes
would.
"""
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey import (  # noqa: E402
    Column,
    IndexColumn,
    Schema,
    SecondaryDeclaration,
    create_collection,
    open_collection,
)
from bytekey.join import JoinSource, open_join, prepare_join  # noqa: E402
from tests import refmodel  # noqa: E402

LSCHEMA = Schema(columns=(
    Column("region", "text", asc=True, nulls_first=True),
    Column("sku", "text", asc=False, nulls_first=True),
    Column("qty", "int64", asc=True, nulls_first=False),
    Column("blob", "bytes", asc=True, nulls_first=True),
    Column("flag", "bool", asc=False, nulls_first=True),
))
RSCHEMA = Schema(columns=(
    Column("product", "text", asc=False, nulls_first=True),
    Column("area", "text", asc=True, nulls_first=True),
    Column("code", "int64", asc=True, nulls_first=True),
    Column("tag", "bytes", asc=False, nulls_first=True),
    Column("active", "bool", asc=True, nulls_first=False),
    Column("label", "text", asc=True, nulls_first=True),
))

# Compatible column pairs (same kind on both sides).
COMPAT = [
    ("region", "area"),
    ("sku", "product"),
    ("qty", "code"),
    ("blob", "tag"),
    ("flag", "active"),
]

TEXT_POOL = ["", "a", "ab", "a\x00b", "region-0", "x", "y", "é", "zz"]
BYTES_POOL = [b"", b"\x00", b"\x00\xff", b"\xff", b"ab", b"\x00\x00"]
INT_POOL = [-(1 << 40), -7, 0, 1, 2, 3, 42, 1 << 50]


def _rand_value(rng, kind):
    if rng.random() < 0.18:
        return None
    if kind == "text":
        return rng.choice(TEXT_POOL)
    if kind == "bytes":
        return rng.choice(BYTES_POOL)
    if kind == "int64":
        return rng.choice(INT_POOL)
    if kind == "bool":
        return rng.random() < 0.5
    raise AssertionError(kind)


def _rand_row(rng, schema):
    return tuple(_rand_value(rng, c.kind) for c in schema.columns)


def _strict_eq(a, b):
    return type(a) is type(b) and a == b


def oracle_join(left_rows, right_rows, mapping_pairs, mode, nulls_equal):
    """Independent double-loop equi-join over plain Python values.

    ``left_rows`` / ``right_rows`` are ``(values, rid)`` lists already in
    their path order.  Null handling: default never matches a null key
    component; ``nulls_equal`` matches null with null only.
    """
    lpos = [p[0] for p in mapping_pairs]
    rpos = [p[1] for p in mapping_pairs]
    results = []
    right_matched = [False] * len(right_rows)
    for lvals, lrid in left_rows:
        lkey = tuple(lvals[p] for p in lpos)
        can = nulls_equal or not any(v is None for v in lkey)
        matched_any = False
        if can:
            for i, (rvals, rrid) in enumerate(right_rows):
                rkey = tuple(rvals[p] for p in rpos)
                if not nulls_equal and any(v is None for v in rkey):
                    continue
                if all(_strict_eq(a, b) for a, b in zip(lkey, rkey)):
                    matched_any = True
                    right_matched[i] = True
                    results.append(("matched", (lvals, lrid), (rvals, rrid)))
        if not matched_any and mode in ("left", "full"):
            results.append(("left_only", (lvals, lrid), None))
    if mode == "full":
        for i, (rvals, rrid) in enumerate(right_rows):
            if not right_matched[i]:
                results.append(("right_only", None, (rvals, rrid)))
    return results


def _rand_query(rng, schema, ordered_names):
    """A random valid query over the given access-path column order."""
    if rng.random() < 0.45:
        return {"shape": "full_tuple", "lower": None, "upper": None}
    k = rng.randrange(0, min(2, len(ordered_names) - 1) + 1)
    prefix = []
    for i in range(k):
        col = schema.column(ordered_names[i])
        prefix.append([col.name, _rand_value(rng, col.kind)])
    rcol = schema.column(ordered_names[k])
    q = {"shape": "prefix_range", "prefix": prefix, "range_column": rcol.name}
    lo = _rand_value(rng, rcol.kind) if rng.random() < 0.8 else None
    hi = _rand_value(rng, rcol.kind) if rng.random() < 0.8 else None
    lo_inc = rng.random() < 0.7
    hi_inc = rng.random() < 0.7
    if lo is not None and hi is not None:
        # Keep the range legal: drop the upper bound if it would invert or
        # empty the range (nulls are never legal endpoints).
        if lo > hi or (lo == hi and not (lo_inc and hi_inc)):
            hi = None
    if lo is not None:
        q["lo"] = lo
        q["lo_inclusive"] = lo_inc
    if hi is not None:
        q["hi"] = hi
        q["hi_inclusive"] = hi_inc
    return q


class RandomJoinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _build_side(self, rng, name, schema, index_decl_cols):
        path = os.path.join(self.tmp.name, name)
        create_collection(path, schema)
        ref = refmodel.RefCollection(schema)
        n_seed = rng.randrange(0, 14)
        seed_ops = []
        rids = set()
        for _ in range(n_seed):
            rid = rng.randbytes(rng.randrange(1, 5))
            if rid in rids:
                continue
            rids.add(rid)
            vals = _rand_row(rng, schema)
            seed_ops.append({"op": "put", "row_id": rid,
                             **{c.name: v for c, v in zip(schema.columns, vals)}})
        with open_collection(path) as coll:
            # Register the index first (coverage start 0), then seed and
            # correct: every generation is covered by the ready index.
            decl = None
            if index_decl_cols:
                decl = SecondaryDeclaration(f"idx_{name}", tuple(
                    IndexColumn(c, asc=rng.random() < 0.5,
                                nulls_first=rng.random() < 0.5)
                    for c in index_decl_cols
                ))
                coll.register_index(decl)
                while coll.advance_index(decl.name, 200).status != "ready":
                    pass
            # The seed is batch #1 on both the reference model and the
            # production collection, keeping generation numbers aligned.
            ref.apply(b"seed", 0, seed_ops)
            coll.apply(b"seed", 0, iter(seed_ops))
            # A few correction generations so snapshots can pin older gens.
            n_batches = rng.randrange(0, 3)
            for b in range(n_batches):
                ops = []
                for rid, vals in list(ref.states[ref.head].items()):
                    if rng.random() < 0.25:
                        new_vals = _rand_row(rng, schema)
                        ops.append({"op": "put", "row_id": rid,
                                    **{c.name: v for c, v in
                                       zip(schema.columns, new_vals)}})
                    elif rng.random() < 0.1:
                        ops.append({"op": "delete", "row_id": rid})
                for _ in range(rng.randrange(0, 3)):
                    rid = rng.randbytes(rng.randrange(1, 5))
                    if rid in ref.states[ref.head]:
                        continue
                    vals = _rand_row(rng, schema)
                    ops.append({"op": "put", "row_id": rid,
                                **{c.name: v for c, v in
                                   zip(schema.columns, vals)}})
                ref.apply(f"fix{b}".encode(), ref.head, ops)
                coll.apply(f"fix{b}".encode(), coll.status().head, iter(ops))
            gen = rng.randrange(0, ref.head + 1)
            token = coll.begin_snapshot(gen)
        return {
            "path": path, "ref": ref, "token": token, "gen": gen,
            "decl": decl,
        }

    def _side_rows(self, side, use_index, query):
        """Reference rows at the pinned generation, in path order, filtered."""
        ref = side["ref"]
        gen = side["gen"]
        if use_index:
            decl_dict = side["decl"].to_dict()
            return refmodel.query_index_at(ref, gen, decl_dict, query)
        return ref.query_at(gen, query)

    def test_random_joins(self):
        rng = random.Random(20261118)
        for case in range(40):
            with self.subTest(case=case):
                self._run_case(rng, case)

    def _run_case(self, rng, case):
        lidx_cols = rng.sample([c.name for c in LSCHEMA.columns],
                               rng.randrange(1, 3)) if rng.random() < 0.7 else None
        ridx_cols = rng.sample([c.name for c in RSCHEMA.columns],
                               rng.randrange(1, 3)) if rng.random() < 0.7 else None
        left = self._build_side(rng, f"l{case}.db", LSCHEMA, lidx_cols)
        right = self._build_side(rng, f"r{case}.db", RSCHEMA, ridx_cols)

        # Mapping: 1..min(3, ...) compatible pairs.
        pairs = rng.sample(COMPAT, rng.randrange(1, 4))
        mapping = [(l, r) for l, r in pairs]
        lpos = [next(i for i, c in enumerate(LSCHEMA.columns) if c.name == l)
                for l, _ in pairs]
        rpos = [next(i for i, c in enumerate(RSCHEMA.columns) if c.name == r)
                for _, r in pairs]

        use_lindex = left["decl"] is not None and rng.random() < 0.6
        use_rindex = right["decl"] is not None and rng.random() < 0.6
        lq = _rand_query(rng, LSCHEMA,
                         [c.name for c in left["decl"].columns] if use_lindex
                         else [c.name for c in LSCHEMA.columns])
        rq = _rand_query(rng, RSCHEMA,
                         [c.name for c in right["decl"].columns] if use_rindex
                         else [c.name for c in RSCHEMA.columns])
        mode = rng.choice(["inner", "left", "full"])
        nulls_equal = rng.random() < 0.4
        page_size = rng.randrange(1, 5)

        work = os.path.join(self.tmp.name, f"w{case}.db")
        result = prepare_join(
            work,
            left=JoinSource(left["path"], left["token"], lq,
                            index=left["decl"].name if use_lindex else None),
            right=JoinSource(right["path"], right["token"], rq,
                             index=right["decl"].name if use_rindex else None),
            mapping=mapping, mode=mode, nulls_equal=nulls_equal,
            page_size=page_size,
        )

        # Independent expectation: reference rows + double loop.
        left_rows = self._side_rows(left, use_lindex, lq)
        right_rows = self._side_rows(right, use_rindex, rq)
        expected = oracle_join(
            left_rows, right_rows,
            list(zip(lpos, rpos)), mode, nulls_equal,
        )

        # Drain production, reopening the handle for every page.
        got = []
        cursor = None
        handle = result["handle_token"]
        total_pairs = 0
        while True:
            join = open_join(work, handle)
            page, stats = join.page(cursor)
            join.close()
            for row in page.rows:
                lv = None if row.left is None else (row.left.values,
                                                    row.left.row_id)
                rv = None if row.right is None else (row.right.values,
                                                     row.right.row_id)
                got.append((row.kind, lv, rv))
            total_pairs += stats.pairs_delivered
            cursor = page.next_cursor
            if cursor is None:
                break
        self.assertEqual(got, expected,
                         f"case {case}: mode={mode} nulls_equal={nulls_equal} "
                         f"mapping={mapping} ps={page_size}")
        self.assertEqual(
            total_pairs,
            sum(1 for k, _, _ in expected if k == "matched"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
