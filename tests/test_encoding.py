"""Encoding-layer property tests: reversibility, collision-freedom and the
equivalence between SQLite BLOB ordering and the independent business order.
"""
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey import (  # noqa: E402
    INT64_MAX,
    INT64_MIN,
    Column,
    EncodingError,
    Schema,
    key_decode,
    key_encode,
)
from bytekey.encoding import decode_component, encode_component  # noqa: E402
from tests.oracle import expected_order  # noqa: E402


def blob_sorted(keys):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(k BLOB PRIMARY KEY) WITHOUT ROWID")
    conn.executemany("INSERT INTO t(k) VALUES (?)", [(k,) for k in keys])
    rows = [r[0] for r in conn.execute("SELECT k FROM t ORDER BY k ASC")]
    conn.close()
    return rows


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema(
            columns=(
                Column("i", "int64"),
                Column("t", "text"),
                Column("b", "bytes"),
                Column("flag", "bool"),
            )
        )

    def test_extreme_and_tricky_values_roundtrip(self):
        cases = [
            [INT64_MIN, "héllo wörld", b"\x00\xff\x00", True],
            [INT64_MAX, "", b"", False],
            [0, "a\x00b", b"plain", None],
            [-1, "ab", b"\xff" * 10, True],
            [1, "日本語", bytes(range(256)), False],
            [None, None, None, None],
        ]
        rids = [b"id-1", b"id-2", b"\x00", b"\xff", b"x", b"y\x00z"]
        seen = set()
        for vals, rid in zip(cases, rids):
            key = key_encode(self.schema, vals, rid)
            self.assertNotIn(key, seen, "key collision detected")
            seen.add(key)
            got_vals, got_rid = key_decode(self.schema, key)
            self.assertEqual(got_vals, tuple(vals))
            self.assertEqual(got_rid, rid)

    def test_int64_boundaries(self):
        for v in (INT64_MIN, INT64_MIN + 1, -1, 0, 1, INT64_MAX - 1, INT64_MAX):
            col = Column("i", "int64")
            enc = encode_component(col, v)
            dec, _ = decode_component(col, enc, 0)
            self.assertEqual(dec, v)

    def test_out_of_range_int_rejected(self):
        with self.assertRaises(EncodingError):
            key_encode(self.schema, [INT64_MAX + 1, "x", b"x", True], b"r")
        with self.assertRaises(EncodingError):
            key_encode(self.schema, [INT64_MIN - 1, "x", b"x", True], b"r")

    def test_lone_surrogate_rejected(self):
        with self.assertRaises(EncodingError):
            key_encode(self.schema, [1, "bad\ud800surrogate", b"x", True], b"r")

    def test_type_discipline_bool_vs_int(self):
        with self.assertRaises(EncodingError):
            key_encode(self.schema, [True, "x", b"x", True], b"r")
        with self.assertRaises(EncodingError):
            key_encode(self.schema, [1, "x", b"x", 1], b"r")
        with self.assertRaises(EncodingError):
            key_encode(self.schema, [1, 2, b"x", True], b"r")

    def test_row_id_must_be_nonempty_bytes(self):
        with self.assertRaises(EncodingError):
            key_encode(self.schema, [1, "x", b"x", True], b"")
        with self.assertRaises(EncodingError):
            key_encode(self.schema, [1, "x", b"x", True], "rid")

    def test_max_key_size_enforced(self):
        schema = Schema(columns=(Column("t", "text"),), max_key_bytes=16)
        with self.assertRaises(EncodingError):
            key_encode(schema, ["too-long-text"], b"rid")
        key = key_encode(schema, ["ab"], b"r")  # within limit
        self.assertEqual(key_decode(schema, key)[0], ("ab",))

    def test_wrong_arity_rejected(self):
        with self.assertRaises(EncodingError):
            key_encode(self.schema, [1, "x"], b"r")


class OrderEquivalenceTests(unittest.TestCase):
    """SQLite BLOB order == independent comparator order, across directions."""

    def _check(self, schema, records):
        keys = {}
        for vals, rid in records:
            key = key_encode(schema, vals, rid)
            keys[(tuple(vals), rid)] = key
        # Distinct records must produce distinct keys.
        self.assertEqual(len(set(keys.values())), len(records))
        blob_order_keys = blob_sorted(list(keys.values()))
        key_to_record = {v: k for k, v in keys.items()}
        blob_order = [key_to_record[k] for k in blob_order_keys]
        want = expected_order(
            schema, [(list(v), rid) for v, rid in records]
        )
        want = [(tuple(v), rid) for v, rid in want]
        self.assertEqual(blob_order, want)

    def test_mixed_directions_and_null_positions(self):
        schema = Schema(
            columns=(
                Column("region", "text", asc=True, nulls_first=True),
                Column("ts", "int64", asc=False, nulls_first=True),
                Column("score", "int64", asc=True, nulls_first=False),
                Column("flag", "bool", asc=False, nulls_first=False),
            )
        )
        vals = [None, -3, 0, 5, 100]
        texts = [None, "", "a", "a\x00", "ab", "b"]
        flags = [None, False, True]
        records = []
        n = 0
        for region in texts:
            for ts in vals:
                for score in vals:
                    for flag in flags:
                        records.append(([region, ts, score, flag], f"r{n}".encode()))
                        n += 1
        self._check(schema, records)

    def test_all_desc_all_nulls_first(self):
        schema = Schema(
            columns=(
                Column("t", "text", asc=False, nulls_first=True),
                Column("i", "int64", asc=False, nulls_first=True),
            )
        )
        records = [
            ([t, i], f"{n}".encode())
            for n, (t, i) in enumerate(
                [(None, None), (None, 1), ("a", None), ("a", 1), ("b", -9),
                 ("", 0), ("a\x00", 2)]
            )
        ]
        self._check(schema, records)

    def test_all_desc_nulls_last(self):
        schema = Schema(
            columns=(
                Column("t", "text", asc=False, nulls_first=False),
                Column("i", "int64", asc=False, nulls_first=False),
            )
        )
        records = [
            ([t, i], f"{n}".encode())
            for n, (t, i) in enumerate(
                [(None, None), (None, 1), ("a", None), ("a", 1), ("b", -9),
                 ("", 0), ("a\x00", 2)]
            )
        ]
        self._check(schema, records)

    def test_text_prefix_and_embedded_nul_order(self):
        schema = Schema(columns=(Column("t", "text"),))
        records = [
            (["a"], b"1"),
            (["a\x00"], b"2"),
            (["a\x00x"], b"3"),
            (["ab"], b"4"),
            ([""], b"5"),
            (["a"], b"6"),  # duplicate business key, different row_id
        ]
        self._check(schema, records)

    def test_bytes_unsigned_byte_order(self):
        schema = Schema(columns=(Column("b", "bytes"),))
        records = [
            ([b""], b"1"),
            ([b"\x00"], b"2"),
            ([b"\x00\x00"], b"3"),
            ([b"\x00\xff"], b"4"),
            ([b"\x01"], b"5"),
            ([b"\xff"], b"6"),
            ([bytes([255, 255])], b"7"),
        ]
        self._check(schema, records)

    def test_bool_order_independent_of_int(self):
        schema = Schema(columns=(Column("f", "bool"),))
        records = [([None], b"1"), ([True], b"2"), ([False], b"3")]
        self._check(schema, records)
        # False < True in the encoding.
        self.assertLess(
            encode_component(schema.columns[0], False),
            encode_component(schema.columns[0], True),
        )

    def test_corrupt_key_decode_fails(self):
        schema = Schema(columns=(Column("i", "int64"),))
        good = key_encode(schema, [1], b"r")
        with self.assertRaises(EncodingError):
            key_decode(schema, good + b"\x00")
        with self.assertRaises(EncodingError):
            key_decode(schema, good[:-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
