"""Order-preserving, reversible composite-key byte encoding.

Encoding layout
---------------
Every column value becomes one self-delimiting *component*::

    NULL        : tag
    BOOL        : tag
    INT64       : tag + 8-byte big-endian sign-flipped integer
    TEXT        : tag + escaped(UTF-8 bytes) + terminator
    BYTES       : tag + escaped(raw bytes)       + terminator
    ROW_ID      : tag + escaped(non-empty bytes) + terminator

Raw (ascending) tags order every domain exactly as the business rules demand::

    0x01 NULL
    0x02 BOOL FALSE
    0x03 BOOL TRUE
    0x04 INT64
    0x05 TEXT
    0x06 BYTES
    0x07 (NULL when nulls_last; also the ROW_ID tag in its own slot)

Distinct tags mean bool never compares as int and text never compares as
bytes.  For ``nulls_last`` the NULL tag is simply emitted as 0x07, which
moves it past every value without touching value encodings.

Variable payloads escape every ``0x00`` as ``0x00 0xFF`` and terminate with
``0x00 0x00``.  A reader only ever sees, after a 0x00, either 0xFF (escaped
zero) or 0x00 (terminator); component tags are never zero so the terminator
stays unambiguous at a component boundary.  The scheme is lexicographically
order preserving, including strict-prefix and embedded-NUL cases, so no
"prefix plus one" trick is applied to business data.

A descending column is produced by bit-inverting the *whole finished
component* (``b ^ 0xFF``).  Inversion is a monotone-reversing bijection on
byte strings, so the DESC business order (including the null placement
chosen through the raw NULL tag) becomes ascending SQLite BLOB order.
"""
from __future__ import annotations

from typing import Sequence

from .schema import Column, Schema

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
_UINT64_WRAP = 1 << 64
_SIGN_FLIP = 1 << 63

# Raw (ascending) component tags.
TAG_NULL_FIRST = 0x01
TAG_BOOL_FALSE = 0x02
TAG_BOOL_TRUE = 0x03
TAG_INT64 = 0x04
TAG_TEXT = 0x05
TAG_BYTES = 0x06
TAG_NULL_LAST = 0x07
TAG_ROW_ID = 0x07  # separate slot, so sharing TAG_NULL_LAST is harmless.

_ESC = b"\x00\xff"
_TERM = b"\x00\x00"


class EncodingError(ValueError):
    """Raised when a value cannot be encoded or a key cannot be decoded."""


# ---------------------------------------------------------------------------
# Value validation
# ---------------------------------------------------------------------------

def _check_scalar(col: Column, value) -> None:
    kind = col.kind
    if value is None:
        return
    if kind == "int64":
        # bool is a subclass of int; the declaration keeps bool independent.
        if isinstance(value, bool) or not isinstance(value, int):
            raise EncodingError(
                f"column {col.name!r} expects int64, got {type(value).__name__}"
            )
        if not INT64_MIN <= value <= INT64_MAX:
            raise EncodingError(
                f"column {col.name!r} value {value} outside int64 range"
            )
    elif kind == "bool":
        if not isinstance(value, bool):
            raise EncodingError(
                f"column {col.name!r} expects bool, got {type(value).__name__}"
            )
    elif kind == "text":
        if not isinstance(value, str):
            raise EncodingError(
                f"column {col.name!r} expects text, got {type(value).__name__}"
            )
        # strict UTF-8 rejects lone surrogates; no normalization is performed.
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise EncodingError(
                f"column {col.name!r} contains illegal Unicode scalar "
                f"(lone surrogate at index {exc.start})"
            ) from None
    elif kind == "bytes":
        if not isinstance(value, (bytes, bytearray)):
            raise EncodingError(
                f"column {col.name!r} expects bytes, got {type(value).__name__}"
            )
    else:  # pragma: no cover - schema validation already guards this
        raise EncodingError(f"column {col.name!r} has unknown kind {kind!r}")


def _check_row_id(row_id) -> bytes:
    if not isinstance(row_id, (bytes, bytearray)):
        raise EncodingError(f"row_id must be bytes, got {type(row_id).__name__}")
    row_id = bytes(row_id)
    if not row_id:
        raise EncodingError("row_id must be a non-empty bytes value")
    return row_id


# ---------------------------------------------------------------------------
# Raw component encoding (ascending business order)
# ---------------------------------------------------------------------------

def _escape(payload: bytes) -> bytes:
    # bytes.replace keeps lexicographic order: 0x00 -> 0x00 0xFF.
    return payload.replace(b"\x00", _ESC)


def _encode_int64(value: int) -> bytes:
    flipped = (value + _SIGN_FLIP) % _UINT64_WRAP
    return bytes([TAG_INT64]) + flipped.to_bytes(8, "big")


def _encode_variable(tag: int, payload: bytes) -> bytes:
    return bytes([tag]) + _escape(payload) + _TERM


def _raw_null_tag(col: Column) -> int:
    """Pick the raw NULL tag so inversion leaves NULL in the requested slot.

    ASC keeps tags as-is; DESC bit-inverts the finished component, which
    reverses the tag order too.  With raw 0x01 the NULL sorts before values
    in raw order, with raw 0x07 after them -- so DESC swaps the choice.
    """
    want_first = col.nulls_first if col.asc else not col.nulls_first
    return TAG_NULL_FIRST if want_first else TAG_NULL_LAST


def _raw_component(col: Column, value) -> bytes:
    if value is None:
        return bytes([_raw_null_tag(col)])
    if col.kind == "bool":
        return bytes([TAG_BOOL_TRUE if value else TAG_BOOL_FALSE])
    if col.kind == "int64":
        return _encode_int64(value)
    if col.kind == "text":
        return _encode_variable(TAG_TEXT, value.encode("utf-8", errors="strict"))
    # bytes
    return _encode_variable(TAG_BYTES, bytes(value))


def _invert(data: bytes) -> bytes:
    return bytes(b ^ 0xFF for b in data)


def encode_component(col: Column, value) -> bytes:
    """Encode one column value into its final (SQLite-order) component."""
    _check_scalar(col, value)
    raw = _raw_component(col, value)
    return _invert(raw) if not col.asc else raw


def encode_row_id(row_id) -> bytes:
    """Encode the mandatory, unique, non-empty row_id tie-breaker."""
    return _encode_variable(TAG_ROW_ID, _check_row_id(row_id))


# ---------------------------------------------------------------------------
# Full key encode / decode
# ---------------------------------------------------------------------------

def key_encode(
    schema: Schema, values: Sequence, row_id, *, max_size: int | None = None
) -> bytes:
    """Encode a full business tuple plus row_id into a comparable BLOB key.

    When ``max_size`` is omitted the schema's own ``max_key_bytes`` limit
    (if any) applies; an explicit integer overrides it for that call.
    """
    if max_size is None:
        max_size = schema.max_key_bytes
    if len(values) != schema.width:
        raise EncodingError(
            f"expected {schema.width} column values, got {len(values)}"
        )
    parts = []
    for col, value in zip(schema.columns, values):
        parts.append(encode_component(col, value))
    parts.append(encode_row_id(row_id))
    key = b"".join(parts)
    if max_size is not None and len(key) > max_size:
        raise EncodingError(
            f"encoded key is {len(key)} bytes, exceeds limit {max_size}"
        )
    return key


class _Reader:
    """Component reader that transparently undoes DESC inversion."""

    def __init__(self, data: bytes, flipped: bool):
        self.data = data
        self.pos = 0
        self.flipped = flipped

    def read_byte(self) -> int:
        if self.pos >= len(self.data):
            raise EncodingError("truncated key: expected another component tag")
        b = self.data[self.pos]
        self.pos += 1
        return b ^ 0xFF if self.flipped else b

    def read_raw(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise EncodingError("truncated key: fixed-width payload incomplete")
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        if self.flipped:
            chunk = _invert(chunk)
        return chunk

    def read_terminated(self) -> bytes:
        """Read an escaped payload up to its 0x00 0x00 terminator."""
        out = bytearray()
        while True:
            b = self.read_byte()
            if b != 0x00:
                out.append(b)
                continue
            nxt = self.read_byte()
            if nxt == 0x00:
                return bytes(out)
            if nxt == 0xFF:
                out.append(0x00)
                continue
            raise EncodingError("corrupt key: invalid escape sequence in payload")


def _expected_tags(col: Column) -> set[int]:
    # The reader already undoes inversion, so the NULL tag it reads back is
    # the *raw* one selected by _raw_null_tag.
    tags = {_raw_null_tag(col)}
    if col.kind == "bool":
        tags.update((TAG_BOOL_FALSE, TAG_BOOL_TRUE))
    elif col.kind == "int64":
        tags.add(TAG_INT64)
    elif col.kind == "text":
        tags.add(TAG_TEXT)
    else:
        tags.add(TAG_BYTES)
    return tags


def decode_component(col: Column, data: bytes, pos: int) -> tuple[object, int]:
    """Decode one component starting at ``pos``; return (value, new_pos)."""
    reader = _Reader(data, flipped=not col.asc)
    reader.pos = pos
    tag = reader.read_byte()
    expected = _expected_tags(col)
    if tag not in expected:
        raise EncodingError(
            f"corrupt key at column {col.name!r}: tag 0x{tag:02x} not in "
            f"{sorted(expected)}"
        )
    if tag in (TAG_NULL_FIRST, TAG_NULL_LAST):
        return None, reader.pos
    if tag in (TAG_BOOL_FALSE, TAG_BOOL_TRUE):
        return tag == TAG_BOOL_TRUE, reader.pos
    if tag == TAG_INT64:
        raw = reader.read_raw(8)
        value = int.from_bytes(raw, "big") - _SIGN_FLIP
        return value, reader.pos
    payload = reader.read_terminated()
    if tag == TAG_TEXT:
        try:
            return payload.decode("utf-8", errors="strict"), reader.pos
        except UnicodeDecodeError:
            raise EncodingError(
                f"corrupt key at column {col.name!r}: payload is not valid UTF-8"
            ) from None
    return payload, reader.pos  # bytes


def key_decode(schema: Schema, key: bytes) -> tuple[tuple, bytes]:
    """Inverse of :func:`key_encode`.  Returns (business_tuple, row_id)."""
    if not isinstance(key, (bytes, bytearray)):
        raise EncodingError("key must be bytes")
    key = bytes(key)
    pos = 0
    values = []
    for col in schema.columns:
        value, pos = decode_component(col, key, pos)
        values.append(value)
    # Row_id slot is always ascending (final tie-break is never inverted).
    reader = _Reader(key, flipped=False)
    reader.pos = pos
    tag = reader.read_byte()
    if tag != TAG_ROW_ID:
        raise EncodingError(
            f"corrupt key: expected row_id tag 0x{TAG_ROW_ID:02x}, got 0x{tag:02x}"
        )
    row_id = reader.read_terminated()
    if reader.pos != len(key):
        raise EncodingError("corrupt key: trailing bytes after row_id component")
    if not row_id:
        raise EncodingError("corrupt key: empty row_id")
    return tuple(values), row_id
