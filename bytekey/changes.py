"""Persistent change-log records and subscription result objects.

The collection stores an append-only *outbox* of change events in the same
SQLite transaction that publishes a generation.  This module owns the storage
names, the canonical type-tagged JSON used for complete before/after images,
the exception vocabulary and the plain result objects handed back to callers.

An **absent** image is distinct from a business row whose columns are null:
absent is stored as SQL ``NULL`` in the image column and rendered as
``{"present": false}`` in JSON, while a present image is always a JSON array
(one type-tagged slot per schema column) rendered as
``{"present": true, "values": [...]}``.  A present tuple therefore never
collides with the absent marker, even if every business column is null.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

CHANGE_TABLE = "bytekey_changes"
CHANGE_BATCH_TABLE = "bytekey_change_batches"
SUBSCRIPTION_TABLE = "bytekey_subscriptions"

EVENT_KIND_INSERT = "insert"
EVENT_KIND_UPDATE = "update"
EVENT_KIND_DELETE = "delete"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ChangeError(RuntimeError):
    """Base class for change-stream and subscription failures."""


class SubscriptionError(ChangeError):
    """Invalid subscription request or state."""


class SubscriptionNotFoundError(SubscriptionError):
    """No live subscription with that name (or an already released identity)."""


class SubscriptionNameConflict(SubscriptionError):
    """An active subscription already owns that name."""


class SubscriptionConflictError(SubscriptionError):
    """Acknowledgment conflict: stale expected position or backwards move."""


class SubscriptionCursorError(ChangeError, ValueError):
    """Malformed, foreign, stale or cross-subscription page credential."""


class SubscriptionSkippedError(SubscriptionError):
    """Acknowledging a position this subscription has never read."""


class ChangeRetentionError(ChangeError):
    """The requested generation is below the retained change boundary."""


class ChangeLogError(ChangeError):
    """The durable change log is missing or corrupt at a locatable position."""


# ---------------------------------------------------------------------------
# Canonical type-tagged values (shared with the idempotency digest)
# ---------------------------------------------------------------------------

def typed_json_value(value: Any) -> dict:
    """Type-tagged canonical JSON value.

    The tag makes null, empty text, empty bytes, bool and integers
    non-confusable even when their plain JSON spelling would collide.
    """
    if value is None:
        return {"t": "null"}
    if isinstance(value, bool):
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        return {"t": "i64", "v": str(value)}
    if isinstance(value, str):
        return {"t": "txt", "v": value}
    if isinstance(value, (bytes, bytearray)):
        return {"t": "bin", "v": bytes(value).hex()}
    raise ValueError(f"value of type {type(value).__name__} is not encodable")


def encode_values(values: Sequence[Any]) -> str:
    """Encode one complete business tuple as canonical type-tagged JSON."""
    return json.dumps(
        [typed_json_value(v) for v in values],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def encode_image(values: Sequence[Any] | None) -> str | None:
    """Encode a before/after image; ``None`` means the row was absent."""
    return None if values is None else encode_values(values)


def _decode_one(slot: Any, gen: int, seq: int) -> Any:
    if not isinstance(slot, dict) or "t" not in slot:
        raise ChangeLogError(
            f"change event ({gen}, {seq}) carries a malformed value slot"
        )
    tag = slot["t"]
    if tag == "null":
        if set(slot) != {"t"}:
            raise ChangeLogError(
                f"change event ({gen}, {seq}) has an invalid null slot"
            )
        return None
    if set(slot) != {"t", "v"}:
        raise ChangeLogError(
            f"change event ({gen}, {seq}) has an invalid {tag!r} slot"
        )
    raw = slot["v"]
    try:
        if tag == "bool":
            if not isinstance(raw, bool):
                raise ValueError("bool slot is not a bool")
            return raw
        if tag == "i64":
            if not isinstance(raw, str):
                raise ValueError("i64 slot is not a string")
            return int(raw)
        if tag == "txt":
            if not isinstance(raw, str):
                raise ValueError("txt slot is not a string")
            return raw
        if tag == "bin":
            if not isinstance(raw, str):
                raise ValueError("bin slot is not a hex string")
            return bytes.fromhex(raw)
    except (TypeError, ValueError) as exc:
        raise ChangeLogError(
            f"change event ({gen}, {seq}) has an invalid {tag!r} value: {exc}"
        ) from None
    raise ChangeLogError(
        f"change event ({gen}, {seq}) carries unknown value tag {tag!r}"
    )


def decode_image(text: str | None, width: int, gen: int, seq: int) -> tuple | None:
    """Decode a stored before/after image; ``None`` stays absent."""
    if text is None:
        return None
    try:
        slots = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ChangeLogError(
            f"change event ({gen}, {seq}) image is not valid JSON: {exc}"
        ) from None
    if not isinstance(slots, list) or len(slots) != width:
        raise ChangeLogError(
            f"change event ({gen}, {seq}) image does not have {width} columns"
        )
    return tuple(_decode_one(slot, gen, seq) for slot in slots)


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

def render_value(value: Any) -> Any:
    """Render a business value for JSON output (bytes as ``{"__hex__": ...}``)."""
    if isinstance(value, (bytes, bytearray)):
        return {"__hex__": bytes(value).hex()}
    return value


def _present_image(values: tuple | None) -> dict:
    if values is None:
        return {"present": False}
    return {"present": True, "values": [render_value(v) for v in values]}


@dataclass(frozen=True)
class ChangeEvent:
    """One state-changing operation inside one generation."""

    gen: int
    seq: int           # stable 1-based position inside the generation
    row_id: bytes
    kind: str          # insert | update | delete
    before: tuple | None
    after: tuple | None

    def to_dict(self, row_id_name: str = "row_id") -> dict:
        return {
            "gen": self.gen,
            "seq": self.seq,
            row_id_name: {"__hex__": self.row_id.hex()},
            "op": self.kind,
            "before": _present_image(self.before),
            "after": _present_image(self.after),
        }


@dataclass(frozen=True)
class ChangeBatchMarker:
    """One successful generation in the change stream (possibly 0 events)."""

    gen: int
    batch_id: bytes
    digest: str
    event_count: int
    cum_events: int

    def to_dict(self) -> dict:
        return {
            "gen": self.gen,
            "batch_id": {"__hex__": self.batch_id.hex()},
            "digest": self.digest,
            "event_count": self.event_count,
            "cum_events": self.cum_events,
        }


@dataclass
class ChangeReadStats:
    start_gen: int
    start_seq: int
    ack_gen: int
    ack_seq: int
    next_gen: int
    next_seq: int
    events_examined: int          # event rows physically pulled from storage
    events_returned: int
    batch_markers_accessed: int   # generation marker rows physically pulled
    acked_skipped: int            # events at/below the resume position
    has_more: bool

    def to_dict(self) -> dict:
        return {
            "start_gen": self.start_gen,
            "start_seq": self.start_seq,
            "ack_gen": self.ack_gen,
            "ack_seq": self.ack_seq,
            "next_gen": self.next_gen,
            "next_seq": self.next_seq,
            "events_examined": self.events_examined,
            "events_returned": self.events_returned,
            "batch_markers_accessed": self.batch_markers_accessed,
            "acked_skipped": self.acked_skipped,
            "has_more": self.has_more,
        }


@dataclass
class ChangePage:
    """A bounded page of change events plus the generation markers crossed."""

    name: str
    instance_id: str
    events: list[ChangeEvent]
    batches: list[ChangeBatchMarker]
    next_cursor: str
    gen: int
    seq: int
    ack_gen: int
    ack_seq: int
    has_more: bool
    head: int
    stats: ChangeReadStats
    collection_id: str = ""
    row_id_name: str = "row_id"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instance_id": self.instance_id,
            "gen": self.gen,
            "seq": self.seq,
            "ack": {"gen": self.ack_gen, "seq": self.ack_seq},
            "head": self.head,
            "events": [e.to_dict(self.row_id_name) for e in self.events],
            "batches": [b.to_dict() for b in self.batches],
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
            "stats": self.stats.to_dict(),
        }


@dataclass(frozen=True)
class SubscriptionInfo:
    name: str
    instance_id: str
    status: str
    start_gen: int
    ack_gen: int
    ack_seq: int
    read_gen: int
    read_seq: int
    created_seq: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instance_id": self.instance_id,
            "status": self.status,
            "start_gen": self.start_gen,
            "ack": {"gen": self.ack_gen, "seq": self.ack_seq},
            "ack_gen": self.ack_gen,
            "ack_seq": self.ack_seq,
            "read": {"gen": self.read_gen, "seq": self.read_seq},
            "read_gen": self.read_gen,
            "read_seq": self.read_seq,
            "created_seq": self.created_seq,
        }
