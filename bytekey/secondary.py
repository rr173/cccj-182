"""Named secondary access indexes over updatable collections.

A secondary index is *not* a Python-side sort cache and it does not add a
business column or expression.  It is an ordered set of storage entries whose
key is the unchanged reversible bytekey encoding of a user-chosen subset of
the collection's existing business columns (with independently re-declared
``asc`` / ``nulls_first``) followed by the original row_id ascending.  Because
the row_id component is part of the key, equal indexed business values for
different identities remain distinct, ordered keys and no uniqueness is
imposed on the business projection.

Three physical tables carry the state:

``bytekey_sec_indexes``
    One row per *instance*.  ``instance_id`` is minted at registration and is
    never reused; ``build_id`` is the resumable build handle; the declaration
    is stored as canonical JSON plus a hash, so the build object can never be
    rewritten in place.  ``start_gen`` is the coverage start ``B``.

``bytekey_sec_queue``
    Per-instance catch-up journal.  Applies publish, in the *same* commit as
    the main version and receipt, one ``open`` / ``close`` event for every
    version they touch while the instance is building.  Advance processing
    consumes this journal with the caller's budget.

``bytekey_sec_entries``
    The materialized ordered entries: ``(instance_id, index_key, cgen)`` with
    ``dgen`` plus the full main key and the main version's creation
    generation (``main_key``, ``main_cgen``).  Visibility mirrors the main
    store (``cgen <= g AND (dgen IS NULL OR dgen > g)``), so every generation
    from ``start_gen`` onwards keeps its own version rather than only the
    latest one.  The stored ``(main_key, rid, main_cgen)`` identifies the
    exact main *version*, so a query locates that version by primary-key
    point seek -- never the current live row masquerading as history.

The module deliberately contains no scanning logic; the collection module
owns transactions, budgets and paging.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .encoding import key_encode
from .planner import QueryError, plan_query
from .schema import Column, Schema

SEC_INDEX_TABLE = "bytekey_sec_indexes"
SEC_ENTRY_TABLE = "bytekey_sec_entries"
SEC_QUEUE_TABLE = "bytekey_sec_queue"

INSTANCE_FORMAT = "bytekey-secondary/v1"
BUILDING = "building"
READY = "ready"
CANCELLED = "cancelled"
DELETED = "deleted"
ACTIVE_STATES = (BUILDING, READY)


class SecondaryError(RuntimeError):
    """Base class for secondary-index failures."""


class IndexDeclarationError(SecondaryError, ValueError):
    """A secondary-index declaration is invalid or refers to unknown columns."""


class IndexNotFoundError(SecondaryError):
    """No live index instance with that name exists."""


class IndexNotReadyError(SecondaryError):
    """The index exists but is not (yet) queryable."""


class IndexConflictError(SecondaryError):
    """Duplicate active name, busy build, or a conflicting concurrent step."""


class IndexCursorError(SecondaryError, ValueError):
    """A cursor/credential does not belong to the requested index instance."""


class CoverageError(SecondaryError, ValueError):
    """The pinned generation predates the index's coverage start B."""


class BuildCancelledError(SecondaryError):
    """The build was cancelled (or deleted) by another caller."""


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexColumn:
    """One column of a secondary index: a business column plus ordering."""

    name: str
    asc: bool = True
    nulls_first: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise IndexDeclarationError("index column name must be a non-empty string")
        if not isinstance(self.asc, bool):
            raise IndexDeclarationError("index column asc must be bool")
        if not isinstance(self.nulls_first, bool):
            raise IndexDeclarationError("index column nulls_first must be bool")

    def to_dict(self) -> dict:
        return {"name": self.name, "asc": self.asc, "nulls_first": self.nulls_first}


@dataclass(frozen=True)
class SecondaryDeclaration:
    """An immutable, named ordering over a subset of existing business columns."""

    name: str
    columns: tuple[IndexColumn, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise IndexDeclarationError("index name must be a non-empty string")
        columns = tuple(self.columns)
        if not columns:
            raise IndexDeclarationError(
                f"index {self.name!r} must declare at least one column"
            )
        if not all(isinstance(c, IndexColumn) for c in columns):
            raise IndexDeclarationError(
                "index columns must be IndexColumn declarations"
            )
        names = [c.name for c in columns]
        if len(set(names)) != len(names):
            raise IndexDeclarationError(
                f"index {self.name!r} references a column more than once"
            )
        object.__setattr__(self, "columns", columns)

    def to_dict(self) -> dict:
        return {
            "format": INSTANCE_FORMAT,
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
        }

    def canonical_json(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def declaration_hash(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SecondaryDeclaration":
        if not isinstance(data, Mapping):
            raise IndexDeclarationError("index declaration must be an object")
        allowed = {"format", "name", "columns"}
        unknown = set(data) - allowed
        if unknown:
            raise IndexDeclarationError(
                f"unknown index declaration field(s): {sorted(unknown)}"
            )
        fmt = data.get("format", INSTANCE_FORMAT)
        if fmt != INSTANCE_FORMAT:
            raise IndexDeclarationError(f"unsupported index declaration format {fmt!r}")
        columns = data.get("columns")
        if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
            raise IndexDeclarationError("index columns must be a list")
        parsed = []
        for item in columns:
            if not isinstance(item, Mapping):
                raise IndexDeclarationError("each index column must be an object")
            extra = set(item) - {"name", "asc", "nulls_first"}
            if extra:
                raise IndexDeclarationError(
                    f"unknown index column field(s): {sorted(extra)}"
                )
            if "name" not in item:
                raise IndexDeclarationError("index column requires 'name'")
            parsed.append(
                IndexColumn(
                    name=item["name"],
                    asc=item.get("asc", True),
                    nulls_first=item.get("nulls_first", True),
                )
            )
        return cls(name=data.get("name"), columns=tuple(parsed))

    @classmethod
    def from_json(cls, text: str | bytes) -> "SecondaryDeclaration":
        return cls.from_dict(json.loads(text))


def resolve_declaration(
    schema: Schema, declaration: SecondaryDeclaration | Mapping[str, Any]
) -> SecondaryDeclaration:
    """Validate a declaration against the collection's business schema.

    Every referenced column must already exist in the collection schema; the
    column *type* is inherited from there (a declaration may only choose
    ordering), and no business column or expression is added.
    """
    if not isinstance(declaration, SecondaryDeclaration):
        try:
            declaration = SecondaryDeclaration.from_dict(declaration)
        except IndexDeclarationError:
            raise
        except (TypeError, ValueError) as exc:
            raise IndexDeclarationError(f"malformed index declaration: {exc}") from None
    for col in declaration.columns:
        try:
            base = schema.column(col.name)
        except KeyError:
            raise IndexDeclarationError(
                f"index {declaration.name!r} references unknown business column "
                f"{col.name!r}"
            ) from None
        if not isinstance(base, Column):  # pragma: no cover - schema guarantees
            raise IndexDeclarationError(f"invalid business column {col.name!r}")
    return declaration


def index_columns(schema: Schema, declaration: SecondaryDeclaration) -> tuple[Column, ...]:
    """Business columns in index order, with kind inherited, ordering overridden."""
    out = []
    for item in declaration.columns:
        base = schema.column(item.name)
        out.append(
            Column(
                name=base.name,
                kind=base.kind,
                asc=item.asc,
                nulls_first=item.nulls_first,
            )
        )
    return tuple(out)


def index_schema(schema: Schema, declaration: SecondaryDeclaration) -> Schema:
    """A schema whose key is the index projection followed by the row_id."""
    return Schema(
        columns=index_columns(schema, declaration),
        row_id_name=schema.row_id_name,
        max_key_bytes=None,
    )


def encode_index_key(
    schema: Schema,
    declaration: SecondaryDeclaration,
    values: Sequence[Any],
    row_id: bytes,
) -> bytes:
    """Encode the full secondary key: projected columns + ascending row_id.

    ``values`` is a full main-schema tuple; only declared columns are used.
    """
    projected = []
    for item in declaration.columns:
        try:
            pos = next(
                i for i, c in enumerate(schema.columns) if c.name == item.name
            )
        except StopIteration:  # pragma: no cover - resolve_declaration guards
            raise IndexDeclarationError(
                f"index column {item.name!r} no longer exists"
            ) from None
        projected.append(values[pos])
    cols = index_columns(schema, declaration)
    return key_encode(
        Schema(columns=cols, row_id_name=schema.row_id_name), projected, row_id
    )


def plan_index_query(
    schema: Schema,
    declaration: SecondaryDeclaration,
    query: Mapping[str, Any],
):
    """Compile a business query in the *index declaration's* column order."""
    if not isinstance(query, Mapping):
        raise QueryError("query must be an object")
    unknown = set(query) - {
        "shape", "lower", "upper", "lower_inclusive", "upper_inclusive",
        "prefix", "range_column", "lo", "hi", "lo_inclusive", "hi_inclusive",
    }
    if unknown:
        raise QueryError(f"unknown query field(s): {sorted(unknown)}")
    projected = index_schema(schema, declaration)
    shape = query.get("shape")
    if shape == "full_tuple":
        q = dict(query)
        q["lower"] = None if query.get("lower") is None else list(query["lower"])
        q["upper"] = None if query.get("upper") is None else list(query["upper"])
        return plan_query(projected, q)
    if shape == "prefix_range":
        q = dict(query)
        prefix = []
        for item in query.get("prefix", []) or []:
            prefix.append(list(item))
        q["prefix"] = prefix
        return plan_query(projected, q)
    raise QueryError(
        f"query.shape must be 'full_tuple' or 'prefix_range', got {shape!r}"
    )


def validate_query_columns(
    schema: Schema,
    declaration: SecondaryDeclaration,
    query: Mapping[str, Any],
) -> None:
    """Check that column names used by the query exist *and are indexed*.

    Full-tuple bounds name no columns (they are positional in the index
    declaration order), but a prefix_range names them explicitly; those names
    must exist in the business schema and participate in this index.
    """
    index_names = {c.name for c in declaration.columns}
    if query.get("shape") == "prefix_range":
        for pair in query.get("prefix", []) or []:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise QueryError("each prefix entry must be a [column, value] pair")
            name = pair[0]
            if name not in index_names:
                raise QueryError(
                    f"prefix column {name!r} is not part of index {declaration.name!r}"
                )
        rname = query.get("range_column")
        if rname is not None and rname not in index_names:
            raise QueryError(
                f"range_column {rname!r} is not part of index "
                f"{declaration.name!r}"
            )
