"""Index schema declarations.

Column kinds are fixed at build time: ``int64``, ``text``, ``bytes`` or
``bool``.  Every column carries its own ``asc``/``desc`` direction and
``nulls_first``/``nulls_last`` position.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

KINDS = ("int64", "text", "bytes", "bool")


@dataclass(frozen=True)
class Column:
    name: str
    kind: str
    asc: bool = True
    nulls_first: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("column name must be a non-empty string")
        if self.kind not in KINDS:
            raise ValueError(f"unsupported column kind {self.kind!r}; expected one of {KINDS}")
        if not isinstance(self.asc, bool):
            raise ValueError("column.asc must be bool")
        if not isinstance(self.nulls_first, bool):
            raise ValueError("column.nulls_first must be bool")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "asc": self.asc,
            "nulls_first": self.nulls_first,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Column":
        if not isinstance(d, Mapping):
            raise ValueError("column declaration must be an object")
        allowed = {"name", "kind", "asc", "nulls_first"}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"unknown column field(s): {sorted(unknown)}")
        if "name" not in d or "kind" not in d:
            raise ValueError("column requires 'name' and 'kind'")
        return cls(
            name=d["name"],
            kind=d["kind"],
            asc=bool(d.get("asc", True)),
            nulls_first=bool(d.get("nulls_first", True)),
        )


@dataclass(frozen=True)
class Schema:
    columns: tuple[Column, ...]
    row_id_name: str = "row_id"
    max_key_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("schema must declare at least one column")
        if not all(isinstance(c, Column) for c in self.columns):
            raise ValueError("schema.columns must be Column instances")
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("duplicate column names are not allowed")
        if not isinstance(self.row_id_name, str) or not self.row_id_name:
            raise ValueError("row_id_name must be a non-empty string")
        if self.row_id_name in names:
            raise ValueError("row_id_name must not collide with a business column name")
        if self.max_key_bytes is not None:
            if not isinstance(self.max_key_bytes, int) or self.max_key_bytes <= 0:
                raise ValueError("max_key_bytes must be a positive integer")
        # frozen dataclass with tuple input is fine; coerce list inputs.
        object.__setattr__(self, "columns", tuple(self.columns))

    @property
    def width(self) -> int:
        return len(self.columns)

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "columns": [c.to_dict() for c in self.columns],
            "row_id_name": self.row_id_name,
            "max_key_bytes": self.max_key_bytes,
        }

    def canonical_json(self) -> bytes:
        """Stable JSON form used for fingerprints / query-spec identity."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Schema":
        if not isinstance(d, Mapping):
            raise ValueError("schema declaration must be an object")
        allowed = {"columns", "row_id_name", "max_key_bytes"}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"unknown schema field(s): {sorted(unknown)}")
        if "columns" not in d or not isinstance(d["columns"], Sequence) or isinstance(
            d["columns"], (str, bytes)
        ):
            raise ValueError("schema.columns must be a list of column declarations")
        cols = tuple(Column.from_dict(c) for c in d["columns"])
        return cls(
            columns=cols,
            row_id_name=str(d.get("row_id_name", "row_id")),
            max_key_bytes=d.get("max_key_bytes"),
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> "Schema":
        return cls.from_dict(json.loads(text))


class SchemaError(ValueError):
    pass
