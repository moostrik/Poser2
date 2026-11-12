"""The settings mirror: a tree described in plain values, and a tree rebuilt from that description.

A process that must not import an app's settings classes (they pull in the app's whole stack) builds a
mirror from ``describe()``: the same instances, fields, hints, labels and wiring, each enum made again from
its members. Values travel as a preset holds them (``Field.encode``) and arrive through ``Field.set``'s
coercion, on either side. A field is addressed by its instance's path, the child names from the root
joined by dots ("" for the root), and its name.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Iterator, get_args, get_origin

from modules.utils import Color, Point2f, Rect

from .base_settings import BaseSettings
from .field import Access, Field
from .group import Group
from .widget import Widget

logger = logging.getLogger(__name__)

Values = dict[str, dict[str, Any]]      # path → field name → encoded value

# The kinds a field's value can have, and a value of each for a rebuilt field's type.
_VALUE_KINDS: dict[type, str] = {
    bool: "bool", int: "int", float: "float", str: "str", Color: "color", Point2f: "point2f", Rect: "rect",
}
_TYPE_VALUES: dict[str, Callable[[], Any]] = {
    "bool": lambda: False, "int": lambda: 0, "float": lambda: 0.0, "str": lambda: "",
    "color": lambda: Color(0.0, 0.0, 0.0), "point2f": lambda: Point2f(0.0, 0.0), "rect": lambda: Rect(0.0, 0.0, 0.0, 0.0),
}
_ELEMENT_KINDS = ("bool", "int", "float", "str", "enum")

# The Field parameters carried as they are, and the ones that name another field of the same instance.
_HINTS = ("min", "max", "step", "description", "label", "unit", "row_label", "width", "visible", "pinned",
          "newline", "color", "saved")
_REFERENCES = ("options", "drag", "seek")

_NO_WIRING: tuple[bool, bool, str | None] = (False, False, None)
_UNSENT = object()      # a value the other side does not hold


class Shown(str):
    """The text of a value whose type the mirror has no kind for; the panel shows it read-only."""


class MirrorSettings(BaseSettings):
    """A mirrored instance: it answers the wiring questions from the description, since the sharing
    itself happens in the tree it mirrors."""

    _wiring: dict[str, tuple[bool, bool, str | None]] = {}     # name → (shared, share source, label)

    def is_shared(self, name: str) -> bool:
        return self._wiring.get(name, _NO_WIRING)[0]

    def is_share_source(self, name: str) -> bool:
        return self._wiring.get(name, _NO_WIRING)[1]

    def wiring_label(self, name: str) -> str | None:
        return self._wiring.get(name, _NO_WIRING)[2]


# -- paths -------------------------------------------------------------------------------

def walk(root: BaseSettings, path: str = "") -> Iterator[tuple[str, BaseSettings]]:
    """Every instance of the tree with its path, the root first, in declaration order."""
    yield path, root
    for name, child in root.children.items():
        yield from walk(child, f"{path}.{name}" if path else name)


def find(root: BaseSettings, path: str) -> BaseSettings:
    """The instance at *path*; raises KeyError when the tree has none there."""
    settings = root
    for name in path.split(".") if path else []:
        settings = settings.children[name]
    return settings


# -- the description ---------------------------------------------------------------------

def _kind(type_: Any) -> tuple[str, str | None, type[Enum] | None]:
    """A field type's kind, its list element's kind and its enum: ("list", "enum", Mode) for list[Mode]."""
    if get_origin(type_) is list:
        args = get_args(type_)
        element, _, enum_type = _kind(args[0]) if args else ("other", None, None)
        if element in _ELEMENT_KINDS:
            return "list", element, enum_type
        return "other", None, None
    if isinstance(type_, type) and issubclass(type_, Enum):
        return "enum", None, type_
    kind = _VALUE_KINDS.get(type_)
    return (kind, None, None) if kind is not None else ("other", None, None)


def encode_value(field: Field, value: Any) -> Any:
    """*value* as it travels: as a preset holds it, or as its text for a type without a kind."""
    if field.type_ is Shown or _kind(field.type_)[0] == "other":
        return str(value)
    return field.encode(value)


def _label(member: Enum) -> str | None:
    label = getattr(member, "label", None)
    return label if isinstance(label, str) else None


def _describe_enum(enum_type: type[Enum], enums: dict[str, Any]) -> str:
    """The enum's key in *enums*, described there once: each member's name, value and label. A value
    that is not an int or a str travels as the member's index; only `min`/`max` filtering reads it."""
    key = f"{enum_type.__module__}.{enum_type.__qualname__}"
    if key not in enums:
        members = list(enum_type)
        plain = all(isinstance(m.value, (int, str)) and not isinstance(m.value, bool) for m in members)
        enums[key] = {
            "name": enum_type.__name__,
            "members": [[m.name, m.value if plain else i, _label(m)] for i, m in enumerate(members)],
        }
    return key


def _describe_field(settings: BaseSettings, name: str, field: Field, enums: dict[str, Any]) -> dict[str, Any]:
    kind, element, enum_type = _kind(field.type_)
    references = {ref: getattr(field, ref) for ref in _REFERENCES}
    return {
        "name": name,
        "kind": kind,
        "element": element,
        "enum": _describe_enum(enum_type, enums) if enum_type is not None else None,
        "access": field.access.name,
        "widget": field.widget.name if kind != "other" else Widget.default.name,
        "default": encode_value(field, field.default),
        "hints": {hint: getattr(field, hint) for hint in _HINTS},
        "references": {ref: other.name if other is not None else None for ref, other in references.items()},
        "wiring": [settings.is_shared(name), settings.is_share_source(name), settings.wiring_label(name)],
    }


def _describe_settings(settings: BaseSettings, enums: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": type(settings).__name__,
        "fields": [_describe_field(settings, name, field, enums) for name, field in settings.fields.items()],
        "children": [[name, _describe_settings(child, enums)] for name, child in settings.children.items()],
    }


def describe(root: BaseSettings) -> dict[str, Any]:
    """The tree in plain values: every instance with its fields (visible or not), their hints and
    wiring, and the enums they use. ``build_mirror`` makes a tree from it; its values are ``values()``."""
    enums: dict[str, Any] = {}
    tree = _describe_settings(root, enums)
    return {"enums": enums, "root": tree}


# -- the mirror --------------------------------------------------------------------------

def _make_enum(description: dict[str, Any]) -> type[Enum]:
    members: list[list[Any]] = description["members"]
    enum_type = Enum(description["name"], [(name, value) for name, value, _ in members])
    labels: dict[str, str] = {name: label for name, _, label in members if label is not None}
    if labels:
        setattr(enum_type, "label", property(lambda member: labels.get(member.name)))
    return enum_type


def _make_field(description: dict[str, Any], enums: dict[str, type[Enum]]) -> Field:
    kind: str = description["kind"]
    type_value: Any
    if kind == "enum":
        type_value = next(iter(enums[description["enum"]]))
    elif kind == "list":
        element: str = description["element"]
        type_value = [next(iter(enums[description["enum"]]))] if element == "enum" else [_TYPE_VALUES[element]()]
    elif kind == "other":
        type_value = Shown("")
    else:
        type_value = _TYPE_VALUES[kind]()
    field: Field = Field(type_value, access=Access[description["access"]], widget=Widget[description["widget"]],
                         **description["hints"])
    default: Any = description["default"]
    field.default = Shown(default) if kind == "other" else field._coerce(default)
    return field


def _make_class(description: dict[str, Any], enums: dict[str, type[Enum]]) -> type[MirrorSettings]:
    attributes: dict[str, Any] = {"__module__": __name__}
    fields: dict[str, Field] = {}
    wiring: dict[str, tuple[bool, bool, str | None]] = {}
    for field_description in description["fields"]:
        name: str = field_description["name"]
        fields[name] = attributes[name] = _make_field(field_description, enums)
        shared, source, label = field_description["wiring"]
        wiring[name] = (shared, source, label)
    for field_description in description["fields"]:
        for ref, other in field_description["references"].items():
            setattr(fields[field_description["name"]], ref, fields.get(other) if other is not None else None)
    for name, child in description["children"]:
        attributes[name] = Group(_make_class(child, enums))
    attributes["_wiring"] = wiring
    return type(description["type"], (MirrorSettings,), attributes)


def build_mirror(description: dict[str, Any]) -> BaseSettings:
    """A tree made from ``describe()``: its instances, fields, hints, labels and wiring, at the fields'
    defaults; ``apply`` gives it the values. Not initialized, so every field takes what it is sent."""
    enums = {key: _make_enum(enum) for key, enum in description["enums"].items()}
    return _make_class(description["root"], enums)()


# -- values ------------------------------------------------------------------------------

def values(root: BaseSettings) -> Values:
    """Every field's value, encoded, by instance path."""
    return {
        path: {name: encode_value(field, getattr(settings, name)) for name, field in settings.fields.items()}
        for path, settings in walk(root)
    }


def apply(root: BaseSettings, values: Values) -> None:
    """Set the encoded *values* on the tree; a value the tree refuses is logged and skipped."""
    for path, fields in values.items():
        try:
            settings = find(root, path)
        except KeyError:
            logger.warning("No settings at '%s'; %d values skipped", path, len(fields))
            continue
        own: dict[str, Field] = settings.fields
        for name, value in fields.items():
            field = own.get(name)
            if field is None:
                logger.warning("No field '%s' at '%s'", name, path)
                continue
            try:
                setattr(settings, name, Shown(value) if field.type_ is Shown else value)
            except (TypeError, ValueError, AttributeError):
                logger.warning("Value %r for '%s' at '%s' refused", value, name, path, exc_info=True)


class ValueDiff:
    """The values of a tree that changed since they were last sent, encoded. Read on one thread."""

    def __init__(self, root: BaseSettings) -> None:
        self._entries: list[tuple[str, str, BaseSettings, Field]] = [
            (path, name, settings, field) for path, settings in walk(root) for name, field in settings.fields.items()
        ]
        self._index: dict[tuple[str, str], int] = {(path, name): i for i, (path, name, _, _) in enumerate(self._entries)}
        self._sent: list[Any] = [getattr(settings, name) for _, name, settings, _ in self._entries]

    def changes(self) -> Values:
        """The values that differ from the last sent, encoded by path; they count as sent from here."""
        changed: Values = {}
        for i, (path, name, settings, field) in enumerate(self._entries):
            value = getattr(settings, name)
            if value != self._sent[i]:
                self._sent[i] = value
                changed.setdefault(path, {})[name] = encode_value(field, value)
        return changed

    def written(self, path: str, name: str, value: Any) -> None:
        """The other side wrote the encoded *value*, and it was set here: what the tree now holds counts
        as sent, unless it is not that value (refused, or coerced to another), which then goes back."""
        i = self._index.get((path, name))
        if i is None:
            return
        _, _, settings, field = self._entries[i]
        held = getattr(settings, name)
        self._sent[i] = held if encode_value(field, held) == value else _UNSENT
