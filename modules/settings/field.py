"""Field descriptor with type coercion, callbacks, and thread safety."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar, overload, Any, cast, get_origin, get_args

from .widget import Widget

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FieldAlias:
    """A field shared to a child under another name, made by `Field.as_('child_name')` in a `Group`'s share list."""
    field: 'Field'
    child_name: str

T = TypeVar("T")


def _enum_member(enum_type: type[Enum], value: Any) -> Enum:
    """The member of *enum_type* named *value* (a str) or holding it; raises TypeError when there is none."""
    if isinstance(value, str):
        try:
            return enum_type[value]
        except KeyError:
            raise TypeError(f"Cannot construct {enum_type.__name__} from name {value!r}") from None
    try:
        return enum_type(value)
    except (ValueError, KeyError):
        raise TypeError(f"Cannot construct {enum_type.__name__} from {value!r}") from None


def _encode_one(value: Any) -> Any:
    """One value as a settings file holds it: a value object as its dict, an Enum by its name, the rest as it is."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.name
    return value


class Access(Enum):
    """Who writes a field, and whether the panel lets the user edit it (SETTINGS.md, Access)."""
    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"
    INIT = "init"


class Field(Generic[T]):
    """One value of a settings class: its type, from the default, its coerced writes and its callbacks; every
    keyword but `default`, `access` and `saved` is a panel hint (SETTINGS.md, Fields)."""

    # Convenience aliases so callers write Field.READ instead of Access.READ
    READ = Access.READ
    WRITE = Access.WRITE
    READWRITE = Access.READWRITE
    INIT = Access.INIT

    def __init__(
        self,
        default: T,
        *,
        min: float | int | None = None,   # UI hint only — not enforced by the descriptor
        max: float | int | None = None,   # UI hint only — not enforced by the descriptor
        step: float | int | None = None,  # UI hint only — not enforced by the descriptor
        description: str = "",
        label: str | None = None,        # UI hint only — the control's title in the panel
        unit: str | None = None,         # UI hint only — shown after the value in pinned chips (e.g. 'ms', 'fps')
        row_label: str | None = None,    # UI hint only — a text title rendered before this field
        width: int | None = None,        # UI hint only — Tailwind width step for a fixed-width control
        access: Access = Access.WRITE,
        visible: bool = True,
        pinned: bool = False,
        newline: bool = False,
        widget: Widget = Widget.default,
        color: str = "primary",  # UI color hint (e.g. 'primary', 'red', '#00f')
        options: Field | None = None,  # the Field a control reads beside its own (text_select, timeline)
        drag: Field | None = None,     # the Field a control writes while dragged (timeline)
        seek: Field | None = None,     # the Field a control writes when pressed beside a handle (timeline)
        saved: bool = True,            # False: left out of to_dict, so a settings file does not hold it
    ) -> None:
        # Infer type from default value
        if isinstance(default, list):
            if not default:
                raise ValueError(
                    "List defaults must contain at least one element "
                    "so the element type can be inferred."
                )
            elem_type = type(default[0])
            type_ = cast(type[T], list[elem_type])
        elif isinstance(default, bool):
            # bool check must come before int (bool is a subclass of int)
            type_ = cast(type[T], bool)
        else:
            # Covers Enum subclasses and plain types (int, float, str, Color, …)
            type_ = cast(type[T], type(default))

        self.type_: type[T] = type_
        self.default: T = default
        self.min = min
        self.max = max
        self.step = step
        self.description = description
        self.label = label
        self.unit = unit
        self.row_label = row_label
        self.width = width
        self.access = access
        self.visible = visible
        self.pinned = pinned
        self.newline = newline
        self.widget = widget
        self.color = color
        self.options: Field | None = options
        self.drag: Field | None = drag
        self.seek: Field | None = seek
        self.saved: bool = saved
        # Validate widget ↔ type compatibility
        if widget is not Widget.default and not widget.accepts(type_):
            raise TypeError(
                f"Widget.{widget.name} is not compatible with type "
                f"{getattr(type_, '__name__', repr(type_))}"
            )
        # Generic list support: list[int], list[str], etc.
        self._origin = get_origin(type_)          # list | None
        self._element_type = get_args(type_)[0] if get_args(type_) else None  # int | str | …
        # Set by __set_name__
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name

    # -- Descriptor protocol -------------------------------------------------

    @overload
    def __get__(self, obj: None, objtype: type) -> Field[T]: ...
    @overload
    def __get__(self, obj: Any, objtype: type) -> T: ...
    def __get__(self, obj: Any | None, objtype: type | None = None) -> Field[T] | T:
        if obj is None:
            return self  # class-level access returns the descriptor itself
        value = obj._values[self.name]
        # Return a shallow copy of lists to prevent silent in-place mutation
        # (in-place changes would bypass __set__ and not fire callbacks).
        if isinstance(value, list):
            return cast(T, list(value))
        return value

    def __set__(self, obj: Any, value: T) -> None:
        self.set(obj, value)

    # -- Public set (respects Access.INIT) ----------------------------------

    def set(self, obj, value) -> None:
        """Set value. Only enforces Access.INIT after initialization."""
        if self.access is Access.INIT and obj._initialized:
            raise AttributeError(
                f"Field '{self.name}' can only be set during initialization"
            )
        self._apply(obj, value)

    def _apply(self, obj, value, from_parent: bool = False) -> None:
        """Write *value* unconditionally: coerce, store, share it with the children and, unless it came from the
        parent, with the parent; then call the callbacks."""
        value = self._coerce(value)
        changed = False
        callbacks: list = []
        with obj._locks[self.name]:
            if obj._values[self.name] != value:
                obj._values[self.name] = value
                callbacks = list(obj._callbacks[self.name])
                changed = True
        if changed:
            obj._propagate_shared_field(self.name)
            if not from_parent:
                obj._propagate_upward_field(self.name)
        self._call(callbacks, value)

    def _call(self, callbacks: list, value: Any) -> None:
        """Call each of *callbacks* with *value*, outside the lock; a failing one is logged and the others go on."""
        for callback in callbacks:
            try:
                callback(value)
            except Exception:
                logger.warning("Callback %r for setting '%s' raised an exception", callback, self.name, exc_info=True)

    # -- Type coercion -------------------------------------------------------

    def _coerce(self, value):
        """Coerce *value* to self.type_, or raise TypeError."""
        # Generic list: list[int], list[str], etc.
        if self._origin is list:
            if not isinstance(value, list):
                raise TypeError(
                    f"Field '{self.name}' expects list, "
                    f"got {type(value).__name__}: {value!r}"
                )
            if self._element_type is not None:
                return [self._coerce_element(v) for v in value]
            return list(value)
        # Reject bool when int or float is expected (bool is a subclass of int)
        if self.type_ in (int, float) and isinstance(value, bool):
            raise TypeError(
                f"Field '{self.name}' expects {self._type_name}, got bool: {value!r}"
            )
        if isinstance(value, self.type_):
            return value
        # Promote int → float (lossless)
        if self.type_ is float and isinstance(value, int):
            return float(value)
        if isinstance(self.type_, type) and issubclass(self.type_, Enum):
            return _enum_member(self.type_, value)
        # tuple/list → positional-arg construction (e.g. Color(r, g, b), Point2f(x, y))
        if isinstance(value, (tuple, list)):
            try:
                return self.type_(*value)
            except Exception:
                raise TypeError(
                    f"Cannot construct {self._type_name} from {type(value).__name__}: {value!r}"
                )
        if isinstance(value, dict) and hasattr(self.type_, "from_dict"):
            try:
                return cast(Any, self.type_).from_dict(value)
            except (TypeError, ValueError, KeyError, AttributeError) as exc:
                raise TypeError(
                    f"Cannot construct {self._type_name} from dict: {value!r}"
                ) from exc
        raise TypeError(
            f"Field '{self.name}' expects {self._type_name}, "
            f"got {type(value).__name__}: {value!r}"
        )

    def _coerce_element(self, v):
        """Coerce a single list element to self._element_type."""
        assert self._element_type is not None
        et: type = self._element_type
        # Reject bool when int or float is expected (bool is a subclass of int)
        if et in (int, float) and isinstance(v, bool):
            raise TypeError(
                f"Field '{self.name}' list element expects {et.__name__}, "
                f"got bool: {v!r}"
            )
        if isinstance(v, et):
            return v
        # Promote int → float for float lists
        if et is float and isinstance(v, int):
            return float(v)
        # Promote float → int only when lossless (e.g. 3.0 → 3, but reject 3.7)
        if et is int and isinstance(v, float):
            iv = int(v)
            if v != iv:
                raise TypeError(
                    f"Field '{self.name}' list element expects int, "
                    f"got non-integer float: {v!r}"
                )
            return iv
        if isinstance(et, type) and issubclass(et, Enum):
            return _enum_member(et, v)
        return et(v)

    # -- Callbacks -----------------------------------------------------------

    def fire(self, obj, from_parent: bool = False) -> None:
        """Press a button: its callbacks are called with ``True`` and its value stays; the press is shared with the
        children and, unless it came from the parent, with the parent."""
        with obj._locks[self.name]:
            callbacks = list(obj._callbacks[self.name])
        self._call(callbacks, True)
        for child, child_name in obj._share_down.get(self.name, []):
            if child_name in child._fields:
                child._fields[child_name].fire(child, from_parent=True)
        if not from_parent:
            upward = obj._share_up.get(self.name)
            if upward is not None:
                parent, parent_name = upward
                if parent_name in parent._fields:
                    parent._fields[parent_name].fire(parent)

    def bind(self, obj, callback) -> None:
        with obj._locks[self.name]:
            if callback not in obj._callbacks[self.name]:
                obj._callbacks[self.name].append(callback)

    def unbind(self, obj, callback) -> None:
        with obj._locks[self.name]:
            try:
                obj._callbacks[self.name].remove(callback)
            except ValueError:
                pass

    # -- Serialization -------------------------------------------------------

    def to_json_value(self, obj) -> Any:
        """Return a JSON-serializable representation of the current value."""
        return self.encode(obj._values[self.name])

    def encode(self, value: Any) -> Any:
        """*value* as a settings file holds it: an Enum by its name, a value object as its dict, a list
        element by element. ``set`` takes it back."""
        if not isinstance(value, list):
            return _encode_one(value)
        result = [_encode_one(v) for v in value]
        for v in result:
            if not isinstance(v, (int, float, str, bool, dict, type(None))):
                logger.warning("Field '%s': list element %r (type %s) may not be JSON-serializable",
                               self.name, v, type(v).__name__)
        return result

    def from_json_value(self, obj, raw) -> None:
        """Restore value from JSON, bypassing the INIT guard.

        Callers (``update_from_dict``) decide whether INIT fields should
        be written; this method just applies the value unconditionally.
        Shared child fields may appear in JSON, but parent re-propagation is
        still the source of truth after the load completes.
        """
        self._apply(obj, raw)

    # -- Repr ----------------------------------------------------------------

    @property
    def _type_name(self) -> str:
        if self._origin is not None:
            return repr(self.type_)  # e.g. "list[int]"
        return getattr(self.type_, '__name__', repr(self.type_))

    def __repr__(self):
        parts = [self._type_name, f"default={self.default!r}"]
        if self.description:
            parts.append(f"description={self.description!r}")
        if self.access is not Access.WRITE:
            parts.append(f"access=Field.{self.access.name}")
        if not self.visible:
            parts.append("visible=False")
        if self.widget != Widget.default:
            parts.append(f"widget={self.widget!r}")
        return f"Field({', '.join(parts)})"

    def as_(self, child_name: str) -> FieldAlias:
        """This field, shared to a child as *child_name*."""
        return FieldAlias(self, child_name)
