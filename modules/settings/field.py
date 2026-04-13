"""Field descriptor with type coercion, callbacks, and thread safety."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar, overload, Any, cast, get_origin, get_args

from modules.settings.widget import Widget

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FieldAlias:
    """A Field reference with an alternate name for sharing to a child.

    Created via ``Field.as_('child_name')`` and used in share lists::

        frequency = Field(30.0)
        interpolator = InterpolatorSettings(share=[frequency.as_('input_frequency')])

    The parent field ``frequency`` is shared to the child as ``input_frequency``.
    """
    field: 'Field'
    child_name: str

T = TypeVar("T")


class Access(Enum):
    """Controls who may read/write a Field and whether the UI polls it.

    ========== ============ ======== ==========================================
    Member     UI editable  Polled   Use case
    ========== ============ ======== ==========================================
    READ       no           yes      Code writes, UI displays (e.g. FPS)
    WRITE      yes          no       UI is the only writer (most fields)
    READWRITE  yes          yes      Both code and UI write (e.g. window size)
    INIT       static label no       Set once at construction, then immutable
    ========== ============ ======== ==========================================
    """
    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"
    INIT = "init"


class Field(Generic[T]):
    """Descriptor-based setting field with type coercion, callbacks, and thread safety.

    Declare as a class attribute on a Settings subclass::

        exposure = Field(1000, min=100, max=10000)

    The type is always inferred from the default value.  For list settings,
    the element type is inferred from the first element::

        tags = Field(["default"])
        modes = Field([RenderMode.SOLID])

    List defaults must contain at least one element for type inference.

    The descriptor handles get/set, type enforcement, init-only guards,
    callbacks, and JSON serialization.

    **Parameters and GUI hints**

    Only *default* affects the runtime behaviour of the class that uses the
    setting.  All other keyword parameters are metadata consumed by the UI
    layer (``nice_panel.py``) and are never enforced by the descriptor itself:

    - ``min``, ``max``, ``step`` — range hints for sliders; the descriptor
      does **not** clamp or validate values against them.
    - ``description`` — tooltip / label text.
    - ``visible`` — whether the field appears in the panel.
    - ``pinned`` — whether the field is shown in a pinned summary section.
        - ``newline`` — start this field on a new compact UI row.
    - ``widget`` — explicit widget override (validated for type compatibility
      at construction time, but has no effect on get/set).
    - ``color`` — UI colour hint for the control.
    - ``options`` — a Field holding a ``list[str]`` of options for
      ``Widget.text_select``.

    **Action buttons (``widget=Widget.button``)**

    A button is a stateless action trigger, declared as ``Field[bool]`` by
    convention::

        start = Field(False, widget=Widget.button, description="Start recording")

    The stored ``bool`` value is never meaningful — it stays ``False``
    forever.  Pressing the button calls ``field.fire(settings)``, which
    invokes all registered callbacks with ``True`` without changing the
    value.  Buttons are excluded from serialization (``to_dict``).
    If the button field is shared to a child via ``Group(..., share=[])``,
    ``fire()`` propagates downstream automatically.

    The one keyword parameter with a runtime effect is ``access``:

    - ``access=Field.INIT`` prevents writes after ``settings.initialize()``
      has been called (raises ``AttributeError``).
    - ``access=Field.READ`` is a UI hint only — code can still write the
      field freely at runtime.

    Custom types used as Field values should implement ``__eq__`` so that
    change detection (skip callback when value unchanged) works correctly.

    .. note:: In-place mutation limitation

        Modifying a mutable value in place (e.g. ``settings.tags.append(x)``)
        will **not** trigger callbacks because the descriptor ``__set__`` is
        never invoked.  To fire callbacks, re-assign the whole value::

            settings.tags = settings.tags + [x]
    """

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
        access: Access = Access.READWRITE,
        visible: bool = True,
        pinned: bool = False,
        newline: bool = False,
        widget: Widget = Widget.default,
        color: str = "primary",  # UI color hint (e.g. 'primary', 'red', '#00f')
        options: Field | None = None,  # Field holding list[str] options (for text_select)
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
        self.access = access
        self.visible = visible
        self.pinned = pinned
        self.newline = newline
        self.widget = widget
        self.color = color
        self.options: Field | None = options
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

    def set(self, obj, value):
        """Set value. Only enforces Access.INIT after initialization."""
        if self.access is Access.INIT and obj._initialized:
            raise AttributeError(
                f"Field '{self.name}' can only be set during initialization"
            )
        self._apply(obj, value)

    def _apply(self, obj, value):
        """Write *value* unconditionally (coerce, store, fire callbacks)."""
        value = self._coerce(value)
        lock = obj._locks[self.name]
        callbacks_to_fire = []
        changed = False

        with lock:
            if obj._values[self.name] != value:
                obj._values[self.name] = value
                callbacks_to_fire = list(obj._callbacks[self.name])
                changed = True

        if changed:
            obj._propagate_shared_field(self.name)
            # Upward propagation: child → parent (only when not already propagating)
            if not obj._is_propagating:
                obj._propagate_upward_field(self.name)

        # Fire callbacks outside the lock
        for cb in callbacks_to_fire:
            try:
                cb(value)
            except Exception:
                logger.warning(
                    "Callback %r for setting '%s' raised an exception",
                    cb, self.name, exc_info=True,
                )

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
        # Enum: reconstruct from name (str) or value (int, etc.)
        if isinstance(self.type_, type) and issubclass(self.type_, Enum):
            if isinstance(value, str):
                try:
                    return self.type_[value]
                except KeyError:
                    raise TypeError(
                        f"Cannot construct {self._type_name} from name {value!r}"
                    )
            try:
                return self.type_(value)
            except (ValueError, KeyError):
                raise TypeError(
                    f"Cannot construct {self._type_name} from {value!r}"
                )
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
                return self.type_.from_dict(value)  # type: ignore[union-attr]
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
        et: type = self._element_type  # type: ignore[assignment]
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
        # Enum element: reconstruct from name (str) or value (int)
        if isinstance(et, type) and issubclass(et, Enum):
            if isinstance(v, str):
                try:
                    return et[v]
                except KeyError:
                    raise TypeError(f"Cannot construct {et.__name__} from name {v!r}")
            try:
                return et(v)
            except (ValueError, KeyError):
                raise TypeError(f"Cannot construct {et.__name__} from {v!r}")
        return et(v)

    # -- Callbacks -----------------------------------------------------------

    def fire(self, obj):
        """Invoke all registered callbacks (for Widget.button actions).

        Unlike set(), fire() does not change the value — it triggers every
        callback with ``True``.  Button callbacks therefore always receive
        ``True``; this keeps the ``callback(value)`` signature uniform across
        all field types, but callers should document which fields are buttons.
        """
        lock = obj._locks[self.name]
        with lock:
            callbacks = list(obj._callbacks[self.name])
        for cb in callbacks:
            try:
                cb(True)
            except Exception:
                logger.warning(
                    "Action callback %r for '%s' raised an exception",
                    cb, self.name, exc_info=True,
                )
        # Propagate to shared children
        for child, child_name in getattr(obj, '_share_down', {}).get(self.name, []):
            if child_name in child._fields:
                previous = child._is_propagating
                object.__setattr__(child, '_is_propagating', True)
                try:
                    child._fields[child_name].fire(child)
                finally:
                    object.__setattr__(child, '_is_propagating', previous)
        # Propagate upward to parent (skip if fire arrived from parent)
        if not obj._is_propagating:
            upward = getattr(obj, '_share_up', {}).get(self.name)
            if upward is not None:
                parent, parent_name = upward
                if parent_name in parent._fields:
                    parent._fields[parent_name].fire(parent)

    def bind(self, obj, callback):
        with obj._locks[self.name]:
            if callback not in obj._callbacks[self.name]:
                obj._callbacks[self.name].append(callback)

    def unbind(self, obj, callback):
        with obj._locks[self.name]:
            try:
                obj._callbacks[self.name].remove(callback)
            except ValueError:
                pass

    # -- Serialization -------------------------------------------------------

    def to_json_value(self, obj):
        """Return a JSON-serializable representation of the current value."""
        value = obj._values[self.name]
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, list):
            result = []
            for v in value:
                if isinstance(v, Enum):
                    result.append(v.name)
                elif hasattr(v, 'to_dict'):
                    result.append(v.to_dict())
                elif isinstance(v, (int, float, str, bool, type(None))):
                    result.append(v)
                else:
                    logger.warning(
                        "Field '%s': list element %r (type %s) may not be "
                        "JSON-serializable",
                        self.name, v, type(v).__name__,
                    )
                    result.append(v)
            return result
        return value

    def from_json_value(self, obj, raw):
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
        """Return an alias for sharing this field under a different name.

        Usage::

            frequency = Field(30.0)
            interp = InterpolatorSettings(share=[frequency.as_('input_frequency')])

        The parent's ``frequency`` value is shared to the child as ``input_frequency``.
        """
        return FieldAlias(self, child_name)
