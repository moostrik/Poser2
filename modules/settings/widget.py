"""Widget — GUI presentation hints for Field descriptors.

Each Widget member declares which Field value types it is compatible with.
The panel uses ``Widget.resolve()`` to map ``Widget.default`` to a concrete
widget based on the field's type and parameters.

Adding a new widget:
    1. Add member to the ``Widget`` enum below.
    2. Register a builder in ``nice_panel.py`` with ``@widget_builder(Widget.xxx)``.
    3. Optionally add it to ``resolve()`` if it should be the auto-pick for a type.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, get_origin, get_args

if TYPE_CHECKING:
    from modules.settings.field import Field

# Lazy import to avoid circular dependency at module level.
# Special types are only needed at runtime inside accepts() / resolve().
_Color = None
_Point2f = None
_Rect = None

def _get_color():
    global _Color
    if _Color is None:
        from modules.utils.Color import Color
        _Color = Color
    return _Color


def _get_point2f():
    global _Point2f
    if _Point2f is None:
        from modules.utils.PointsAndRects import Point2f
        _Point2f = Point2f
    return _Point2f


def _get_rect():
    global _Rect
    if _Rect is None:
        from modules.utils.PointsAndRects import Rect
        _Rect = Rect
    return _Rect


class Widget(Enum):
    """GUI hint for how a Setting should be rendered in the panel.

    Each member carries a ``types`` tuple declaring compatible Setting
    value types, or ``None`` meaning *any type* (used by ``default``,
    ``color``, ``color_alpha``, ``point2f``, ``rect``).

    List widgets:
        - ``checklist`` — checkboxes only (pick items, no reorder)
        - ``playlist``  — checkboxes + reorder arrows (pick and arrange)
        - ``order``     — reorder arrows only (arrange, no selection)

    The ``resolve(field)`` class method maps ``Widget.default`` to a concrete
    widget based on the field's ``type_``, ``min``, and ``max``.
    """

    # Use __new__ to give each member a unique auto-incrementing int value
    # so that members with the same compatible types (e.g. switch/toggle/button)
    # remain distinct instead of collapsing into aliases.
    # Enum unpacks tuple values as positional args, so we accept *args.
    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__) + 1
        return obj

    def __init__(self, *args) -> None:
        # args == (None,) for 'default = None'; (bool,) for 'switch = (bool,)';
        # (int, float) for 'slider = (int, float)'; etc.
        if len(args) == 1 and args[0] is None:
            self._types = None
        else:
            self._types = args

    # -- Members (argument = compatible types, or None) ----------------------
    # bool widgets
    default     = None
    switch      = (bool,)
    toggle      = (bool,)
    button      = (bool,)
    # numeric widgets
    slider      = (int, float)
    number      = (int, float)
    knob        = (int, float)
    number_field = (int, float)
    # enum widgets
    select      = (Enum,)
    radio       = (Enum,)
    # string widgets
    input       = (str,)
    ip_field          = (str,)
    textarea    = (str,)
    text_select = (str,)
    # color widgets (accepts() uses lazy Color import)
    color       = None
    color_alpha = None
    point2f     = None
    rect        = None
    # list widgets
    checklist   = (list,)
    playlist    = (list,)
    order       = (list,)
    number_list = (list,)

    # -- Public API ----------------------------------------------------------

    @property
    def types(self) -> tuple[type, ...] | None:
        """Compatible field types, or ``None`` for *any type*."""
        return self._types

    def accepts(self, field_type: type) -> bool:
        """Return True if this widget is compatible with *field_type*."""
        types = self._types
        if types is None:
            # Widget.default accepts anything; special widgets need exact type matches.
            if self.name in ("color", "color_alpha"):
                return field_type is _get_color()
            if self.name == "point2f":
                return field_type is _get_point2f()
            if self.name == "rect":
                return field_type is _get_rect()
            return True  # Widget.default accepts anything
        origin = get_origin(field_type)
        if origin is not None:
            field_type = origin  # list[Enum] → list
        # bool is a subclass of int — reject it for numeric widgets
        if field_type is bool and types and bool not in types:
            return False
        for t in types:
            if t is Enum:
                # Accept any Enum subclass
                if isinstance(field_type, type) and issubclass(field_type, Enum):
                    return True
            elif field_type is t:
                return True
            elif isinstance(field_type, type) and issubclass(field_type, t):
                return True
        return False

    @classmethod
    def resolve(cls, field: Field) -> Widget:
        """Map ``Widget.default`` to a concrete widget based on field metadata.

        If the field already has a non-default widget, return it unchanged.
        """
        if field.widget is not cls.default:
            return field.widget

        ft = field.type_
        Color = _get_color()
        Point2f = _get_point2f()
        Rect = _get_rect()

        # Generic list → checklist (enum elements) or number_list (numeric elements)
        if get_origin(ft) is list:
            args = get_args(ft)
            if args and args[0] in (int, float):
                return cls.number_list
            return cls.checklist

        # bool → toggle
        if ft is bool:
            return cls.toggle

        # int / float with min+max → slider, otherwise → number
        if ft in (int, float):
            if field.min is not None and field.max is not None:
                return cls.slider
            return cls.number

        # Enum subclass → select
        if isinstance(ft, type) and issubclass(ft, Enum):
            return cls.select

        # str → input
        if ft is str:
            return cls.input

        # Color → color (no alpha)
        if ft is Color:
            return cls.color

        if ft is Point2f:
            return cls.point2f

        if ft is Rect:
            return cls.rect

        # Fallback — return default (panel renders a read-only label)
        return cls.default

    # -- Dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Widget.{self.name}"

