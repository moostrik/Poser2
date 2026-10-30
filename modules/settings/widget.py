"""Widget — the panel hint of a Field: which control shows it.

The hint is pure data; the panel (``modules/panel/controls.py``) holds one builder per member.
``Widget.resolve()`` maps ``Widget.default`` to the member a field's type and parameters call for.

Adding a widget: add the member here with its types in ``_TYPES``, register its builder in the
panel, and add it to ``resolve()`` when it should be the pick for a type.
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import TYPE_CHECKING, get_origin, get_args

from modules.utils import Color, Point2f, Rect

if TYPE_CHECKING:
    from .field import Field


class Widget(Enum):
    """The control a field is shown with; ``default`` lets ``resolve()`` pick one.

    List widgets:
        - ``checklist`` — checkboxes only (pick items, no reorder)
        - ``playlist``  — checkboxes + reorder arrows (pick and arrange)
        - ``order``     — reorder arrows only (arrange, no selection)
    """

    default = auto()
    # bool widgets
    switch = auto()
    toggle = auto()
    button = auto()
    status = auto()         # read-only badge: "<label> OK" in green when True, "<label> WARNING" in red when False
    # numeric widgets
    slider = auto()
    log_slider = auto()     # logarithmic slider for values spanning decades; needs min > 0
    number = auto()
    knob = auto()
    number_field = auto()
    # enum widgets
    select = auto()
    radio = auto()
    # string widgets
    input = auto()
    ip_field = auto()
    textarea = auto()
    text_select = auto()
    # value-object widgets
    color = auto()
    color_alpha = auto()
    point2f = auto()
    rect = auto()
    # list widgets
    checklist = auto()
    playlist = auto()
    order = auto()
    number_list = auto()

    @property
    def types(self) -> tuple[type, ...] | None:
        """Compatible field types, or ``None`` for *any type*."""
        return _TYPES[self]

    def accepts(self, field_type: type) -> bool:
        """Return True if this widget is compatible with *field_type*."""
        if self is Widget.default:
            return True
        exact = _EXACT_TYPES.get(self)
        if exact is not None:
            return field_type is exact
        types = _TYPES[self]
        assert types is not None    # None only for default and the exact types, both handled above
        origin = get_origin(field_type)
        if origin is not None:
            field_type = origin  # list[Enum] → list
        # bool is a subclass of int — reject it for numeric widgets
        if field_type is bool and bool not in types:
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

    # -- Log slider mapping --------------------------------------------------

    @staticmethod
    def log_position(value: float) -> float:
        """Log slider position of a value: its base-10 logarithm."""
        return math.log10(value)

    @staticmethod
    def log_value(position: float) -> float:
        """Value at a log slider position, rounded to 3 significant digits."""
        return float(f"{10.0 ** position:.3g}")

    @staticmethod
    def log_step(value: float) -> float:
        """Arrow step of a log slider's number input: a tenth of the value's decade."""
        return float(f"{10.0 ** (math.floor(math.log10(value)) - 1):.1g}")

    # -- Dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Widget.{self.name}"


# The field types each widget shows; None for the ones that take any type or one exact type.
_TYPES: dict[Widget, tuple[type, ...] | None] = {
    Widget.default: None,
    Widget.switch: (bool,),
    Widget.toggle: (bool,),
    Widget.button: (bool,),
    Widget.status: (bool,),
    Widget.slider: (int, float),
    Widget.log_slider: (int, float),
    Widget.number: (int, float),
    Widget.knob: (int, float),
    Widget.number_field: (int, float),
    Widget.select: (Enum,),
    Widget.radio: (Enum,),
    Widget.input: (str,),
    Widget.ip_field: (str,),
    Widget.textarea: (str,),
    Widget.text_select: (str,),
    Widget.color: None,
    Widget.color_alpha: None,
    Widget.point2f: None,
    Widget.rect: None,
    Widget.checklist: (list,),
    Widget.playlist: (list,),
    Widget.order: (list,),
    Widget.number_list: (list,),
}

# The widgets that take exactly one value-object type.
_EXACT_TYPES: dict[Widget, type] = {
    Widget.color: Color,
    Widget.color_alpha: Color,
    Widget.point2f: Point2f,
    Widget.rect: Rect,
}
