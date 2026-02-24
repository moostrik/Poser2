"""Widget — GUI presentation hints for Field descriptors.

Each Widget constant declares which Field value types it is compatible with.
The panel uses ``Widget.resolve()`` to map ``Widget.default`` to a concrete
widget based on the field's type and parameters.

Adding a new widget:
    1. Add a ``ClassVar[Widget]`` annotation + assignment below.
    2. Register a builder in ``nice_panel.py`` with ``@widget_builder(Widget.xxx)``.
    3. Optionally add it to ``_DEFAULTS`` if it should be the auto-pick for a type.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, ClassVar, get_origin

if TYPE_CHECKING:
    from modules.settings.field import Field


class WidgetSize(Enum):
    """Layout size returned by panel builders."""
    full = "full"    # needs a full row (slider, text input, checklist)
    small = "small"  # compact inline control (switch, select, number)

# Lazy import to avoid circular dependency at module level.
# Color is only needed at runtime inside accepts() / resolve().
_Color = None

def _get_color():
    global _Color
    if _Color is None:
        from modules.utils.Color import Color
        _Color = Color
    return _Color


class Widget:
    """GUI hint for how a Setting should be rendered in the panel.

    Each constant carries a ``types`` tuple declaring compatible Setting
    value types.  ``None`` means *any type* (used only by ``Widget.default``).

    The ``resolve(field)`` class method maps ``Widget.default`` to a concrete
    widget based on the field's ``type_``, ``min``, and ``max``.
    """

    # -- ClassVar annotations for Pylance autocompletion ---------------------
    # bool widgets
    default:      ClassVar[Widget]
    switch:       ClassVar[Widget]
    toggle:       ClassVar[Widget]
    button:       ClassVar[Widget]
    # numeric widgets
    slider:       ClassVar[Widget]
    number:       ClassVar[Widget]
    knob:         ClassVar[Widget]
    # enum widgets
    select:       ClassVar[Widget]
    radio:        ClassVar[Widget]
    # string widgets
    input:        ClassVar[Widget]
    ip:           ClassVar[Widget]
    textarea:     ClassVar[Widget]
    # color widgets
    color:        ClassVar[Widget]
    color_alpha:  ClassVar[Widget]
    # list widgets
    checklist:    ClassVar[Widget]
    order:        ClassVar[Widget]

    # -- Instance -----------------------------------------------------------

    __slots__ = ("_name", "_types")

    def __init__(self, name: str, types: tuple[type, ...] | None = None) -> None:
        self._name = name
        self._types = types

    # -- Public API ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def types(self) -> tuple[type, ...] | None:
        return self._types

    def accepts(self, field_type: type) -> bool:
        """Return True if this widget is compatible with *field_type*."""
        if self._types is None:
            # Widget.default accepts anything; color/color_alpha need special handling
            if self._name in ("color", "color_alpha"):
                return field_type is _get_color()
            return True  # Widget.default accepts anything
        origin = get_origin(field_type)
        if origin is not None:
            field_type = origin  # list[Enum] → list
        # bool is a subclass of int — reject it for numeric widgets
        if field_type is bool and self._types and bool not in self._types:
            return False
        for t in self._types:
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

        # Generic list → checklist
        if get_origin(ft) is list:
            return cls.checklist

        # bool → switch
        if ft is bool:
            return cls.switch

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

        # Fallback — return default (panel renders a read-only label)
        return cls.default

    # -- Dunder --------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Widget):
            return self._name == other._name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"Widget.{self._name}"


# -- Singleton constants -----------------------------------------------------
# Each is created once; equality/hash is by name.

Widget.default     = Widget("default",     None)
Widget.switch      = Widget("switch",      (bool,))
Widget.toggle      = Widget("toggle",      (bool,))
Widget.button      = Widget("button",      (bool,))
Widget.slider      = Widget("slider",      (int, float))
Widget.number      = Widget("number",      (int, float))
Widget.knob        = Widget("knob",        (int, float))
Widget.select      = Widget("select",      (Enum,))
Widget.radio       = Widget("radio",       (Enum,))
Widget.input       = Widget("input",       (str,))
Widget.ip          = Widget("ip",          (str,))
Widget.textarea    = Widget("textarea",    (str,))
# Color types use a lazy getter so we don't import at module level
Widget.color       = Widget("color",       None)  # accepts() overridden by resolve()
Widget.color_alpha = Widget("color_alpha", None)  # accepts() overridden by resolve()
Widget.checklist   = Widget("checklist",   (list,))
Widget.order       = Widget("order",       (list,))
