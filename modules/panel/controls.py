"""The controls: one ``Control`` class per Widget, each tying a Field to the NiceGUI elements that show it.

The base class owns what every control shares: the label and tooltip, the read-only state, whether
the control is polled, the commit of a value the user entered and the echo guard that keeps the
poll from pushing that value back. A subclass writes only its elements: ``build`` creates them in
the current slot, ``show`` puts a value into them. ``CONTROLS`` maps each Widget to its class.

Design principles for settings widgets and layout:
- Prefer vanilla NiceGUI and Quasar primitives over custom-composed pseudo-widgets.
- Do not invent new visual widget types when an existing NiceGUI control can
    express the behavior.
- Keep builder logic thin: layout, labels, descriptions, and spacing should be
    centralized instead of reimplemented inside each builder.
- Controls should have one obvious title or label area when the control already
    has one; descriptions belong to that existing title area only, and should
    not cause a new title or label area to be introduced just to host a
    description.
- Avoid absolute positioning, magic offsets, and one-off spacing hacks for
    field metadata.
- Compound controls are allowed only when one vanilla control cannot express
    the behavior cleanly; when used, they should still read like a standard
    field, not a special gadget.
- Reuse shared field structure for label, description, body, and optional
    actions so widgets align consistently.
- Prefer fewer nested rows and columns; use the smallest NiceGUI structure that
    expresses the layout clearly.
- Preserve declaration order and predictable placement.
- Treat mobile collapse behavior as a first-class requirement.
- Prefer simple CSS-based solutions (parent-class toggling, cascading
    selectors) over client-side JavaScript for show/hide and styling changes.
    Resort to ``ui.run_javascript`` only when no CSS or NiceGUI API equivalent
    exists.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Callable, ClassVar, get_args

from nicegui import ui

from modules.settings import BaseSettings, Field, Access, Widget
from modules.utils import Color, Point2f, Rect

from .knob import Knob, CHANGE_THROTTLE

SLIDER_WIDTH: int = 48      # Tailwind width step of a slider without a `width` hint


# -- Shared helpers ----------------------------------------------------------

def generate_label(name: str) -> str:
    """Convert field/config name to a human-readable label.

    Examples:
        "min_cutoff"  -> "Min Cutoff"
        "TCP_PORT"    -> "TCP PORT"
        "x"           -> "X"
    """
    parts = name.split("_")
    result = []
    for part in parts:
        if part.isupper() and len(part) > 1:
            result.append(part)
        else:
            result.append(part.capitalize())
    return " ".join(result)


def access_css_class(field: Field) -> str:
    """Return a CSS marker class for the field's access level."""
    if field.access is Access.INIT:
        return "poser-init"
    if field.access is Access.READ:
        return "poser-feedback"
    return "poser-input"  # WRITE or READWRITE


def format_readout(value: Any) -> str:
    """A read-only value as short text for a pinned chip."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _wiring_css_class(settings: BaseSettings, name: str) -> str:
    """Return a CSS class for the field's wiring state, or empty string."""
    if settings.is_shared(name):
        return "poser-synced"
    if settings.is_share_source(name):
        return "poser-source"
    return ""


def _lock_prop(locked: bool, *, readonly: bool = True) -> str:
    """Return the Quasar prop suffix that locks a read-only control.

    ``readonly`` keeps hover events alive (tooltips) and the value legible;
    ``disable`` sets ``pointer-events:none`` on q-field controls, so it is used
    only for components without a ``readonly`` prop (QToggle, QBtn).
    """
    if not locked:
        return ""
    return " readonly" if readonly else " disable"


def _attach_tooltip(element, description: str | None):
    """Attach a tooltip to a native labeled control when a description exists."""
    if description:
        with element:
            ui.tooltip(description).props('delay=500')
    return element


def _commit_typed(element, commit: Callable[[Any], None], *, on_enter: bool = True) -> None:
    """Commit a typed input's value on commit-style events rather than per keystroke.

    Uses the native ``change`` DOM event, which bubbles (so it reaches NiceGUI even for the
    custom ``ui.input`` component) and fires on defocus-with-change *and* on each spinner-arrow
    click of a ``<input type=number>``. ``keydown.enter`` covers explicit Enter and ``blur`` is a
    harmless backup for the plain ``q-input`` path. All are idempotent — ``Field._apply`` no-ops
    when the value is unchanged."""
    if on_enter:
        element.on("keydown.enter", lambda _e: commit(element.value))
    element.on("change", lambda _e: commit(element.value), [])
    element.on("blur", lambda _e: commit(element.value), [])


def _commit_while_dragging(element, commit: Callable[[Any], None]) -> None:
    """Commit a slider's value as it is dragged, throttled; the last value always arrives."""
    element.on("update:model-value", lambda e: commit(e.args), [None], throttle=CHANGE_THROTTLE)


def _enum_type(field: Field) -> type[Enum] | None:
    """The element type of a ``list[Enum]`` field, None for any other field."""
    args = get_args(field.type_)
    elem = args[0] if args else None
    return elem if isinstance(elem, type) and issubclass(elem, Enum) else None


# -- The base class ------------------------------------------------------------

class Control:
    """One field on the page: a Widget's elements, built in the current slot at construction.

    ``shown`` is the value the control last showed, by ``show`` or ``commit``; the poll compares the
    field with it, so a value the user entered is never pushed back into the control.
    """

    widget: ClassVar[Widget | None] = None      # the Widget this class is registered for

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.widget is not None:
            CONTROLS[cls.widget] = cls

    def __init__(self, settings: BaseSettings, name: str, field: Field) -> None:
        self.settings = settings
        self.name = name
        self.field = field
        self.label = field.label if field.label is not None else generate_label(name)
        self.description = self._wiring_tooltip(field.description)
        self.locked = field.access is Access.READ
        self.polled = self.polls()
        self.shown = self.value()
        self.build()

    # -- what a subclass writes ------------------------------------------------

    def build(self) -> None:
        """Create the elements in the current slot."""
        raise NotImplementedError

    def show(self, value: Any) -> None:
        """Put *value* into the elements."""
        raise NotImplementedError

    def polls(self) -> bool:
        """Whether the page polls this control: a field that something else than the panel may write."""
        return self.settings.is_shared(self.name) or self.field.access is not Access.WRITE

    # -- what every control shares --------------------------------------------

    def value(self) -> Any:
        """What the field holds; a list comes as a copy (``Field.__get__``)."""
        return getattr(self.settings, self.name)

    def commit(self, value: Any) -> None:
        """Write *value* to the field and show what the field holds after it, which is the value
        the poll then expects: the echo guard."""
        setattr(self.settings, self.name, value)
        self.shown = self.value()
        self.show(self.shown)

    def poll(self) -> None:
        """Show the field's value when it changed since the control last showed one."""
        value = self.value()
        if value != self.shown:
            self.shown = value
            self.show(value)

    def _wiring_tooltip(self, description: str) -> str:
        """The description with the field's wiring (shared or source) appended."""
        label = self.settings.wiring_label(self.name)
        if label is None:
            return description
        if description:
            return f"{description}\n({label})"
        return label

    def title(self, *, classes: str = ""):
        """A title label with the description as its hover tooltip."""
        title = ui.label(self.label).classes(classes)
        if self.description:
            with title:
                ui.tooltip(self.description).props('delay=500')
        return title

    def tooltip(self, element):
        """Attach the description as a tooltip to a control that carries its own label."""
        return _attach_tooltip(element, self.description)

    def lock(self, *, readonly: bool = True) -> str:
        """The Quasar prop suffix that locks this control when its field is read-only."""
        return _lock_prop(self.locked, readonly=readonly)

    def step(self) -> int | float:
        """The field's step, or 1 for an int field and 0.01 for a float field."""
        if self.field.step is not None:
            return self.field.step
        return 1 if self.field.type_ is int else 0.01


CONTROLS: dict[Widget, type[Control]] = {}     # filled by Control.__init_subclass__


# -- bool controls -------------------------------------------------------------

class Switch(Control):
    widget = Widget.switch

    def build(self) -> None:
        self.element = self.tooltip(ui.switch(self.label, value=self.shown).props(
            "dense" + self.lock(readonly=False)
        ))
        if not self.locked:
            self.element.on_value_change(lambda e: self.commit(e.value))

    def show(self, value: Any) -> None:
        self.element.set_value(value)


class Status(Control):
    """Read-only good/bad indicator: one badge carrying the field's label, green with OK or red
    with WARNING.

    Both texts sit in the same grid cell and only one is visible, so the badge is always as wide as
    the longer one and does not jump in size when the value flips.
    """
    widget = Widget.status

    def build(self) -> None:
        with ui.badge().classes("text-weight-bold px-2 py-1").style("display: inline-grid") as badge:
            self.ok_text = ui.label(f"{self.label} OK").classes("text-center").style("grid-area: 1 / 1")
            self.warning_text = ui.label(f"{self.label} WARNING").classes("text-center").style("grid-area: 1 / 1")
        self.badge = self.tooltip(badge)
        self.show(self.shown)

    def show(self, value: Any) -> None:
        shown, hidden = (self.ok_text, self.warning_text) if value else (self.warning_text, self.ok_text)
        shown.classes(remove="invisible")
        hidden.classes(add="invisible")
        self.badge._props["color"] = "positive" if value else "negative"
        self.badge.update()


class Toggle(Control):
    widget = Widget.toggle

    def build(self) -> None:
        self.button = self.tooltip(ui.button(self.label).props("dense" + self.lock(readonly=False)))
        self.show(self.shown)
        if not self.locked:
            self.button.on_click(lambda: self.commit(not self.value()))

    def show(self, value: Any) -> None:
        if value:
            self.button.props(remove="outline")
            self.button._props["color"] = "primary"
        else:
            self.button.props(add="outline")
            self.button._props["color"] = "grey"
        self.button.update()


class ActionButton(Control):
    """A button that fires the field's callbacks; it holds no value and is not polled."""
    widget = Widget.button

    def build(self) -> None:
        props = "dense disable" if self.locked else "dense"
        self.button = ui.button(self.label, on_click=lambda: self.field.fire(self.settings)).props(props)
        _attach_tooltip(self.button, self.description or None)
        if self.locked:
            self.button.disable()

    def show(self, value: Any) -> None:
        pass

    def polls(self) -> bool:
        return False


# -- numeric controls ----------------------------------------------------------

def _slider_width(field: Field) -> str:
    return f"w-{field.width}" if field.width is not None else f"w-{SLIDER_WIDTH}"


class Slider(Control):
    """A slider with a number input beside its title; the input commits typed values, clamped."""
    widget = Widget.slider

    def build(self) -> None:
        field = self.field
        step = self.step()
        color = getattr(field, "color", "primary")
        with ui.column().classes(f"{_slider_width(field)} max-w-full gap-1"):
            with ui.row().classes("w-full items-center justify-between flex-nowrap"):
                self.title(classes="flex-1 truncate")
                self.input = ui.number(
                    value=self.shown, step=step, format="%.0f" if field.type_ is int else "%.2f",
                ).props("dense borderless" + self.lock()).classes("w-16")
            self.slider = ui.slider(
                min=field.min, max=field.max, step=step, value=self.shown,
            ).props(f"dense color={color}" + self.lock()).classes("w-full")

        # The relay between the slider and its input, not an echo guard: each one updates the
        # other, and the lock keeps that from going round.
        self._updating = False
        if self.locked:
            return

        def on_slider_change(e):
            if not self._updating:
                self._updating = True
                self.input.set_value(e.value)
                self._updating = False
        self.slider.on_value_change(on_slider_change)

        def commit_slider(val):
            if val is not None:
                self.commit(field.type_(val))
        _commit_while_dragging(self.slider, commit_slider)

        def commit_input(val):
            if self._updating or val is None:
                return
            self._updating = True
            self.commit(max(field.min, min(field.max, field.type_(val))))
            self._updating = False
        _commit_typed(self.input, commit_input)

    def show(self, value: Any) -> None:
        self.slider.set_value(value)
        self.input.set_value(value)


class LogSlider(Control):
    """Slider over log10(min)..log10(max); the number input holds the real value."""
    widget = Widget.log_slider

    def build(self) -> None:
        field = self.field
        color = getattr(field, "color", "primary")
        with ui.column().classes(f"{_slider_width(field)} max-w-full gap-1"):
            with ui.row().classes("w-full items-center justify-between flex-nowrap"):
                self.title(classes="flex-1 truncate")
                # HTML min is 0, not field.min: the browser steps in multiples of step counted
                # from min, so 0 keeps the values round. field.min is enforced on commit.
                self.input = ui.number(
                    value=self.shown, min=0, max=field.max,
                    step=Widget.log_step(self._clamp(self.shown)), format="%.3g",
                ).props("dense borderless" + self.lock()).classes("w-16")
            self.slider = ui.slider(
                min=Widget.log_position(field.min), max=Widget.log_position(field.max),
                step=0.01, value=Widget.log_position(self._clamp(self.shown)),
            ).props(f"dense color={color}" + self.lock()).classes("w-full")

        self._updating = False

        def on_input_change(e):
            # The arrow step follows the value (a tenth of its decade) on every change, not
            # only on commit: a held arrow keeps stepping in the browser without committing.
            if e.value is not None and e.value > 0:
                self.input.props(f"step={Widget.log_step(self._clamp(e.value))}")
        self.input.on_value_change(on_input_change)

        if self.locked:
            return

        def on_slider_change(e):
            if not self._updating:
                self._updating = True
                self.input.set_value(Widget.log_value(e.value))
                self._updating = False
        self.slider.on_value_change(on_slider_change)

        def commit_slider(pos):
            if pos is not None:
                self.commit(field.type_(self._clamp(Widget.log_value(pos))))
        _commit_while_dragging(self.slider, commit_slider)

        def commit_input(val):
            if self._updating or val is None:
                return
            self._updating = True
            self.commit(field.type_(self._clamp(float(f"{val:.3g}"))))
            self.slider.set_value(Widget.log_position(self.shown))
            self._updating = False
        _commit_typed(self.input, commit_input)

    def _clamp(self, v: float) -> float:
        return max(self.field.min, min(self.field.max, v))

    def commit(self, value: Any) -> None:
        """Write the value and show it in the input only: the slider stands where the user put it,
        and the committed value's position lies between its steps."""
        setattr(self.settings, self.name, value)
        self.shown = self.value()
        self.input.set_value(self.shown)

    def show(self, value: Any) -> None:
        self._updating = True
        self.slider.set_value(Widget.log_position(self._clamp(value)))
        self.input.set_value(value)
        self._updating = False


class Number(Control):
    widget = Widget.number

    def build(self) -> None:
        self.input = self.tooltip(ui.number(
            label=self.label, value=self.shown, step=self.step(),
            format="%.0f" if self.field.type_ is int else "%.2f",
        ).props("dense outlined" + self.lock()).classes("w-24"))
        if not self.locked:
            _commit_typed(self.input, lambda val: self.commit(self.field.type_(val)) if val is not None else None)

    def show(self, value: Any) -> None:
        self.input.set_value(value)


class KnobControl(Control):
    """The knob shows its own value while the user turns it; every change is answered with the
    setting's value (``commit`` shows it), so the knob never keeps a value the field refused."""
    widget = Widget.knob

    def build(self) -> None:
        field = self.field
        step = self.step()
        # The knob shows its value as it is, so a value that is not on the step's grid (a read-only
        # knob fed by the app) is shown to the step's decimals and not with a float's tail.
        self.decimals = 0 if field.type_ is int else max(0, -math.floor(math.log10(step))) if step > 0 else 2
        # The title goes into the knob's caption, above the value and beside the dial.
        with Knob(
            self._rounded(self.shown),
            min=field.min if field.min is not None else 0,
            max=field.max if field.max is not None else 100,
            step=step, default=field.default, decimals=self.decimals, readonly=self.locked,
        ) as self.knob:
            self.title(classes="truncate max-w-full")
        if not self.locked:
            def on_change(val):
                if val is not None:
                    self.commit(field.type_(val))
                else:
                    self.show(self.value())
            self.knob.on_change(on_change)

    def _rounded(self, v):
        return self.field.type_(round(v, self.decimals)) if v is not None else v

    def show(self, value: Any) -> None:
        self.knob.set_value(self._rounded(value))


# -- enum controls -------------------------------------------------------------

class Select(Control):
    widget = Widget.select

    def build(self) -> None:
        options = {m: generate_label(m.name) for m in self.field.type_}
        width = f"w-{self.field.width}" if self.field.width is not None else "w-48"
        self.select = self.tooltip(ui.select(options=options, value=self.shown, label=self.label).props(
            "dense outlined" + self.lock()
        ).classes(f"{width} max-w-full"))
        if not self.locked:
            self.select.on_value_change(lambda e: self.commit(e.value))

    def show(self, value: Any) -> None:
        self.select.set_value(value)


class TextSelect(Control):
    """A select over the current list of another field (``options``); polls both, always."""
    widget = Widget.text_select

    def build(self) -> None:
        self.options_field = self.field.options
        self.options = list(getattr(self.settings, self.options_field.name)) if self.options_field else []
        self.select = self.tooltip(ui.select(
            options=self.options, value=self.shown if self.shown in self.options else None, label=self.label,
        ).props("dense outlined" + self.lock()).classes("w-48 max-w-full"))
        if not self.locked:
            def on_change(e):
                if e.value is None:     # the select clears itself when its options change; not a choice
                    return
                self.commit(e.value)
            self.select.on_value_change(on_change)

    def polls(self) -> bool:
        return True

    def poll(self) -> None:
        if self.options_field:
            options = getattr(self.settings, self.options_field.name)
            if options != self.options:
                self.options = options
                self.select.options = options
                self.select.update()
                self.show(self.value())     # re-apply the value the new options may have cleared
        super().poll()

    def show(self, value: Any) -> None:
        self.select.set_value(value if value in self.options else None)


class Radio(Control):
    widget = Widget.radio

    def build(self) -> None:
        options = {m: generate_label(m.name) for m in self.field.type_}
        with ui.column().classes("gap-1"):
            self.title()
            self.toggle = ui.toggle(options=options, value=self.shown).props("dense" + self.lock())
        if not self.locked:
            self.toggle.on_value_change(lambda e: self.commit(e.value))

    def show(self, value: Any) -> None:
        self.toggle.set_value(value)


# -- string controls -----------------------------------------------------------

class Input(Control):
    widget = Widget.input

    def build(self) -> None:
        self.input = self.tooltip(ui.input(label=self.label, value=self.shown).props("dense outlined" + self.lock()))
        if not self.locked:
            _commit_typed(self.input, self.commit)

    def show(self, value: Any) -> None:
        self.input.set_value(value)


class IpField(Control):
    """A text input that only commits a dotted IPv4 address."""
    widget = Widget.ip_field

    @staticmethod
    def is_valid(v: str) -> bool:
        parts = v.split(".")
        return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)

    def build(self) -> None:
        self.input = self.tooltip(ui.input(
            label=self.label, value=self.shown, validation={"": self.is_valid},
        ).props('dense outlined hide-bottom-space' + self.lock()).classes("w-36"))
        if not self.locked:
            _commit_typed(self.input, lambda val: self.commit(val) if self.is_valid(val) else None)

    def show(self, value: Any) -> None:
        self.input.set_value(value)


class NumberField(Control):
    """A text input for a number, validated against the field's type and range."""
    widget = Widget.number_field

    def is_valid(self, v: str) -> bool:
        try:
            n = self.field.type_(v)
        except (ValueError, TypeError):
            return False
        if self.field.min is not None and n < self.field.min:
            return False
        if self.field.max is not None and n > self.field.max:
            return False
        return True

    def build(self) -> None:
        self.input = self.tooltip(ui.input(
            label=self.label, value=str(self.shown), validation={"": self.is_valid},
        ).props('dense outlined hide-bottom-space' + self.lock()).classes("w-24"))
        if not self.locked:
            _commit_typed(self.input, lambda val: self.commit(self.field.type_(val)) if self.is_valid(val) else None)

    def show(self, value: Any) -> None:
        self.input.set_value(str(value))


class Textarea(Control):
    widget = Widget.textarea

    def build(self) -> None:
        self.textarea = self.tooltip(ui.textarea(label=self.label, value=self.shown).props(
            "dense outlined" + self.lock()
        ).classes("w-full"))
        if not self.locked:
            # Enter inserts a newline in a textarea, so commit on focus-out / change only.
            _commit_typed(self.textarea, self.commit, on_enter=False)

    def show(self, value: Any) -> None:
        self.textarea.set_value(value)


# -- color controls ------------------------------------------------------------

def _hex(value: Any) -> str:
    return value.to_hex() if isinstance(value, Color) else '#000000'


class ColorControl(Control):
    """A color input shown in its own color, with black or white text by its luminance."""
    widget = Widget.color

    @staticmethod
    def _style(hex_val: str) -> str:
        c = Color.from_hex(hex_val)
        luminance = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
        text = '#000000' if luminance > 0.5 else '#ffffff'
        return f'background:{hex_val};border-radius:4px;color:{text}'

    def build(self) -> None:
        hex_val = _hex(self.shown)
        self.input = self.tooltip(ui.color_input(label=self.label, value=hex_val).props(
            "dense outlined" + self.lock()
        ).classes("w-36").style(self._style(hex_val)))
        if self.locked:
            self.input.button.disable()  # readonly does not block the picker button
        else:
            def on_change(e):
                if e.value:
                    c = Color.from_hex(e.value)
                    self.commit(Color(c.r, c.g, c.b, 1.0))
            self.input.on_value_change(on_change)

    def show(self, value: Any) -> None:
        hex_val = _hex(value)
        self.input.set_value(hex_val)
        self.input.style(self._style(hex_val))


class ColorAlpha(Control):
    """A color input and a number input for the alpha, behind one value."""
    widget = Widget.color_alpha

    def build(self) -> None:
        with ui.column().classes("w-full gap-1"):
            self.title()
            with ui.row().classes("items-end gap-2"):
                self.input = ui.color_input(label="Color", value=_hex(self.shown)).props(
                    "dense outlined" + self.lock()
                ).classes("w-36")
                self.alpha = ui.number(
                    label="A", value=self.shown.a if isinstance(self.shown, Color) else 1.0,
                    min=0.0, max=1.0, step=0.01, format="%.2f",
                ).props("dense outlined" + self.lock()).classes("w-24")
        if self.locked:
            self.input.button.disable()  # readonly does not block the picker button
            return

        def on_color_change(e):
            if e.value:
                c = Color.from_hex(e.value)
                self.commit(Color(c.r, c.g, c.b, float(self.alpha.value or 1.0)))
        self.input.on_value_change(on_color_change)

        def commit_alpha(val):
            if val is not None:
                cur = self.value()
                self.commit(Color(cur.r, cur.g, cur.b, float(val)))
        _commit_typed(self.alpha, commit_alpha)

    def show(self, value: Any) -> None:
        if isinstance(value, Color):
            self.input.set_value(value.to_hex())
            self.alpha.set_value(value.a)


# -- list controls -------------------------------------------------------------

class SortableList(Control):
    """A list of enum members as rows: checkboxes pick them, arrows order them.

    The rows are rebuilt from the control's own order on every change, since the order of the
    inactive members is more than the field's value and would be lost in a rebuild from it.
    """
    widget = None
    with_checkboxes: ClassVar[bool] = True
    with_order: ClassVar[bool] = True

    def build(self) -> None:
        self.members = list(_enum_type(self.field))
        with ui.column().classes("w-72 max-w-full gap-1"):
            with ui.row().classes("w-full items-center gap-1"):
                self.title(classes="flex-1")
            self.container = ui.column().classes("w-full gap-0 border rounded p-1")
        self._take(self.shown)
        self._rebuild()

    def _take(self, value: list) -> None:
        """Set the order and the active set from a field value."""
        self.active = set(value) if self.with_checkboxes else set(self.members)
        if self.with_checkboxes and self.with_order:
            # Playlist: active first, then inactive
            self.order = list(value) + [m for m in self.members if m not in self.active]
        elif self.with_checkboxes:
            # Checklist: stable enum order, no reordering
            self.order = list(self.members)
        else:
            # Order-only: all members always active, ordered as in value
            self.order = list(value)

    def _rebuild(self) -> None:
        self.container.clear()
        with self.container:
            for idx, member in enumerate(self.order):
                is_active = member in self.active
                with ui.row().classes(
                    "w-full items-center gap-2 px-2 py-0.5 rounded"
                    + ("" if self.locked else " cursor-move")
                    + ("" if is_active else " opacity-50")
                ).style(
                    "background: color-mix(in srgb, var(--q-primary) 15%, transparent);" if is_active else ""
                ):
                    if self.with_checkboxes and not self.locked:
                        ui.checkbox(
                            value=is_active, on_change=lambda e, m=member: self._check(m, e.value),
                        ).props("dense").classes("my-0")
                    ui.label(generate_label(member.name)).classes("flex-1")
                    if self.with_order and not self.locked:
                        up = ui.button(
                            icon="arrow_upward", on_click=lambda _, i=idx: self._move(i, -1),
                        ).props("dense flat size=xs").classes("my-0")
                        down = ui.button(
                            icon="arrow_downward", on_click=lambda _, i=idx: self._move(i, 1),
                        ).props("dense flat size=xs").classes("my-0")
                        if idx == 0:
                            up.props("disable")
                        if idx == len(self.order) - 1:
                            down.props("disable")

    def _check(self, member: Enum, active: bool) -> None:
        if active:
            self.active.add(member)
        else:
            self.active.discard(member)
        self._apply()

    def _move(self, idx: int, direction: int) -> None:
        new_idx = idx + direction
        if 0 <= new_idx < len(self.order):
            self.order[idx], self.order[new_idx] = self.order[new_idx], self.order[idx]
            self._apply()

    def _apply(self) -> None:
        if self.with_checkboxes and self.with_order:
            # Playlist: active items first, then inactive (preserving relative order)
            self.order = [m for m in self.order if m in self.active] + [m for m in self.order if m not in self.active]
        self.commit([m for m in self.order if m in self.active])

    def commit(self, value: Any) -> None:
        """Write the value and rebuild from the control's own order, inactive members included."""
        setattr(self.settings, self.name, value)
        self.shown = self.value()
        self._rebuild()

    def show(self, value: Any) -> None:
        self._take(value)
        self._rebuild()


class Checklist(SortableList):
    widget = Widget.checklist
    with_order = False


class Playlist(SortableList):
    widget = Widget.playlist


class Order(SortableList):
    widget = Widget.order
    with_checkboxes = False


class NumberList(Control):
    """A fixed-length list of numbers as a row of number inputs, each committing on change."""
    widget = Widget.number_list

    def build(self) -> None:
        args = get_args(self.field.type_)
        self.elem_type = args[0] if args else float
        step = self.field.step if self.field.step is not None else (1 if self.elem_type is int else 0.01)
        fmt = "%.0f" if self.elem_type is int else "%.2f"
        self.inputs: list = []
        with ui.column().classes("w-72 max-w-full gap-1"):
            self.title()
            with ui.column().classes("w-full gap-0 border rounded p-1"):
                for i, v in enumerate(self.shown):
                    with ui.row().classes("w-full items-center gap-2 px-2 py-0.5 rounded").style(
                        "background: color-mix(in srgb, var(--q-primary) 15%, transparent);"
                    ):
                        num = ui.number(
                            value=v, step=step, format=fmt, min=self.field.min, max=self.field.max,
                        ).props("dense borderless" + self.lock()).classes("flex-1")
                        self.inputs.append(num)
                        if not self.locked:
                            _commit_typed(num, lambda val, i=i: self._commit_item(i, val))

    def _commit_item(self, index: int, val) -> None:
        if val is None:
            return
        values = self.value()
        values[index] = self.elem_type(val)
        self.commit(values)

    def show(self, value: Any) -> None:
        for num, v in zip(self.inputs, value):
            num.set_value(v)


class Point2fControl(Control):
    """Two number inputs, X and Y, behind one value."""
    widget = Widget.point2f

    def build(self) -> None:
        value = self.shown if isinstance(self.shown, Point2f) else Point2f(0.0, 0.0)
        with ui.column().classes("w-full gap-1"):
            self.title()
            with ui.row().classes("items-end gap-2"):
                self.x = self._input("X", value.x)
                self.y = self._input("Y", value.y)
        if not self.locked:
            _commit_typed(self.x, lambda v: self.commit(Point2f(float(v), self.value().y)) if v is not None else None)
            _commit_typed(self.y, lambda v: self.commit(Point2f(self.value().x, float(v))) if v is not None else None)

    def _input(self, label: str, value: float):
        return ui.number(label=label, value=value, step=0.01, format="%.3f").props(
            "dense outlined" + self.lock()
        ).classes("w-24")

    def show(self, value: Any) -> None:
        if isinstance(value, Point2f):
            self.x.set_value(value.x)
            self.y.set_value(value.y)


class RectControl(Control):
    """Four number inputs, X, Y, W and H, behind one value."""
    widget = Widget.rect

    def build(self) -> None:
        value = self.shown if isinstance(self.shown, Rect) else Rect(0.0, 0.0, 0.0, 0.0)
        with ui.column().classes("w-full gap-1"):
            self.title()
            with ui.row().classes("items-end gap-2 flex-wrap"):
                self.x = self._input("X", value.x)
                self.y = self._input("Y", value.y)
                self.w = self._input("W", value.width)
                self.h = self._input("H", value.height)
        if self.locked:
            return

        def commit_part(part: str, v) -> None:
            if v is None:
                return
            cur = self.value()
            parts = {"x": cur.x, "y": cur.y, "w": cur.width, "h": cur.height}
            parts[part] = float(v)
            self.commit(Rect(parts["x"], parts["y"], parts["w"], parts["h"]))
        for part, element in (("x", self.x), ("y", self.y), ("w", self.w), ("h", self.h)):
            _commit_typed(element, lambda v, p=part: commit_part(p, v))

    def _input(self, label: str, value: float):
        return ui.number(label=label, value=value, step=0.01, format="%.3f").props(
            "dense outlined" + self.lock()
        ).classes("w-24")

    def show(self, value: Any) -> None:
        if isinstance(value, Rect):
            self.x.set_value(value.x)
            self.y.set_value(value.y)
            self.w.set_value(value.width)
            self.h.set_value(value.height)


# -- the controls no Widget names ----------------------------------------------

class Fallback(Control):
    """A read-only label for a type without a control of its own."""
    widget = Widget.default

    def build(self) -> None:
        with ui.row().classes("items-center gap-2"):
            self.title()
            self.text = ui.label(str(self.shown)).classes("text-secondary")

    def show(self, value: Any) -> None:
        self.text.set_text(str(value))


class InitField(Control):
    """An init-only field as its title and value; it cannot change, so it is not polled."""

    def build(self) -> None:
        with ui.row().classes("items-center gap-2 poser-init"):
            self.title()
            ui.label(str(self.shown)).classes("text-secondary italic")

    def show(self, value: Any) -> None:
        pass

    def polls(self) -> bool:
        return False


class PinnedReadout(Control):
    """A read-only pinned field as one compact chip, "<Label> <value> <unit>", always polled."""

    def build(self) -> None:
        self.unit = f" {self.field.unit}" if self.field.unit else ""
        self.chip = self.tooltip(ui.label(self._text(self.shown)).classes("bg-grey-8 rounded px-2 py-1"))

    def _text(self, value: Any) -> str:
        return f"{self.label} {format_readout(value)}{self.unit}"

    def show(self, value: Any) -> None:
        self.chip.set_text(self._text(value))

    def polls(self) -> bool:
        return True


# -- the page's entry points -----------------------------------------------------

def control_class(field: Field) -> type[Control]:
    """The class for a field's resolved Widget; a list of anything but an Enum gets the fallback."""
    cls = CONTROLS.get(Widget.resolve(field), Fallback)
    if issubclass(cls, SortableList) and _enum_type(field) is None:
        return Fallback
    return cls


def build_control(settings: BaseSettings, name: str, field: Field) -> Control:
    """Build the control of one field in the current slot."""
    return control_class(field)(settings, name, field)


def build_entry(settings: BaseSettings, name: str, field: Field) -> Control:
    """Build one visible field of a card: the control in a div carrying its access and wiring classes."""
    css = access_css_class(field)
    wiring = _wiring_css_class(settings, name)
    if wiring:
        css = f"{css} {wiring}"
    if Widget.resolve(field) is Widget.textarea:
        css = f"{css} w-full"   # a text area takes a whole row of its card
    if field.widget == Widget.button:
        with ui.element("div").classes(css):
            return ActionButton(settings, name, field)
    if field.access is Access.INIT:
        return InitField(settings, name, field)
    with ui.element("div").classes(css):
        return build_control(settings, name, field)
