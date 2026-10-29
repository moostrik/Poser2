"""The controls: one builder per Widget, each turning a Field into a NiceGUI control.

Each builder takes ``(settings, name, field, polls)`` and appends a poll entry
``(settings, name, [last_value], show)`` for every control that follows the setting's value;
``poll.tick`` pushes changed values into the controls (see ``poll.py``).

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
from typing import Callable, get_args

from nicegui import ui

from modules.settings import Field, Access, Widget
from modules.utils import Color, Point2f, Rect

from .knob import Knob, CHANGE_THROTTLE


def generate_label(name):
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


def _field_label(name: str, field: Field) -> str:
    """The control's title: the field's own ``label`` when it has one, else its name made readable."""
    return field.label if field.label is not None else generate_label(name)


def _build_field_title(label: str, description: str | None, *, classes: str = ""):
    """Create a shared field title area with an optional hover description."""
    title = ui.label(label).classes(classes)
    if description:
        with title:
            ui.tooltip(description).props('delay=500')
    return title


def _attach_description_tooltip(element, description: str | None):
    """Attach a tooltip to a native labeled control when a description exists."""
    if description:
        with element:
            ui.tooltip(description).props('delay=500')
    return element


def _is_field_read_only(settings, name: str, field: Field) -> bool:
    """Return True when a field should render read-only in this settings instance."""
    return field.access is Access.READ


def _lock_prop(locked: bool, *, readonly: bool = True) -> str:
    """Return the Quasar prop suffix that locks a read-only control.

    ``readonly`` keeps hover events alive (tooltips) and the value legible;
    ``disable`` sets ``pointer-events:none`` on q-field controls, so it is used
    only for components without a ``readonly`` prop (QToggle, QBtn).
    """
    if not locked:
        return ""
    return " readonly" if readonly else " disable"


def _field_needs_poll(settings, name: str, field: Field) -> bool:
    """Return True when UI should poll external updates for this field."""
    if settings.is_shared(name):
        return True
    return field.access is not Access.WRITE


def _wiring_css_class(settings, name: str) -> str:
    """Return a CSS class for the field's wiring state, or empty string."""
    if settings.is_shared(name):
        return "poser-synced"
    if settings.is_share_source(name):
        return "poser-source"
    return ""


def _wiring_tooltip(settings, name: str, description: str) -> str:
    """Append wiring info to a field's description for the tooltip."""
    label = settings.wiring_label(name)
    if label is None:
        return description
    if description:
        return f"{description}\n({label})"
    return label


def _build_field_header(
    label: str,
    description: str | None,
    *,
    title_classes: str = "",
    row_classes: str = "w-full items-center justify-between gap-2",
    actions: Callable[[], None] | None = None,
    action_classes: str = "items-center gap-0",
):
    """Create a shared header row for compound fields with title and optional actions."""
    with ui.row().classes(row_classes):
        _build_field_title(label, description, classes=title_classes)
        if actions is not None:
            with ui.row().classes(action_classes):
                actions()


def _build_init_field(settings, name: str, field: Field) -> None:
    """Render a single init-only field without forcing it into a separate section."""
    with ui.row().classes("items-center gap-2 poser-init"):
        _build_field_title(_field_label(name, field), _wiring_tooltip(settings, name, field.description))
        ui.label(str(getattr(settings, name))).classes("text-secondary italic")


def access_css_class(field: Field) -> str:
    """Return a CSS marker class for the field's access level."""
    if field.access is Access.INIT:
        return "poser-init"
    if field.access is Access.READ:
        return "poser-feedback"
    return "poser-input"  # WRITE or READWRITE


def build_entry(settings, name: str, field: Field, polls) -> None:
    """Render one visible field in declaration order."""
    css = access_css_class(field)
    wiring = _wiring_css_class(settings, name)
    if wiring:
        css = f"{css} {wiring}"
    if Widget.resolve(field) is Widget.textarea:
        css = f"{css} w-full"   # a text area takes a whole row of its card
    if field.widget == Widget.button:
        with ui.element("div").classes(css):
            build_action_button(settings, name, field)
        return
    if field.access is Access.INIT:
        _build_init_field(settings, name, field)
        return
    with ui.element("div").classes(css):
        build_control(settings, name, field, polls)



def build_control(settings, name, field, polls):
    """Create a NiceGUI control for a single Setting field.

    *polls* collects ``(settings, name, [last_value], setter)`` tuples.
    A timer created by the caller will poll these periodically to push
    external changes into the UI — thread-safe by construction.
    """
    resolved = Widget.resolve(field)
    builder = _BUILDERS.get(resolved)
    if builder is not None:
        return builder(settings, name, field, polls)
    return _build_fallback(settings, name, field, polls)


# ---------------------------------------------------------------------------
# Builder registry — maps Widget constants to NiceGUI builder functions.
# Each builder receives (settings, name, field, polls) and emits its control.
# Use @widget_builder(Widget.xxx) to register.
# ---------------------------------------------------------------------------

_BUILDERS: dict[Widget, Callable] = {}


def widget_builder(widget: Widget):
    """Decorator: register a panel builder for a Widget constant."""
    def decorator(fn: Callable) -> Callable:
        _BUILDERS[widget] = fn
        return fn
    return decorator


def _commit_typed(element, commit, *, on_enter=True):
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


def _commit_while_dragging(element, commit):
    """Commit a slider's value as it is dragged, throttled; the last value always arrives."""
    element.on("update:model-value", lambda e: commit(e.args), [None], throttle=CHANGE_THROTTLE)


# -- bool builders -----------------------------------------------------------

@widget_builder(Widget.switch)
def _build_switch(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)

    sw = _attach_description_tooltip(ui.switch(label, value=value).props(
        "dense" + _lock_prop(is_disabled, readonly=False)
    ), desc)

    if not is_disabled:
        def on_switch_change(e):
            setattr(settings, name, e.value)
        sw.on_value_change(on_switch_change)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, sw=sw: sw.set_value(v)))


@widget_builder(Widget.status)
def _build_status(settings, name, field, polls):
    """Read-only good/bad indicator: one badge carrying the field's label, green with OK or red
    with WARNING.

    Both texts sit in the same grid cell and only one is visible, so the badge is always as wide as
    the longer one and does not jump in size when the value flips.
    """
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)

    with ui.badge().classes("text-weight-bold px-2 py-1").style("display: inline-grid") as badge:
        ok_text = ui.label(f"{label} OK").classes("text-center").style("grid-area: 1 / 1")
        warning_text = ui.label(f"{label} WARNING").classes("text-center").style("grid-area: 1 / 1")
    _attach_description_tooltip(badge, desc)

    def _apply_style(ok: bool):
        shown, hidden = (ok_text, warning_text) if ok else (warning_text, ok_text)
        shown.classes(remove="invisible")
        hidden.classes(add="invisible")
        badge._props["color"] = "positive" if ok else "negative"
        badge.update()

    _apply_style(value)

    polls.append((settings, name, [value], lambda v: _apply_style(v)))


@widget_builder(Widget.toggle)
def _build_toggle(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)

    def _apply_style(btn, active: bool):
        if active:
            btn.props(remove="outline")
            btn._props["color"] = "primary"
        else:
            btn.props(add="outline")
            btn._props["color"] = "grey"
        btn.update()

    btn = _attach_description_tooltip(ui.button(label).props(
        "dense" + _lock_prop(is_disabled, readonly=False)
    ), desc)
    _apply_style(btn, value)

    if not is_disabled:
        def on_click(b=btn):
            new_val = not getattr(settings, name)
            setattr(settings, name, new_val)
            _apply_style(b, new_val)
        btn.on_click(on_click)

    if _field_needs_poll(settings, name, field):
        def _poll_toggle(v, b=btn):
            _apply_style(b, v)
        polls.append((settings, name, [value], _poll_toggle))


# -- numeric builders --------------------------------------------------------

SLIDER_WIDTH: int = 48


def _slider_width(field) -> str:
    return f"w-{field.width}" if field.width is not None else f"w-{SLIDER_WIDTH}"



@widget_builder(Widget.slider)
def _build_slider(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)
    step = field.step if field.step is not None else (1 if field.type_ is int else 0.01)
    color = getattr(field, "color", "primary")

    with ui.column().classes(f"{_slider_width(field)} max-w-full gap-1"):
        with ui.row().classes("w-full items-center justify-between flex-nowrap"):
            _build_field_title(label, desc, classes="flex-1 truncate")
            fmt = "%.0f" if field.type_ is int else "%.2f"
            val_input = ui.number(
                value=value,
                step=step,
                format=fmt,
            ).props(
                "dense borderless"
                + _lock_prop(is_disabled)
            ).classes("w-16")
        sl = ui.slider(
            min=field.min, max=field.max, step=step, value=value
        ).props(
            f"dense color={color}"
            + _lock_prop(is_disabled)
        ).classes("w-full")

    _updating = {"lock": False}
    # The poll's last-seen value: a commit records what it wrote, so the poll does not push it
    # back into the slider while the user is still dragging.
    last = [value]

    if not is_disabled:
        def on_slider_change(e):
            if not _updating["lock"]:
                _updating["lock"] = True
                val_input.set_value(e.value)
                _updating["lock"] = False
        sl.on_value_change(on_slider_change)

        def commit_slider(val):
            if val is None:
                return
            setattr(settings, name, field.type_(val))
            last[0] = getattr(settings, name)
        _commit_while_dragging(sl, commit_slider)

        def commit_input(val):
            if _updating["lock"] or val is None:
                return
            _updating["lock"] = True
            clamped = max(field.min, min(field.max, field.type_(val)))
            setattr(settings, name, clamped)
            last[0] = getattr(settings, name)
            sl.set_value(clamped)
            val_input.set_value(clamped)
            _updating["lock"] = False
        _commit_typed(val_input, commit_input)

    if _field_needs_poll(settings, name, field):
        def _poll_slider(v, _sl=sl, _vi=val_input):
            _sl.set_value(v)
            _vi.set_value(v)
        polls.append((settings, name, last, _poll_slider))


@widget_builder(Widget.log_slider)
def _build_log_slider(settings, name, field, polls):
    """Slider over log10(min)..log10(max); the number input holds the real value."""
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)
    color = getattr(field, "color", "primary")

    def clamp(v: float) -> float:
        return max(field.min, min(field.max, v))

    with ui.column().classes(f"{_slider_width(field)} max-w-full gap-1"):
        with ui.row().classes("w-full items-center justify-between flex-nowrap"):
            _build_field_title(label, desc, classes="flex-1 truncate")
            # HTML min is 0, not field.min: the browser steps in multiples of step counted
            # from min, so 0 keeps the values round. field.min is enforced on commit.
            val_input = ui.number(
                value=value,
                min=0,
                max=field.max,
                step=Widget.log_step(clamp(value)),
                format="%.3g",
            ).props(
                "dense borderless"
                + _lock_prop(is_disabled)
            ).classes("w-16")
        sl = ui.slider(
            min=Widget.log_position(field.min),
            max=Widget.log_position(field.max),
            step=0.01,
            value=Widget.log_position(clamp(value)),
        ).props(
            f"dense color={color}"
            + _lock_prop(is_disabled)
        ).classes("w-full")

    _updating = {"lock": False}

    def set_input(v: float) -> None:
        val_input.set_value(v)

    def on_input_change(e):
        # The arrow step follows the value (a tenth of its decade) on every change, not
        # only on commit: a held arrow keeps stepping in the browser without committing.
        if e.value is not None and e.value > 0:
            val_input.props(f"step={Widget.log_step(clamp(e.value))}")
    val_input.on_value_change(on_input_change)

    # The poll's last-seen value: a commit records what it wrote, so the poll does not push it
    # back into the slider while the user is still dragging.
    last = [value]

    if not is_disabled:
        def on_slider_change(e):
            if not _updating["lock"]:
                _updating["lock"] = True
                set_input(Widget.log_value(e.value))
                _updating["lock"] = False
        sl.on_value_change(on_slider_change)

        def commit_slider(pos):
            if pos is None:
                return
            setattr(settings, name, field.type_(clamp(Widget.log_value(pos))))
            last[0] = getattr(settings, name)
        _commit_while_dragging(sl, commit_slider)

        def commit_input(val):
            if _updating["lock"] or val is None:
                return
            _updating["lock"] = True
            clamped = field.type_(clamp(float(f"{val:.3g}")))
            setattr(settings, name, clamped)
            last[0] = getattr(settings, name)
            sl.set_value(Widget.log_position(clamped))
            set_input(clamped)
            _updating["lock"] = False
        _commit_typed(val_input, commit_input)

    if _field_needs_poll(settings, name, field):
        def _poll_log_slider(v, _sl=sl):
            _updating["lock"] = True
            _sl.set_value(Widget.log_position(clamp(v)))
            set_input(v)
            _updating["lock"] = False
        polls.append((settings, name, last, _poll_log_slider))


@widget_builder(Widget.number)
def _build_number(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)

    num = _attach_description_tooltip(ui.number(
        label=label,
        value=value,
        step=field.step if field.step is not None else (1 if field.type_ is int else 0.01),
        format="%.0f" if field.type_ is int else "%.2f",
    ).props("dense outlined" + _lock_prop(is_disabled)).classes("w-24"), desc)

    if not is_disabled:
        def commit_num(val):
            if val is not None:
                setattr(settings, name, field.type_(val))
        _commit_typed(num, commit_num)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, num=num: num.set_value(v)))


@widget_builder(Widget.knob)
def _build_knob(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)
    step = field.step if field.step is not None else (1 if field.type_ is int else 0.01)
    min_val = field.min if field.min is not None else 0
    max_val = field.max if field.max is not None else 100
    # The knob shows its value as it is, so a value that is not on the step's grid (a read-only
    # knob fed by the app) is shown to the step's decimals and not with a float's tail.
    decimals = 0 if field.type_ is int else max(0, -math.floor(math.log10(step))) if step > 0 else 2

    def shown(v):
        return field.type_(round(v, decimals)) if v is not None else v

    # The title goes into the knob's caption, above the value and beside the dial.
    with Knob(
        shown(value), min=min_val, max=max_val, step=step,
        default=field.default, decimals=decimals, readonly=is_disabled,
    ) as kn:
        _build_field_title(label, desc, classes="truncate max-w-full")

    # The poll's last-seen value: a change records what it wrote, so the poll does not echo it.
    last = [value]

    if not is_disabled:
        # Answer every change with the setting's value: the knob shows its own value until then.
        def commit_knob(val):
            if val is not None:
                setattr(settings, name, field.type_(val))
            last[0] = getattr(settings, name)
            kn.set_value(shown(last[0]))
        kn.on_change(commit_knob)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, last, lambda v, kn=kn: kn.set_value(shown(v))))


# -- enum builders -----------------------------------------------------------

@widget_builder(Widget.select)
def _build_select(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)

    options = {m: generate_label(m.name) for m in field.type_}
    width = f"w-{field.width}" if field.width is not None else "w-48"
    sel = _attach_description_tooltip(ui.select(
        options=options,
        value=value,
        label=label,
    ).props("dense outlined" + _lock_prop(is_disabled)).classes(f"{width} max-w-full"), desc)

    if not is_disabled:
        def on_select_change(e):
            setattr(settings, name, e.value)
        sel.on_value_change(on_select_change)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, s=sel: s.set_value(v)))


@widget_builder(Widget.text_select)
def _build_text_select(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)

    # Resolve the options Field → current list[str] value
    options_field = field.options
    option_list = getattr(settings, options_field.name) if options_field else []
    sel = _attach_description_tooltip(ui.select(
        options=option_list,
        value=value if value in option_list else None,
        label=label,
    ).props("dense outlined" + _lock_prop(is_disabled)).classes("w-48 max-w-full"), desc)

    if not is_disabled:
        def on_select_change(e):
            if e.value is None:         # the select clears itself when its options change; not a choice
                return
            setattr(settings, name, e.value)
        sel.on_value_change(on_select_change)

    polls.append((settings, name, [value], lambda v, s=sel: s.set_value(v)))
    # Add a custom poll for the options field too; re-apply the value the new options may have cleared
    if options_field:
        def on_options_change(v, s=sel):
            s.options = v
            s.update()
            current = getattr(settings, name)
            s.set_value(current if current in v else None)
        polls.append((settings, options_field.name, [list(option_list)], on_options_change))


@widget_builder(Widget.radio)
def _build_radio(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)

    options = {m: generate_label(m.name) for m in field.type_}
    with ui.column().classes("gap-1"):
        _build_field_title(label, desc)
        rg = ui.toggle(options=options, value=value).props(
            "dense" + _lock_prop(is_disabled)
        )

    if not is_disabled:
        def on_radio_change(e):
            setattr(settings, name, e.value)
        rg.on_value_change(on_radio_change)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, rg=rg: rg.set_value(v)))


# -- string builders ---------------------------------------------------------

@widget_builder(Widget.input)
def _build_input(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)

    inp = _attach_description_tooltip(ui.input(label=label, value=value).props(
        "dense outlined" + _lock_prop(is_disabled)
    ), desc)

    if not is_disabled:
        def commit_input(val):
            setattr(settings, name, val)
        _commit_typed(inp, commit_input)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, inp=inp: inp.set_value(v)))


@widget_builder(Widget.ip_field)
def _build_ip(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)

    def is_valid_ip(v: str) -> bool:
        parts = v.split(".")
        return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)

    inp = _attach_description_tooltip(ui.input(
        label=label, value=value,
        validation={"": is_valid_ip},
    ).props(
        'dense outlined hide-bottom-space' + _lock_prop(is_disabled)
    ).classes("w-36"), desc)

    if not is_disabled:
        def commit_ip(val):
            if is_valid_ip(val):
                setattr(settings, name, val)
        _commit_typed(inp, commit_ip)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, inp=inp: inp.set_value(v)))


@widget_builder(Widget.number_field)
def _build_number_input(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)
    lo = field.min
    hi = field.max
    ft = field.type_

    def is_valid(v: str) -> bool:
        try:
            n = ft(v)
            if lo is not None and n < lo:
                return False
            if hi is not None and n > hi:
                return False
            return True
        except (ValueError, TypeError):
            return False

    inp = _attach_description_tooltip(ui.input(
        label=label, value=str(value),
        validation={"": is_valid},
    ).props(
        'dense outlined hide-bottom-space' + _lock_prop(is_disabled)
    ).classes("w-24"), desc)

    if not is_disabled:
        def commit_field(val):
            if is_valid(val):
                setattr(settings, name, ft(val))
        _commit_typed(inp, commit_field)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, inp=inp: inp.set_value(str(v))))


@widget_builder(Widget.textarea)
def _build_textarea(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)

    ta = _attach_description_tooltip(ui.textarea(label=label, value=value).props(
        "dense outlined" + _lock_prop(is_disabled)
    ).classes("w-full"), desc)

    if not is_disabled:
        # Enter inserts a newline in a textarea, so commit on focus-out / change only.
        _commit_typed(ta, lambda val: setattr(settings, name, val), on_enter=False)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, ta=ta: ta.set_value(v)))


# -- color builders ----------------------------------------------------------

@widget_builder(Widget.color)
def _build_color(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)

    def _color_style(hex_val: str) -> str:
        c = Color.from_hex(hex_val)
        luminance = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
        text = '#000000' if luminance > 0.5 else '#ffffff'
        return f'background:{hex_val};border-radius:4px;color:{text}'

    hex_val = value.to_hex() if isinstance(value, Color) else '#000000'

    ci = _attach_description_tooltip(ui.color_input(
        label=label, value=hex_val
    ).props("dense outlined" + _lock_prop(is_disabled)).classes("w-36").style(
        _color_style(hex_val)
    ), desc)

    if is_disabled:
        ci.button.disable()  # readonly does not block the picker button
    else:
        def on_color_change(e):
            if e.value:
                c = Color.from_hex(e.value)
                setattr(settings, name, Color(c.r, c.g, c.b, 1.0))
                ci.style(_color_style(e.value))
        ci.on_value_change(on_color_change)

    if _field_needs_poll(settings, name, field):
        def _poll_color(v, _ci=ci):
            h = v.to_hex() if isinstance(v, Color) else '#000000'
            _ci.set_value(h)
            _ci.style(_color_style(h))
        polls.append((settings, name, [value], _poll_color))


@widget_builder(Widget.color_alpha)
def _build_color_alpha(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    is_disabled = _is_field_read_only(settings, name, field)

    hex_val = value.to_hex() if isinstance(value, Color) else '#000000'
    alpha = value.a if isinstance(value, Color) else 1.0

    with ui.column().classes("w-full gap-1"):
        _build_field_title(label, _wiring_tooltip(settings, name, field.description))
        with ui.row().classes("items-end gap-2"):
            ci = ui.color_input(
                label="Color", value=hex_val
            ).props("dense outlined" + _lock_prop(is_disabled)).classes("w-36")

            alpha_num = ui.number(
                label="A", value=alpha,
                min=0.0, max=1.0, step=0.01,
                format="%.2f",
            ).props("dense outlined" + _lock_prop(is_disabled)).classes("w-24")

    if is_disabled:
        ci.button.disable()  # readonly does not block the picker button
    else:
        def on_color_change(e):
            if e.value:
                c = Color.from_hex(e.value)
                cur_alpha = alpha_num.value or 1.0
                setattr(settings, name, Color(c.r, c.g, c.b, float(cur_alpha)))
        ci.on_value_change(on_color_change)

        def commit_alpha(val):
            if val is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Color(cur.r, cur.g, cur.b, float(val)))
        _commit_typed(alpha_num, commit_alpha)

    def _color_alpha_setter(v, _ci=ci, _an=alpha_num):
        if isinstance(v, Color):
            _ci.set_value(v.to_hex())
            _an.set_value(v.a)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], _color_alpha_setter))


# -- list builders -----------------------------------------------------------

def _build_sortable_list(settings, name, field, polls, *, with_checkboxes: bool, with_order: bool = True):
    """Shared implementation for checklist, playlist, and order widgets."""
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)

    elem_type = get_args(field.type_)[0] if get_args(field.type_) else None
    if elem_type is None or not (isinstance(elem_type, type) and issubclass(elem_type, Enum)):
        # Fallback for non-enum lists
        return _build_fallback(settings, name, field, polls)

    all_members = list(elem_type)
    active_set = set(value)

    with ui.column().classes("w-72 max-w-full gap-1"):
        _build_field_header(
            label,
            desc,
            title_classes="flex-1",
            row_classes="w-full items-center gap-1",
        )
        container = ui.column().classes("w-full gap-0 border rounded p-1")

        if with_checkboxes and with_order:
            # Playlist: active first, then inactive
            ordered = list(value) + [m for m in all_members if m not in active_set]
        elif with_checkboxes:
            # Checklist: stable enum order, no reordering
            ordered = list(all_members)
        else:
            # Order-only: all members always active, ordered as in value
            ordered = list(value)
        state = {
            "order": ordered,
            "active": set(value) if with_checkboxes else set(all_members),
        }

        def _rebuild(cont, st, _settings=settings, _name=name, _elem=elem_type, _cb=with_checkboxes, _ord=with_order, _ro=is_disabled):
            cont.clear()
            with cont:
                visible_members = list(st["order"])
                for idx, member in enumerate(visible_members):
                    is_active = member in st["active"]
                    with ui.row().classes(
                        "w-full items-center gap-2 px-2 py-0.5 rounded"
                        + ("" if _ro else " cursor-move")
                        + ("" if is_active else " opacity-50")
                    ).style(
                        "background: color-mix(in srgb, var(--q-primary) 15%, transparent);"
                        if is_active else ""
                    ):
                        if _cb and not _ro:
                            cb_widget = ui.checkbox(
                                value=is_active,
                            ).props("dense").classes("my-0")

                        ui.label(generate_label(member.name)).classes("flex-1")

                        if _ord and not _ro:
                            # Find the real index in st["order"] for move operations
                            real_idx = st["order"].index(member)

                            up_btn = ui.button(
                                icon="arrow_upward",
                                on_click=lambda _, ri=real_idx: _move(cont, st, ri, -1),
                            ).props("dense flat size=xs").classes("my-0")
                            down_btn = ui.button(
                                icon="arrow_downward",
                                on_click=lambda _, ri=real_idx: _move(cont, st, ri, 1),
                            ).props("dense flat size=xs").classes("my-0")

                            if idx == 0:
                                up_btn.props("disable")
                            if idx == len(visible_members) - 1:
                                down_btn.props("disable")

                        if _cb and not _ro:
                            def _on_check(e, m=member):
                                if e.value:
                                    st["active"].add(m)
                                else:
                                    st["active"].discard(m)
                                _apply(cont, st)
                            cb_widget.on_value_change(_on_check)

        def _move(cont, st, idx, direction):
            new_idx = idx + direction
            if 0 <= new_idx < len(st["order"]):
                st["order"][idx], st["order"][new_idx] = st["order"][new_idx], st["order"][idx]
                _apply(cont, st)

        def _apply(cont, st):
            # Playlist: active items first, then inactive (preserving relative order)
            if with_checkboxes and with_order:
                st["order"] = (
                    [m for m in st["order"] if m in st["active"]]
                    + [m for m in st["order"] if m not in st["active"]]
                )
            new_list = [m for m in st["order"] if m in st["active"]]
            setattr(settings, name, new_list)
            _rebuild(cont, st)

        _rebuild(container, state)

    def _list_setter(v, _cont=container, _st=state, _elem=elem_type, _cb=with_checkboxes, _ord=with_order):
        active_set_inner = set(v)
        _st["active"] = active_set_inner
        if _cb and _ord:
            # Playlist: active first, then inactive
            _st["order"] = list(v) + [m for m in list(_elem) if m not in active_set_inner]
        elif _cb:
            # Checklist: keep stable enum order
            _st["order"] = list(_elem)
        else:
            _st["order"] = list(v)
        _rebuild(_cont, _st)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [list(value)], _list_setter))


@widget_builder(Widget.checklist)
def _build_checklist(settings, name, field, polls):
    return _build_sortable_list(settings, name, field, polls, with_checkboxes=True, with_order=False)


@widget_builder(Widget.playlist)
def _build_playlist(settings, name, field, polls):
    return _build_sortable_list(settings, name, field, polls, with_checkboxes=True, with_order=True)


@widget_builder(Widget.order)
def _build_order(settings, name, field, polls):
    return _build_sortable_list(settings, name, field, polls, with_checkboxes=False, with_order=True)


@widget_builder(Widget.number_list)
def _build_number_list(settings, name, field, polls):
    """Render a fixed-length list of numbers as styled rows (matches sortable list look)."""
    value = getattr(settings, name)
    label = _field_label(name, field)
    desc = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)
    elem_type = get_args(field.type_)[0] if get_args(field.type_) else float
    step = field.step if field.step is not None else (1 if elem_type is int else 0.01)
    fmt = "{:.0f}" if elem_type is int else "{:.2f}"

    labels: list = []
    with ui.column().classes("w-72 max-w-full gap-1"):
        _build_field_title(label, desc)
        with ui.column().classes("w-full gap-0 border rounded p-1"):
            for i, v in enumerate(value):
                with ui.row().classes(
                    "w-full items-center gap-2 px-2 py-0.5 rounded"
                ).style(
                    "background: color-mix(in srgb, var(--q-primary) 15%, transparent);"
                ):
                    lbl = ui.label(fmt.format(v)).classes("flex-1")
                    labels.append(lbl)

                    if not is_disabled:
                        def on_click(e, idx=i, _lbl=lbl):
                            cur = getattr(settings, name)
                            slot = _lbl.parent_slot
                            if slot is None:
                                return
                            parent = slot.parent

                            parent.clear()
                            with parent:
                                num = ui.number(
                                    value=cur[idx], step=step,
                                    min=field.min, max=field.max,
                                ).props("dense autofocus borderless").classes("flex-1")

                                def on_blur(_e, _idx=idx, _parent=parent, _num=num):
                                    val = _num.value
                                    if val is not None:
                                        cur2 = getattr(settings, name)
                                        cur2[_idx] = elem_type(val)
                                        setattr(settings, name, cur2)
                                    _parent.clear()
                                    with _parent:
                                        new_lbl = ui.label(fmt.format(getattr(settings, name)[_idx])).classes("flex-1")
                                        new_lbl.on("click", lambda _e, _i=_idx, _nl=new_lbl: on_click(_e, _i, _nl))
                                        labels[_idx] = new_lbl
                                num.on("blur", on_blur)
                        lbl.on("click", on_click)

    if _field_needs_poll(settings, name, field):
        def _poll_number_list(v, _labels=labels, _fmt=fmt):
            for idx, lbl in enumerate(_labels):
                if idx < len(v):
                    lbl.set_text(_fmt.format(v[idx]))
        polls.append((settings, name, [list(value)], _poll_number_list))


@widget_builder(Widget.point2f)
def _build_point2f(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    is_disabled = _is_field_read_only(settings, name, field)

    with ui.column().classes("w-full gap-1"):
        _build_field_title(label, _wiring_tooltip(settings, name, field.description))
        with ui.row().classes("items-end gap-2"):
            x_num = ui.number(
                label="X", value=value.x if isinstance(value, Point2f) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + _lock_prop(is_disabled)).classes("w-24")
            y_num = ui.number(
                label="Y", value=value.y if isinstance(value, Point2f) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + _lock_prop(is_disabled)).classes("w-24")

    if not is_disabled:
        def commit_x(val):
            if val is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Point2f(float(val), cur.y))
        _commit_typed(x_num, commit_x)

        def commit_y(val):
            if val is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Point2f(cur.x, float(val)))
        _commit_typed(y_num, commit_y)

    def _point_setter(v, _x=x_num, _y=y_num):
        if isinstance(v, Point2f):
            _x.set_value(v.x)
            _y.set_value(v.y)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], _point_setter))


@widget_builder(Widget.rect)
def _build_rect(settings, name, field, polls):
    value = getattr(settings, name)
    label = _field_label(name, field)
    is_disabled = _is_field_read_only(settings, name, field)

    with ui.column().classes("w-full gap-1"):
        _build_field_title(label, _wiring_tooltip(settings, name, field.description))
        with ui.row().classes("items-end gap-2 flex-wrap"):
            rx = ui.number(
                label="X", value=value.x if isinstance(value, Rect) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + _lock_prop(is_disabled)).classes("w-24")
            ry = ui.number(
                label="Y", value=value.y if isinstance(value, Rect) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + _lock_prop(is_disabled)).classes("w-24")
            rw = ui.number(
                label="W", value=value.width if isinstance(value, Rect) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + _lock_prop(is_disabled)).classes("w-24")
            rh = ui.number(
                label="H", value=value.height if isinstance(value, Rect) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + _lock_prop(is_disabled)).classes("w-24")

    if not is_disabled:
        def commit_rx(val):
            if val is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Rect(float(val), cur.y, cur.width, cur.height))
        _commit_typed(rx, commit_rx)

        def commit_ry(val):
            if val is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Rect(cur.x, float(val), cur.width, cur.height))
        _commit_typed(ry, commit_ry)

        def commit_rw(val):
            if val is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Rect(cur.x, cur.y, float(val), cur.height))
        _commit_typed(rw, commit_rw)

        def commit_rh(val):
            if val is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Rect(cur.x, cur.y, cur.width, float(val)))
        _commit_typed(rh, commit_rh)

    def _rect_setter(v, _rx=rx, _ry=ry, _rw=rw, _rh=rh):
        if isinstance(v, Rect):
            _rx.set_value(v.x)
            _ry.set_value(v.y)
            _rw.set_value(v.width)
            _rh.set_value(v.height)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], _rect_setter))


# -- Pinned read-outs ----------------------------------------------------------

def _format_readout(value) -> str:
    """A read-only value as short text for a pinned chip."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def build_pinned_readout(settings, name, field, polls):
    """A read-only pinned field as one compact chip, "<Label> <value> <unit>", instead of a
    locked input."""
    value = getattr(settings, name)
    label = _field_label(name, field)
    unit = f" {field.unit}" if field.unit else ""
    desc = _wiring_tooltip(settings, name, field.description)

    chip = _attach_description_tooltip(
        ui.label(f"{label} {_format_readout(value)}{unit}").classes("bg-grey-8 rounded px-2 py-1"),
        desc,
    )
    polls.append((settings, name, [value],
                  lambda v, c=chip: c.set_text(f"{label} {_format_readout(v)}{unit}")))


# -- Fallback for unregistered / unsupported types --------------------------

def _build_fallback(settings, name, field, polls):
    """Build a control for types without a registered Widget builder."""
    value = getattr(settings, name)
    label = _field_label(name, field)

    # -- Generic fallback: read-only label -----------------------------------
    with ui.row().classes("items-center gap-2"):
        _build_field_title(label, _wiring_tooltip(settings, name, field.description))
        lbl = ui.label(str(value)).classes("text-secondary")

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, lbl=lbl: lbl.set_text(str(v))))


def build_action_button(settings, name, field):
    """Create a NiceGUI button for a Widget.button Setting."""
    label = _field_label(name, field)
    description = _wiring_tooltip(settings, name, field.description)
    is_disabled = _is_field_read_only(settings, name, field)
    props = "dense disable" if is_disabled else "dense"
    btn = ui.button(label, on_click=lambda: field.fire(settings)).props(props)
    _attach_description_tooltip(btn, description or None)
    if is_disabled:
        btn.disable()

