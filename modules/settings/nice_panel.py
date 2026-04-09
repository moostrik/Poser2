# Deferred: revisit Shift+Arrow 10x stepping after the panel layout structure settles.



"""NiceGUI settings panel — auto-generates a tabbed UI from a Settings root.

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

import json
import socket
from enum import Enum
from pathlib import Path
from typing import Callable, get_origin, get_args

from nicegui import ui

from modules.settings.base_settings import BaseSettings
from modules.settings.nice_log import LOG_DRAWER_DEFAULT_HEIGHT, build_log_drawer
from modules.settings import presets
from modules.settings.field import Field, Access
from modules.settings.widget import Widget
from modules.utils import Color, Point2f, Rect

# ---------------------------------------------------------------------------
# Poll rate for setting → UI synchronisation.
# All UI updates run on NiceGUI's event-loop timer, never on the thread that
# writes the setting.  This avoids WebSocket flooding (bind fired per-write)
# and thread-safety issues (GL thread calling NiceGUI API).
# ---------------------------------------------------------------------------
POLL_INTERVAL = 0.25  # seconds
# ---------------------------------------------------------------------------
# Expansion state persistence — remembers which sections are open/closed.
# ---------------------------------------------------------------------------
_EXPANSION_STATE_FILE = Path("files/settings/.ui_state.json")
_expansion_state: dict[str, bool | str] = {}

def _load_expansion_state() -> None:
    global _expansion_state
    if _EXPANSION_STATE_FILE.exists():
        try:
            _expansion_state = json.loads(_EXPANSION_STATE_FILE.read_text())
        except Exception:
            _expansion_state = {}

def _save_expansion_state() -> None:
    try:
        _EXPANSION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _EXPANSION_STATE_FILE.write_text(json.dumps(_expansion_state))
    except Exception:
        pass

_load_expansion_state()


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
    return field.access is Access.READ or settings.is_incoming_shared(name)


def _field_needs_poll(settings, name: str, field: Field) -> bool:
    """Return True when UI should poll external updates for this field."""
    return field.access is not Access.WRITE or settings.is_incoming_shared(name)


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


def _build_init_field(settings, name: str) -> None:
    """Render a single init-only field without forcing it into a separate section."""
    with ui.row().classes("items-center gap-2 poser-init"):
        ui.label(generate_label(name))
        ui.label(str(getattr(settings, name))).classes("text-secondary italic")


def _access_css_class(field: Field) -> str:
    """Return a CSS marker class for the field's access level."""
    if field.access is Access.INIT:
        return "poser-init"
    if field.access is Access.READ:
        return "poser-feedback"
    return "poser-input"  # WRITE or READWRITE


def _build_settings_entry(settings, name: str, field: Field, polls) -> None:
    """Render one visible field in declaration order."""
    css = _access_css_class(field)
    if field.widget == Widget.button:
        with ui.element("div").classes(css):
            _build_action_button(settings, name, field)
        return
    if field.access is Access.INIT:
        _build_init_field(settings, name)
        return
    with ui.element("div").classes(css):
        _build_field_control(settings, name, field, polls)



def _build_field_control(settings, name, field, polls):
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


# -- bool builders -----------------------------------------------------------

@widget_builder(Widget.switch)
def _build_switch(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
    is_disabled = _is_field_read_only(settings, name, field)

    sw = _attach_description_tooltip(ui.switch(label, value=value).props(
        "dense" + (" disable" if is_disabled else "")
    ), desc)

    if not is_disabled:
        def on_switch_change(e):
            setattr(settings, name, e.value)
        sw.on_value_change(on_switch_change)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, sw=sw: sw.set_value(v)))


@widget_builder(Widget.toggle)
def _build_toggle(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
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
        "dense" + (" disable" if is_disabled else "")
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



@widget_builder(Widget.slider)
def _build_slider(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
    is_disabled = _is_field_read_only(settings, name, field)
    step = field.step if field.step is not None else (1 if field.type_ is int else 0.01)
    color = getattr(field, "color", "primary")

    with ui.column().classes("w-48 max-w-full gap-1"):
        with ui.row().classes("w-full items-center justify-between flex-nowrap"):
            _build_field_title(label, desc, classes="flex-1 truncate")
            fmt = "%.0f" if field.type_ is int else "%.2f"
            val_input = ui.number(
                value=value,
                step=step,
                format=fmt,
            ).props(
                "dense borderless"
                + (" disable" if is_disabled else "")
            ).classes("w-16")
        sl = ui.slider(
            min=field.min, max=field.max, step=step, value=value
        ).props(
            f"dense color={color}"
            + (" disable" if is_disabled else "")
        ).classes("w-full")

    _updating = {"lock": False}

    if not is_disabled:
        def on_slider_change(e):
            if not _updating["lock"]:
                _updating["lock"] = True
                setattr(settings, name, field.type_(e.value))
                val_input.set_value(e.value)
                _updating["lock"] = False
        sl.on_value_change(on_slider_change)

        def on_input_change(e):
            if not _updating["lock"] and e.value is not None:
                _updating["lock"] = True
                clamped = max(field.min, min(field.max, field.type_(e.value)))
                setattr(settings, name, clamped)
                sl.set_value(clamped)
                if clamped != e.value:
                    val_input.set_value(clamped)
                _updating["lock"] = False
        val_input.on_value_change(on_input_change)

    if _field_needs_poll(settings, name, field):
        def _poll_slider(v, _sl=sl, _vi=val_input):
            _sl.set_value(v)
            _vi.set_value(v)
        polls.append((settings, name, [value], _poll_slider))


@widget_builder(Widget.number)
def _build_number(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
    is_disabled = _is_field_read_only(settings, name, field)

    num = _attach_description_tooltip(ui.number(
        label=label,
        value=value,
        step=field.step if field.step is not None else (1 if field.type_ is int else 0.01),
        format="%.0f" if field.type_ is int else "%.2f",
    ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24"), desc)

    if not is_disabled:
        def on_num_change(e):
            if e.value is not None:
                setattr(settings, name, field.type_(e.value))
        num.on_value_change(on_num_change)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, num=num: num.set_value(v)))


@widget_builder(Widget.knob)
def _build_knob(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
    is_disabled = _is_field_read_only(settings, name, field)
    step = field.step if field.step is not None else (1 if field.type_ is int else 0.01)
    min_val = field.min if field.min is not None else 0
    max_val = field.max if field.max is not None else 100

    with ui.column().classes("gap-1"):
        _build_field_title(label, desc)
        kn = ui.knob(
            value=value, min=min_val, max=max_val, step=step,
            show_value=True, size="lg",
        ).props(
            "thickness=0.2" + (" disable" if is_disabled else "")
        )

    if not is_disabled:
        def on_knob_change(e):
            setattr(settings, name, field.type_(e.value))
        kn.on_value_change(on_knob_change)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, kn=kn: kn.set_value(v)))


# -- enum builders -----------------------------------------------------------

@widget_builder(Widget.select)
def _build_select(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
    is_disabled = _is_field_read_only(settings, name, field)

    options = {m: generate_label(m.name) for m in field.type_}
    sel = _attach_description_tooltip(ui.select(
        options=options,
        value=value,
        label=label,
    ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-48 max-w-full"), desc)

    if not is_disabled:
        def on_select_change(e):
            setattr(settings, name, e.value)
        sel.on_value_change(on_select_change)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, s=sel: s.set_value(v)))


@widget_builder(Widget.text_select)
def _build_text_select(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
    is_disabled = _is_field_read_only(settings, name, field)

    # Resolve the options Field → current list[str] value
    options_field = field.options
    option_list = getattr(settings, options_field.name) if options_field else []
    sel = _attach_description_tooltip(ui.select(
        options=option_list,
        value=value if value in option_list else None,
        label=label,
    ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-48 max-w-full"), desc)

    if not is_disabled:
        def on_select_change(e):
            setattr(settings, name, e.value)
        sel.on_value_change(on_select_change)

    polls.append((settings, name, [value], lambda v, s=sel: s.set_value(v)))
    # Add a custom poll for the options field too
    if options_field:
        polls.append((settings, options_field.name, [list(option_list)],
                       lambda v, s=sel: (setattr(s, 'options', v), s.update())))


@widget_builder(Widget.radio)
def _build_radio(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
    is_disabled = _is_field_read_only(settings, name, field)

    options = {m: generate_label(m.name) for m in field.type_}
    with ui.column().classes("gap-1"):
        _build_field_title(label, desc)
        rg = ui.toggle(options=options, value=value).props(
            "dense" + (" disable" if is_disabled else "")
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
    label = generate_label(name)
    desc = field.description
    is_disabled = _is_field_read_only(settings, name, field)

    inp = _attach_description_tooltip(ui.input(label=label, value=value).props(
        "dense outlined" + (" disable" if is_disabled else "")
    ), desc)

    if not is_disabled:
        def on_input_change(e):
            setattr(settings, name, e.value)
        inp.on_value_change(on_input_change)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, inp=inp: inp.set_value(v)))


@widget_builder(Widget.ip_field)
def _build_ip(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
    is_disabled = _is_field_read_only(settings, name, field)

    def is_valid_ip(v: str) -> bool:
        parts = v.split(".")
        return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)

    inp = _attach_description_tooltip(ui.input(
        label=label, value=value,
        validation={"": is_valid_ip},
    ).props(
        'dense outlined hide-bottom-space' + (" disable" if is_disabled else "")
    ).classes("w-36"), desc)

    if not is_disabled:
        def on_ip_change(e):
            if is_valid_ip(e.value):
                setattr(settings, name, e.value)
        inp.on_value_change(on_ip_change)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, inp=inp: inp.set_value(v)))


@widget_builder(Widget.number_field)
def _build_number_input(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
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
        'dense outlined hide-bottom-space' + (" disable" if is_disabled else "")
    ).classes("w-24"), desc)

    if not is_disabled:
        def on_change(e, inp=inp):
            if is_valid(e.value):
                setattr(settings, name, ft(e.value))
        inp.on_value_change(on_change)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, inp=inp: inp.set_value(str(v))))


@widget_builder(Widget.textarea)
def _build_textarea(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
    is_disabled = _is_field_read_only(settings, name, field)

    ta = _attach_description_tooltip(ui.textarea(label=label, value=value).props(
        "dense outlined" + (" disable" if is_disabled else "")
    ), desc)

    if not is_disabled:
        def on_ta_change(e):
            setattr(settings, name, e.value)
        ta.on_value_change(on_ta_change)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, ta=ta: ta.set_value(v)))


# -- color builders ----------------------------------------------------------

@widget_builder(Widget.color)
def _build_color(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description
    is_disabled = _is_field_read_only(settings, name, field)

    def _color_style(hex_val: str) -> str:
        c = Color.from_hex(hex_val)
        luminance = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
        text = '#000000' if luminance > 0.5 else '#ffffff'
        return f'background:{hex_val};border-radius:4px;color:{text}'

    hex_val = value.to_hex() if isinstance(value, Color) else '#000000'

    ci = _attach_description_tooltip(ui.color_input(
        label=label, value=hex_val
    ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-36").style(
        _color_style(hex_val)
    ), desc)

    if not is_disabled:
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
    label = generate_label(name)
    is_disabled = _is_field_read_only(settings, name, field)

    hex_val = value.to_hex() if isinstance(value, Color) else '#000000'
    alpha = value.a if isinstance(value, Color) else 1.0

    with ui.column().classes("w-full gap-1"):
        _build_field_title(label, field.description)
        with ui.row().classes("items-end gap-2"):
            ci = ui.color_input(
                label="Color", value=hex_val
            ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-36")

            alpha_num = ui.number(
                label="A", value=alpha,
                min=0.0, max=1.0, step=0.01,
                format="%.2f",
            ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24")

    if not is_disabled:
        def on_color_change(e):
            if e.value:
                c = Color.from_hex(e.value)
                cur_alpha = alpha_num.value or 1.0
                setattr(settings, name, Color(c.r, c.g, c.b, float(cur_alpha)))
        ci.on_value_change(on_color_change)

        def on_alpha_change(e):
            if e.value is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Color(cur.r, cur.g, cur.b, float(e.value)))
        alpha_num.on_value_change(on_alpha_change)

    def _color_alpha_setter(v, _ci=ci, _an=alpha_num):
        if isinstance(v, Color):
            _ci.set_value(v.to_hex())
            _an.set_value(v.a)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], _color_alpha_setter))


# -- list builders -----------------------------------------------------------

def _build_sortable_list(settings, name, field, polls, *, with_checkboxes: bool):
    """Shared implementation for checklist (with checkboxes) and order (without)."""
    value = getattr(settings, name)
    label = generate_label(name)
    desc = field.description

    elem_type = get_args(field.type_)[0] if get_args(field.type_) else None
    if elem_type is None or not (isinstance(elem_type, type) and issubclass(elem_type, Enum)):
        # Fallback for non-enum lists
        return _build_fallback(settings, name, field, polls)

    all_members = list(elem_type)
    active_set = set(value)

    fold_btn = None

    def _build_sortable_actions() -> None:
        nonlocal fold_btn
        if with_checkboxes:
            fold_btn = ui.button(icon="visibility_off", on_click=lambda: None).props(
                "dense flat round size=xs"
            ).tooltip("Show/hide unchecked items")

    with ui.column().classes("w-72 max-w-full gap-1"):
        _build_field_header(
            label,
            desc,
            title_classes="flex-1",
            row_classes="w-full items-center gap-1",
            actions=_build_sortable_actions if with_checkboxes else None,
        )
        container = ui.column().classes("w-full gap-0 border rounded p-1")

        if with_checkboxes:
            ordered = list(value) + [m for m in all_members if m not in active_set]
        else:
            # Order-only: all members always active, ordered as in value
            ordered = list(value)
        state = {
            "order": ordered,
            "active": set(value) if with_checkboxes else set(all_members),
            "folded": True,
        }

        def _rebuild(cont, st, _settings=settings, _name=name, _elem=elem_type, _cb=with_checkboxes):
            cont.clear()
            folded = st.get("folded", False)
            with cont:
                visible_members = [
                    m for m in st["order"]
                    if not (folded and _cb and m not in st["active"])
                ]
                for idx, member in enumerate(visible_members):
                    is_active = member in st["active"]
                    with ui.row().classes(
                        "w-full items-center gap-2 px-2 py-0.5 rounded cursor-move"
                        + ("" if is_active else " opacity-50")
                    ).style(
                        "background: color-mix(in srgb, var(--q-primary) 15%, transparent);"
                        if is_active else ""
                    ):
                        if _cb:
                            cb_widget = ui.checkbox(
                                value=is_active,
                            ).props("dense").classes("my-0")

                        ui.label(generate_label(member.name)).classes("flex-1")

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

                        if _cb:
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
            # Reorder: active items first, then inactive (preserving relative order)
            if with_checkboxes:
                st["order"] = (
                    [m for m in st["order"] if m in st["active"]]
                    + [m for m in st["order"] if m not in st["active"]]
                )
            new_list = [m for m in st["order"] if m in st["active"]]
            setattr(settings, name, new_list)
            _rebuild(cont, st)

        _rebuild(container, state)

        # Wire up fold button
        if with_checkboxes:
            assert fold_btn is not None
            fold_button = fold_btn

            def _toggle_fold():
                state["folded"] = not state["folded"]
                fold_button.props(
                    f'icon={"visibility_off" if state["folded"] else "visibility"}'
                )
                fold_button.update()
                _rebuild(container, state)
            fold_button.on_click(_toggle_fold)

    def _list_setter(v, _cont=container, _st=state, _elem=elem_type, _cb=with_checkboxes):
        active_set_inner = set(v)
        _st["active"] = active_set_inner
        if _cb:
            _st["order"] = list(v) + [m for m in list(_elem) if m not in active_set_inner]
        else:
            _st["order"] = list(v)
        _rebuild(_cont, _st)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [list(value)], _list_setter))


@widget_builder(Widget.checklist)
def _build_checklist(settings, name, field, polls):
    return _build_sortable_list(settings, name, field, polls, with_checkboxes=True)


@widget_builder(Widget.order)
def _build_order(settings, name, field, polls):
    return _build_sortable_list(settings, name, field, polls, with_checkboxes=False)


@widget_builder(Widget.point2f)
def _build_point2f(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = _is_field_read_only(settings, name, field)

    with ui.column().classes("w-full gap-1"):
        _build_field_title(label, field.description)
        with ui.row().classes("items-end gap-2"):
            x_num = ui.number(
                label="X", value=value.x if isinstance(value, Point2f) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24")
            y_num = ui.number(
                label="Y", value=value.y if isinstance(value, Point2f) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24")

    if not is_disabled:
        def on_x_change(e, _y=y_num):
            if e.value is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Point2f(float(e.value), cur.y))
        x_num.on_value_change(on_x_change)

        def on_y_change(e, _x=x_num):
            if e.value is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Point2f(cur.x, float(e.value)))
        y_num.on_value_change(on_y_change)

    def _point_setter(v, _x=x_num, _y=y_num):
        if isinstance(v, Point2f):
            _x.set_value(v.x)
            _y.set_value(v.y)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], _point_setter))


@widget_builder(Widget.rect)
def _build_rect(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = _is_field_read_only(settings, name, field)

    with ui.column().classes("w-full gap-1"):
        _build_field_title(label, field.description)
        with ui.row().classes("items-end gap-2 flex-wrap"):
            rx = ui.number(
                label="X", value=value.x if isinstance(value, Rect) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24")
            ry = ui.number(
                label="Y", value=value.y if isinstance(value, Rect) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24")
            rw = ui.number(
                label="W", value=value.width if isinstance(value, Rect) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24")
            rh = ui.number(
                label="H", value=value.height if isinstance(value, Rect) else 0.0,
                step=0.01, format="%.3f",
            ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24")

    if not is_disabled:
        def on_rx(e, _ry=ry, _rw=rw, _rh=rh):
            if e.value is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Rect(float(e.value), cur.y, cur.width, cur.height))
        rx.on_value_change(on_rx)

        def on_ry(e, _rx=rx, _rw=rw, _rh=rh):
            if e.value is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Rect(cur.x, float(e.value), cur.width, cur.height))
        ry.on_value_change(on_ry)

        def on_rw(e, _rx=rx, _ry=ry, _rh=rh):
            if e.value is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Rect(cur.x, cur.y, float(e.value), cur.height))
        rw.on_value_change(on_rw)

        def on_rh(e, _rx=rx, _ry=ry, _rw=rw):
            if e.value is not None:
                cur = getattr(settings, name)
                setattr(settings, name, Rect(cur.x, cur.y, cur.width, float(e.value)))
        rh.on_value_change(on_rh)

    def _rect_setter(v, _rx=rx, _ry=ry, _rw=rw, _rh=rh):
        if isinstance(v, Rect):
            _rx.set_value(v.x)
            _ry.set_value(v.y)
            _rw.set_value(v.width)
            _rh.set_value(v.height)

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], _rect_setter))


# -- Fallback for unregistered / unsupported types --------------------------

def _build_fallback(settings, name, field, polls):
    """Build a control for types without a registered Widget builder."""
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = _is_field_read_only(settings, name, field)

    # -- Generic fallback: read-only label -----------------------------------
    with ui.row().classes("items-center gap-2"):
        ui.label(label)
        lbl = ui.label(str(value)).classes("text-secondary")

    if _field_needs_poll(settings, name, field):
        polls.append((settings, name, [value], lambda v, lbl=lbl: lbl.set_text(str(v))))


def _build_action_button(settings, name, field):
    """Create a NiceGUI button for a Widget.button Setting."""
    label = generate_label(name)
    ui.button(label, on_click=lambda: field.fire(settings)).props("dense")


def _has_visible_content(settings):
    """Return True if settings has any visible fields, actions, or children with visible content."""
    if any(f.visible for f in settings.fields.values()):
        return True
    if settings.actions:
        return True
    if any(_has_visible_content(c) for c in settings.children.values()):
        return True
    return False


def _content_access_types(settings) -> frozenset:
    """Return the set of Access types present in visible fields (recursive).

    Actions are not counted — groups with actions are never auto-hidden.
    """
    if settings.actions:
        return frozenset({Access.WRITE, Access.READ, Access.INIT})  # mixed — never hide
    types: set = set()
    for f in settings.fields.values():
        if f.visible:
            if f.access in (Access.WRITE, Access.READWRITE):
                types.add(Access.WRITE)  # normalize READWRITE → WRITE bucket
            else:
                types.add(f.access)
    for child in settings.children.values():
        if _has_visible_content(child):
            types |= _content_access_types(child)
    return frozenset(types)


# Map access-type combinations to CSS class names for group hiding.
_ACCESS_CLASS_MAP: dict[frozenset, str] = {
    frozenset({Access.INIT}):                          "poser-only-init",
    frozenset({Access.READ}):                          "poser-only-feedback",
    frozenset({Access.WRITE}):                         "poser-only-input",
    frozenset({Access.INIT, Access.READ}):              "poser-only-init-feedback",
    frozenset({Access.INIT, Access.WRITE}):             "poser-only-init-input",
    frozenset({Access.READ, Access.WRITE}):             "poser-only-feedback-input",
    frozenset({Access.INIT, Access.READ, Access.WRITE}): "poser-only-all",
}


def _register_polls(polls, all_polls):
    """Append local poll entries into the page-level poll list.

    Each entry is ``(settings, name, [last_value], setter)``.
    The actual timer is created once per page after all controls have been built.
    """
    if polls:
        all_polls.extend(polls)


def _make_poll_timer(polls, timers):
    """Create a single page-level ``ui.timer`` that polls all collected entries.

    Each entry is ``(settings, name, [last_value], setter)``.
    The timer runs on NiceGUI's event loop so UI calls are thread-safe.
    *timers* collects the timer for later cleanup reference.
    Cancels itself if the client has been deleted (parent slot gone).
    """
    if not polls:
        return

    t_holder: list = [None]

    def _tick():
        try:
            for s, fname, last, setter in polls:
                cur = getattr(s, fname)
                if cur != last[0]:
                    last[0] = cur
                    setter(cur)
        except RuntimeError:
            # Parent slot deleted — client is gone, cancel to stop recurring errors
            if t_holder[0] is not None:
                t_holder[0].cancel()
        except Exception:
            pass  # Transient error (e.g. WebSocket drop) — skip this tick, retry next

    t = ui.timer(POLL_INTERVAL, _tick)
    t_holder[0] = t
    timers.append(t)


def _build_settings_body(settings, all_polls, *, depth=0, expansions=None, path=""):
    """Emit the controls for a single Settings instance (no wrapper).

    Renders fields strictly in **declaration order** inside a single flex-wrap
    container.  ``field.newline`` inserts a full-width zero-height break element
    that carries the field's access CSS class so it hides with the field.

    *depth* tracks nesting level (0 = top, 1+ = child of child).
    *expansions* collects ``ui.expansion`` elements for expand/collapse-all.
    """
    polls: list[tuple] = []

    visible_fields = [
        (field_name, field)
        for field_name, field in settings.fields.items()
        if field.visible
    ]

    with ui.row().classes("w-full gap-4 flex-wrap items-center content-start"):
        for field_name, field in visible_fields:
            if field.newline:
                # Full-width break element with the field's access class so it
                # hides together with the field when that access type is toggled off.
                ui.element("div").classes(
                    f"{_access_css_class(field)} pt-2"
                ).style("flex-basis: 100%; height: 0")
            _build_settings_entry(settings, field_name, field, polls)

    # Register this card's polls with the page-level timer.
    _register_polls(polls, all_polls)

    # Children (recursive)
    for child_name, child in settings.children.items():
        if _has_visible_content(child):
            _build_settings_card(child_name, child, all_polls, depth=depth, expansions=expansions, path=path)


# Depth-based Quasar background classes (dark mode: increasingly lighter).
# _DEPTH_BG = ["bg-grey-10", "bg-grey-9", "bg-grey-8"]
_DEPTH_BG = ["bg-grey-9", "bg-grey-8", "bg-grey-7", "bg-grey-6", "bg-grey-5"]  # extended for deeper nesting


def _build_settings_card(name, settings, all_polls, *, depth=0, expansions=None, path=""):
    """Build a card for one Settings instance.

    Always renders as a collapsible ``ui.expansion`` at any depth.
    Every layer gets a consistent custom header with an icon and an
    expand/collapse-all button (functional when children exist, hidden
    otherwise so the layout stays uniform).
    """
    has_children = bool(settings.children)
    key = f"{path}.{name}" if path else name
    is_open: bool = _expansion_state.get(key) is True

    # Tag expansion with the access types it contains so it hides when
    # all its types are toggled off.
    _access_cls = _ACCESS_CLASS_MAP.get(_content_access_types(settings), "")
    bg = _DEPTH_BG[min(depth, len(_DEPTH_BG) - 1)]
    exp = ui.expansion(
        generate_label(name),
        value=is_open,
    ).props("duration=0 dense dense-toggle").classes(f"w-full rounded {bg}" + (f" {_access_cls}" if _access_cls else ""))
    exp.on("show", lambda: (_expansion_state.update({key: True}), _save_expansion_state()))
    exp.on("hide", lambda: (_expansion_state.update({key: False}), _save_expansion_state()))

    # Always use a custom header slot for consistent appearance.
    child_expansions_ref: list = []
    _state = {"all": False}

    def _toggle_children():
        _state["all"] = not _state["all"]
        if _state["all"]:
            # Ensure the parent expansion is open so children are visible
            exp.open()
            for e in child_expansions_ref:
                e.open()
        else:
            for e in child_expansions_ref:
                e.close()
            exp.close()

    with exp.add_slot("header"):
        with ui.row().classes("w-full items-center gap-1"):
            ui.label(generate_label(name).upper()).classes("flex-1")
            if has_children:
                unfold_btn = ui.button(
                    icon="unfold_more",
                    on_click=_toggle_children,
                ).props("dense flat round size=xs").tooltip(
                    "Expand / Collapse All"
                )
                # Prevent the button click from toggling the expansion header
                unfold_btn.on("click.stop", lambda: None)

    if expansions is not None:
        expansions.append(exp)

    with exp:
        child_expansions: list = []

        _build_settings_body(
            settings, all_polls,
            depth=depth + 1, expansions=child_expansions, path=key,
        )

        if expansions is not None:
            expansions.extend(child_expansions)

        # Wire up the header button to the actual child expansions
        if has_children:
            child_expansions_ref.extend(child_expansions)


def _build_preset_controls(root):
    """Emit preset dropdown + action buttons (no wrapping row — caller provides layout)."""
    preset_list = presets.scan()
    startup = presets.get_startup()
    initial = startup if startup in preset_list else (preset_list[0] if preset_list else None)

    # -- Star button (left of dropdown) ------------------------------------
    def do_set_startup():
        preset_name = dropdown.value
        if not preset_name:
            ui.notify("Select a preset first", type="warning")
            return
        presets.set_startup(preset_name)
        _update_star(preset_name)
        ui.notify(f"'{preset_name}' will load on next startup", type="positive")

    star_btn = ui.button(
        icon="star", on_click=do_set_startup,
    ).props("dense flat").tooltip("Set as startup preset")

    def _update_star(selected):
        if selected and selected == presets.get_startup():
            star_btn._props["color"] = "warning"
            star_btn._props.pop("outline", None)
        else:
            star_btn._props["color"] = "grey"
            star_btn._props["outline"] = True
        star_btn.update()

    _update_star(initial)

    # -- Preset dropdown ---------------------------------------------------
    dropdown = ui.select(
        options=preset_list,
        value=initial,
        label="Preset",
    ).props("dense outlined").classes("min-w-[180px]")

    def _refresh_dropdown():
        dropdown.set_options(presets.scan())

    def _on_preset_change(e):
        name = e.value
        _update_star(name)
        if not name:
            return
        p = presets.path(name)
        if p.exists():
            presets.load(root, p)
            ui.notify(f"Loaded '{name}'", type="positive")

    dropdown.on_value_change(_on_preset_change)

    # -- Action buttons ----------------------------------------------------
    def do_load():
        name = dropdown.value
        if not name:
            ui.notify("Select a preset first", type="warning")
            return
        p = presets.path(name)
        if p.exists():
            presets.load(root, p)
            ui.notify(f"Loaded '{name}'", type="positive")
        else:
            ui.notify(f"Preset '{name}' not found", type="negative")

    def do_save():
        preset_name = dropdown.value
        if not preset_name:
            ui.notify("Select a preset to overwrite, or use Save As", type="warning")
            return
        presets.save(root, presets.path(preset_name))
        ui.notify(f"Saved '{preset_name}'", type="positive")

    async def do_save_as():
        with ui.dialog() as dialog, ui.card().classes("min-w-[300px]"):
            ui.label("Save As").classes("text-lg font-bold")
            name_input = ui.input(label="Preset name", value=dropdown.value or "").props("dense outlined autofocus")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def confirm():
                    preset_name = (name_input.value or "").strip()
                    if not preset_name:
                        ui.notify("Enter a preset name", type="warning")
                        return
                    try:
                        presets.validate_name(preset_name)
                    except ValueError:
                        ui.notify("Invalid preset name — no slashes or leading dots", type="negative")
                        return
                    presets.save(root, presets.path(preset_name))
                    _refresh_dropdown()
                    dropdown.set_value(preset_name)
                    ui.notify(f"Saved '{preset_name}'", type="positive")
                    dialog.close()

                ui.button("Save", on_click=confirm).props("flat color=primary")

            # Allow Enter key to confirm
            name_input.on("keydown.enter", lambda _: confirm())

        dialog.open()

    async def do_delete():
        preset_name = dropdown.value
        if not preset_name:
            ui.notify("Select a preset to delete", type="warning")
            return

        with ui.dialog() as dialog, ui.card():
            ui.label(f"Delete '{preset_name}'?")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def confirm_delete():
                    p = presets.path(preset_name)
                    if p.exists():
                        p.unlink()
                    _refresh_dropdown()
                    dropdown.set_value(None)
                    ui.notify(f"Deleted '{preset_name}'", type="info")
                    dialog.close()

                ui.button("Delete", on_click=confirm_delete).props("flat color=negative")

        dialog.open()

    ui.button(icon="refresh", on_click=do_load).props("dense flat").tooltip("Reload selected preset")
    ui.button(icon="save", on_click=do_save).props("dense flat").tooltip("Save (overwrite selected)")
    ui.button(icon="save_as", on_click=do_save_as).props("dense flat").tooltip("Save as new preset")
    ui.button(icon="delete", on_click=do_delete).props("dense flat color=negative").tooltip("Delete preset")


def _get_local_ips() -> list[str]:
    """Return all non-loopback IPv4 addresses for this machine."""
    try:
        hostname = socket.gethostname()
        _, _, ips = socket.gethostbyname_ex(hostname)
        return [ip for ip in ips if not ip.startswith("127.") and not ip.startswith("169.254.")]
    except Exception:
        return []


def create_settings_panel(
    root,
    *,
    title: str = "",
    port: int | None = None,
    on_exit=None,
) -> None:
    """Build a full tabbed settings panel from a root Settings.

    Call this inside a NiceGUI page context::

        @ui.page("/")
        def index():
            create_settings_panel(app_settings, title="POSER", port=8080, on_exit=stop)
    """

    # Force dark mode for consistent styling
    ui.dark_mode(True)

    # Timers for this client session — self-cancel if parent slot is deleted.
    timers: list = []
    all_polls: list[tuple] = []

    # -- Responsive CSS via @media (works on all browsers) -----------------
    ui.add_css('''
    html { overflow-y: scroll; }
    #popup.nicegui-error-popup { display: none !important; }
    .hide-init .poser-init { display: none !important; }
    .hide-feedback .poser-feedback { display: none !important; }
    .hide-input .poser-input { display: none !important; }
    .hide-init .poser-only-init { display: none !important; }
    .hide-feedback .poser-only-feedback { display: none !important; }
    .hide-input .poser-only-input { display: none !important; }
    .hide-init.hide-feedback .poser-only-init-feedback { display: none !important; }
    .hide-init.hide-input .poser-only-init-input { display: none !important; }
    .hide-feedback.hide-input .poser-only-feedback-input { display: none !important; }
    .hide-init.hide-feedback.hide-input .poser-only-all { display: none !important; }
    ''')

    # -- Shutdown overlay (client-side JS) ---------------------------------
    # Defines showShutdownScreen() and auto-shows it when the WebSocket
    # disconnects (external quit, Ctrl-C, stop() from code, etc.).
    _title_esc = title.replace('\\', '\\\\').replace("'", "\\'") if title else ''
    _title_markup = (
        f'<div style="font-size:1.5rem;font-weight:bold;color:#3f3f46;'
        f'margin-bottom:1rem">{_title_esc}</div>'
        if title else ''
    )
    ui.add_head_html(f'''<script>(function() {{
      window.showShutdownScreen = function() {{
        if (document.getElementById("shutdown-overlay")) return;
        var d = document.createElement("div");
        d.id = "shutdown-overlay";
                d.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:9998;"
                    + "backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);";
                d.innerHTML = '<div style="position:absolute;left:50%;top:calc((100vh - {LOG_DRAWER_DEFAULT_HEIGHT}) / 2);transform:translate(-50%, -50%);text-align:center;color:#e4e4e7">'
          + '{_title_markup}'
          + '<span class="material-icons" style="font-size:64px;color:#71717a;margin-bottom:0.75rem;display:block">'
          + 'cloud_off</span>'
          + '<div style="font-size:1rem;font-weight:500;color:#a1a1aa">No connection</div>'
          + '<div style="font-size:0.8rem;color:#71717a;margin-top:0.25rem">Waiting for server\u2026</div>'
          + '</div>';
        document.body.appendChild(d);
      }};
      var _t = null;
      var _i = setInterval(function() {{
        if (window.socket) {{
          window.socket.on("disconnect", function() {{
            _t = setTimeout(window.showShutdownScreen, 1500);
          }});
          window.socket.on("connect", function() {{
            if (_t) {{ clearTimeout(_t); _t = null; }}
            var el = document.getElementById("shutdown-overlay");
            if (el) el.remove();
          }});
          clearInterval(_i);
        }}
      }}, 100);
    }})();</script>''')

    # -- Header row: title | preset controls | exit button -----------------
    # Collect tab entries early so we can build tabs inside the sticky header.
    tab_entries: list[tuple[str, BaseSettings]] = []  # (name, settings)
    for child_name, child in root.children.items():
        if _has_visible_content(child):
            tab_entries.append((child_name, child))

    with ui.column().classes("w-full sticky top-0 z-50 bg-[#121212] text-white gap-2 px-3 pt-3 pb-2"):
        with ui.row().classes("w-full items-center flex-wrap gap-1"):
            if title:
                ui.label(title).classes("text-2xl font-bold")

            # Log drawer toggle — drawer itself is built outside the header below
            _log_toggle_holder: list = []
            ui.button(icon="terminal", on_click=lambda: _log_toggle_holder[0]() if _log_toggle_holder else None).props(
                "dense flat"
            ).tooltip("Log panel")

            if port is not None:
                local_ip_links_loaded = {"value": False}

                connection_btn = ui.button(icon="lan").props("dense flat").tooltip("Connection info")
                with connection_btn:
                    with ui.menu().props('anchor="bottom middle" self="top middle"'):
                        with ui.card().classes("gap-1").props("flat"):
                            ui.label("Connect").classes("text-base font-bold")
                            ui.separator()
                            local_ip_links = ui.column().classes("gap-0")
                            _url_local = f"http://localhost:{port}"
                            ui.link(_url_local, _url_local, new_tab=True).classes("text-sm")

                def _populate_local_ip_links() -> None:
                    if local_ip_links_loaded["value"]:
                        return
                    local_ip_links_loaded["value"] = True
                    with local_ip_links:
                        for _ip in _get_local_ips():
                            _url = f"http://{_ip}:{port}"
                            ui.link(_url, _url, new_tab=True).classes("text-sm")

                connection_btn.on("click", lambda: _populate_local_ip_links())

            with ui.row().classes("flex-1 items-center gap-1 flex-nowrap justify-center"):
                _build_preset_controls(root)

            if on_exit:
                async def _do_exit():
                    await ui.run_javascript('showShutdownScreen()')
                    on_exit()
                ui.button(icon="power_settings_new", on_click=_do_exit).props(
                    "dense flat color=negative"
                ).tooltip("Exit application")

        # Collect pinned fields and actions from all registered modules
        pinned_fields: list[tuple[BaseSettings, str, Field]] = []
        pinned_actions: list[tuple[BaseSettings, str, Field]] = []

        def _collect_pinned(settings: BaseSettings) -> None:
            for field_name, field in settings.fields.items():
                if field.pinned and field.visible:
                    pinned_fields.append((settings, field_name, field))
            for action_name, action_field in settings.actions.items():
                if action_field.pinned and action_field.visible:
                    pinned_actions.append((settings, action_name, action_field))
            for child in settings.children.values():
                _collect_pinned(child)

        _collect_pinned(root)

        # Render pinned fields and actions in a compact row above the tabs
        if pinned_fields or pinned_actions:
            pinned_polls: list[tuple] = []
            with ui.row().classes("w-full gap-4 flex-wrap items-end bg-grey-9 rounded px-3 py-2 mt-2"):
                for settings, field_name, field in pinned_fields:
                    if field.access is Access.INIT:
                        with ui.row().classes("items-center gap-1"):
                            ui.label(generate_label(field_name))
                            ui.label(str(getattr(settings, field_name))).classes(
                                "text-secondary italic"
                            )
                    else:
                        _build_field_control(settings, field_name, field, pinned_polls)
                for settings, action_name, action_field in pinned_actions:
                    _build_action_button(settings, action_name, action_field)
            _register_polls(pinned_polls, all_polls)

        # Tabs inside the sticky header
        if not tab_entries:
            ui.label("No settings registered.")

        # Visibility toggle definitions: (icon, state_suffix, css_class, tooltip, default_hidden)
        _TOGGLES = [
            ("construction",   "__hide_init__",     "hide-init",     "Init fields",     True),
            ("monitor_heart",  "__hide_feedback__", "hide-feedback", "Feedback fields",  False),
            ("tune",           "__hide_input__",    "hide-input",    "Input fields",     False),
        ]

        # Forward-populated by tab panel loop below.
        _tab_data: dict[str, dict] = {}   # label -> {"panel": ..., "expansions": [...]}
        _active = {"label": ""}

        if tab_entries:
            saved_tab = _expansion_state.get("__active_tab__")
            initial_tab_label = saved_tab if saved_tab in dict(tab_entries) else tab_entries[0][0]
            _active["label"] = initial_tab_label

            with ui.tabs().classes("w-full bg-[#121212] rounded").props("dense active-color=primary no-indicator") as tabs:
                tab_map = {}
                for label, _ in tab_entries:
                    t = ui.tab(generate_label(label))
                    def _on_tab_click(l=label):
                        _expansion_state["__active_tab__"] = l
                        _save_expansion_state()
                        _active["label"] = l
                        _sync_toolbar_to_tab()
                    t.on("click", _on_tab_click)
                    tab_map[label] = t

            # Toolbar row: visibility toggles (left) + expand/collapse-all (right)
            _toggle_btns: list[tuple] = []   # [(btn, suffix, css, default), ...]
            _expand_btn_ref: list = [None]

            def _sync_toolbar_to_tab():
                """Update toggle button appearance for the active tab."""
                lbl = _active["label"]
                for btn, suffix, css, default in _toggle_btns:
                    is_hidden = _expansion_state.get(f"{lbl}.{suffix}", default)
                    if is_hidden:
                        btn.classes(add="text-grey-7")
                    else:
                        btn.classes(remove="text-grey-7")

            with ui.row().classes("w-full items-center"):
                for icon, suffix, css, tip, default in _TOGGLES:
                    btn_ref: list = [None]

                    def _make_toggle(sf=suffix, cc=css, dh=default, br=btn_ref):
                        def _toggle():
                            lbl = _active["label"]
                            td = _tab_data.get(lbl)
                            if not td:
                                return
                            state_key = f"{lbl}.{sf}"
                            is_hidden = not _expansion_state.get(state_key, dh)
                            _expansion_state[state_key] = is_hidden
                            _save_expansion_state()
                            if is_hidden:
                                td["panel"].classes(add=cc)
                                br[0].classes(add="text-grey-7")
                            else:
                                td["panel"].classes(remove=cc)
                                br[0].classes(remove="text-grey-7")
                        return _toggle

                    btn = ui.button(
                        icon=icon, on_click=_make_toggle(),
                    ).props("dense flat size=sm").tooltip(tip)
                    is_hidden = _expansion_state.get(f"{initial_tab_label}.{suffix}", default)
                    if is_hidden:
                        btn.classes(add="text-grey-7")
                    btn_ref[0] = btn
                    _toggle_btns.append((btn, suffix, css, default))

                # Spacer pushes expand/collapse to the right
                ui.element("div").classes("flex-1")

                def _toggle_all_expansions():
                    td = _tab_data.get(_active["label"])
                    if not td or not td["expansions"]:
                        return
                    _expand_state = td.setdefault("_expanded", False)
                    td["_expanded"] = not _expand_state
                    for e in td["expansions"]:
                        if td["_expanded"]:
                            e.open()
                        else:
                            e.close()

                ui.button(
                    icon="unfold_more",
                    on_click=_toggle_all_expansions,
                ).props("dense flat size=sm").tooltip("Expand / Collapse All")

    # -- end of sticky header --

    # Build log drawer outside the header so it's a top-level fixed element
    _, _toggle_log_fn = build_log_drawer(timers, poll_interval=POLL_INTERVAL)
    _log_toggle_holder.append(_toggle_log_fn)

    _make_poll_timer(all_polls, timers)

    if not tab_entries:
        return

    with ui.tab_panels(tabs, value=tab_map[initial_tab_label]).classes("w-full bg-grey-10 rounded ").style("padding-bottom: 320px;"):
        for label, root_settings in tab_entries:
            with ui.tab_panel(tab_map[label]) as panel:
                # Apply persisted visibility classes to this panel
                for _, suffix, css, _, default in _TOGGLES:
                    if _expansion_state.get(f"{label}.{suffix}", default):
                        panel.classes(add=css)

                expansions: list = []

                _build_settings_body(
                    root_settings, all_polls,
                    depth=0, expansions=expansions, path=label,
                )

                _tab_data[label] = {"panel": panel, "expansions": expansions}

