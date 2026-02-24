"""NiceGUI settings panel — auto-generates a tabbed UI from a SettingsRegistry."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Callable, get_origin, get_args

from nicegui import app, ui

from modules.settings.base_settings import BaseSettings
from modules.settings import presets
from modules.settings.registry import SettingsRegistry
from modules.settings.setting import Setting
from modules.settings.widget import Widget
from modules.utils import Color, Point2f, Rect

# ---------------------------------------------------------------------------
# Poll rate for setting → UI synchronisation.
# All UI updates run on NiceGUI's event-loop timer, never on the thread that
# writes the setting.  This avoids WebSocket flooding (bind fired per-write)
# and thread-safety issues (GL thread calling NiceGUI API).
# ---------------------------------------------------------------------------
POLL_INTERVAL = 0.25  # seconds


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


def _build_field_control(settings, name, field, polls):
    """Create a NiceGUI control for a single Setting field.

    Returns the control's "size class":
        'full'  — needs a full row (slider, text input)
        'small' — compact inline control (switch, select, number)

    *polls* collects ``(settings, name, [last_value], setter)`` tuples.
    A timer created by the caller will poll these periodically to push
    external changes into the UI — thread-safe by construction.
    """
    resolved = Widget.resolve(field)
    builder = _BUILDERS.get(resolved)
    if builder is not None:
        return builder(settings, name, field, polls)
    # Fallback for types without a registered builder (Point2f, Rect, etc.)
    return _build_fallback(settings, name, field, polls)


# ---------------------------------------------------------------------------
# Builder registry — maps Widget constants to NiceGUI builder functions.
# Each builder receives (settings, name, field, polls) and returns 'full'
# or 'small'.  Use @widget_builder(Widget.xxx) to register.
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
    is_disabled = field.readonly

    sw = ui.switch(label, value=value).props(
        "dense" + (" disable" if is_disabled else "")
    )

    if not is_disabled:
        def on_switch_change(e):
            setattr(settings, name, e.value)
        sw.on_value_change(on_switch_change)

    polls.append((settings, name, [value], lambda v, sw=sw: sw.set_value(v)))
    return 'small'


@widget_builder(Widget.toggle)
def _build_toggle(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    tg = ui.toggle({True: label, False: label}, value=value).props(
        "dense" + (" disable" if is_disabled else "")
    )

    if not is_disabled:
        def on_toggle_change(e):
            setattr(settings, name, e.value)
        tg.on_value_change(on_toggle_change)

    polls.append((settings, name, [value], lambda v, tg=tg: tg.set_value(v)))
    return 'small'


# -- numeric builders --------------------------------------------------------

@widget_builder(Widget.slider)
def _build_slider(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly
    step = field.step if field.step is not None else (1 if field.type_ is int else 0.01)

    with ui.column().classes("w-full gap-0 pt-2 pb-1"):
        ui.label(label).classes("text-xs text-caption")
        sl = ui.slider(
            min=field.min, max=field.max, step=step, value=value
        ).props("dense label-always" + (" disable" if is_disabled else ""))

    if not is_disabled:
        def on_slider_change(e):
            setattr(settings, name, field.type_(e.value))
        sl.on_value_change(on_slider_change)

    polls.append((settings, name, [value], lambda v, sl=sl: sl.set_value(v)))
    return 'full'


@widget_builder(Widget.number)
def _build_number(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    num = ui.number(
        label=label,
        value=value,
        step=field.step or (1 if field.type_ is int else 0.01),
        format="%.0f" if field.type_ is int else "%.2f",
    ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24")

    if not is_disabled:
        def on_num_change(e):
            if e.value is not None:
                setattr(settings, name, field.type_(e.value))
        num.on_value_change(on_num_change)

    polls.append((settings, name, [value], lambda v, num=num: num.set_value(v)))
    return 'small'


@widget_builder(Widget.knob)
def _build_knob(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly
    step = field.step if field.step is not None else (1 if field.type_ is int else 0.01)
    min_val = field.min if field.min is not None else 0
    max_val = field.max if field.max is not None else 100

    with ui.column().classes("items-center gap-0"):
        kn = ui.knob(
            value=value, min=min_val, max=max_val, step=step,
            show_value=True, size="lg",
        ).props(
            "thickness=0.2" + (" disable" if is_disabled else "")
        )
        ui.label(label).classes("text-xs text-caption")

    if not is_disabled:
        def on_knob_change(e):
            setattr(settings, name, field.type_(e.value))
        kn.on_value_change(on_knob_change)

    polls.append((settings, name, [value], lambda v, kn=kn: kn.set_value(v)))
    return 'small'


# -- enum builders -----------------------------------------------------------

@widget_builder(Widget.select)
def _build_select(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    options = {m.name: m.name for m in field.type_}
    sel = ui.select(
        options=options,
        value=value.name,
        label=label,
    ).props("dense outlined" + (" disable" if is_disabled else "")).classes("min-w-[120px]")

    if not is_disabled:
        def on_select_change(e, f=field):
            setattr(settings, name, f.type_[e.value])
        sel.on_value_change(on_select_change)

    polls.append((settings, name, [value], lambda v, s=sel: s.set_value(v.name if isinstance(v, Enum) else v)))
    return 'small'


@widget_builder(Widget.radio)
def _build_radio(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    options = {m: generate_label(m.name) for m in field.type_}
    with ui.column().classes("gap-0 pt-1 pb-1"):
        ui.label(label).classes("text-xs text-caption")
        rg = ui.toggle(options=options, value=value).props(
            "dense" + (" disable" if is_disabled else "")
        )

    if not is_disabled:
        def on_radio_change(e):
            setattr(settings, name, e.value)
        rg.on_value_change(on_radio_change)

    polls.append((settings, name, [value], lambda v, rg=rg: rg.set_value(v)))
    return 'small'


# -- string builders ---------------------------------------------------------

@widget_builder(Widget.input)
def _build_input(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    inp = ui.input(label=label, value=value).props(
        "dense outlined" + (" disable" if is_disabled else "")
    )

    if not is_disabled:
        def on_input_change(e):
            setattr(settings, name, e.value)
        inp.on_value_change(on_input_change)

    polls.append((settings, name, [value], lambda v, inp=inp: inp.set_value(v)))
    return 'full'


@widget_builder(Widget.ip)
def _build_ip(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    inp = ui.input(
        label=label, value=value,
        validation={"Invalid IP": lambda v: all(
            p.isdigit() and 0 <= int(p) <= 255 for p in v.split(".")
        ) if v.count(".") == 3 else False},
    ).props(
        'dense outlined mask="###.###.###.###"' + (" disable" if is_disabled else "")
    ).classes("w-40")

    if not is_disabled:
        def on_ip_change(e):
            setattr(settings, name, e.value)
        inp.on_value_change(on_ip_change)

    polls.append((settings, name, [value], lambda v, inp=inp: inp.set_value(v)))
    return 'small'


@widget_builder(Widget.textarea)
def _build_textarea(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    ta = ui.textarea(label=label, value=value).props(
        "dense outlined" + (" disable" if is_disabled else "")
    )

    if not is_disabled:
        def on_ta_change(e):
            setattr(settings, name, e.value)
        ta.on_value_change(on_ta_change)

    polls.append((settings, name, [value], lambda v, ta=ta: ta.set_value(v)))
    return 'full'


# -- color builders ----------------------------------------------------------

@widget_builder(Widget.color)
def _build_color(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    hex_val = value.to_hex() if isinstance(value, Color) else '#000000'
    ci = ui.color_input(
        label=label, value=hex_val
    ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-40")

    if not is_disabled:
        def on_color_change(e):
            if e.value:
                c = Color.from_hex(e.value)
                setattr(settings, name, Color(c.r, c.g, c.b, 1.0))
        ci.on_value_change(on_color_change)

    polls.append((settings, name, [value], lambda v, _ci=ci: _ci.set_value(v.to_hex() if isinstance(v, Color) else '#000000')))
    return 'small'


@widget_builder(Widget.color_alpha)
def _build_color_alpha(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    hex_val = value.to_hex() if isinstance(value, Color) else '#000000'
    alpha = value.a if isinstance(value, Color) else 1.0

    with ui.row().classes("items-end gap-2"):
        ci = ui.color_input(
            label=label, value=hex_val
        ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-32")

        alpha_num = ui.number(
            label="A", value=alpha,
            min=0.0, max=1.0, step=0.01,
            format="%.2f",
        ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-20")

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

    polls.append((settings, name, [value], _color_alpha_setter))
    return 'small'


# -- list builders -----------------------------------------------------------

def _build_sortable_list(settings, name, field, polls, *, with_checkboxes: bool):
    """Shared implementation for checklist (with checkboxes) and order (without)."""
    value = getattr(settings, name)
    label = generate_label(name)

    elem_type = get_args(field.type_)[0] if get_args(field.type_) else None
    if elem_type is None or not (isinstance(elem_type, type) and issubclass(elem_type, Enum)):
        # Fallback for non-enum lists
        return _build_fallback(settings, name, field, polls)

    all_members = list(elem_type)
    active_set = set(value)

    with ui.column().classes("w-full gap-1 pt-2 pb-1"):
        ui.label(label).classes("text-xs text-caption")
        container = ui.column().classes("w-full gap-0 border rounded p-1")

        if with_checkboxes:
            ordered = list(value) + [m for m in all_members if m not in active_set]
        else:
            # Order-only: all members always active, ordered as in value
            ordered = list(value)
        state = {"order": ordered, "active": set(value) if with_checkboxes else set(all_members)}

        def _rebuild(cont, st, _settings=settings, _name=name, _elem=elem_type, _cb=with_checkboxes):
            cont.clear()
            with cont:
                for idx, member in enumerate(st["order"]):
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

                        ui.label(generate_label(member.name)).classes("text-sm flex-1")

                        up_btn = ui.button(
                            icon="arrow_upward",
                            on_click=lambda _, i=idx: _move(cont, st, i, -1),
                        ).props("dense flat size=xs").classes("my-0")
                        down_btn = ui.button(
                            icon="arrow_downward",
                            on_click=lambda _, i=idx: _move(cont, st, i, 1),
                        ).props("dense flat size=xs").classes("my-0")

                        if idx == 0:
                            up_btn.props("disable")
                        if idx == len(st["order"]) - 1:
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
            new_list = [m for m in st["order"] if m in st["active"]]
            setattr(settings, name, new_list)
            _rebuild(cont, st)

        _rebuild(container, state)

    def _list_setter(v, _cont=container, _st=state, _elem=elem_type, _cb=with_checkboxes):
        active_set_inner = set(v)
        _st["active"] = active_set_inner
        if _cb:
            _st["order"] = list(v) + [m for m in list(_elem) if m not in active_set_inner]
        else:
            _st["order"] = list(v)
        _rebuild(_cont, _st)

    polls.append((settings, name, [list(value)], _list_setter))
    return 'full'


@widget_builder(Widget.checklist)
def _build_checklist(settings, name, field, polls):
    return _build_sortable_list(settings, name, field, polls, with_checkboxes=True)


@widget_builder(Widget.order)
def _build_order(settings, name, field, polls):
    return _build_sortable_list(settings, name, field, polls, with_checkboxes=False)


# -- Fallback for unregistered types (Point2f, Rect, unknown) ----------------

def _build_fallback(settings, name, field, polls):
    """Build a control for types without a registered Widget builder."""
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    # -- Point2f → x/y number row --------------------------------------------
    if field.type_ is Point2f:
        with ui.row().classes("items-end gap-2"):
            ui.label(label).classes("text-xs text-caption self-center")
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
        polls.append((settings, name, [value], _point_setter))
        return 'full'

    # -- Rect → x/y/w/h number row -------------------------------------------
    if field.type_ is Rect:
        with ui.row().classes("items-end gap-2"):
            ui.label(label).classes("text-xs text-caption self-center")
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
        polls.append((settings, name, [value], _rect_setter))
        return 'full'

    # -- Generic fallback: read-only label -----------------------------------
    with ui.row().classes("items-center gap-2"):
        ui.label(label).classes("text-sm")
        lbl = ui.label(str(value)).classes("text-sm text-secondary")

    polls.append((settings, name, [value], lambda v, lbl=lbl: lbl.set_text(str(v))))
    return 'small'


def _build_action_button(settings, name, field):
    """Create a NiceGUI button for a Widget.button Setting."""
    label = generate_label(name)
    ui.button(label, on_click=lambda: field.fire(settings)).props("dense")


def _has_visible_content(settings):
    """Return True if settings has any visible fields, actions, or children."""
    if any(f.visible for f in settings.fields.values()):
        return True
    if settings.actions:
        return True
    if settings.children:
        return True
    return False


def _make_poll_timer(polls, timers):
    """Create a single ``ui.timer`` that polls all entries in *polls*.

    Each entry is ``(settings, name, [last_value], setter)``.
    The timer runs on NiceGUI's event loop so UI calls are thread-safe.
    *timers* collects the timer for later cleanup reference.
    """
    if not polls:
        return

    def _tick():
        for s, fname, last, setter in polls:
            cur = getattr(s, fname)
            if cur != last[0]:
                last[0] = cur
                setter(cur)

    timers.append(ui.timer(POLL_INTERVAL, _tick))


def _build_settings_body(settings, timers):
    """Emit the controls for a single BaseSettings instance (no wrapper).

    Uses a 2-column grid for sliders and flows small controls (switches,
    selects, numbers) inline in a wrapping row for a compact layout.
    """
    polls: list[tuple] = []

    # Separate fields into full-width (sliders) and small (inline) controls
    full_fields = []
    small_fields = []
    init_only_fields = []

    # -- Full / small classification via Widget.resolve() --------------------
    _FULL_WIDGETS = {Widget.slider, Widget.input, Widget.textarea, Widget.checklist, Widget.order}

    for field_name, field in settings.fields.items():
        if not field.visible:
            continue
        if field.widget == Widget.button:
            continue  # rendered in the actions section
        if field.init_only:
            init_only_fields.append(field_name)
            continue
        resolved = Widget.resolve(field)
        if resolved in _FULL_WIDGETS or field.type_ in (Point2f, Rect):
            full_fields.append(field_name)
        else:
            small_fields.append(field_name)

    # Init-only fields as compact read-only labels
    if init_only_fields:
        with ui.row().classes("w-full gap-4 flex-wrap"):
            for field_name in init_only_fields:
                with ui.row().classes("items-center gap-1"):
                    ui.label(generate_label(field_name)).classes("text-xs text-caption")
                    ui.label(str(getattr(settings, field_name))).classes(
                        "text-sm text-secondary italic"
                    )

    # Small controls (switches, selects, numbers) in a wrapping row
    if small_fields:
        with ui.row().classes("w-full gap-4 flex-wrap items-end"):
            for field_name in small_fields:
                _build_field_control(settings, field_name, settings.fields[field_name], polls)

    # Full-width controls (sliders, text) in a 2-column grid
    if full_fields:
        with ui.grid(columns=2).classes("w-full gap-x-4 gap-y-2"):
            for field_name in full_fields:
                _build_field_control(settings, field_name, settings.fields[field_name], polls)

    # Create one poll timer for all fields in this card
    _make_poll_timer(polls, timers)

    # Actions (Widget.button fields)
    action_items = [
        (n, f) for n, f in settings.actions.items() if f.visible
    ]
    if action_items:
        ui.separator()
        with ui.row().classes("gap-2"):
            for action_name, action_field in action_items:
                _build_action_button(settings, action_name, action_field)

    # Children (recursive)
    for child_name, child in settings.children.items():
        _build_settings_card(child_name, child, timers)


def _build_settings_card(name, settings, timers):
    """Build a collapsible card for one BaseSettings instance."""
    with ui.expansion(generate_label(name), icon="settings").props("duration=0").classes("w-full"):
        _build_settings_body(settings, timers)


def _build_preset_controls(registry):
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
            presets.load(registry, p)
            ui.notify(f"Loaded '{name}'", type="positive")

    dropdown.on_value_change(_on_preset_change)

    # -- Action buttons ----------------------------------------------------
    def do_save():
        preset_name = dropdown.value
        if not preset_name:
            ui.notify("Select a preset to overwrite, or use Save As", type="warning")
            return
        presets.save(registry, presets.path(preset_name))
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
                    presets.save(registry, presets.path(preset_name))
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

    ui.button(icon="save", on_click=do_save).props("dense flat").tooltip("Save (overwrite selected)")
    ui.button(icon="save_as", on_click=do_save_as).props("dense flat").tooltip("Save as new preset")
    ui.button(icon="delete", on_click=do_delete).props("dense flat color=negative").tooltip("Delete preset")


def create_settings_panel(
    registry,
    *,
    title: str = "",
    on_exit=None,
) -> None:
    """Build a full tabbed settings panel from a SettingsRegistry.

    Call this inside a NiceGUI page context::

        @ui.page("/")
        def index():
            create_settings_panel(registry, title="POSER", on_exit=stop)
    """

    # Force dark mode for consistent styling
    ui.dark_mode(True)

    # Timers for this client session — NiceGUI auto-cleans on disconnect.
    timers: list = []

    # -- Header row: title | preset controls | exit button -----------------
    with ui.row().classes("w-full items-center flex-nowrap gap-0"):
        if title:
            ui.label(title).classes("text-2xl font-bold")
        with ui.row().classes("flex-1 items-center gap-1 flex-nowrap justify-center"):
            _build_preset_controls(registry)
        if on_exit:
            ui.button(icon="power_settings_new", on_click=on_exit).props(
                "dense flat color=negative"
            ).tooltip("Exit application")

    # Collect pinned fields and actions from all registered modules (recursing into children)
    pinned_fields: list[tuple[BaseSettings, str, Setting]] = []
    pinned_actions: list[tuple[BaseSettings, str, Setting]] = []

    def _collect_pinned(settings: BaseSettings) -> None:
        for field_name, field in settings.fields.items():
            if field.pinned and field.visible:
                pinned_fields.append((settings, field_name, field))
        for action_name, action_field in settings.actions.items():
            if action_field.pinned and action_field.visible:
                pinned_actions.append((settings, action_name, action_field))
        for child in settings.children.values():
            _collect_pinned(child)

    for module_name in registry._modules:
        _collect_pinned(registry.get(module_name))

    # Render pinned fields and actions in a compact row above the tabs
    if pinned_fields or pinned_actions:
        pinned_polls: list[tuple] = []
        with ui.row().classes("w-full gap-4 flex-wrap items-end"):
            for settings, field_name, field in pinned_fields:
                _build_field_control(settings, field_name, field, pinned_polls)
            for settings, action_name, action_field in pinned_actions:
                _build_action_button(settings, action_name, action_field)
        _make_poll_timer(pinned_polls, timers)

    # Filter out groups where all settings have no visible content
    group_map = {
        g: [n for n in names if _has_visible_content(registry.get(n))]
        for g, names in registry.groups().items()
    }
    group_map = {g: names for g, names in group_map.items() if names}

    if not group_map:
        ui.label("No settings registered.")
        return

    group_names = list(group_map.keys())

    with ui.tabs().classes("w-full") as tabs:
        tab_map = {}
        for group in group_names:
            tab_map[group] = ui.tab(generate_label(group))

    with ui.tab_panels(tabs, value=tab_map[group_names[0]]).classes("w-full"):
        for group, config_names in group_map.items():
            with ui.tab_panel(tab_map[group]):
                if len(config_names) == 1:
                    # Single config in group — render directly, no
                    # redundant expansion with the same name as the tab.
                    _build_settings_body(registry.get(config_names[0]), timers)
                else:
                    for config_name in config_names:
                        settings = registry.get(config_name)
                        _build_settings_card(config_name, settings, timers)
