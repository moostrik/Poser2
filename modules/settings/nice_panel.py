"""NiceGUI settings panel — auto-generates a tabbed UI from a Settings root."""

from __future__ import annotations

import socket
from enum import Enum
from typing import Callable, get_origin, get_args

from nicegui import ui

from modules.settings.settings import Settings
from modules.settings import presets
from modules.settings.field import Field, Access
from modules.settings.widget import Widget, WidgetSize
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

    Returns a ``WidgetSize`` indicating the layout class:
        WidgetSize.full  — needs a full row (slider, text input)
        WidgetSize.small — compact inline control (switch, select, number)

    *polls* collects ``(settings, name, [last_value], setter)`` tuples.
    A timer created by the caller will poll these periodically to push
    external changes into the UI — thread-safe by construction.

    If the field has a non-empty ``description``, the control is wrapped
    in a container with a hover tooltip showing that description.
    """
    desc = field.description
    resolved = Widget.resolve(field)
    builder = _BUILDERS.get(resolved)
    if desc:
        with ui.element('div').classes('relative'):
            if builder is not None:
                result = builder(settings, name, field, polls)
            else:
                result = _build_fallback(settings, name, field, polls)
            with ui.icon('info_outline').classes(
                'text-grey-6 cursor-help absolute'
            ).style('font-size: 14px; bottom: -18px; right: -2px'):
                ui.tooltip(desc).props('anchor="bottom left" self="top right" :offset="[2, -14]" delay=500')
            return result
    else:
        if builder is not None:
            return builder(settings, name, field, polls)
        return _build_fallback(settings, name, field, polls)


# ---------------------------------------------------------------------------
# Builder registry — maps Widget constants to NiceGUI builder functions.
# Each builder receives (settings, name, field, polls) and returns a WidgetSize.
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
    is_disabled = field.access is Access.READ

    sw = ui.switch(label, value=value).props(
        "dense" + (" disable" if is_disabled else "")
    )

    if not is_disabled:
        def on_switch_change(e):
            setattr(settings, name, e.value)
        sw.on_value_change(on_switch_change)

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, sw=sw: sw.set_value(v)))
    return WidgetSize.small


@widget_builder(Widget.toggle)
def _build_toggle(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.access is Access.READ

    tg = ui.toggle({True: label, False: label}, value=value).props(
        "dense" + (" disable" if is_disabled else "")
    )

    if not is_disabled:
        def on_toggle_change(e):
            setattr(settings, name, e.value)
        tg.on_value_change(on_toggle_change)

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, tg=tg: tg.set_value(v)))
    return WidgetSize.small


# -- numeric builders --------------------------------------------------------

# Cycling accent colors for slider controls.
_SLIDER_COLORS = ["blue", "purple", "amber-8"]
_slider_color_index = [0]

def _next_slider_color() -> str:
    """Return the next color in the blue → purple → yellow cycle."""
    c = _SLIDER_COLORS[_slider_color_index[0] % len(_SLIDER_COLORS)]
    _slider_color_index[0] += 1
    return c


@widget_builder(Widget.slider)
def _build_slider(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.access is Access.READ
    step = field.step if field.step is not None else (1 if field.type_ is int else 0.01)
    big_step = step * 10
    color = _next_slider_color()

    with ui.column().classes("w-full gap-0 pt-2 pb-1"):
        with ui.row().classes("w-full items-center justify-between gap-0"):
            ui.label(label).classes("text-xs text-caption")
            if not is_disabled:
                with ui.row().classes("gap-0 items-center"):
                    btn_big_down = ui.button(
                        icon="keyboard_double_arrow_left", on_click=lambda: None,
                    ).props(f"flat dense round size=xs color={color}")
                    btn_down = ui.button(
                        icon="chevron_left", on_click=lambda: None,
                    ).props(f"flat dense round size=xs color={color}")
                    btn_up = ui.button(
                        icon="chevron_right", on_click=lambda: None,
                    ).props(f"flat dense round size=xs color={color}")
                    btn_big_up = ui.button(
                        icon="keyboard_double_arrow_right", on_click=lambda: None,
                    ).props(f"flat dense round size=xs color={color}")
        sl = ui.slider(
            min=field.min, max=field.max, step=step, value=value
        ).props(
            f"dense label-always color={color}"
            + (" disable" if is_disabled else "")
        )



    if not is_disabled:
        def nudge(delta):
            cur = sl.value
            new = max(field.min, min(field.max, field.type_(cur + delta)))
            sl.set_value(new)
            setattr(settings, name, new)

        btn_big_down.on_click(lambda: nudge(-big_step))
        btn_down.on_click(lambda: nudge(-step))
        btn_up.on_click(lambda: nudge(step))
        btn_big_up.on_click(lambda: nudge(big_step))

        def on_slider_change(e):
            setattr(settings, name, field.type_(e.value))
        sl.on_value_change(on_slider_change)

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, sl=sl: sl.set_value(v)))
    return WidgetSize.full


@widget_builder(Widget.number)
def _build_number(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.access is Access.READ

    num = ui.number(
        label=label,
        value=value,
        step=field.step if field.step is not None else (1 if field.type_ is int else 0.01),
        format="%.0f" if field.type_ is int else "%.2f",
    ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24")

    if not is_disabled:
        def on_num_change(e):
            if e.value is not None:
                setattr(settings, name, field.type_(e.value))
        num.on_value_change(on_num_change)

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, num=num: num.set_value(v)))
    return WidgetSize.small


@widget_builder(Widget.knob)
def _build_knob(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.access is Access.READ
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

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, kn=kn: kn.set_value(v)))
    return WidgetSize.small


# -- enum builders -----------------------------------------------------------

@widget_builder(Widget.select)
def _build_select(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.access is Access.READ

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

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, s=sel: s.set_value(v.name if isinstance(v, Enum) else v)))
    return WidgetSize.small


@widget_builder(Widget.radio)
def _build_radio(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.access is Access.READ

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

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, rg=rg: rg.set_value(v)))
    return WidgetSize.small


# -- string builders ---------------------------------------------------------

@widget_builder(Widget.input)
def _build_input(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.access is Access.READ

    inp = ui.input(label=label, value=value).props(
        "dense outlined" + (" disable" if is_disabled else "")
    )

    if not is_disabled:
        def on_input_change(e):
            setattr(settings, name, e.value)
        inp.on_value_change(on_input_change)

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, inp=inp: inp.set_value(v)))
    return WidgetSize.full


@widget_builder(Widget.ip)
def _build_ip(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.access is Access.READ

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

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, inp=inp: inp.set_value(v)))
    return WidgetSize.small


@widget_builder(Widget.textarea)
def _build_textarea(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.access is Access.READ

    ta = ui.textarea(label=label, value=value).props(
        "dense outlined" + (" disable" if is_disabled else "")
    )

    if not is_disabled:
        def on_ta_change(e):
            setattr(settings, name, e.value)
        ta.on_value_change(on_ta_change)

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, ta=ta: ta.set_value(v)))
    return WidgetSize.full


# -- color builders ----------------------------------------------------------

@widget_builder(Widget.color)
def _build_color(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.access is Access.READ

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

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, _ci=ci: _ci.set_value(v.to_hex() if isinstance(v, Color) else '#000000')))
    return WidgetSize.small


@widget_builder(Widget.color_alpha)
def _build_color_alpha(settings, name, field, polls):
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.access is Access.READ

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

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], _color_alpha_setter))
    return WidgetSize.small


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

    if field.access is not Access.WRITE:
        polls.append((settings, name, [list(value)], _list_setter))
    return WidgetSize.full


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
    is_disabled = field.access is Access.READ

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
        if field.access is not Access.WRITE:
            polls.append((settings, name, [value], _point_setter))
        return WidgetSize.full

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
        if field.access is not Access.WRITE:
            polls.append((settings, name, [value], _rect_setter))
        return WidgetSize.full

    # -- Generic fallback: read-only label -----------------------------------
    with ui.row().classes("items-center gap-2"):
        ui.label(label).classes("text-sm")
        lbl = ui.label(str(value)).classes("text-sm text-secondary")

    if field.access is not Access.WRITE:
        polls.append((settings, name, [value], lambda v, lbl=lbl: lbl.set_text(str(v))))
    return WidgetSize.small


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


def _build_settings_body(settings, timers, *, depth=0, expansions=None):
    """Emit the controls for a single Settings instance (no wrapper).

    Uses a 2-column grid for sliders and flows small controls (switches,
    selects, numbers) inline in a wrapping row for a compact layout.

    *depth* tracks nesting level (0 = top, 1+ = child of child).
    *expansions* collects ``ui.expansion`` elements for expand/collapse-all.
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
        if field.access is Access.INIT:
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
        with ui.grid(columns=2).classes("w-full gap-x-4 gap-y-2 poser-grid"):
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
        _build_settings_card(child_name, child, timers, depth=depth, expansions=expansions)


def _build_settings_card(name, settings, timers, *, depth=0, expansions=None):
    """Build a card for one Settings instance.

    At *depth* 0 the card is a collapsible ``ui.expansion``.
    At *depth* >= 1 it renders flat with a section divider + bold label.
    """
    if depth >= 1:
        # Flat rendering — no nested expansion
        ui.separator().classes("mt-2")
        ui.label(generate_label(name)).classes("text-sm font-bold text-primary")
        _build_settings_body(settings, timers, depth=depth + 1, expansions=expansions)
    else:
        exp = ui.expansion(
            generate_label(name), icon="settings",
        ).props("duration=0").classes("w-full")
        if expansions is not None:
            expansions.append(exp)
        with exp:
            _build_settings_body(settings, timers, depth=depth + 1, expansions=expansions)


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

    # -- Responsive CSS via @media (works on all browsers) -----------------
    ui.add_css('''
    @media (max-width: 639px) {
        .poser-grid { grid-template-columns: 1fr !important; }
    }
    .q-slider__pin { opacity: 0.65 !important; }
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
        d.style.cssText = "position:fixed;inset:0;background:black;z-index:99999;"
          + "display:flex;align-items:center;justify-content:center;pointer-events:none;";
        d.innerHTML = '<div style="text-align:center">'
          + '{_title_markup}'
          + '<span class="material-icons" style="font-size:80px;color:#3f3f46">'
          + 'power_settings_new</span></div>';
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
    tab_entries: list[tuple[str, Settings]] = []  # (name, settings)
    root_has_fields = any(
        f.visible and f.widget != Widget.button
        for f in root.fields.values()
    ) or any(f.visible for f in root.actions.values())
    if root_has_fields:
        tab_entries.append((type(root).__name__, root))
    for child_name, child in root.children.items():
        if _has_visible_content(child):
            tab_entries.append((child_name, child))

    with ui.column().classes("w-full sticky top-0 z-50 gap-0 bg-dark text-white").style(
        "background: #1d1d1d"
    ):
        with ui.row().classes("w-full items-center flex-wrap gap-1"):
            if title:
                ui.label(title).classes("text-2xl font-bold")
            if port is not None:
                with ui.button(icon="info").props("dense flat").tooltip("Show connection info"):
                    with ui.menu().props('anchor="bottom middle" self="top middle"'):
                        with ui.card().classes("gap-1").props("flat"):
                            ui.label("Connect").classes("text-base font-bold")
                            ui.separator()
                            for _ip in _get_local_ips():
                                _url = f"http://{_ip}:{port}"
                                ui.link(_url, _url, new_tab=True).classes("text-sm")
                            _url_local = f"http://localhost:{port}"
                            ui.link(_url_local, _url_local, new_tab=True).classes("text-sm")
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
        pinned_fields: list[tuple[Settings, str, Field]] = []
        pinned_actions: list[tuple[Settings, str, Field]] = []

        def _collect_pinned(settings: Settings) -> None:
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
            with ui.row().classes("w-full gap-4 flex-wrap items-end"):
                for settings, field_name, field in pinned_fields:
                    _build_field_control(settings, field_name, field, pinned_polls)
                for settings, action_name, action_field in pinned_actions:
                    _build_action_button(settings, action_name, action_field)
            _make_poll_timer(pinned_polls, timers)

        # Tabs inside the sticky header
        if not tab_entries:
            ui.label("No settings registered.")

        if tab_entries:
            with ui.tabs().classes("w-full") as tabs:
                tab_map = {}
                for label, _ in tab_entries:
                    tab_map[label] = ui.tab(generate_label(label))
    # -- end of sticky header --

    if not tab_entries:
        return

    with ui.tab_panels(tabs, value=tab_map[tab_entries[0][0]]).classes("w-full"):
        for label, root_settings in tab_entries:
            with ui.tab_panel(tab_map[label]):
                expansions: list = []

                # Expand / Collapse-all placeholder — filled after body build
                toggle_row = ui.row().classes("w-full justify-end mb-1")

                _build_settings_body(
                    root_settings, timers,
                    depth=0, expansions=expansions,
                )

                # Expand / Collapse-all toggle (only if there are expansions)
                if expansions:
                    _expanded = {"all": False}
                    def _toggle_all(exps=expansions, state=_expanded):
                        state["all"] = not state["all"]
                        for e in exps:
                            if state["all"]:
                                e.open()
                            else:
                                e.close()
                    with toggle_row:
                        ui.button(
                            "Expand / Collapse All",
                            icon="unfold_more",
                            on_click=_toggle_all,
                        ).props("dense flat size=sm")
                else:
                    toggle_row.delete()
