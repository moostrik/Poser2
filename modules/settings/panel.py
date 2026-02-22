"""NiceGUI settings panel — auto-generates a tabbed UI from a SettingsRegistry."""

from enum import Enum

from nicegui import ui

from modules.settings.base_settings import BaseSettings
from modules.settings.setting import Setting
from modules.settings.action import Action
from modules.settings.registry import SettingsRegistry


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


def _build_field_control(settings, name, field):
    """Create a NiceGUI control for a single Setting field."""
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    # -- Enum → select -------------------------------------------------------
    if isinstance(field.type_, type) and issubclass(field.type_, Enum):
        options = {m.value: m.name for m in field.type_}
        sel = ui.select(
            options=options,
            value=value.value,
            label=label,
        ).props("dense outlined" + (" disable" if is_disabled else ""))

        def on_select_change(e, f=field):
            setattr(settings, name, f.type_(e.value))

        sel.on_value_change(on_select_change)

        def update_select(v, sel=sel):
            sel.set_value(v.value if isinstance(v, Enum) else v)

        settings.on_change(name, update_select)
        return

    # -- bool → switch -------------------------------------------------------
    if field.type_ is bool:
        sw = ui.switch(label, value=value).props(
            "dense" + (" disable" if is_disabled else "")
        )

        def on_switch_change(e):
            setattr(settings, name, e.value)

        sw.on_value_change(on_switch_change)

        def update_switch(v, sw=sw):
            sw.set_value(v)

        settings.on_change(name, update_switch)
        return

    # -- int/float with min+max → slider + number ----------------------------
    if field.type_ in (int, float) and field.min is not None and field.max is not None:
        step = field.step if field.step is not None else (1 if field.type_ is int else 0.01)

        with ui.row().classes("items-center w-full gap-2"):
            ui.label(label).classes("w-32 text-sm")
            sl = ui.slider(
                min=field.min, max=field.max, step=step, value=value
            ).props("dense label-always" + (" disable" if is_disabled else "")).classes("flex-grow")
            num = ui.number(
                value=value, min=field.min, max=field.max, step=step,
                format=f"%.0f" if field.type_ is int else f"%.2f",
            ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24")

        def on_slider_change(e, num=num):
            val = field.type_(e.value)
            setattr(settings, name, val)
            num.set_value(val)

        def on_number_change(e, sl=sl):
            if e.value is not None:
                val = field.type_(e.value)
                setattr(settings, name, val)
                sl.set_value(val)

        sl.on_value_change(on_slider_change)
        num.on_value_change(on_number_change)

        def update_slider(v, sl=sl, num=num):
            sl.set_value(v)
            num.set_value(v)

        settings.on_change(name, update_slider)
        return

    # -- int/float without range → number input ------------------------------
    if field.type_ in (int, float):
        with ui.row().classes("items-center w-full gap-2"):
            ui.label(label).classes("w-32 text-sm")
            num = ui.number(
                value=value,
                step=field.step or (1 if field.type_ is int else 0.01),
                format=f"%.0f" if field.type_ is int else f"%.2f",
            ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-32")

        def on_num_change(e):
            if e.value is not None:
                setattr(settings, name, field.type_(e.value))

        num.on_value_change(on_num_change)

        def update_num(v, num=num):
            num.set_value(v)

        settings.on_change(name, update_num)
        return

    # -- str → text input ----------------------------------------------------
    if field.type_ is str:
        inp = ui.input(label=label, value=value).props(
            "dense outlined" + (" disable" if is_disabled else "")
        )

        def on_input_change(e):
            setattr(settings, name, e.value)

        inp.on_value_change(on_input_change)

        def update_input(v, inp=inp):
            inp.set_value(v)

        settings.on_change(name, update_input)
        return

    # -- Fallback: read-only label -------------------------------------------
    with ui.row().classes("items-center w-full gap-2"):
        ui.label(label).classes("w-32 text-sm")
        lbl = ui.label(str(value)).classes("text-sm text-gray-500")

    def update_label(v, lbl=lbl):
        lbl.set_text(str(v))

    settings.on_change(name, update_label)


def _build_action_button(settings, name, action):
    """Create a NiceGUI button for an Action."""
    label = generate_label(name)
    ui.button(label, on_click=lambda: action.fire(settings)).props("dense")


def _build_settings_card(name, settings):
    """Build a collapsible card for one BaseSettings instance."""
    with ui.expansion(generate_label(name), icon="settings").classes("w-full"):
        with ui.column().classes("w-full gap-1 p-2"):
            for field_name, field in settings.fields.items():
                if not field.visible:
                    continue
                if field.init_only:
                    # Show as read-only label
                    with ui.row().classes("items-center w-full gap-2"):
                        ui.label(generate_label(field_name)).classes("w-32 text-sm")
                        ui.label(str(getattr(settings, field_name))).classes(
                            "text-sm text-gray-500 italic"
                        )
                    continue
                _build_field_control(settings, field_name, field)

            # Actions
            action_items = [
                (n, a) for n, a in settings.actions.items() if a.visible
            ]
            if action_items:
                ui.separator()
                with ui.row().classes("gap-2"):
                    for action_name, action in action_items:
                        _build_action_button(settings, action_name, action)

            # Children (recursive)
            for child_name, child in settings.children.items():
                _build_settings_card(child_name, child)


def create_settings_panel(registry):
    """Build a full tabbed settings panel from a SettingsRegistry.

    Call this inside a NiceGUI page context::

        @ui.page("/")
        def index():
            create_settings_panel(registry)
    """
    group_map = registry.groups()
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
                for config_name in config_names:
                    settings = registry.get(config_name)
                    _build_settings_card(config_name, settings)
