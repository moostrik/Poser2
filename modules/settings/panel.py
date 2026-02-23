"""NiceGUI settings panel — auto-generates a tabbed UI from a SettingsRegistry."""

from enum import Enum
from pathlib import Path

from nicegui import app, ui

from modules.settings.base_settings import BaseSettings
from modules.settings.setting import Setting
from modules.settings.action import Action
from modules.settings.registry import SettingsRegistry

SETTINGS_DIR = Path("files/settings")
PRESET_SUFFIX = ".reactive.json"


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


def _build_field_control(settings, name, field, cleanup):
    """Create a NiceGUI control for a single Setting field.

    Returns the control's "size class":
        'full'  — needs a full row (slider, text input)
        'small' — compact inline control (switch, select, number)

    *cleanup* collects (settings, field, callback) tuples so the caller
    can deregister them when the browser client disconnects.
    """
    value = getattr(settings, name)
    label = generate_label(name)
    is_disabled = field.readonly

    # -- Enum → select -------------------------------------------------------
    if isinstance(field.type_, type) and issubclass(field.type_, Enum):
        options = {m.name: m.name for m in field.type_}
        sel = ui.select(
            options=options,
            value=value.name,
            label=label,
        ).props("dense outlined" + (" disable" if is_disabled else "")).classes("min-w-[120px]")

        def on_select_change(e, f=field):
            setattr(settings, name, f.type_[e.value])

        sel.on_value_change(on_select_change)

        def update_select(v, sel=sel):
            sel.set_value(v.name if isinstance(v, Enum) else v)

        settings.bind(field, update_select)
        cleanup.append((settings, field, update_select))
        return 'small'

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

        settings.bind(field, update_switch)
        cleanup.append((settings, field, update_switch))
        return 'small'

    # -- int/float with min+max → slider -------------------------------------
    if field.type_ in (int, float) and field.min is not None and field.max is not None:
        step = field.step if field.step is not None else (1 if field.type_ is int else 0.01)

        with ui.column().classes("w-full gap-0 pt-2 pb-1"):
            ui.label(label).classes("text-xs text-gray-400")
            sl = ui.slider(
                min=field.min, max=field.max, step=step, value=value
            ).props("dense label-always" + (" disable" if is_disabled else ""))

        def on_slider_change(e):
            setattr(settings, name, field.type_(e.value))

        sl.on_value_change(on_slider_change)

        def update_slider(v, sl=sl):
            sl.set_value(v)

        settings.bind(field, update_slider)
        cleanup.append((settings, field, update_slider))
        return 'full'

    # -- int/float without range → number input ------------------------------
    if field.type_ in (int, float):
        num = ui.number(
            label=label,
            value=value,
            step=field.step or (1 if field.type_ is int else 0.01),
            format=f"%.0f" if field.type_ is int else f"%.2f",
        ).props("dense outlined" + (" disable" if is_disabled else "")).classes("w-24")

        def on_num_change(e):
            if e.value is not None:
                setattr(settings, name, field.type_(e.value))

        num.on_value_change(on_num_change)

        def update_num(v, num=num):
            num.set_value(v)

        settings.bind(field, update_num)
        cleanup.append((settings, field, update_num))
        return 'small'

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

        settings.bind(field, update_input)
        cleanup.append((settings, field, update_input))
        return 'full'

    # -- Fallback: read-only label -------------------------------------------
    with ui.row().classes("items-center gap-2"):
        ui.label(label).classes("text-sm")
        lbl = ui.label(str(value)).classes("text-sm text-gray-500")

    def update_label(v, lbl=lbl):
        lbl.set_text(str(v))

    settings.bind(field, update_label)
    cleanup.append((settings, field, update_label))
    return 'small'


def _build_action_button(settings, name, action):
    """Create a NiceGUI button for an Action."""
    label = generate_label(name)
    ui.button(label, on_click=lambda: action.fire(settings)).props("dense")


def _build_settings_card(name, settings, cleanup):
    """Build a collapsible card for one BaseSettings instance.

    Uses a 2-column grid for sliders and flows small controls (switches,
    selects, numbers) inline in a wrapping row for a compact layout.
    """
    with ui.expansion(generate_label(name), icon="settings").props("duration=0").classes("w-full"):
        # Separate fields into full-width (sliders) and small (inline) controls
        full_fields = []
        small_fields = []
        init_only_fields = []

        for field_name, field in settings.fields.items():
            if not field.visible:
                continue
            if field.init_only:
                init_only_fields.append(field_name)
                continue
            # Classify by type
            if field.type_ in (int, float) and field.min is not None and field.max is not None:
                full_fields.append(field_name)
            elif field.type_ is str:
                full_fields.append(field_name)
            else:
                small_fields.append(field_name)

        # Init-only fields as compact read-only labels
        if init_only_fields:
            with ui.row().classes("w-full gap-4 flex-wrap"):
                for field_name in init_only_fields:
                    with ui.row().classes("items-center gap-1"):
                        ui.label(generate_label(field_name)).classes("text-xs text-gray-400")
                        ui.label(str(getattr(settings, field_name))).classes(
                            "text-sm text-gray-500 italic"
                        )

        # Small controls (switches, selects, numbers) in a wrapping row
        if small_fields:
            with ui.row().classes("w-full gap-4 flex-wrap items-end"):
                for field_name in small_fields:
                    _build_field_control(settings, field_name, settings.fields[field_name], cleanup)

        # Full-width controls (sliders, text) in a 2-column grid
        if full_fields:
            with ui.grid(columns=2).classes("w-full gap-x-4 gap-y-2"):
                for field_name in full_fields:
                    _build_field_control(settings, field_name, settings.fields[field_name], cleanup)

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
            _build_settings_card(child_name, child, cleanup)


def _scan_presets():
    """Return sorted list of preset names (without suffix) from the settings directory."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        p.name.removesuffix(PRESET_SUFFIX)
        for p in SETTINGS_DIR.glob(f"*{PRESET_SUFFIX}")
    )


def _build_save_load_toolbar(registry):
    """Build a preset toolbar: dropdown, name input, Save / Load / Delete buttons."""
    presets = _scan_presets()

    with ui.row().classes("w-full items-end gap-2 flex-nowrap"):
        dropdown = ui.select(
            options=presets,
            value=presets[0] if presets else None,
            label="Preset",
        ).props("dense outlined clearable").classes("min-w-[180px] flex-1")

        name_input = ui.input(
            label="Name",
            value=dropdown.value or "",
        ).props("dense outlined").classes("min-w-[140px] flex-1")

        # Sync: selecting a preset fills the name input
        def on_dropdown_change(e):
            if e.value:
                name_input.set_value(e.value)

        dropdown.on_value_change(on_dropdown_change)

        def _refresh_dropdown():
            updated = _scan_presets()
            dropdown.set_options(updated)

        def do_save():
            preset_name = (name_input.value or "").strip()
            if not preset_name:
                ui.notify("Enter a preset name", type="warning")
                return
            path = SETTINGS_DIR / f"{preset_name}{PRESET_SUFFIX}"
            registry.save(path)
            _refresh_dropdown()
            dropdown.set_value(preset_name)
            ui.notify(f"Saved '{preset_name}'", type="positive")

        def do_load():
            preset_name = dropdown.value
            if not preset_name:
                ui.notify("Select a preset to load", type="warning")
                return
            path = SETTINGS_DIR / f"{preset_name}{PRESET_SUFFIX}"
            if not path.exists():
                ui.notify(f"Preset '{preset_name}' not found", type="negative")
                return
            registry.load(path)
            ui.notify(f"Loaded '{preset_name}'", type="positive")

        def do_delete():
            preset_name = dropdown.value
            if not preset_name:
                ui.notify("Select a preset to delete", type="warning")
                return
            path = SETTINGS_DIR / f"{preset_name}{PRESET_SUFFIX}"
            if path.exists():
                path.unlink()
            _refresh_dropdown()
            dropdown.set_value(None)
            name_input.set_value("")
            ui.notify(f"Deleted '{preset_name}'", type="info")

        ui.button(icon="save", on_click=do_save).props("dense flat").tooltip("Save preset")
        ui.button(icon="file_open", on_click=do_load).props("dense flat").tooltip("Load preset")
        ui.button(icon="delete", on_click=do_delete).props("dense flat color=negative").tooltip("Delete preset")


def create_settings_panel(registry):
    """Build a full tabbed settings panel from a SettingsRegistry.

    Call this inside a NiceGUI page context::

        @ui.page("/")
        def index():
            create_settings_panel(registry)
    """
    # Track all (settings, field_name, callback) for this client session
    cleanup: list[tuple] = []

    _build_save_load_toolbar(registry)

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
                    _build_settings_card(config_name, settings, cleanup)

    # Deregister UI callbacks when the browser tab / client disconnects
    def _on_disconnect():
        for s, fld, cb in cleanup:
            try:
                s.unbind(fld, cb)
            except (KeyError, ValueError):
                pass
        cleanup.clear()

    app.on_disconnect(_on_disconnect)
