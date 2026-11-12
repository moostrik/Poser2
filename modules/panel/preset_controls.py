"""The preset controls of the header: the startup star, the preset dropdown and its buttons."""

from __future__ import annotations

import logging
from typing import Protocol

from nicegui import ui

from modules.settings import presets

logger = logging.getLogger(__name__)


class PresetStore(Protocol):
    """Where a preset is loaded into and saved from: the app's settings tree, in the app's process.
    The files themselves (the list, the startup star, a delete) are the panel's."""

    def load(self, name: str) -> bool: ...

    def save(self, name: str) -> bool: ...


def preset_controls(store: PresetStore) -> None:
    """The star that marks the startup preset, the dropdown that loads a preset when picked, and
    the reload, save, save-as and delete buttons, in the caller's row."""
    names = presets.scan()
    startup = presets.get_startup()
    initial = startup if startup in names else (names[0] if names else None)

    def set_startup() -> None:
        name = dropdown.value
        if not name:
            ui.notify("Select a preset first", type="warning")
            return
        presets.set_startup(name)
        show_star(name)
        ui.notify(f"'{name}' will load on next startup", type="positive")

    star = ui.button(icon="star", on_click=set_startup).props("dense flat").tooltip("Set as startup preset")

    def show_star(selected: str | None) -> None:
        """The star lit when the selected preset is the startup one, grey and outlined otherwise."""
        if selected and selected == presets.get_startup():
            star.props("color=warning", remove="outline")
        else:
            star.props("color=grey outline")

    show_star(initial)

    dropdown = ui.select(options=names, value=initial, label="Preset").props("dense outlined").classes("min-w-[180px]")

    def refresh_dropdown() -> None:
        dropdown.set_options(presets.scan())

    def on_preset_change(e) -> None:
        name = e.value
        show_star(name)
        if not name:
            return
        if presets.path(name).exists():
            notify_load(name)

    dropdown.on_value_change(on_preset_change)

    def notify_load(name: str) -> None:
        if store.load(name):
            ui.notify(f"Loaded '{name}'", type="positive")
        else:
            ui.notify(f"Preset '{name}' could not be loaded", type="negative")

    def load() -> None:
        name = dropdown.value
        if not name:
            ui.notify("Select a preset first", type="warning")
            return
        if presets.path(name).exists():
            notify_load(name)
        else:
            ui.notify(f"Preset '{name}' not found", type="negative")

    def save() -> None:
        name = dropdown.value
        if not name:
            ui.notify("Select a preset to overwrite, or use Save As", type="warning")
            return
        if store.save(name):
            ui.notify(f"Saved '{name}'", type="positive")
        else:
            ui.notify(f"Preset '{name}' could not be saved", type="negative")

    def save_as() -> None:
        with ui.dialog() as dialog, ui.card().classes("min-w-[300px]"):
            ui.label("Save As").classes("text-lg font-bold")
            name_input = ui.input(label="Preset name", value=dropdown.value or "").props("dense outlined autofocus")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def confirm() -> None:
                    name = (name_input.value or "").strip()
                    if not name:
                        ui.notify("Enter a preset name", type="warning")
                        return
                    try:
                        presets.validate_name(name)
                    except ValueError:
                        ui.notify("Invalid preset name — no slashes or leading dots", type="negative")
                        return
                    if not store.save(name):
                        ui.notify(f"Preset '{name}' could not be saved", type="negative")
                        return
                    refresh_dropdown()
                    dropdown.set_value(name)
                    ui.notify(f"Saved '{name}'", type="positive")
                    dialog.close()

                ui.button("Save", on_click=confirm).props("flat color=primary")

            name_input.on("keydown.enter", lambda _: confirm())

        dialog.open()

    def delete() -> None:
        name = dropdown.value
        if not name:
            ui.notify("Select a preset to delete", type="warning")
            return

        with ui.dialog() as dialog, ui.card():
            ui.label(f"Delete '{name}'?")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def confirm() -> None:
                    path = presets.path(name)
                    if path.exists():
                        path.unlink()
                    refresh_dropdown()
                    dropdown.set_value(None)
                    ui.notify(f"Deleted '{name}'", type="info")
                    dialog.close()

                ui.button("Delete", on_click=confirm).props("flat color=negative")

        dialog.open()

    ui.button(icon="refresh", on_click=load).props("dense flat").tooltip("Reload selected preset")
    ui.button(icon="save", on_click=save).props("dense flat").tooltip("Save (overwrite selected)")
    ui.button(icon="save_as", on_click=save_as).props("dense flat").tooltip("Save as new preset")
    ui.button(icon="delete", on_click=delete).props("dense flat color=negative").tooltip("Delete preset")
