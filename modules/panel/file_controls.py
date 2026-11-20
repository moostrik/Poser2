"""The settings file controls of the header, a row for each root group: the startup star, the file dropdown and
its buttons."""

from __future__ import annotations

import logging
from typing import Protocol

from nicegui import ui

from modules.settings import SettingsStore

from .controls import generate_label

logger = logging.getLogger(__name__)


class SettingsLink(Protocol):
    """Where a root group's file is loaded into and saved from: the app's settings tree, in the app's process.
    The files themselves (the list, the startup star, a delete) are the panel's `SettingsStore`."""

    def load(self, group: str, name: str) -> bool: ...

    def save(self, group: str, name: str) -> bool: ...


def file_controls(link: SettingsLink, store: SettingsStore, group: str) -> None:
    """For root group *group*: the star that marks its startup file, the dropdown, labelled by the group, that
    loads a file when picked, and the reload, save, save-as and delete buttons, in the caller's row."""
    label = generate_label(group)
    names = store.scan(group)
    startup = store.startup(group)
    initial = startup if startup in names else (names[0] if names else None)

    def set_startup() -> None:
        name = dropdown.value
        if not name:
            ui.notify(f"Select a {label} file first", type="warning")
            return
        store.set_startup(group, name)
        show_star(name)
        ui.notify(f"{label} '{name}' will load on next startup", type="positive")

    star = ui.button(icon="star", on_click=set_startup).props("dense flat").tooltip(f"Set as startup {label} file")

    def show_star(selected: str | None) -> None:
        """The star lit when the selected file is the startup one, grey and outlined otherwise."""
        if selected and selected == store.startup(group):
            star.props("color=warning", remove="outline")
        else:
            star.props("color=grey outline")

    show_star(initial)

    dropdown = ui.select(options=names, value=initial, label=label).props("dense outlined").classes("min-w-[180px]")

    def refresh_dropdown() -> None:
        dropdown.set_options(store.scan(group))

    def on_change(e) -> None:
        name = e.value
        show_star(name)
        if not name:
            return
        if store.path(group, name).exists():
            notify_load(name)

    dropdown.on_value_change(on_change)

    def notify_load(name: str) -> None:
        if link.load(group, name):
            ui.notify(f"Loaded {label} '{name}'", type="positive")
        else:
            ui.notify(f"{label} '{name}' could not be loaded", type="negative")

    def load() -> None:
        name = dropdown.value
        if not name:
            ui.notify(f"Select a {label} file first", type="warning")
            return
        if store.path(group, name).exists():
            notify_load(name)
        else:
            ui.notify(f"{label} '{name}' not found", type="negative")

    def save() -> None:
        name = dropdown.value
        if not name:
            ui.notify(f"Select a {label} file to overwrite, or use Save As", type="warning")
            return
        if link.save(group, name):
            ui.notify(f"Saved {label} '{name}'", type="positive")
        else:
            ui.notify(f"{label} '{name}' could not be saved", type="negative")

    def save_as() -> None:
        with ui.dialog() as dialog, ui.card().classes("min-w-[300px]"):
            ui.label(f"Save {label} As").classes("text-lg font-bold")
            name_input = ui.input(label=f"{label} name", value=dropdown.value or "").props("dense outlined autofocus")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def confirm() -> None:
                    name = (name_input.value or "").strip()
                    if not name:
                        ui.notify(f"Enter a {label} name", type="warning")
                        return
                    try:
                        store.validate_name(name)
                    except ValueError:
                        ui.notify("Invalid name — no slashes or leading dots", type="negative")
                        return
                    if not link.save(group, name):
                        ui.notify(f"{label} '{name}' could not be saved", type="negative")
                        return
                    refresh_dropdown()
                    dropdown.set_value(name)
                    ui.notify(f"Saved {label} '{name}'", type="positive")
                    dialog.close()

                ui.button("Save", on_click=confirm).props("flat color=primary")

            name_input.on("keydown.enter", lambda _: confirm())

        dialog.open()

    def delete() -> None:
        name = dropdown.value
        if not name:
            ui.notify(f"Select a {label} file to delete", type="warning")
            return

        with ui.dialog() as dialog, ui.card():
            ui.label(f"Delete {label} '{name}'?")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def confirm() -> None:
                    store.delete(group, name)
                    refresh_dropdown()
                    dropdown.set_value(None)
                    ui.notify(f"Deleted {label} '{name}'", type="info")
                    dialog.close()

                ui.button("Delete", on_click=confirm).props("flat color=negative")

        dialog.open()

    ui.button(icon="refresh", on_click=load).props("dense flat").tooltip(f"Reload the selected {label} file")
    ui.button(icon="save", on_click=save).props("dense flat").tooltip(f"Save (overwrite the selected {label} file)")
    ui.button(icon="save_as", on_click=save_as).props("dense flat").tooltip(f"Save as a new {label} file")
    ui.button(icon="delete", on_click=delete).props("dense flat color=negative").tooltip(f"Delete the {label} file")
