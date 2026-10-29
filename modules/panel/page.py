"""The page: the panel's header, preset controls, pinned fields, tabs and cards, built from a settings tree.

Every function builds its part inside the current NiceGUI page context, for one client; what the
panel remembers between page loads lives in the `PanelState` they are given. The page's stylesheet
and script are the static files in `static/`, served under `STATIC_URL`.
"""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path
from typing import Callable, NamedTuple

from nicegui import ui

from modules.log_config import LogRingBuffer
from modules.settings import BaseSettings, Field, Access, Widget

from .controls import ActionButton, Control, PinnedReadout, access_css_class, build_control, build_entry, generate_label
from .log_drawer import LOG_DRAWER_DEFAULT_HEIGHT, build_log_drawer
from .poll import POLL_INTERVAL, tick
from .preset_controls import preset_controls
from .state import PanelState

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_URL = "/panel"

# The background of a card by its depth, lighter the deeper it sits.
DEPTH_BG = ["bg-grey-9", "bg-grey-8", "bg-grey-7", "bg-grey-6", "bg-grey-5"]


class HideToggle(NamedTuple):
    """One of the three toggles of a tab: hides the fields of one access type."""
    icon: str
    key: str        # the state key within the tab
    css: str        # the class on the tab's panel that hides the fields
    tooltip: str
    default: bool   # hidden until the state says otherwise


TOGGLES = [
    HideToggle("construction",  "__hide_init__",     "hide-init",     "Init fields",     True),
    HideToggle("monitor_heart", "__hide_feedback__", "hide-feedback", "Feedback fields", False),
    HideToggle("tune",          "__hide_input__",    "hide-input",    "Input fields",    False),
]


# -- the settings tree -------------------------------------------------------------------

def has_visible_content(settings: BaseSettings) -> bool:
    """Whether a settings instance holds a visible field, an action or a child with one."""
    if any(f.visible for f in settings.fields.values()):
        return True
    if settings.actions:
        return True
    return any(has_visible_content(c) for c in settings.children.values())


def content_access_types(settings: BaseSettings) -> frozenset[Access]:
    """The access types of the visible fields, down the tree, with READWRITE counted as WRITE.

    A settings instance with actions counts as holding every type, so its card is never hidden.
    """
    if settings.actions:
        return frozenset({Access.WRITE, Access.READ, Access.INIT})
    types: set[Access] = set()
    for f in settings.fields.values():
        if f.visible:
            types.add(Access.WRITE if f.access in (Access.WRITE, Access.READWRITE) else f.access)
    for child in settings.children.values():
        if has_visible_content(child):
            types |= content_access_types(child)
    return frozenset(types)


# The class that hides a card when the toggles hide every access type it holds.
ACCESS_CLASS = {
    frozenset({Access.INIT}):                            "poser-only-init",
    frozenset({Access.READ}):                            "poser-only-feedback",
    frozenset({Access.WRITE}):                           "poser-only-input",
    frozenset({Access.INIT, Access.READ}):               "poser-only-init-feedback",
    frozenset({Access.INIT, Access.WRITE}):              "poser-only-init-input",
    frozenset({Access.READ, Access.WRITE}):              "poser-only-feedback-input",
    frozenset({Access.INIT, Access.READ, Access.WRITE}): "poser-only-all",
}


def tab_entries(root: BaseSettings) -> list[tuple[str, BaseSettings]]:
    """The children of the root that get a tab: those with visible content."""
    return [(name, child) for name, child in root.children.items() if has_visible_content(child)]


# -- cards -------------------------------------------------------------------------------

def body(settings: BaseSettings, controls: list[Control], state: PanelState, *,
         depth: int, expansions: list[ui.expansion], path: str) -> None:
    """The controls of one settings instance, in declaration order in one wrapping row, then a
    card per child with visible content.

    A field's `newline` is a full-width break that carries the field's access class, so it hides
    with the field. *controls* collects the polled controls, *expansions* the cards, for the
    page's timer and the expand/collapse-all buttons.
    """
    with ui.row().classes("w-full gap-4 flex-wrap items-center content-start"):
        for name, field in settings.fields.items():
            if not field.visible:
                continue
            if field.newline:
                ui.element("div").classes(f"{access_css_class(field)} pt-2").style("flex-basis: 100%; height: 0")
            if field.row_label is not None:
                # The row's title: a fixed width, so the controls of successive rows line up.
                ui.label(field.row_label).classes(f"{access_css_class(field)} w-24 font-bold")
            control = build_entry(settings, name, field)
            if control.polled:
                controls.append(control)

    for name, child in settings.children.items():
        if has_visible_content(child):
            card(name, child, controls, state, depth=depth, expansions=expansions, path=path)


def card(name: str, settings: BaseSettings, controls: list[Control], state: PanelState, *,
         depth: int, expansions: list[ui.expansion], path: str) -> None:
    """One settings instance as a collapsible card: a header with its name and, when it has
    children, an expand/collapse-all button; its body inside. The state remembers whether it is open.
    """
    key = state.card_key(path, name)
    access_class = ACCESS_CLASS.get(content_access_types(settings), "")
    bg = DEPTH_BG[min(depth, len(DEPTH_BG) - 1)]
    exp = ui.expansion(generate_label(name), value=state.card_open(key)).props(
        "duration=0 dense dense-toggle"
    ).classes(f"w-full rounded {bg}" + (f" {access_class}" if access_class else ""))
    exp.on("show", lambda: state.set_card_open(key, True))
    exp.on("hide", lambda: state.set_card_open(key, False))
    expansions.append(exp)

    children: list[ui.expansion] = []
    all_open = [False]

    def toggle_children() -> None:
        all_open[0] = not all_open[0]
        if all_open[0]:
            exp.open()
            for child in children:
                child.open()
        else:
            for child in children:
                child.close()
            exp.close()

    with exp.add_slot("header"):
        with ui.row().classes("w-full items-center gap-1"):
            ui.label(generate_label(name).upper()).classes("flex-1")
            if settings.children:
                button = ui.button(icon="unfold_more", on_click=toggle_children).props(
                    "dense flat round size=xs"
                ).tooltip("Expand / Collapse All")
                button.on("click.stop", lambda: None)   # the click does not toggle the card itself

    with exp:
        body(settings, controls, state, depth=depth + 1, expansions=children, path=key)
    expansions.extend(children)


# -- the header --------------------------------------------------------------------------

def local_ips() -> list[str]:
    """All non-loopback IPv4 addresses of this machine."""
    try:
        hostname = socket.gethostname()
        _, _, ips = socket.gethostbyname_ex(hostname)
        return [ip for ip in ips if not ip.startswith("127.") and not ip.startswith("169.254.")]
    except OSError:
        return []


def head(title: str) -> None:
    """The page's stylesheet and script, and the overlay the script shows while the server is gone."""
    ui.add_head_html(
        f'<link rel="stylesheet" href="{STATIC_URL}/panel.css">'
        f'<script src="{STATIC_URL}/panel.js"></script>'
        f"<script>panel.installShutdownOverlay({_js(title)}, {_js(LOG_DRAWER_DEFAULT_HEIGHT)});</script>"
    )


def _js(value: str) -> str:
    """*value* as a JavaScript string literal that cannot close a script element."""
    return json.dumps(value).replace("</", "<\\/")


def header(root: BaseSettings, *, title: str, port: int, on_exit: Callable[[], None] | None,
           toggle_log: list[Callable[[], None]]) -> None:
    """The title, the log and connection buttons, the preset controls and the exit button, in one row.

    *toggle_log* is filled by the caller once the log drawer exists, since the drawer is built
    outside the header.
    """
    with ui.row().classes("w-full items-center flex-wrap gap-1"):
        if title:
            ui.label(title).classes("text-2xl font-bold")

        ui.button(icon="terminal", on_click=lambda: toggle_log[0]() if toggle_log else None).props(
            "dense flat"
        ).tooltip("Log panel")

        connection = ui.button(icon="lan").props("dense flat").tooltip("Connection info")
        with connection:
            with ui.menu().props('anchor="bottom middle" self="top middle"'):
                with ui.card().classes("gap-1").props("flat"):
                    ui.label("Connect").classes("text-base font-bold")
                    ui.separator()
                    links = ui.column().classes("gap-0")
                    url = f"http://localhost:{port}"
                    ui.link(url, url, new_tab=True).classes("text-sm")

        loaded = [False]

        def add_ip_links() -> None:
            """The machine's addresses, looked up at the first click: the lookup can take a moment."""
            if loaded[0]:
                return
            loaded[0] = True
            with links:
                for ip in local_ips():
                    url = f"http://{ip}:{port}"
                    ui.link(url, url, new_tab=True).classes("text-sm")

        connection.on("click", add_ip_links)

        with ui.row().classes("flex-1 items-center gap-1 flex-nowrap justify-center"):
            preset_controls(root)

        if on_exit:
            async def do_exit() -> None:
                await ui.run_javascript("showShutdownScreen()")
                on_exit()
            ui.button(icon="power_settings_new", on_click=do_exit).props(
                "dense flat color=negative"
            ).tooltip("Exit application")


def pinned_bar(root: BaseSettings, controls: list[Control]) -> None:
    """The pinned fields of the whole tree above the tabs, scaled down: a row of what can be
    changed and a row of what can only be read, status badges first. Nothing when nothing is pinned."""
    fields: list[tuple[BaseSettings, str, Field]] = []
    actions: list[tuple[BaseSettings, str, Field]] = []

    def collect(settings: BaseSettings) -> None:
        for name, field in settings.fields.items():
            # a button is in `fields` too; it goes to the actions only
            if field.pinned and field.visible and field.widget is not Widget.button:
                fields.append((settings, name, field))
        for name, field in settings.actions.items():
            if field.pinned and field.visible:
                actions.append((settings, name, field))
        for child in settings.children.values():
            collect(child)

    collect(root)
    if not fields and not actions:
        return

    inputs = [p for p in fields if p[2].access in (Access.WRITE, Access.READWRITE)]
    readouts = [p for p in fields if p[2].access in (Access.READ, Access.INIT)]
    readouts.sort(key=lambda p: p[2].widget is not Widget.status)   # stable: tree order otherwise
    # The frame follows the bar's scaled height (see .poser-pinned in panel.css).
    ui.add_head_html("<script>panel.watchPinnedFrame();</script>")
    with ui.element("div").classes("poser-pinned-frame w-full mt-2"), \
            ui.column().classes("poser-pinned gap-2 bg-grey-9 rounded px-3 py-2"):
        if inputs or actions:
            with ui.row().classes("w-full gap-4 flex-wrap items-end"):
                for settings, name, field in inputs:
                    control = build_control(settings, name, field)
                    if control.polled:
                        controls.append(control)
                for settings, name, field in actions:
                    ActionButton(settings, name, field)
        if readouts:
            with ui.row().classes("w-full gap-3 flex-wrap items-center"):
                for settings, name, field in readouts:
                    if field.widget is Widget.status:
                        controls.append(build_control(settings, name, field))
                    else:
                        controls.append(PinnedReadout(settings, name, field))


# -- the tabs ----------------------------------------------------------------------------

class TabBar:
    """The tabs of one client and their toolbar: the three toggles and expand/collapse-all act on
    the active tab. The state remembers the active tab and each tab's toggles."""

    def __init__(self, names: list[str], state: PanelState) -> None:
        self.state = state
        self.active = state.active_tab(names)
        self.panels: dict[str, ui.tab_panel] = {}
        self.expansions: dict[str, list[ui.expansion]] = {}
        self.all_open: dict[str, bool] = {}
        self.tab_of: dict[str, ui.tab] = {}
        self.buttons: dict[HideToggle, ui.button] = {}

        with ui.tabs().classes("w-full bg-[#121212] rounded").props("dense active-color=primary no-indicator") as self.tabs:
            for name in names:
                tab = ui.tab(generate_label(name))
                tab.on("click", lambda name=name: self.select(name))
                self.tab_of[name] = tab

        with ui.row().classes("w-full items-center"):
            for toggle in TOGGLES:
                self.buttons[toggle] = ui.button(icon=toggle.icon, on_click=lambda toggle=toggle: self.toggle(toggle)).props(
                    "dense flat size=sm"
                ).tooltip(toggle.tooltip)
            ui.element("div").classes("flex-1")
            ui.button(icon="unfold_more", on_click=self.toggle_all).props("dense flat size=sm").tooltip("Expand / Collapse All")
        self.show_toggles()

    def panels_for(self, entries: list[tuple[str, BaseSettings]], controls: list[Control]) -> None:
        """The tab panels, one per entry, each holding its settings' body with the tab's hidden
        types applied; the active tab's panel is shown."""
        with ui.tab_panels(self.tabs, value=self.tab_of[self.active]).classes("w-full bg-grey-10 rounded ").style("padding-bottom: 320px;"):
            for name, settings in entries:
                with ui.tab_panel(self.tab_of[name]) as panel:
                    for toggle in TOGGLES:
                        if self.state.hidden(name, toggle.key, toggle.default):
                            panel.classes(add=toggle.css)
                    expansions: list[ui.expansion] = []
                    body(settings, controls, self.state, depth=0, expansions=expansions, path=name)
                self.panels[name] = panel
                self.expansions[name] = expansions

    def select(self, name: str) -> None:
        """A tab was clicked: remember it and show its toggles."""
        self.state.set_active_tab(name)
        self.active = name
        self.show_toggles()

    def show_toggles(self) -> None:
        """The toggle buttons as the active tab has them: grey when hiding."""
        for toggle, button in self.buttons.items():
            if self.state.hidden(self.active, toggle.key, toggle.default):
                button.classes(add="text-grey-7")
            else:
                button.classes(remove="text-grey-7")

    def toggle(self, toggle: HideToggle) -> None:
        """Flip one toggle of the active tab, on its panel and in the state."""
        panel = self.panels.get(self.active)
        if panel is None:
            return
        hidden = not self.state.hidden(self.active, toggle.key, toggle.default)
        self.state.set_hidden(self.active, toggle.key, hidden)
        if hidden:
            panel.classes(add=toggle.css)
        else:
            panel.classes(remove=toggle.css)
        self.show_toggles()

    def toggle_all(self) -> None:
        """Open every card of the active tab, or close them all the next time."""
        expansions = self.expansions.get(self.active)
        if not expansions:
            return
        self.all_open[self.active] = not self.all_open.get(self.active, False)
        for exp in expansions:
            if self.all_open[self.active]:
                exp.open()
            else:
                exp.close()


# -- the page ----------------------------------------------------------------------------

def build_page(root: BaseSettings, state: PanelState, *, title: str, port: int,
               on_exit: Callable[[], None] | None, log_buffer: LogRingBuffer | None) -> int:
    """Build the panel page for one client inside the current NiceGUI page context.

    Returns the number of polled controls. *on_exit* is called by the exit button; *log_buffer*
    feeds the log drawer."""
    controls: list[Control] = []    # the polled controls of this client
    head(title)
    entries = tab_entries(root)
    toggle_log: list[Callable[[], None]] = []

    with ui.column().classes("w-full sticky top-0 z-50 bg-[#121212] text-white gap-2 px-3 pt-3 pb-2"):
        header(root, title=title, port=port, on_exit=on_exit, toggle_log=toggle_log)
        pinned_bar(root, controls)
        if entries:
            tab_bar = TabBar([name for name, _ in entries], state)
        else:
            ui.label("No settings registered.")

    # The log drawer is a fixed element at the page's top level, outside the header.
    _, toggle = build_log_drawer(log_buffer, poll_interval=POLL_INTERVAL)
    toggle_log.append(toggle)

    # One timer per client; NiceGUI stops it when the client is gone.
    failed: set[int] = set()
    ui.timer(POLL_INTERVAL, lambda: tick(controls, failed))

    if entries:
        tab_bar.panels_for(entries, controls)
    return len(controls)
