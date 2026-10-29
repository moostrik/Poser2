"""The page: the panel's header, preset controls, pinned fields, tabs and cards, built from a settings tree."""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path

from nicegui import ui

from modules.log_config import LogRingBuffer
from modules.settings import BaseSettings, Field, Access, Widget, presets

from .controls import ActionButton, Control, PinnedReadout, access_css_class, build_control, build_entry, generate_label
from .log_drawer import LOG_DRAWER_DEFAULT_HEIGHT, build_log_drawer
from .poll import POLL_INTERVAL, tick

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expansion state persistence — remembers which sections are open/closed.
# ---------------------------------------------------------------------------
_UI_STATE_FILE = ".ui_state.json"
_expansion_state: dict[str, bool | str] = {}

def _state_path() -> Path:
    """Return the app-scoped UI state file path."""
    return presets._app_dir() / _UI_STATE_FILE

def _load_expansion_state() -> None:
    global _expansion_state
    p = _state_path()
    if p.exists():
        try:
            _expansion_state = json.loads(p.read_text())
        except Exception:
            _expansion_state = {}

def _save_expansion_state() -> None:
    try:
        p = _state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_expansion_state))
    except Exception:
        pass

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

def _build_settings_body(settings, controls: list[Control], *, depth=0, expansions=None, path=""):
    """Emit the controls for a single Settings instance (no wrapper).

    Renders fields strictly in **declaration order** inside a single flex-wrap
    container.  ``field.newline`` inserts a full-width zero-height break element
    that carries the field's access CSS class so it hides with the field.

    *controls* collects the polled controls for the page's timer.
    *depth* tracks nesting level (0 = top, 1+ = child of child).
    *expansions* collects ``ui.expansion`` elements for expand/collapse-all.
    """

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
                    f"{access_css_class(field)} pt-2"
                ).style("flex-basis: 100%; height: 0")
            if field.row_label is not None:
                # The row's title: a fixed width, so the controls of successive rows line up.
                ui.label(field.row_label).classes(f"{access_css_class(field)} w-24 font-bold")
            control = build_entry(settings, field_name, field)
            if control.polled:
                controls.append(control)

    # Children (recursive)
    for child_name, child in settings.children.items():
        if _has_visible_content(child):
            _build_settings_card(child_name, child, controls, depth=depth, expansions=expansions, path=path)


# Depth-based Quasar background classes (dark mode: increasingly lighter).
# _DEPTH_BG = ["bg-grey-10", "bg-grey-9", "bg-grey-8"]
_DEPTH_BG = ["bg-grey-9", "bg-grey-8", "bg-grey-7", "bg-grey-6", "bg-grey-5"]  # extended for deeper nesting


def _build_settings_card(name, settings, controls: list[Control], *, depth=0, expansions=None, path=""):
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
            settings, controls,
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


def local_ips() -> list[str]:
    """All non-loopback IPv4 addresses of this machine."""
    try:
        hostname = socket.gethostname()
        _, _, ips = socket.gethostbyname_ex(hostname)
        return [ip for ip in ips if not ip.startswith("127.") and not ip.startswith("169.254.")]
    except OSError:
        return []


def build_page(root: BaseSettings, *, title: str, port: int, on_exit, log_buffer: LogRingBuffer | None) -> int:
    """Build the panel page for one client inside the current NiceGUI page context.

    Returns the number of polled controls. *on_exit* is called by the exit button; *log_buffer*
    feeds the log drawer."""

    # Load UI state (expansion, active tab) from app-scoped file
    _load_expansion_state()

    controls: list[Control] = []    # the polled controls of this client

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

    .poser-source { border-left: 2px solid #26a69a; padding-left: 6px; }
    .poser-synced { border-left: 2px solid #ffa726; padding-left: 6px; }

    /* A number list's inputs as low as the rows of a checklist beside it. */
    .poser-number-list .q-field--dense .q-field__control, .poser-number-list .q-field--dense .q-field__native,
    .poser-number-list .q-field--dense .q-field__marginal { height: 24px; min-height: 24px; padding-top: 0; padding-bottom: 0; }

    .poser-knob { display: flex; align-items: center; gap: 6px; }
    .poser-knob-caption { display: flex; flex-direction: column; align-items: flex-end; min-width: 56px; max-width: 110px; }
    .poser-knob-dial { width: 40px; height: 40px; flex: none; cursor: grab; touch-action: none; user-select: none; }
    .poser-knob-dragging .poser-knob-dial { cursor: grabbing; }
    .poser-knob-readonly .poser-knob-dial { cursor: default; }
    .poser-knob-track { fill: none; stroke: #424242; stroke-width: 8; stroke-linecap: round; }
    .poser-knob-value { fill: none; stroke: var(--q-primary); stroke-width: 8; stroke-linecap: round; }
    .poser-knob-body { stroke: #0e0e0e; stroke-width: 2; }
    .poser-knob-pointer { stroke: #e0e0e0; stroke-width: 5; stroke-linecap: round; }
    /* The value and its editor share one box (border, padding, font), so the digits do not move
       when editing starts; the editor only lights the border. */
    .poser-knob-text { width: 100%; box-sizing: border-box; margin: 0; padding: 0 3px;
                       border: 1px solid #424242; border-radius: 3px;
                       font-family: inherit; font-size: 12px; line-height: 18px; text-align: right;
                       font-variant-numeric: tabular-nums; cursor: text; }
    .poser-knob-readonly .poser-knob-text { cursor: default; border-color: transparent; }
    .poser-knob-value-box { position: relative; width: 100%; }
    .poser-knob-editor { position: absolute; inset: 0; min-width: 0; height: 100%;
                         background: #2a2a2a; color: inherit; border-color: var(--q-primary); outline: none; }

    /* Scaled with transform, not zoom: zoom puts getBoundingClientRect and pointer clientX in
       different spaces in some browsers, and Quasar sliders compare the two. A transform keeps
       its layout size, so the bar is laid out wider and the frame is sized by script. */
    .poser-pinned-frame { --scale: 0.8; overflow: hidden; }
    .poser-pinned {
        width: calc(100% / var(--scale));
        transform: scale(var(--scale));
        transform-origin: top left;
    }
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
                    for _ip in local_ips():
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
                # a button is in `fields` too; it goes to the actions only
                if field.pinned and field.visible and field.widget is not Widget.button:
                    pinned_fields.append((settings, field_name, field))
            for action_name, action_field in settings.actions.items():
                if action_field.pinned and action_field.visible:
                    pinned_actions.append((settings, action_name, action_field))
            for child in settings.children.values():
                _collect_pinned(child)

        _collect_pinned(root)

        # Render pinned fields above the tabs, scaled down: a controls row (what can be changed)
        # and a status row (what can only be read), status indicators first.
        if pinned_fields or pinned_actions:
            pinned_controls = [p for p in pinned_fields if p[2].access in (Access.WRITE, Access.READWRITE)]
            readouts = [p for p in pinned_fields if p[2].access in (Access.READ, Access.INIT)]
            readouts.sort(key=lambda p: p[2].widget is not Widget.status)   # stable: tree order otherwise
            # The frame follows the bar's scaled height (see .poser-pinned in the CSS above).
            ui.add_head_html('''<script>(function() {
              var _i = setInterval(function() {
                var frame = document.querySelector(".poser-pinned-frame");
                var bar = frame && frame.querySelector(".poser-pinned");
                if (!bar) return;
                clearInterval(_i);
                var scale = parseFloat(getComputedStyle(frame).getPropertyValue("--scale"));
                new ResizeObserver(function() {
                  frame.style.height = bar.offsetHeight * scale + "px";
                }).observe(bar);
              }, 100);
            })();</script>''')
            with ui.element("div").classes("poser-pinned-frame w-full mt-2"), \
                    ui.column().classes("poser-pinned gap-2 bg-grey-9 rounded px-3 py-2"):
                if pinned_controls or pinned_actions:
                    with ui.row().classes("w-full gap-4 flex-wrap items-end"):
                        for settings, field_name, field in pinned_controls:
                            control = build_control(settings, field_name, field)
                            if control.polled:
                                controls.append(control)
                        for settings, action_name, action_field in pinned_actions:
                            ActionButton(settings, action_name, action_field)
                if readouts:
                    with ui.row().classes("w-full gap-3 flex-wrap items-center"):
                        for settings, field_name, field in readouts:
                            if field.widget is Widget.status:
                                controls.append(build_control(settings, field_name, field))
                            else:
                                controls.append(PinnedReadout(settings, field_name, field))

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
    _, _toggle_log_fn = build_log_drawer(log_buffer, poll_interval=POLL_INTERVAL)
    _log_toggle_holder.append(_toggle_log_fn)

    # One timer per client; NiceGUI stops it when the client is gone.
    failed: set[int] = set()
    ui.timer(POLL_INTERVAL, lambda: tick(controls, failed))

    if not tab_entries:
        return len(controls)

    with ui.tab_panels(tabs, value=tab_map[initial_tab_label]).classes("w-full bg-grey-10 rounded ").style("padding-bottom: 320px;"):
        for label, root_settings in tab_entries:
            with ui.tab_panel(tab_map[label]) as panel:
                # Apply persisted visibility classes to this panel
                for _, suffix, css, _, default in _TOGGLES:
                    if _expansion_state.get(f"{label}.{suffix}", default):
                        panel.classes(add=css)

                expansions: list = []

                _build_settings_body(
                    root_settings, controls,
                    depth=0, expansions=expansions, path=label,
                )

                _tab_data[label] = {"panel": panel, "expansions": expansions}

    return len(controls)
