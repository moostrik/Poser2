from __future__ import annotations

import logging
from typing import Callable

from nicegui import ui

LOG_DRAWER_DEFAULT_HEIGHT = '33vh'

_LOG_LEVEL_COLORS = {
    'DEBUG':    '#888888',
    'INFO':     '#ffffff',
    'WARNING':  '#f59e0b',
    'ERROR':    '#ef4444',
    'CRITICAL': '#ef4444',
}

_LOG_LEVEL_BUTTON_COLORS = {
    'DEBUG':    'grey-6',
    'INFO':     'grey-1',
    'WARNING':  'orange-5',
    'ERROR':    'red-5',
    'CRITICAL': 'red-5',
}

_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
_LOG_SOURCE_ICON = 'code'


def build_log_drawer(timers: list, *, poll_interval: float) -> tuple:
    """Build a slide-up log drawer anchored to the bottom of the page.

    Returns (drawer_element, toggle_function) so the caller can place
    a toggle button in the header.
    """
    from modules.log_config import get_log_buffer

    state = {
        'cursor': 0,
        'filter_level': logging.INFO,
        'auto_scroll': True,
        'show_time': False,
        'show_source': False,
    }

    ui.add_css('''
    .log-drawer { font-family: 'Roboto Mono', monospace; font-size: 13px; }
    .log-drawer .log-line {
        padding: 1px 8px;
        white-space: pre-wrap; word-break: break-all; line-height: 1.4;
    }
    .log-chip-active { opacity: 1.0 !important; }
    .log-chip-inactive { opacity: 0.35 !important; }
    .log-drag-handle {
        height: 6px; cursor: ns-resize; background: #333;
        flex-shrink: 0; transition: background 0.15s;
    }
    .log-drag-handle:hover, .log-drag-handle:active { background: #4af; }
    ''')

    drawer = ui.element('div').classes('log-drawer').style(
        f'position: fixed; bottom: 0; left: 0; right: 0; height: {LOG_DRAWER_DEFAULT_HEIGHT};'
        'background: #141414; border-top: 2px solid #333; z-index: 9999;'
        'display: flex; flex-direction: column;'
    )
    drawer.set_visibility(False)
    toggle_holder: list[Callable[[], None]] = []

    with drawer:
        drag_handle = ui.element('div').classes('log-drag-handle')

        with ui.row().classes('w-full items-center gap-2').style(
            'padding: 4px 12px; background: #1a1a1a; border-bottom: 1px solid #333; flex-shrink: 0;'
        ):
            ui.button(icon='terminal', on_click=lambda: toggle_holder[0]() if toggle_holder else None).props(
                'dense flat size=sm color=primary'
            ).tooltip('Hide panel')

            time_btn = ui.button(icon='schedule', on_click=lambda: _toggle_time()).props(
                'dense flat size=sm'
            ).tooltip('Show Timestamps').classes('log-chip-inactive')

            source_btn = ui.button(icon=_LOG_SOURCE_ICON, on_click=lambda: _toggle_source()).props(
                'dense flat size=sm'
            ).tooltip('Show Source')

            chips: dict[str, ui.element] = {}
            for level_name in _LOG_LEVELS:
                button_color = _LOG_LEVEL_BUTTON_COLORS[level_name]

                chip = ui.button(level_name, on_click=lambda _, ln=level_name: _set_filter(ln)).props(
                    f'dense flat size=sm no-caps color={button_color}'
                ).style('font-size: 13px; min-height: 30px; padding: 0 10px;')
                chips[level_name] = chip

            ui.element('div').classes('flex-1')

            ui.button(icon='delete_sweep', on_click=lambda: _clear_log()).props(
                'dense flat size=sm color=primary'
            ).tooltip('Clear log')

        log_area = ui.element('div').style(
            'flex: 1; overflow-y: auto; padding: 4px 0;'
        )

    def _update_chip_styles():
        for name, chip in chips.items():
            level_val = getattr(logging, name)
            if level_val >= state['filter_level']:
                chip.classes(remove='log-chip-inactive', add='log-chip-active')
            else:
                chip.classes(remove='log-chip-active', add='log-chip-inactive')

    def _update_toggle_styles():
        time_btn.props(f'color={"primary" if state["show_time"] else "grey-7"}')
        if state['show_time']:
            time_btn.classes(remove='log-chip-inactive', add='log-chip-active')
        else:
            time_btn.classes(remove='log-chip-active', add='log-chip-inactive')
        time_btn.update()
        source_btn.props(f'color={"primary" if state["show_source"] else "grey-7"}')
        if state['show_source']:
            source_btn.classes(remove='log-chip-inactive', add='log-chip-active')
        else:
            source_btn.classes(remove='log-chip-active', add='log-chip-inactive')
        source_btn.update()

    _update_chip_styles()
    _update_toggle_styles()

    def _set_filter(level_name: str):
        state['filter_level'] = getattr(logging, level_name)
        _update_chip_styles()
        _full_refresh()

    def _toggle_time():
        state['show_time'] = not state['show_time']
        _update_toggle_styles()
        _full_refresh()

    def _toggle_source():
        state['show_source'] = not state['show_source']
        _update_toggle_styles()
        _full_refresh()

    def _clear_log():
        log_area.clear()
        buf = get_log_buffer()
        if buf:
            state['cursor'] = buf._counter
        state['auto_scroll'] = True

    def _make_line_text(entry) -> tuple[str, str]:
        color = _LOG_LEVEL_COLORS.get(entry.level, '#cccccc')
        time_prefix = f'{entry.timestamp} ' if state['show_time'] else ''
        source_prefix = f'{entry.source}: ' if state['show_source'] else ''
        text = f'{time_prefix}{source_prefix}{entry.message}'
        return text, color

    def _append_entries(entries):
        with log_area:
            for entry in entries:
                if getattr(logging, entry.level) >= state['filter_level']:
                    text, color = _make_line_text(entry)
                    ui.label(text).classes('log-line w-full').style(
                        f'color: {color}; white-space: pre-wrap;'
                    )
        if state['auto_scroll']:
            _scroll_to_bottom()

    def _scroll_to_bottom():
        ui.run_javascript(f'''
            var el = document.getElementById("c{log_area.id}");
            if (el) {{ el.scrollTop = el.scrollHeight; }}
        ''')

    def _full_refresh():
        log_area.clear()
        buf = get_log_buffer()
        if not buf:
            return
        _, all_entries = buf.get_entries_since(0)
        _append_entries(all_entries)
        state['auto_scroll'] = True
        _scroll_to_bottom()

    def _poll_log():
        buf = get_log_buffer()
        if not buf:
            return
        try:
            new_cursor, entries = buf.get_entries_since(state['cursor'])
            if entries:
                state['cursor'] = new_cursor
                _append_entries(entries)
        except RuntimeError:
            pass

    t = ui.timer(poll_interval, _poll_log)
    timers.append(t)

    log_area.on('scroll', lambda: ui.run_javascript(f'''
        var el = document.getElementById("c{log_area.id}");
        if (el) {{
            var atBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) < 30;
            emitEvent("log_scroll", {{atBottom: atBottom}});
        }}
    '''))
    ui.on('log_scroll', lambda e: state.update({'auto_scroll': e.args.get('atBottom', True)}))

    ui.run_javascript(f'''
        (function() {{
            var handle = document.getElementById("c{drag_handle.id}");
            var drawer = document.getElementById("c{drawer.id}");
            if (!handle || !drawer) return;
            var startY = 0, startH = 0, dragging = false;
            handle.addEventListener("mousedown", function(e) {{
                dragging = true; startY = e.clientY;
                startH = drawer.offsetHeight;
                e.preventDefault();
            }});
            document.addEventListener("mousemove", function(e) {{
                if (!dragging) return;
                var newH = startH + (startY - e.clientY);
                newH = Math.max(100, Math.min(window.innerHeight - 60, newH));
                drawer.style.height = newH + "px";
            }});
            document.addEventListener("mouseup", function() {{ dragging = false; }});
        }})();
    ''')

    visible = {'v': False}

    def _toggle():
        visible['v'] = not visible['v']
        drawer.set_visibility(visible['v'])
        if visible['v']:
            _full_refresh()

    toggle_holder.append(_toggle)

    return drawer, _toggle