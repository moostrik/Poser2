from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import glfw

from .context import Context
from .monitor import Monitor, MonitorId
from .window_settings import WindowSettings

logger = logging.getLogger(__name__)

FALLBACK_SIZE: int = 200        # px: a window whose monitor is missing waits this small on the primary monitor
FALLBACK_STEP: int = 240        # px between the waiting windows: the size, a title bar and a gap
FALLBACK_MARGIN: int = 40       # px from the primary monitor's corner


@dataclass(frozen=True)
class Placement:
    """Where a window is on the desktop (screen coordinates), and how it shows."""
    shown: bool
    decorated: bool
    x: int
    y: int
    width: int
    height: int
    cursor_hidden: bool


class Window:
    """One GLFW window with its `gl.Context` and its settings. The window manager applies its settings at the
    tick; a move or resize by the user is stored by GLFW's callbacks and written to the settings at the tick."""

    def __init__(self, index: int, settings: WindowSettings, title: str, share: Window | None,
                 on_close: Callable[[], None]) -> None:
        self.index: int = index
        self.settings: WindowSettings = settings
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)      # shown when its placement is applied
        handle = glfw.create_window(max(1, settings.width), max(1, settings.height), title, None,
                                    share.handle if share is not None else None)  # pyright: ignore[reportArgumentType]
        if not handle:
            raise RuntimeError(f"Failed to create window {index}")
        self.handle: glfw._GLFWwindowPointerT = handle
        self._context: Context | None = None       # made at its first use, in its own GL context
        self._applied: Placement | None = None
        self._applying: bool = False                # our own calls: their move and size callbacks are not the user's
        self._moved: bool = False                   # the user moved or resized the window since the last tick
        self._on_close: Callable[[], None] = on_close
        glfw.set_window_pos_callback(handle, self._on_move)
        glfw.set_window_size_callback(handle, self._on_move)
        glfw.set_window_close_callback(handle, lambda _window: self._on_close())

    @property
    def shown(self) -> bool:
        """Shown as last applied."""
        return self._applied is not None and self._applied.shown

    def make_current(self) -> None:
        """This window's GL context made current, and its `gl.Context` with it."""
        glfw.make_context_current(self.handle)
        if self._context is None:
            self._context = Context()
            self._context.allocate()
        else:
            self._context.make_current()

    def framebuffer_size(self) -> tuple[int, int]:
        width, height = glfw.get_framebuffer_size(self.handle)
        return int(width), int(height)

    def placement(self, monitors: list[Monitor], fullscreen_mouse: bool) -> Placement:
        """Where the settings put the window on these monitors; small on the primary monitor while its own is missing."""
        s = self.settings
        monitor: Monitor | None = monitors[s.monitor] if 0 <= s.monitor < len(monitors) else None
        if monitor is None:
            px, py = (monitors[0].x, monitors[0].y) if monitors else (0, 0)
            return Placement(s.enabled, True, px + FALLBACK_MARGIN, py + FALLBACK_MARGIN + self.index * FALLBACK_STEP,
                             FALLBACK_SIZE, FALLBACK_SIZE, False)
        if s.fullscreen:
            return Placement(s.enabled, False, monitor.x, monitor.y, monitor.width, monitor.height, not fullscreen_mouse)
        return Placement(s.enabled, True, monitor.x + s.x, monitor.y + s.y, max(1, s.width), max(1, s.height), False)

    def apply(self, placement: Placement) -> None:
        """Apply what differs from the last applied placement."""
        last: Placement | None = self._applied
        if placement == last:
            return
        self._applying = True
        try:
            geometry = (placement.decorated, placement.x, placement.y, placement.width, placement.height)
            if last is None or geometry != (last.decorated, last.x, last.y, last.width, last.height):
                glfw.set_window_attrib(self.handle, glfw.DECORATED, glfw.TRUE if placement.decorated else glfw.FALSE)
                glfw.set_window_monitor(self.handle, None, placement.x, placement.y,  # pyright: ignore[reportArgumentType]
                                        placement.width, placement.height, 0)
            if last is None or placement.cursor_hidden != last.cursor_hidden:
                glfw.set_input_mode(self.handle, glfw.CURSOR, glfw.CURSOR_HIDDEN if placement.cursor_hidden else glfw.CURSOR_NORMAL)
            if last is None or placement.shown != last.shown:
                if placement.shown:
                    glfw.show_window(self.handle)
                else:
                    glfw.hide_window(self.handle)
        finally:
            self._applying = False
        self._applied = placement

    def write_move(self, monitors: list[Monitor], fullscreen_mouse: bool) -> None:
        """The user's move or resize since the last tick written to the settings, once, as applied: the monitor
        under the window's center, the place on it and the size. Only a window on its own monitor, windowed."""
        if not self._moved:
            return
        self._moved = False
        s = self.settings
        own: bool = 0 <= s.monitor < len(monitors)
        if s.fullscreen or not own or glfw.get_window_attrib(self.handle, glfw.ICONIFIED):
            return
        x, y = glfw.get_window_pos(self.handle)
        width, height = glfw.get_window_size(self.handle)
        if width <= 0 or height <= 0:
            return
        index: int = next((i for i, m in enumerate(monitors[:len(MonitorId)]) if m.contains(x + width // 2, y + height // 2)),
                          int(s.monitor))
        monitor: Monitor = monitors[index]
        for name, value in (('monitor', MonitorId(index)), ('x', x - monitor.x), ('y', y - monitor.y),
                            ('width', width), ('height', height)):
            if getattr(s, name) != value:
                setattr(s, name, value)
        self._applied = self.placement(monitors, fullscreen_mouse)      # where the window is: not applied back

    def destroy(self) -> None:
        """The `gl.Context` freed in its own GL context, then the window."""
        if self._context is not None:
            glfw.make_context_current(self.handle)
            self._context.deallocate()
            self._context = None
        glfw.destroy_window(self.handle)

    def _on_move(self, _window: glfw._GLFWwindowPointerT, _a: int, _b: int) -> None:
        """GLFW's move and size callbacks: a user's move is stored for the tick."""
        if not self._applying:
            self._moved = True
