from __future__ import annotations

from enum import IntEnum
from typing import Callable

import glfw

from modules.utils import Broadcast
from .window import Window


class MouseButton(IntEnum):
    LEFT    = 0
    RIGHT   = 1
    MIDDLE  = 2
    BACK    = 3
    FORWARD = 4


_GLFW_BUTTON_MAP: dict[int, MouseButton] = {
    glfw.MOUSE_BUTTON_LEFT:   MouseButton.LEFT,
    glfw.MOUSE_BUTTON_RIGHT:  MouseButton.RIGHT,
    glfw.MOUSE_BUTTON_MIDDLE: MouseButton.MIDDLE,
    glfw.MOUSE_BUTTON_4:      MouseButton.BACK,
    glfw.MOUSE_BUTTON_5:      MouseButton.FORWARD,
}

KeyCallback = Callable[[int, int, int, int], None]                  # (window, key, action, mods)
MouseMoveCallback = Callable[[int, float, float], None]             # (window, x, y), 0..1 over the window
MouseButtonCallback = Callable[[int, float, float, MouseButton, bool], None]     # (window, x, y, button, pressed)
MouseScrollCallback = Callable[[int, float], None]                  # (window, steps), positive away from the user


class WindowInput:
    """The windows' key, mouse and scroll input, each callback told the window's index. Callbacks run in registration
    order, on the main thread, during the loop's event processing; a raising callback is logged and the next one runs."""

    def __init__(self) -> None:
        self._key_callbacks: Broadcast[[int, int, int, int]] = Broadcast(name='key')
        self._mouse_move_callbacks: Broadcast[[int, float, float]] = Broadcast(name='mouse move')
        self._mouse_button_callbacks: Broadcast[[int, float, float, MouseButton, bool]] = Broadcast(name='mouse button')
        self._mouse_scroll_callbacks: Broadcast[[int, float]] = Broadcast(name='mouse scroll')
        self._mouse: dict[int, tuple[float, float]] = {}     # each window's cursor, in screen coordinates

    def attach(self, window: Window) -> None:
        """GLFW's input callbacks of `window`."""
        index: int = window.index
        glfw.set_key_callback(window.handle, lambda handle, key, scancode, action, mods: self._on_key(index, key, action, mods))
        glfw.set_cursor_pos_callback(window.handle, lambda handle, x, y: self._on_cursor(index, handle, x, y))
        glfw.set_mouse_button_callback(window.handle, lambda handle, button, action, mods: self._on_button(index, handle, button, action))
        glfw.set_scroll_callback(window.handle, lambda handle, x, y: self._on_scroll(index, y))

    def add_key_callback(self, callback: KeyCallback) -> None:
        self._key_callbacks.add_callback(callback)

    def add_mouse_move_callback(self, callback: MouseMoveCallback) -> None:
        self._mouse_move_callbacks.add_callback(callback)

    def add_mouse_button_callback(self, callback: MouseButtonCallback) -> None:
        self._mouse_button_callbacks.add_callback(callback)

    def add_mouse_scroll_callback(self, callback: MouseScrollCallback) -> None:
        self._mouse_scroll_callbacks.add_callback(callback)

    def clear_callbacks(self) -> None:
        for callbacks in (self._key_callbacks, self._mouse_move_callbacks, self._mouse_button_callbacks,
                          self._mouse_scroll_callbacks):
            callbacks.clear()

    def _on_key(self, index: int, key: int, action: int, mods: int) -> None:
        self._key_callbacks(index, key, action, mods)

    @staticmethod
    def _normalized(handle: glfw._GLFWwindowPointerT, x: float, y: float) -> tuple[float, float]:
        width, height = glfw.get_window_size(handle)
        return x / max(1, width), y / max(1, height)

    def _on_cursor(self, index: int, handle: glfw._GLFWwindowPointerT, x: float, y: float) -> None:
        self._mouse[index] = (x, y)
        self._mouse_move_callbacks(index, *self._normalized(handle, x, y))

    def _on_button(self, index: int, handle: glfw._GLFWwindowPointerT, button: int, action: int) -> None:
        mouse_button: MouseButton | None = _GLFW_BUTTON_MAP.get(button)
        if mouse_button is None or action not in (glfw.PRESS, glfw.RELEASE):
            return
        x, y = self._normalized(handle, *self._mouse.get(index, (0.0, 0.0)))
        self._mouse_button_callbacks(index, x, y, mouse_button, action == glfw.PRESS)

    def _on_scroll(self, index: int, steps: float) -> None:
        self._mouse_scroll_callbacks(index, float(steps))
