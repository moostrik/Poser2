from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum

import glfw


class MonitorId(IntEnum):
    """A monitor's place in the connected monitors: the primary first, then by position."""
    M0 = 0
    M1 = 1
    M2 = 2
    M3 = 3


@dataclass(frozen=True)
class Monitor:
    """A connected monitor: its name, its position and size on the desktop (screen coordinates), its GLFW handle."""
    name: str
    x: int
    y: int
    width: int
    height: int
    handle: glfw._GLFWmonitorPointerT

    def contains(self, x: int, y: int) -> bool:
        """Whether the desktop point (x, y) lies on this monitor."""
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def is_handle(self, handle: glfw._GLFWmonitorPointerT) -> bool:
        """Whether `handle` is this monitor's (GLFW hands out a new handle object per call)."""
        return ctypes.addressof(handle.contents) == ctypes.addressof(self.handle.contents)

    @staticmethod
    def connected() -> list[Monitor]:
        """The connected monitors in `MonitorId` order: the primary first, then the others by x, then y. A monitor
        without a video mode, one that is being unplugged, is left out."""
        handles: list[glfw._GLFWmonitorPointerT] = glfw.get_monitors() or []
        primary = glfw.get_primary_monitor()
        monitors: list[tuple[bool, Monitor]] = []
        for handle in handles:
            try:
                mode = glfw.get_video_mode(handle)
                x, y = glfw.get_monitor_pos(handle)
                name = glfw.get_monitor_name(handle)
            except Exception:
                continue        # pyGLFW raises for a monitor that is being disconnected
            if not mode:
                continue
            monitor = Monitor(name.decode('utf-8', 'replace') if isinstance(name, bytes) else str(name or ''),
                              x, y, mode.size.width, mode.size.height, handle)
            is_primary: bool = bool(primary) and monitor.is_handle(primary)
            monitors.append((is_primary, monitor))
        monitors.sort(key=lambda item: (not item[0], item[1].x, item[1].y))
        return [monitor for _, monitor in monitors]
