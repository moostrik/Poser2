"""Knob — the settings panel's rotary control, turned by circling the pointer (see ``nice_knob.js``).

Quasar's QKnob sets the value from the angle of the pointer, so grabbing it jumps the value.
This knob keeps the value where it is on grab and turns by how far the pointer circles from there.
"""

from __future__ import annotations

from typing import Callable

from nicegui.element import Element


CHANGE_THROTTLE = 0.05  # seconds between changes sent while turning; the last one always arrives


class Knob(Element, component="nice_knob.js"):
    """SVG knob that reports its value through ``on_change`` while the user turns it.

    The client shows its own value while the user changes it; ``set_value`` both updates the
    shown value and tells the client to drop its own, so answer every change with ``set_value``.
    """

    def __init__(self, value: float, *, min: float, max: float, step: float,
                 default: float | None, decimals: int, readonly: bool) -> None:
        super().__init__()
        self._props["value"] = value
        self._props["min"] = min
        self._props["max"] = max
        self._props["step"] = step
        self._props["default-value"] = default
        self._props["decimals"] = decimals
        self._props["readonly"] = readonly
        self._props["revision"] = 0

    def set_value(self, value: float) -> None:
        """Show *value* and drop any value the client holds from the user's last change."""
        self._props["value"] = value
        self._props["revision"] += 1
        self.update()

    def on_change(self, callback: Callable[[float], None]) -> Knob:
        """Call *callback* with the value as the user changes it, throttled."""
        self.on("change", lambda e: callback(e.args), [None], throttle=CHANGE_THROTTLE)
        return self
