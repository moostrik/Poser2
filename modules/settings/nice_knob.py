"""Knob — the settings panel's rotary control, turned by circling the pointer (see ``nice_knob.js``).

Quasar's QKnob sets the value from the angle of the pointer, so grabbing it jumps the value.
This knob keeps the value where it is on grab and turns by how far the pointer circles from there.
"""

from __future__ import annotations

from typing import Callable

from nicegui.element import Element


class Knob(Element, component="nice_knob.js"):
    """SVG knob that reports a finished change through ``on_commit``.

    The client shows its own value only while the user changes it; ``set_value`` both updates the
    shown value and tells the client to drop its own, so answer every commit with ``set_value``.
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

    def on_commit(self, callback: Callable[[float], None]) -> Knob:
        """Call *callback* with the value when the user finishes a change."""
        self.on("commit", lambda e: callback(e.args), [None])
        return self
