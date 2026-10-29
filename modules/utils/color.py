"""Color utility class for the settings system.

Canonical internal representation is RGBA floats in the 0.0–1.0 range.
Factory classmethods provide construction from int (0–255), hex strings,
HSV and HSL colour spaces.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Any, Generator


@dataclass
class Color:
    """RGBA colour stored as floats in [0, 1]."""

    r: float
    g: float
    b: float
    a: float = 1.0

    # ------------------------------------------------------------------
    # Iteration / indexing — enables tuple-splat coercion in Setting
    # ------------------------------------------------------------------

    def __iter__(self) -> Generator[float, Any, None]:
        yield self.r
        yield self.g
        yield self.b
        yield self.a

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.r
        elif index == 1:
            return self.g
        elif index == 2:
            return self.b
        elif index == 3:
            return self.a
        else:
            raise IndexError("Color index out of range (0-3)")

    def __len__(self) -> int:
        return 4

    # ------------------------------------------------------------------
    # Equality — required for Setting change detection
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Color):
            return NotImplemented
        return (self.r, self.g, self.b, self.a) == (other.r, other.g, other.b, other.a)

    def __repr__(self) -> str:
        if self.a == 1.0:
            return f"Color({self.r}, {self.g}, {self.b})"
        return f"Color({self.r}, {self.g}, {self.b}, {self.a})"

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> Color:
        return Color(self.r, self.g, self.b, self.a)

    # ------------------------------------------------------------------
    # Serialization — required for Setting JSON round-trip
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"r": self.r, "g": self.g, "b": self.b, "a": self.a}

    @classmethod
    def from_dict(cls, data: dict) -> Color:
        return cls(data["r"], data["g"], data["b"], data.get("a", 1.0))

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)

    @classmethod
    def from_tuple(cls, t: tuple | list) -> Color:
        if len(t) == 3:
            return cls(float(t[0]), float(t[1]), float(t[2]))
        elif len(t) >= 4:
            return cls(float(t[0]), float(t[1]), float(t[2]), float(t[3]))
        raise ValueError(f"Expected 3 or 4 elements, got {len(t)}")

    # ------------------------------------------------------------------
    # Factory: integer 0-255
    # ------------------------------------------------------------------

    @classmethod
    def from_int(cls, r: int, g: int, b: int, a: int = 255) -> Color:
        """Create from 8-bit integer channels (0-255)."""
        return cls(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    def to_int(self) -> tuple[int, int, int, int]:
        """Return channels as 8-bit integers (0-255)."""
        return (
            round(self.r * 255),
            round(self.g * 255),
            round(self.b * 255),
            round(self.a * 255),
        )

    # ------------------------------------------------------------------
    # Factory: hex string
    # ------------------------------------------------------------------

    @classmethod
    def from_hex(cls, hex_str: str) -> Color:
        """Parse ``#RGB``, ``#RRGGBB``, or ``#RRGGBBAA`` hex strings."""
        s = hex_str.lstrip("#")
        if len(s) == 3:
            # #RGB → #RRGGBB
            s = s[0] * 2 + s[1] * 2 + s[2] * 2
        if len(s) == 4:
            # #RGBA → #RRGGBBAA
            s = s[0] * 2 + s[1] * 2 + s[2] * 2 + s[3] * 2
        if len(s) == 6:
            r = int(s[0:2], 16) / 255.0
            g = int(s[2:4], 16) / 255.0
            b = int(s[4:6], 16) / 255.0
            return cls(r, g, b)
        elif len(s) == 8:
            r = int(s[0:2], 16) / 255.0
            g = int(s[2:4], 16) / 255.0
            b = int(s[4:6], 16) / 255.0
            a = int(s[6:8], 16) / 255.0
            return cls(r, g, b, a)
        raise ValueError(f"Invalid hex color string: {hex_str!r}")

    def to_hex(self, include_alpha: bool = False) -> str:
        """Return ``#RRGGBB`` or ``#RRGGBBAA`` hex string."""
        ri, gi, bi, ai = self.to_int()
        if include_alpha:
            return f"#{ri:02X}{gi:02X}{bi:02X}{ai:02X}"
        return f"#{ri:02X}{gi:02X}{bi:02X}"

    # ------------------------------------------------------------------
    # Factory: HSV
    # ------------------------------------------------------------------

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float, a: float = 1.0) -> Color:
        """Create from HSV (all values 0-1)."""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return cls(r, g, b, a)

    def to_hsv(self) -> tuple[float, float, float, float]:
        """Return ``(h, s, v, a)`` with all values in [0, 1]."""
        h, s, v = colorsys.rgb_to_hsv(self.r, self.g, self.b)
        return (h, s, v, self.a)

    # ------------------------------------------------------------------
    # Factory: HSL
    # ------------------------------------------------------------------

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float, a: float = 1.0) -> Color:
        """Create from HSL (all values 0-1).  Note: Python's colorsys uses HLS order."""
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return cls(r, g, b, a)

    def to_hsl(self) -> tuple[float, float, float, float]:
        """Return ``(h, s, l, a)`` with all values in [0, 1]."""
        h, l, s = colorsys.rgb_to_hls(self.r, self.g, self.b)
        return (h, s, l, self.a)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def clamped(self) -> Color:
        """Return a copy with all channels clamped to [0, 1]."""
        return Color(
            max(0.0, min(1.0, self.r)),
            max(0.0, min(1.0, self.g)),
            max(0.0, min(1.0, self.b)),
            max(0.0, min(1.0, self.a)),
        )

    def lerp(self, other: Color, t: float) -> Color:
        """Linearly interpolate between *self* and *other*."""
        return Color(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
            self.a + (other.a - self.a) * t,
        )

    @property
    def luminance(self) -> float:
        """Perceived luminance (ITU BT.709)."""
        return 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b

    def with_alpha(self, a: float) -> Color:
        """Return a copy with a different alpha value."""
        return Color(self.r, self.g, self.b, a)

    @property
    def rgb(self) -> tuple[float, float, float]:
        """Return just the RGB channels."""
        return (self.r, self.g, self.b)
