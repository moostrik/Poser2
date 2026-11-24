from dataclasses import dataclass
from math import atan2, cos, sin
from typing import Any, Generator

import numpy as np


@dataclass
class Point2f:
    """A 2D point or vector."""
    x: float
    y: float

    def __iter__(self) -> Generator[float, Any, None]:
        """Iterate over x and y coordinates."""
        yield self.x
        yield self.y

    def __getitem__(self, index: int) -> float:
        """Get x or y by index (0 for x, 1 for y)."""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Point2f index out of range (0-1)")

    def copy(self) -> "Point2f":
        """Return a copy of this point."""
        return Point2f(x=self.x, y=self.y)

    def __add__(self, other: "float | Point2f") -> "Point2f":
        """Add another Point2f or scalar to this point."""
        if isinstance(other, Point2f):
            return Point2f(self.x + other.x, self.y + other.y)
        else:
            return Point2f(self.x + other, self.y + other)

    def __sub__(self, other: "float | Point2f") -> "Point2f":
        """Subtract another Point2f or scalar from this point."""
        if isinstance(other, Point2f):
            return Point2f(self.x - other.x, self.y - other.y)
        else:
            return Point2f(self.x - other, self.y - other)

    def __mul__(self, other: "float | Point2f") -> "Point2f":
        """Multiply this point by another Point2f or scalar."""
        if isinstance(other, Point2f):
            return Point2f(self.x * other.x, self.y * other.y)
        else:
            return Point2f(self.x * other, self.y * other)

    def __truediv__(self, other: "float | Point2f") -> "Point2f":
        """Divide this point by another Point2f or scalar."""
        if isinstance(other, Point2f):
            return Point2f(self.x / other.x, self.y / other.y)
        else:
            return Point2f(self.x / other, self.y / other)

    def __eq__(self, other: object) -> bool:
        """Check if two points are equal."""
        if not isinstance(other, Point2f):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    @property
    def length(self) -> float:
        """Return the magnitude of the vector."""
        return (self.x ** 2 + self.y ** 2) ** 0.5

    @property
    def is_zero(self) -> bool:
        """Whether both coordinates are zero."""
        return self.x == 0 and self.y == 0

    def normalized(self) -> "Point2f":
        """Return a normalized (unit length) vector."""
        l = self.length
        if l == 0:
            return Point2f(0, 0)
        return Point2f(self.x / l, self.y / l)

    def dot(self, other: "Point2f") -> float:
        """Return the dot product with another point."""
        return self.x * other.x + self.y * other.y

    def clamp(self, min_point: "Point2f", max_point: "Point2f") -> "Point2f":
        """Clamp this point between min_point and max_point."""
        return Point2f(
            x=max(min_point.x, min(self.x, max_point.x)),
            y=max(min_point.y, min(self.y, max_point.y))
        )

    def distance_to(self, other: "Point2f") -> float:
        """Return the Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx ** 2 + dy ** 2) ** 0.5

    def angle_to(self, other: "Point2f") -> float:
        """Return the angle to another point in radians."""
        return atan2(other.y - self.y, other.x - self.x)

    def rotate(self, angle_rad: float) -> "Point2f":
        """Return a new Point2f rotated by angle_rad radians around the origin."""
        cos_a = cos(angle_rad)
        sin_a = sin(angle_rad)
        return Point2f(
            x=self.x * cos_a - self.y * sin_a,
            y=self.x * sin_a + self.y * cos_a
        )

    def rotate_around(self, center: "Point2f", angle_rad: float) -> "Point2f":
        """Return a new Point2f rotated by angle_rad radians around another point."""
        translated = self - center
        rotated = translated.rotate(angle_rad)
        return rotated + center

    def lerp(self, other: "Point2f", t: float) -> "Point2f":
        """Linearly interpolate between this point and another by t (0.0 to 1.0)."""
        return Point2f(
            x=self.x + (other.x - self.x) * t,
            y=self.y + (other.y - self.y) * t
        )

    @staticmethod
    def zero() -> "Point2f":
        """Return a zero point (0.0, 0.0)."""
        return Point2f(0.0, 0.0)

    def to_array(self) -> np.ndarray:
        """Return the point as a numpy array [x, y]."""
        return np.array([self.x, self.y], dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> "Point2f":
        """Create a Point2f from a numpy array [x, y]."""
        return Point2f(float(arr[0]), float(arr[1]))

    def to_tuple(self) -> tuple[float, float]:
        """Return the point as a tuple (x, y)."""
        return (self.x, self.y)

    @staticmethod
    def from_tuple(coords: tuple[float, float]) -> "Point2f":
        """Create a Point2f from a tuple (x, y)."""
        return Point2f(coords[0], coords[1])

    def to_dict(self) -> dict:
        """Return the point as a dict for JSON serialization."""
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict) -> "Point2f":
        """Create a Point2f from a dict."""
        return cls(data["x"], data["y"])
