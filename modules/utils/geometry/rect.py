from dataclasses import dataclass
from typing import Any, Generator

import numpy as np

from .fit import fill, fit
from .point import Point2f
from .quad import Quad


@dataclass
class Rect:
    """Rectangle defined by top-left corner and dimensions"""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def __iter__(self) -> Generator[float, Any, None]:
        yield self.x
        yield self.y
        yield self.width
        yield self.height

    def __getitem__(self, index: int) -> float:
        """Get x, y, width, or height by index (0-3)."""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.width
        elif index == 3:
            return self.height
        else:
            raise IndexError("Rect index out of range (0-3)")

    def copy(self) -> "Rect":
        """Return a copy of this rectangle."""
        return Rect(x=self.x, y=self.y, width=self.width, height=self.height)

    @property
    def size(self) -> Point2f:
        """Return the size (width, height) as a Point2f."""
        return Point2f(x=self.width, y=self.height)

    @property
    def top_left(self) -> Point2f:
        """Return the top-left corner as a Point2f."""
        return Point2f(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point2f:
        """Return the bottom-right corner as a Point2f."""
        return Point2f(
            x=self.x + self.width,
            y=self.y + self.height
        )

    @property
    def center(self) -> Point2f:
        """Return the center of the rectangle as a Point2f."""
        return Point2f(
            x=self.x + self.width / 2,
            y=self.y + self.height / 2
        )

    @property
    def top(self) -> float:
        """Return the y-coordinate of the top edge."""
        return self.y

    @property
    def bottom(self) -> float:
        """Return the y-coordinate of the bottom edge."""
        return self.y + self.height

    @property
    def left(self) -> float:
        """Return the x-coordinate of the left edge."""
        return self.x

    @property
    def right(self) -> float:
        """Return the x-coordinate of the right edge."""
        return self.x + self.width

    @property
    def area(self) -> float:
        """Return the area of the rectangle."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Return the aspect ratio (width / height) of the rectangle."""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    @property
    def is_empty(self) -> bool:
        """Check if the rectangle has zero area."""
        return self.width == 0 or self.height == 0

    def intersect(self, other: "Rect") -> "Rect":
        """Return the intersection of this rectangle with another."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        if x1 < x2 and y1 < y2:
            return Rect(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        else:
            return Rect(x=0, y=0, width=0, height=0)  # No intersection

    def union(self, other: "Rect") -> "Rect":
        """Return the union of this rectangle with another."""
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x + self.width, other.x + other.width)
        y2 = max(self.y + self.height, other.y + other.height)

        return Rect(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def overlaps(self, other: "Rect") -> bool:
        """Check if this rectangle overlaps with another."""
        return not (
            self.right <= other.left or
            self.left >= other.right or
            self.bottom <= other.top or
            self.top >= other.bottom
        )

    def contains_rect(self, other: "Rect") -> bool:
        """Check if the rectangle completely contains another rectangle."""
        return (self.x <= other.x and
                self.y <= other.y and
                self.x + self.width >= other.x + other.width and
                self.y + self.height >= other.y + other.height)

    def overflow(self, container: "Rect") -> tuple[int, int, int, int]:
        """
        Calculate how much this rectangle overflows beyond a container's bounds.

        Returns:
            Tuple of (left, top, right, bottom) overflow amounts.
            Each value is positive if this rect extends beyond the container, 0 otherwise.
        """
        left = max(0, int(container.left - self.left))
        top = max(0, int(container.top - self.top))
        right = max(0, int(self.right - container.right))
        bottom = max(0, int(self.bottom - container.bottom))
        return (left, top, right, bottom)

    def contains_point(self, point: Point2f) -> bool:
        """Check if the rectangle contains a given point."""
        return (self.x <= point.x <= self.x + self.width) and (self.y <= point.y <= self.y + self.height)

    def expand(self, amount: float) -> "Rect":
        """Expand the rectangle by a given amount in all directions."""
        return Rect(
            x=self.x - amount,
            y=self.y - amount,
            width=self.width + 2 * amount,
            height=self.height + 2 * amount
        )

    def shrink(self, amount: float) -> "Rect":
        """Shrink the rectangle by a given amount in all directions."""
        return Rect(
            x=self.x + amount,
            y=self.y + amount,
            width=self.width - 2 * amount,
            height=self.height - 2 * amount
        )

    def move(self, delta: Point2f | float) -> "Rect":
        """Move the rectangle by a delta (float moves equally in x and y)."""
        if isinstance(delta, Point2f):
            dx, dy = delta.x, delta.y
        else:
            dx, dy = delta, delta
        return Rect(
            x=self.x + dx,
            y=self.y + dy,
            width=self.width,
            height=self.height
        )

    def scale(self, factor: Point2f | float) -> "Rect":
        """Scale the rectangle (float scales uniformly)."""
        if isinstance(factor, Point2f):
            sx, sy = factor.x, factor.y
        else:
            sx, sy = factor, factor
        return Rect(
            x=self.x * sx,
            y=self.y * sy,
            width=self.width * sx,
            height=self.height * sy
        )

    def zoom(self, factor: Point2f | float) -> "Rect":
        """Scale the rectangle from its center (float zooms uniformly)."""
        if isinstance(factor, Point2f):
            sx, sy = factor.x, factor.y
        else:
            sx, sy = factor, factor

        center = self.center
        new_width = self.width * sx
        new_height = self.height * sy
        new_x = center.x - new_width / 2
        new_y = center.y - new_height / 2

        return Rect(x=new_x, y=new_y, width=new_width, height=new_height)

    def inset(self, margins: "Rect") -> "Rect":
        """Shrink the rectangle by margins specified in another Rect."""
        return Rect(
            x=self.x + margins.x,
            y=self.y + margins.y,
            width=self.width - margins.x - margins.width,
            height=self.height - margins.y - margins.height
        )

    def affine_transform(self, target: "Rect") -> "Rect":
        """Map this normalized rectangle into the coordinate space of another rectangle."""
        return Rect(
            x = target.x + self.x * target.width,
            y = target.y + self.y * target.height,
            width = self.width * target.width,
            height = self.height * target.height
        )

    def aspect_fit(self, target: "Rect") -> "Rect":
        """Return a rectangle that fits this rectangle into the target rectangle while preserving aspect ratio."""
        x, y, width, height = fit(self.width, self.height, target.width, target.height)
        return Rect(x=target.x + x, y=target.y + y, width=width, height=height)

    def aspect_fill(self, target: "Rect") -> "Rect":
        """Scale this rectangle to completely fill the target rectangle while maintaining aspect ratio."""
        x, y, width, height = fill(self.width, self.height, target.width, target.height)
        return Rect(x=target.x + x, y=target.y + y, width=width, height=height)

    def flip_x(self, axis: float) -> "Rect":
        """Flip the rectangle horizontally around a given axis (x coordinate)."""
        return Rect(
            x=2 * axis - self.x - self.width,
            y=self.y,
            width=self.width,
            height=self.height
        )

    def flip_y(self, axis: float) -> "Rect":
        """Flip the rectangle vertically around a given axis (y coordinate)."""
        return Rect(
            x=self.x,
            y=2 * axis - self.y - self.height,
            width=self.width,
            height=self.height
        )

    def to_points(self) -> tuple[Point2f, Point2f]:
        """Return the top-left and bottom-right points of the rectangle."""
        return (self.top_left, self.bottom_right)

    @staticmethod
    def from_points(top_left: Point2f, bottom_right: Point2f) -> "Rect":
        """Create a Rect from top-left and bottom-right points."""
        return Rect(
            x=top_left.x,
            y=top_left.y,
            width=bottom_right.x - top_left.x,
            height=bottom_right.y - top_left.y
        )

    def to_array(self) -> np.ndarray:
        """Return the rectangle as a numpy array [x, y, width, height]."""
        return np.array([self.x, self.y, self.width, self.height], dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> "Rect":
        """Create a Rect from a numpy array [x, y, width, height]."""
        return Rect(
            x=float(arr[0]),
            y=float(arr[1]),
            width=float(arr[2]),
            height=float(arr[3])
        )

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return the rectangle as a tuple (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @staticmethod
    def from_tuple(coords: tuple[float, float, float, float]) -> "Rect":
        """Create a Rect from a tuple (x, y, width, height)."""
        return Rect(coords[0], coords[1], coords[2], coords[3])

    def to_dict(self) -> dict:
        """Return the rectangle as a dict for JSON serialization."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict) -> "Rect":
        """Create a Rect from a dict."""
        return cls(data["x"], data["y"], data["width"], data["height"])

    def corners(self) -> Quad:
        """The rectangle's four corners, as a `Quad`; a negative width or height gives them mirrored."""
        right: float = self.x + self.width
        bottom: float = self.y + self.height
        return Quad(Point2f(self.x, self.y), Point2f(right, self.y), Point2f(right, bottom), Point2f(self.x, bottom))

    def turned(self, angle: float, center: Point2f, aspect: float = 1.0) -> Quad:
        """The rectangle's corners turned by `angle` (radians, the image's top-left frame) about `center`, the turn
        round in pixels of an image `aspect` wide per unit high; all in 0..1 with the origin top-left."""
        cos_a: float = float(np.cos(angle))
        sin_a: float = float(np.sin(angle))

        def turn(point: Point2f) -> Point2f:
            x: float = (point.x - center.x) * aspect
            y: float = point.y - center.y
            return Point2f((cos_a * x + sin_a * y) / aspect + center.x, -sin_a * x + cos_a * y + center.y)

        quad: Quad = self.corners()
        return Quad(turn(quad.top_left), turn(quad.top_right), turn(quad.bottom_right), turn(quad.bottom_left))
