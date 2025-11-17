from dataclasses import dataclass
from typing import Any, Generator

@dataclass
class Point2f:
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
        """Check if either coordinate is zero."""
        return self.x == 0 or self.y == 0

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
        from math import atan2
        return atan2(other.y - self.y, other.x - self.x)

    def rotate(self, angle_rad: float) -> "Point2f":
        """Return a new Point2f rotated by angle_rad radians around the origin."""
        from math import cos, sin
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

    def __eq__(self, other: object) -> bool:
        """Check if two rectangles are equal."""
        if not isinstance(other, Rect):
            return NotImplemented
        return (
            self.x == other.x and
            self.y == other.y and
            self.width == other.width and
            self.height == other.height
        )

    def copy(self) -> "Rect":
        """Return a copy of this rectangle."""
        return Rect(x=self.x, y=self.y, width=self.width, height=self.height)

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

    def contains_point(self, point: Point2f | tuple[float, float]) -> bool:
        """Check if the rectangle contains a given point."""
        if isinstance(point, Point2f):
            px, py = point.x, point.y
        else:
            px, py = point
        return (self.x <= px <= self.x + self.width) and (self.y <= py <= self.y + self.height)

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

    def move(self, delta: Point2f | tuple[float, float]) -> "Rect":
        """Move the rectangle by a given delta Point2f or tuple."""
        if isinstance(delta, Point2f):
            dx, dy = delta.x, delta.y
        else:
            dx, dy = delta
        return Rect(
            x=self.x + dx,
            y=self.y + dy,
            width=self.width,
            height=self.height
        )

    def scale(self, factor: Point2f | tuple[float, float]) -> "Rect":
        """Scale the rectangle by given Point2f or tuple factors."""
        if isinstance(factor, Point2f):
            sx, sy = factor.x, factor.y
        else:
            sx, sy = factor
        return Rect(
            x=self.x * sx,
            y=self.y * sy,
            width=self.width * sx,
            height=self.height * sy
        )

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
        aspect_ratio = self.width / self.height
        if target.width / target.height > aspect_ratio:
            # Outer is wider than inner aspect ratio
            new_height = target.height
            new_width = new_height * aspect_ratio
        else:
            # Outer is taller than inner aspect ratio
            new_width = target.width
            new_height = new_width / aspect_ratio

        new_x = target.x + (target.width - new_width) / 2
        new_y = target.y + (target.height - new_height) / 2

        return Rect(x=new_x, y=new_y, width=new_width, height=new_height)

    def aspect_fill(self, target: "Rect") -> "Rect":
        """Scale this rectangle to completely fill the target rectangle while maintaining aspect ratio."""
        aspect_ratio = self.width / self.height
        if target.width / target.height < aspect_ratio:
            # Outer is narrower than inner aspect ratio
            new_height = target.height
            new_width = new_height * aspect_ratio
        else:
            # Outer is shorter than inner aspect ratio
            new_width = target.width
            new_height = new_width / aspect_ratio

        new_x = target.x + (target.width - new_width) / 2
        new_y = target.y + (target.height - new_height) / 2

        return Rect(x=new_x, y=new_y, width=new_width, height=new_height)

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
