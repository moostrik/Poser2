from dataclasses import dataclass

@dataclass
class Point2f:
    x: float
    y: float

@dataclass
class Point3f:
    """3D point with float coordinates"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Rect:
    """Rectangle defined by top-left corner and dimensions"""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def top_left(self) -> Point2f:
        return Point2f(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point2f:
        return Point2f(
            x=self.x + self.width,
            y=self.y + self.height
        )

    @property
    def center(self) -> Point2f:
        return Point2f(
            x=self.x + self.width / 2,
            y=self.y + self.height / 2
        )

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def left(self) -> float:
        return self.x

    def right(self) -> float:
        return self.x + self.width

    @property
    def area(self) -> float:
        return self.width * self.height
