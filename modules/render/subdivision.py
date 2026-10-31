import math
from dataclasses import dataclass, field, replace

from modules.utils import Rect, Point2f


@dataclass (frozen=True)
class SubdivisionRow:
    name: str
    columns: int = field(default=1)
    rows: int = field(default=1)
    padding: Point2f = field(default_factory=lambda: Point2f(0, 0))
    src_aspect_ratio: float = field(default=1.0)
    tot_aspect_ratio: float = field(default=0.0, init=False)

    def __post_init__(self):
        if self.columns < 1:
            object.__setattr__(self, 'columns', 1)
        if self.rows < 1:
            object.__setattr__(self, 'rows', 1)
        if self.src_aspect_ratio <= 0:
            object.__setattr__(self, 'src_aspect_ratio', 1.0)
        object.__setattr__(self, 'tot_aspect_ratio', self.src_aspect_ratio * self.columns / self.rows)

@dataclass
class Subdivision:
    x: int
    y: int
    width: int
    height: int
    _rows:dict[str, list[Rect]]

    def has(self, key: str) -> bool:
        """Whether this layout holds that row at all. A row a composition leaves out has no rect
        to draw into, so its owner should skip both its compositing and its draw rather than
        render into the fallback below."""
        return key in self._rows

    def get_rect(self, key: str, index: int = 0) -> Rect:
        if key not in self._rows or index >= len(self._rows[key]):
            return Rect(0, 0, 128, 128)
        return self._rows[key][index]

    def get_allocation_size(self, key: str, index: int = 0) -> tuple[int, int]:
        rect: Rect = self.get_rect(key, index)
        return int(rect.width), int(rect.height)

def _fit_size(aspect_ratio: float, dst_width: float, dst_height: float) -> tuple[float, float]:
    width: float = min(dst_width, dst_height * aspect_ratio)
    return width, width / aspect_ratio


def best_grid(cell_aspect_ratios: list[float], dst_width: int, dst_height: int) -> tuple[int, int]:
    """The (columns, rows) that show cells of these aspect ratios with the largest total area in the destination."""
    count: int = len(cell_aspect_ratios)
    best: tuple[int, int] = (count, 1)
    best_area: float = -1.0
    for columns in range(1, count + 1):
        rows: int = math.ceil(count / columns)
        sizes = [_fit_size(aspect_ratio, dst_width / columns, dst_height / rows) for aspect_ratio in cell_aspect_ratios]
        area: float = sum(width * height for width, height in sizes)
        if area > best_area:
            best, best_area = (columns, rows), area
    return best


def grid_cell_aspect_ratio(cell_aspect_ratios: list[float], columns: int, rows: int, dst_width: int, dst_height: int) -> float:
    """The aspect ratio of the smallest grid cell that holds each of these cells at its fitted size."""
    sizes = [_fit_size(aspect_ratio, dst_width / columns, dst_height / rows) for aspect_ratio in cell_aspect_ratios]
    return max(width for width, _ in sizes) / max(height for _, height in sizes)


def make_subdivision(subdivision_rows: list[SubdivisionRow], dst_width: int, dst_height: int, align_center: bool) -> Subdivision:

    dst_aspect_ratio: float = dst_width / dst_height
    tot_aspect_ratio: float = 1.0 / sum(1.0 / cell.tot_aspect_ratio for cell in subdivision_rows)

    fit_width: float = float(dst_width) if tot_aspect_ratio > dst_aspect_ratio else float(dst_height) * tot_aspect_ratio
    fit_height: float = float(dst_width) / tot_aspect_ratio if tot_aspect_ratio > dst_aspect_ratio else float(dst_height)
    fit_x: float = 0
    fit_y: float = 0
    if align_center:
        fit_x: float = (dst_width - fit_width) / 2.0
        fit_y: float = (dst_height - fit_height) / 2.0

    subdivision: Subdivision = Subdivision(x=int(fit_x), y=int(fit_y), width=int(fit_width), height=int(fit_height), _rows={})
    cell_y: float = fit_y
    for cell in subdivision_rows:
        row_height: float = fit_height * (tot_aspect_ratio / cell.tot_aspect_ratio)
        cell_width: float = fit_width / cell.columns
        pad: Point2f = cell.padding
        subdivision._rows[cell.name] = []
        for row in range(cell.rows):
            for col in range(cell.columns):
                x: float = fit_x + col * cell_width + pad.x * 0.5
                y: float = cell_y + row * (row_height / cell.rows) + pad.y * 0.5
                rect_width: float = cell_width - pad.x
                rect_height: float = (row_height / cell.rows) - pad.y
                rect = Rect(x=x, y=y, width=rect_width, height=rect_height)
                subdivision._rows[cell.name].append(rect)
        cell_y += row_height

    return subdivision