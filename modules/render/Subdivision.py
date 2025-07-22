from enum import IntEnum
import math
from dataclasses import dataclass, field, replace

from modules.utils.PointsAndRects import Rect, Point2f

class SubdivisionType(IntEnum):
    TOT = 0
    CAM = 1
    SIM = 2
    PSN = 3
    VIS = 4


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

    def get_rect(self, key: str, index: int = 0) -> Rect:
        if key not in self._rows or index >= len(self._rows[key]):
            return Rect(0, 0, 128, 128)
        return self._rows[key][index]

    def get_allocation_size(self, key: str, index: int = 0) -> tuple[int, int]:
        rect: Rect = self.get_rect(key, index)
        return int(rect.width), int(rect.height)

def make_subdivision(subdivision_rows: list[SubdivisionRow], dst_width: int, dst_height: int) -> Subdivision:

    dst_aspect_ratio: float = dst_width / dst_height
    tot_aspect_ratio: float = 1.0 / sum(1.0 / cell.tot_aspect_ratio for cell in subdivision_rows)

    fit_width: float = float(dst_width) if tot_aspect_ratio > dst_aspect_ratio else float(dst_height * tot_aspect_ratio)
    fit_height: float = dst_width / tot_aspect_ratio if tot_aspect_ratio > dst_aspect_ratio else float(dst_height)
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