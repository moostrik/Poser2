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

@dataclass (frozen=True)
class Subdivision:
    x: int
    y: int
    width: int
    height: int
    rows:dict[str, list[Rect]]


def make_subdivision(subdivision_rows: list[SubdivisionRow], dst_width: int, dst_height: int) -> Subdivision:

    dst_aspect_ratio: float = dst_width / dst_height
    tot_aspect_ratio: float = 1.0 / sum(1.0 / cell.tot_aspect_ratio for cell in subdivision_rows)

    fit_width: float = float(dst_width) if tot_aspect_ratio > dst_aspect_ratio else float(dst_height * tot_aspect_ratio)
    fit_height: float = dst_width / tot_aspect_ratio if tot_aspect_ratio > dst_aspect_ratio else float(dst_height)
    fit_x: float = (dst_width - fit_width) / 2.0
    fit_y: float = (dst_height - fit_height) / 2.0

    print(f"Render composition: {fit_x}, {fit_y}, {fit_width}, {fit_height}, from {dst_width}x{dst_height}, aspect ratio {tot_aspect_ratio}")

    subdivision: Subdivision = Subdivision(x=int(fit_x), y=int(fit_y), width=int(fit_width), height=int(fit_height), rows={})
    cell_y: float = fit_y
    for cell in subdivision_rows:
        row_height: float = fit_height * (tot_aspect_ratio / cell.tot_aspect_ratio)
        cell_width: float = fit_width / cell.columns
        pad: Point2f = cell.padding
        subdivision.rows[cell.name] = []
        for row in range(cell.rows):
            for col in range(cell.columns):
                x: float = fit_x + col * cell_width + pad.x * 0.5
                y: float = cell_y + row * (row_height / cell.rows) + pad.y * 0.5
                rect_width: float = cell_width - pad.x
                rect_height: float = (row_height / cell.rows) - pad.y
                rect = Rect(x=x, y=y, width=rect_width, height=rect_height)
                subdivision.rows[cell.name].append(rect)
        cell_y += row_height

    return subdivision


Composition_Subdivision = dict[SubdivisionType, dict[int, tuple[int, int, int, int]]]

def make_ws_subdivision(dst_width: int, dst_height: int,
                        num_cams: int, num_sims: int, max_players: int, num_viss: int,
                        cam_aspect_ratio: float = 16.0 / 9.0,
                        sim_aspect_ratio: float = 10.0,
                        psn_aspect_ratio: float = 1.0,
                        vis_aspect_ratio: float = 20.0) -> Composition_Subdivision:
    ret: Composition_Subdivision = {}
    ret[SubdivisionType.TOT] = {}
    ret[SubdivisionType.CAM] = {}
    ret[SubdivisionType.SIM] = {}
    ret[SubdivisionType.PSN] = {}
    ret[SubdivisionType.VIS] = {}
    cams_per_row: int = 4
    cam_rows: int = math.ceil(num_cams / cams_per_row)
    cam_columns: int = 0 if cam_rows == 0 else math.ceil(num_cams / cam_rows)
    dst_aspect_ratio: float = dst_width / dst_height
    cam_grid_aspect_ratio: float = 100.0 if cam_rows == 0 else cam_aspect_ratio * cam_columns / cam_rows
    sim_grid_aspect_ratio: float = sim_aspect_ratio / num_sims
    vis_grid_aspect_ratio: float = vis_aspect_ratio / num_viss
    psn_grid_aspect_ratio: float = psn_aspect_ratio * max_players
    tot_aspect_ratio: float = 1.0 / (1.0 / cam_grid_aspect_ratio + 1.0 / sim_grid_aspect_ratio + 1.0 / vis_grid_aspect_ratio + 1.0 / psn_grid_aspect_ratio)
    fit_width: float
    fit_height: float
    if tot_aspect_ratio > dst_aspect_ratio:
        fit_width = dst_width
        fit_height = dst_width / tot_aspect_ratio
    else:
        fit_width = dst_height * tot_aspect_ratio
        fit_height = dst_height
    fit_x: float = (dst_width - fit_width) / 2.0
    fit_y: float = (dst_height - fit_height) / 2.0

    ret[SubdivisionType.TOT][0] = (0, 0, int(fit_width), int(fit_height))
    cam_width: float = fit_width if cam_columns == 0 else fit_width / cam_columns
    cam_height: float = cam_width / cam_aspect_ratio
    sim_height: float = fit_width / sim_aspect_ratio
    psn_width: float =  fit_width / max_players
    psn_height: float = psn_width / psn_aspect_ratio
    vis_height: float = fit_width / vis_aspect_ratio
    y_start: float = fit_y
    for i in range(num_cams):
        cam_x: float = (i % cam_columns) * cam_width + fit_x
        cam_y: float = (i // cam_columns) * cam_height + fit_y
        ret[SubdivisionType.CAM][i] = (int(cam_x), int(cam_y), int(cam_width), int(cam_height))
    y_start += cam_height * cam_rows
    for i in range(num_sims):
        sim_y: float = y_start + i * sim_height
        ret[SubdivisionType.SIM][i] = (int(fit_x), int(sim_y), int(fit_width), int(sim_height))
    y_start += sim_height * num_sims
    for i in range(max_players):
        psn_x: float = i * psn_width + fit_x
        psn_y: float = y_start
        ret[SubdivisionType.PSN][i] = (int(psn_x), int(psn_y), int(psn_width), int(psn_height))
    y_start += psn_height
    for i in range(num_viss):
        sim_y: float = y_start + i * vis_height
        ret[SubdivisionType.VIS][i] = (int(fit_x), int(sim_y), int(fit_width), int(vis_height))
    # Fill the last Vis till the bottom of the window
    if num_viss > 0:
        last_Vis = ret[SubdivisionType.VIS][num_viss - 1]
        if last_Vis[1] + last_Vis[3] < dst_height:
            ret[SubdivisionType.VIS][num_viss - 1] = (last_Vis[0], last_Vis[1], last_Vis[2], dst_height - last_Vis[1])
    return ret
