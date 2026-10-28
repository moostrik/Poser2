"""PoseFigureLayer — each pose as a small figure over the projection row, where it stands.

Every LERP pose with an azimuth and keypoints is drawn as its skeleton in its track colour, its
eyes at its azimuth on the row's 360° x axis (the azimuth is the eyes', `LAYERS.md` *Inputs*), so
a person, a ghost's origin or the dummy can be seen where the light places them. The figure keeps
the crop's aspect: its height is a fraction of the row, its width follows from the row's pixel
aspect and the crop's.
"""

import math
from typing import Protocol

import numpy as np

from modules.board import HasFrames
from modules.pose.features import Azimuth, Points2D, PointLandmark
from modules.pose.nodes import AngleExtractorSettings
from modules.render import ColorSettings
from modules.render.layers import LayerBase
from modules.render.shaders import PosePointLines

from ...light.layers import normalize_azimuth
from ...settings import Stage

_HEIGHT: float = 0.8        # of the row
_LINE_PX: float = 2.0
_SMOOTH_PX: float = 1.0


def figure_spans(x: float, width: float) -> list[float]:
    """The left edges a figure is drawn at: its own, and once more a turn over when it runs off
    the row's 0/360 join, so it shows whole on both sides of the join."""
    spans = [x]
    if x < 0.0:
        spans.append(x + 1.0)
    elif x + width > 1.0:
        spans.append(x - 1.0)
    return spans


def eye_column(points: Points2D) -> float:
    """Where the eyes are across the crop (0..1): the eyes' mean, the nose when they are missing,
    the centre when both are. The azimuth is the eyes', so this column is placed at it."""
    eyes = points.values[[PointLandmark.left_eye, PointLandmark.right_eye], 0]
    if not np.isnan(eyes).any():
        return float(eyes.mean())
    nose = float(points.values[PointLandmark.nose, 0])
    return 0.5 if math.isnan(nose) else nose


class PoseFigureBoard(HasFrames, Protocol):
    ...


class PoseFigureLayer(LayerBase):
    """The poses' figures over the projection row; see the module docstring.

    Owns no FBO: it draws into whatever viewport is current when `draw()` is called, which must be
    the projection row's, sized by the last `allocate`.
    """

    def __init__(self, board: PoseFigureBoard, colors: ColorSettings, angle_extractor: AngleExtractorSettings) -> None:
        self._board: PoseFigureBoard = board
        self._colors: ColorSettings = colors
        self._angle_extractor: AngleExtractorSettings = angle_extractor   # its aspect_ratio is the crop's
        self._shader: PosePointLines = PosePointLines()
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        # The shader is ref-counted: allocate it once, or a single deallocate would never free it.
        if not self._shader.allocated:
            self._shader.allocate()
        self._width = max(1, width)
        self._height = max(1, height)

    def deallocate(self) -> None:
        if self._shader.allocated:
            self._shader.deallocate()

    def update(self) -> None:
        pass

    def draw(self) -> None:
        frames = self._board.get_frames(int(Stage.LERP))
        if not frames:
            return
        aspect: float = float(self._angle_extractor.aspect_ratio)
        height: float = _HEIGHT
        width: float = _HEIGHT * self._height / self._width * aspect     # the figure keeps the crop's aspect
        y: float = (1.0 - height) / 2.0                                  # centred, so the same from either edge
        line_width: float = _LINE_PX / (height * self._height)
        line_smooth: float = _SMOOTH_PX / (height * self._height)
        colors = self._colors.track_color_tuples

        for track_id, frame in frames.items():
            azimuth: float = frame[Azimuth].value
            points = frame[Points2D]
            if math.isnan(azimuth) or points.valid_count == 0:
                continue
            color = colors[track_id % len(colors)]
            for x in figure_spans(normalize_azimuth(azimuth) - eye_column(points) * width, width):
                self._shader.use(points, line_width=line_width, line_smooth=line_smooth, color=color,
                                 use_scores=False, rect=(x, y, width, height), aspect_ratio=aspect)
