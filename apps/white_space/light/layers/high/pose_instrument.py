"""PoseInstrument — PLACEHOLDER for the heart of the piece (see ``data/LAYERS.md``).

The real instrument is a deferred major work package: each person stands in a blue
light, surrounded by a white-and-blue line pattern derived from their pose (pose →
pattern as pose → sound), with all patterns living in one shared phase world so that
matched patterns filling the space between people merge seamlessly.

What this placeholder does now: it **receives the complete input contract** the real
instrument will need — per participant: azimuth, pose length (BBox), the four arm
angles, presence; per pair: pairwise Similarity — and visualises each input in the
simplest legible way:
  - white: a band per participant (width from pose length, brightness plainly modulated
    by the arm angles — enough to see the data move)
  - white: a flat arc between each similarity-matched pair (the simplest sync fill)
  - blue: a marker per participant (the "stands in a blue light" spot)
The real instrument grows inside this layer with all plumbing in place; any striping it
adds must anchor to the shared phase world (a global grid, never per-person phase).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from modules.pose import features
from modules.settings import Field

from .._base_layer import HighLayer, LayerSettings
from .._utilities import BlendType, angle_to_strip_position, draw_field
from ...frame import Frame

if TYPE_CHECKING:
    from ....board import Board


class PoseInstrumentSettings(LayerSettings):
    band_width:     Field[float] = Field(21.6, min=0.0, max=180.0, step=0.5, description="White band base width (deg; scaled by pose length)")
    band_level:     Field[float] = Field(0.8,  min=0.0, max=1.0,  step=0.01, description="White band level (modulated by the arm angles)")
    band_edge:      Field[float] = Field(7.2,  min=0.0, max=72.0, step=0.1,  description="Band edge softness (deg)")
    marker_width:   Field[float] = Field(7.2,  min=0.0, max=72.0, step=0.1,  description="Blue marker width (deg — the person's blue light)", newline=True)
    marker_level:   Field[float] = Field(0.8,  min=0.0, max=1.0,  step=0.01, description="Blue marker level")
    fill_threshold: Field[float] = Field(0.75, min=0.0, max=1.0,  step=0.01, description="Pairwise similarity at which the sync fill lights", newline=True)
    fill_level:     Field[float] = Field(0.5,  min=0.0, max=1.0,  step=0.01, description="Sync fill level (scaled by the pair's similarity)")
    fill_edge:      Field[float] = Field(10.8, min=0.0, max=72.0, step=0.1,  description="Sync fill end softness (deg)")


def _shorter_arc(a: float, b: float) -> tuple[float, float]:
    """Centre and width (both strip fractions) of the shorter arc between ring
    positions ``a`` and ``b`` in [0, 1)."""
    delta = ((b - a + 0.5) % 1.0) - 0.5     # signed shortest offset a → b
    return (a + delta / 2.0) % 1.0, abs(delta)


class PoseInstrument(HighLayer):
    """The placeholder instrument; see the module docstring."""

    def __init__(self, resolution: int, config: PoseInstrumentSettings,
                 board: Board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config
        tracklets = self._board.get_tracklets()
        frames = self._board.get_frames(self._pose_stage)

        # -- Per participant: azimuth position + pose length + angle modulation ------
        positions: dict[int, float] = {}
        for id, pose in frames.items():
            tracklet = tracklets.get(id)
            if tracklet is None or not tracklet.is_active:
                continue
            azimuth = pose[features.Azimuth].value
            if math.isnan(azimuth):
                continue
            position = angle_to_strip_position(azimuth)
            positions[id] = position

            height = pose[features.BBox][features.BBoxElement.height]
            length = height if not math.isnan(height) and height > 0.0 else 1.0

            angles = pose[features.Angles].values
            arm = [angles[features.AngleLandmark.left_shoulder],
                   angles[features.AngleLandmark.right_shoulder],
                   angles[features.AngleLandmark.left_elbow],
                   angles[features.AngleLandmark.right_elbow]]
            raised = [0.5 + 0.5 * math.cos(a) for a in arm if not math.isnan(a)]
            modulation = 0.5 + 0.5 * (sum(raised) / len(raised)) if raised else 0.75

            width = P.band_width / 360.0 * (0.5 + 0.5 * length)   # deg → strip fraction
            edge = int(P.band_edge / 360.0 * self.resolution)
            draw_field(white, position, width, P.band_level * modulation, edge, BlendType.MAX)
            draw_field(blue, position, P.marker_width / 360.0, P.marker_level, edge, BlendType.MAX)

        # -- Per pair: the sync fill — a flat arc between similarity-matched pairs ---
        ids = sorted(positions)
        for i, id_a in enumerate(ids):
            sim_feature = frames[id_a][features.Similarity]
            for id_b in ids[i + 1:]:
                if id_b >= len(sim_feature.values):
                    continue
                sim = float(sim_feature.values[id_b])
                if math.isnan(sim) or sim < P.fill_threshold:
                    continue
                centre, width = _shorter_arc(positions[id_a], positions[id_b])
                if width <= 0.0:
                    continue
                edge = int(P.fill_edge / 360.0 * self.resolution)
                draw_field(white, centre, width, P.fill_level * sim, edge, BlendType.MAX)
