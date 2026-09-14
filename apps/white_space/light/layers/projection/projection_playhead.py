"""ProjectionPlayhead — visualises the content playhead as a bright marker in the projection.

Draws a marker at the normalized azimuth of ``frame.playhead`` (the continuous content playhead,
radians [-π, π); NaN → nothing drawn). Inside a person's **mask** (the pose instrument's dim blue
band at their azimuth, ``PI.mask``) the marker dims itself to ``playhead_at_mask`` of its level: it
reads the poses' azimuth and length from the LERP frames and the mask width from the same settings
the instrument draws with, so the instrument never blinds and no layer shares a mix at a person.
Distinct from the beam-mode ``BeamPlayhead`` and the motor/content ``Playhead`` (the NCO in
``light/playhead.py``).
"""

import math

import numpy as np

from modules.pose import features
from modules.settings import Field

from .._base_layer import ProjectionLayer, LayerSettings
from .._utilities import normalize_azimuth, mask_half_width
from .pose_instrument import MaskSettings
from ...frame import Frame


class ProjectionPlayheadSettings(LayerSettings):
    level: Field[float] = Field(1.0, min=0.0, max=1.0,  step=0.01, description="Marker brightness")
    width: Field[float] = Field(3.6, min=0.1, max=36.0, step=0.1,  description="Marker width (deg)")


class ProjectionPlayhead(ProjectionLayer):
    """A bright marker at the playhead's position in the projection, dimmed inside the masks."""

    def __init__(self, resolution: int, config: ProjectionPlayheadSettings, mask: MaskSettings, board,
                 pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._mask = mask
        self._pose_stage = pose_stage
        self._masked = np.zeros(resolution, dtype=bool)

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        ph = frame.playhead
        if math.isnan(ph):
            return
        P = self._config
        center = int(normalize_azimuth(ph) * self.resolution)   # [0, R)
        w      = max(1, round(P.width / 360.0 * self.resolution))     # deg → pixel count
        start  = center - w // 2
        idx    = np.arange(start, start + w) % self.resolution
        level  = np.full(w, P.level, dtype=white.dtype)
        level[self._in_masks()[idx]] *= self._mask.playhead_at_mask
        white[idx] += level

    def _in_masks(self) -> np.ndarray:
        """The pixels under a person's mask this tick, as the instrument draws it."""
        R = self.resolution
        masked = self._masked
        masked.fill(False)
        for pose in self._board.get_frames(self._pose_stage).values():
            azimuth = pose[features.Azimuth].value
            if math.isnan(azimuth):
                continue
            centre = int(round(normalize_azimuth(azimuth) * R)) % R
            height = pose[features.BBox][features.BBoxElement.height]
            length = height if not math.isnan(height) and height > 0.0 else 1.0
            half = mask_half_width(self._mask.width, length, R)
            masked[(centre + np.arange(-half, half + 1)) % R] = True
        return masked
