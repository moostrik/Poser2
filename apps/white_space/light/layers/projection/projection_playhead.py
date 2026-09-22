"""ProjectionPlayhead — the playhead's marker alone (debug-only, never in a state's mix).

The show's marker is drawn by the pose instrument over its masks (``pose_instrument.py``); this
layer draws the same ``PlayheadMarker``, with the same ``PI.playhead`` settings, on an otherwise
dark projection, so the playhead can be looked at by itself (``docs/CALIBRATION.md``: the
spin-up hands the beam over to it). It finds the masks from the LERP poses' azimuth and the mask
width the instrument draws with.
"""

import math

import numpy as np

from modules.pose import features

from .._base_layer import ProjectionLayer, LayerSettings
from .._utilities import normalize_azimuth, mask_half_width
from .playhead_marker import PlayheadMarker, PlayheadMarkerSettings
from .pose_instrument import MaskSettings
from ...frame import Frame


class ProjectionPlayhead(ProjectionLayer):
    """The marker by itself; see the module docstring."""

    def __init__(self, resolution: int, config: LayerSettings, marker: PlayheadMarkerSettings,
                 mask: MaskSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._marker = marker
        self._mask = mask
        self._pose_stage = pose_stage
        self._masked = np.zeros(resolution, dtype=bool)

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        PlayheadMarker.draw(white, blue, self.resolution, frame.playhead, self._marker, self._in_masks())

    def _in_masks(self) -> np.ndarray:
        """The pixels under a person's mask this tick, as the instrument draws it."""
        R = self.resolution
        masked = self._masked
        masked.fill(False)
        half = mask_half_width(self._mask.width, R)
        for pose in self._board.get_frames(self._pose_stage).values():
            azimuth = pose[features.Azimuth].value
            if math.isnan(azimuth):
                continue
            centre = int(round(normalize_azimuth(azimuth) * R)) % R
            masked[(centre + np.arange(-half, half + 1)) % R] = True
        return masked
