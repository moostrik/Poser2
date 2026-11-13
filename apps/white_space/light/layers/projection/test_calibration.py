"""TestCalibration composition — projects live camera slices into the projection, so a camera's
columns can be checked on the wall against the azimuths the tracker assigns them.

Each camera contributes a horizontal luminance slice of its uploaded image, reduced to one row on the GPU and read
back a tick later (`CameraColumns`). Columns that pass the brightness threshold are projected as a hard ON marker at
``level``, making it easy to verify that edges align across camera seams.

``show_overlap`` optionally renders the geometric overlap zones in blue so the
operator can verify seam positions without needing a feature there.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING

import numpy as np

from modules.settings import Field

from .._base_layer import ProjectionLayer, LayerSettings
from ...frame import LightFrame
from .camera_columns import CameraColumns, SampleMethod

if TYPE_CHECKING:
    from ....board import RenderBoard


class DetectMode(IntEnum):
    BRIGHT = auto()
    DARK   = auto()


class TestCalibrationSettings(LayerSettings):
    """Settings for the Calibration composition."""
    slice_center:  Field[float]        = Field(0.5,  min=0.0,  max=1.0,   step=0.01, description="Vertical center of the sample band (0=top, 1=bottom)")
    slice_height:  Field[float]        = Field(0.2,  min=0.01, max=1.0,   step=0.01, description="Height of the sample band as a fraction of frame height")
    threshold:     Field[float]        = Field(0.5,  min=0.0,  max=1.0,   step=0.01, description="BRIGHT: column value > threshold is ON. DARK: column value < threshold is ON")
    detect:        Field[DetectMode]   = Field(DetectMode.BRIGHT,                     description="Compare column value against threshold from above (BRIGHT) or below (DARK)")
    method:        Field[SampleMethod] = Field(SampleMethod.MAX,                      description="How to reduce each column to a single value: MAX, MIN or AVG")
    show_overlap:  Field[bool]         = Field(True,                                  description="Highlight geometric overlap zones", newline=True)
    show_center:   Field[bool]         = Field(False,                                 description="Highlight camera center lines")
    white_markers: Field[bool]         = Field(False,                                 description="Show markers in white instead of blue")
    fov:           Field[float]        = Field(110.0, description="Camera horizontal FOV (shared from compositor)", access=Field.INIT)


class TestCalibration(ProjectionLayer):
    """Projects horizontal camera slices into the projection, to check on the wall that a camera's
    columns land at the azimuths the tracker thinks they do."""

    def __init__(
        self,
        resolution:  int,
        settings:    TestCalibrationSettings,
        num_cams:    int,
        board:       RenderBoard,
    ) -> None:
        super().__init__(resolution, settings, board)
        self._settings: TestCalibrationSettings = settings
        self._num_cams = num_cams
        self._columns: CameraColumns = CameraColumns()

    # ------------------------------------------------------------------
    # Layer interface
    # ------------------------------------------------------------------

    def _draw(self, frame: LightFrame, white: np.ndarray, blue: np.ndarray) -> None:
        settings = self._settings
        fov  = settings.fov
        nc   = self._num_cams
        res  = self.resolution

        # Geometry constants
        target_fov  = 360.0 / nc
        fov_overlap = (fov - target_fov) / 2.0

        thresh = settings.threshold
        band_top: float = settings.slice_center - settings.slice_height * 0.5
        band_bottom: float = settings.slice_center + settings.slice_height * 0.5

        for cam_id in range(nc):
            image = self._board.get_camera_image(cam_id)
            if image is not None:
                self._columns.submit(image, band_top, band_bottom, settings.method)
            col_val: np.ndarray | None = self._columns.newest(cam_id)
            if col_val is None:
                continue
            w: int = col_val.shape[0]

            if settings.detect == DetectMode.BRIGHT:
                on_mask = col_val > thresh
            else:
                on_mask = col_val < thresh

            if not on_mask.any():
                continue

            # ----------------------------------------------------------
            # Project ON columns into the projection. The camera's warp delivers an
            # equirectangular frame, so a column IS an azimuth: the mapping is linear and
            # there is nothing to undistort. (It used to carry an inlined copy of the
            # tracker's `undistort_x`, which the projection change made redundant.)
            # ----------------------------------------------------------
            cam_x = np.linspace(0.0, 1.0, w, endpoint=False, dtype=np.float32)
            on_x  = cam_x[on_mask]

            world_angle = (target_fov * cam_id + on_x * fov - fov_overlap) % 360.0
            position    = world_angle / 360.0                       # normalized azimuth
            pixel       = (position * res).astype(np.int32) % res

            white[pixel] = np.maximum(white[pixel], 1.0)

        # ----------------------------------------------------------
        # Overlap zone indicator
        # ----------------------------------------------------------
        marker = white if settings.white_markers else blue
        if settings.show_overlap and fov_overlap > 0.0:
            half_w = fov_overlap / 360.0
            for cam_id in range(nc):
                for edge in (target_fov * cam_id, target_fov * (cam_id + 1)):
                    center = (edge / 360.0) % 1.0
                    lo = int(((center - half_w) % 1.0) * res) % res
                    hi = int(((center + half_w) % 1.0) * res) % res
                    if lo < hi:
                        marker[lo:hi] = np.maximum(marker[lo:hi], 1.0)
                    elif lo > hi:
                        marker[lo:] = np.maximum(marker[lo:], 1.0)
                        marker[:hi] = np.maximum(marker[:hi], 1.0)

        if settings.show_center:
            half_c = 5
            for cam_id in range(nc):
                center_angle = (target_fov * cam_id + target_fov * 0.5) % 360.0
                mid = int(center_angle / 360.0 * res) % res
                lo  = (mid - half_c) % res
                hi  = (mid + half_c) % res
                if lo < hi:
                    marker[lo:hi] = np.maximum(marker[lo:hi], 1.0)
                else:
                    marker[lo:] = np.maximum(marker[lo:], 1.0)
                    marker[:hi] = np.maximum(marker[:hi], 1.0)

