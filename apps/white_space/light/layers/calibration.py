"""Calibration composition — projects live camera slices onto the LED strip
for visual distortion tuning.

Each camera contributes a horizontal luminance slice (rows slice_top..slice_bottom).
Columns that pass the brightness threshold are projected as a hard ON marker at
``level``, making it easy to verify that edges align across camera seams.

``show_overlap`` optionally renders the geometric overlap zones in blue so the
operator can verify seam positions without needing a feature there.
"""

from enum import IntEnum, auto
from threading import Lock

import numpy as np

from modules.settings import Field
from modules.tracker.panoramic.settings import DistortionSettings, DistortAlgorithm

from ..base_layer import BaseLayer, LayerSettings
from ..frame import Frame


class DetectMode(IntEnum):
    BRIGHT = auto()
    DARK   = auto()


class SampleMethod(IntEnum):
    MAX = auto()
    MIN = auto()
    AVG = auto()


class CalibrationSettings(LayerSettings):
    """Settings for the Calibration composition."""
    slice_centre:  Field[float]        = Field(0.5,  min=0.0,  max=1.0,   step=0.01, description="Vertical centre of the sample band (0=top, 1=bottom)")
    slice_height:  Field[float]        = Field(0.2,  min=0.01, max=1.0,   step=0.01, description="Height of the sample band as a fraction of frame height")
    threshold:     Field[float]        = Field(0.5,  min=0.0,  max=1.0,   step=0.01, description="BRIGHT: column value > threshold is ON. DARK: column value < threshold is ON")
    detect:        Field[DetectMode]   = Field(DetectMode.BRIGHT,                     description="Compare column value against threshold from above (BRIGHT) or below (DARK)")
    method:        Field[SampleMethod] = Field(SampleMethod.MAX,                      description="How to reduce each column to a single value: MAX, MIN or AVG")
    show_overlap:  Field[bool]         = Field(True,                                  description="Highlight geometric overlap zones", newline=True)
    show_centre:   Field[bool]         = Field(False,                                 description="Highlight camera centre lines")
    white_markers: Field[bool]         = Field(False,                                 description="Show markers in white instead of blue")
    fov:           Field[float]        = Field(110.0, min=60.0, max=180.0, step=0.5,  description="Camera horizontal FOV (shared from compositor)", access=Field.READ)


class Calibration(BaseLayer):
    """Projects horizontal camera slices onto the LED strip for distortion calibration."""

    def __init__(
        self,
        resolution:  int,
        config:      CalibrationSettings,
        distortion:  DistortionSettings,
        num_cameras: int,
    ) -> None:
        super().__init__(resolution, config)
        self._config      = config
        self._distortion  = distortion
        self._num_cameras = num_cameras
        self._lock        = Lock()
        self._images: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Layer interface
    # ------------------------------------------------------------------

    def set_camera_image(self, cam_id: int, image: np.ndarray) -> None:
        """Store the latest VIDEO frame for a camera; called once per render tick."""
        with self._lock:
            self._images[cam_id] = image

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        cfg = self._config
        fov  = cfg.fov
        nc   = self._num_cameras
        res  = self.resolution

        with self._lock:
            images = dict(self._images)

        # DistortionSettings snapshot (live from tracker)
        dist    = self._distortion
        algo    = dist.algorithm
        tanh_s  = dist.tanh.slope
        tanh_c  = dist.tanh.cubic
        poly_k1 = dist.poly.k1
        poly_k2 = dist.poly.k2

        # Geometry constants
        target_fov  = 360.0 / nc
        fov_overlap = (fov - target_fov) / 2.0

        thresh = cfg.threshold

        for cam_id, img in images.items():
            if img is None or img.size == 0:
                continue

            # ----------------------------------------------------------
            # Extract horizontal luminance slice
            # ----------------------------------------------------------
            h, w = img.shape[:2]
            row_top    = max(0, int((cfg.slice_centre - cfg.slice_height * 0.5) * h))
            row_bottom = min(h, int((cfg.slice_centre + cfg.slice_height * 0.5) * h))
            if row_bottom <= row_top:
                continue

            slice_rows = img[row_top:row_bottom]
            if slice_rows.ndim == 3:
                lum = slice_rows.mean(axis=2).astype(np.float32)
            else:
                lum = slice_rows.astype(np.float32)

            if lum.max() > 1.0:
                lum /= 255.0

            # Reduce each column to a single value using the chosen method
            if cfg.method == SampleMethod.MAX:
                col_val = lum.max(axis=0)
            elif cfg.method == SampleMethod.MIN:
                col_val = lum.min(axis=0)
            else:
                col_val = lum.mean(axis=0)

            if cfg.detect == DetectMode.BRIGHT:
                on_mask = col_val > thresh
            else:
                on_mask = col_val < thresh

            if not on_mask.any():
                continue

            # ----------------------------------------------------------
            # Inline undistort + project ON columns onto strip
            # ----------------------------------------------------------
            cam_x = np.linspace(0.0, 1.0, w, endpoint=False, dtype=np.float32)
            on_x  = cam_x[on_mask]

            if algo == DistortAlgorithm.POLY:
                d   = on_x - 0.5
                ux  = on_x + poly_k1 * d + poly_k2 * (d ** 3)
            elif algo == DistortAlgorithm.TANH:
                t   = 2.0 * on_x - 1.0
                ux  = 0.5 * (1.0 + np.tanh(tanh_s * t + tanh_c * (t ** 3)))
            else:
                ux  = on_x

            world_angle = (target_fov * cam_id + ux * fov - fov_overlap) % 360.0
            strip_pos   = world_angle / 360.0
            strip_idx   = (strip_pos * res).astype(np.int32) % res

            white[strip_idx] = np.maximum(white[strip_idx], 1.0)

        # ----------------------------------------------------------
        # Overlap zone indicator
        # ----------------------------------------------------------
        marker = white if cfg.white_markers else blue
        if cfg.show_overlap and fov_overlap > 0.0:
            half_w = fov_overlap / 360.0
            for cam_id in range(nc):
                for edge in (target_fov * cam_id, target_fov * (cam_id + 1)):
                    centre = (edge / 360.0) % 1.0
                    lo = int(((centre - half_w) % 1.0) * res) % res
                    hi = int(((centre + half_w) % 1.0) * res) % res
                    if lo < hi:
                        marker[lo:hi] = np.maximum(marker[lo:hi], 1.0)
                    elif lo > hi:
                        marker[lo:] = np.maximum(marker[lo:], 1.0)
                        marker[:hi] = np.maximum(marker[:hi], 1.0)

        if cfg.show_centre:
            half_c = 5
            for cam_id in range(nc):
                centre_angle = (target_fov * cam_id + target_fov * 0.5) % 360.0
                mid = int(centre_angle / 360.0 * res) % res
                lo  = (mid - half_c) % res
                hi  = (mid + half_c) % res
                if lo < hi:
                    marker[lo:hi] = np.maximum(marker[lo:hi], 1.0)
                else:
                    marker[lo:] = np.maximum(marker[lo:], 1.0)
                    marker[:hi] = np.maximum(marker[:hi], 1.0)

