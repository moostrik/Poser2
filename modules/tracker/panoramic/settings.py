from modules.oak import CameraResolution
from modules.settings import BaseSettings, Field, Group


class SeamAngles(BaseSettings):
    fov: Field[float] = Field(0.0, access=Field.READ, description="Camera FOV (°)")
    overlap: Field[float] = Field(0.0, access=Field.READ, description="Camera FOV overlap zone (°)")
    reject: Field[float] = Field(0.0, access=Field.READ, description="Dead zone at camera edges (°)")
    reach: Field[float] = Field(0.0, access=Field.READ, description="Cross-camera matching zone from camera edges (°)")


class SeamSettings(BaseSettings):
    reject: Field[float] = Field(0.5, min=0.0, max=0.75, step=0.05,
                                 description="Dead zone size as a fraction of the overlap zone.")
    reach: Field[float] = Field(1.3, min=1.0, max=1.5, step=0.05,
                                description="Matching zone size as a fraction of the overlap zone.")
    hysteresis: Field[float] = Field(0.9, min=0.1, max=1.0, step=0.05,
                                     description="Lower values make active camera stickier.")
    max_height_diff: Field[float] = Field(0.15, min=0.0, max=0.5, step=0.01,
                                          description="Maximum ROI height difference for matching two observations.")
    relink_angle: Field[float] = Field(5.0, min=0.0, max=20.0, step=0.5,
                                       description="How far (°) a re-acquired person may be from the lost one and still be them")
    angles: Group[SeamAngles] = Group(SeamAngles)


class ParallaxSettings(BaseSettings):
    """
    Corrects for the cameras sitting on a ring rather than at a shared optical
    centre. Each camera is ``ring_radius`` metres from the rig centre, so the
    same person is seen at different world angles by neighbouring cameras — a
    disagreement of several degrees at the seams. Distance to the person is
    estimated from where their feet meet the floor, which needs only the lens
    height above it, and that is enough to re-project each observation to the
    shared centre.

    Both numbers are *measured with a tape*, not tuned. At ``ring_radius = 0``
    the correction is disabled (identity).
    """
    ring_radius: Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01,
                                      description="Camera distance from rig centre (m), measured. 0 disables the correction")
    camera_height: Field[float] = Field(0.5, min=0.1, max=3.0, step=0.01,
                                        description="Lens height above the floor (m), measured")
    # The delivered frame's row model, published by the tracker for whatever draws with its
    # numbers (the panorama). Rows are tangents of elevation: row = horizon_row - focal_rows * tan(e).
    vfov: Field[float] = Field(79.5, access=Field.READ,
                              description="Elevation span (°) of the delivered frame, bottom row to top row")
    elevation_bottom: Field[float] = Field(-39.7, access=Field.READ,
                                          description="Elevation (°) of the frame's bottom row, at the camera")
    elevation_top: Field[float] = Field(39.7, access=Field.READ,
                                       description="Elevation (°) of the frame's top row, at the camera")
    horizon_row: Field[float] = Field(0.5, access=Field.READ,
                                     description="Normalised row (0 = top) of the horizon; may fall outside 0..1")
    focal_rows: Field[float] = Field(0.72, access=Field.READ,
                                    description="Focal length in frame heights: rows below the horizon = focal_rows · tan(depression)")


class TrackerSettings(BaseSettings):
    fov: Field[float] = Field(110.0, access=Field.INIT)
    # The frame geometry, shared from the camera group: what the warp was built with, so the
    # tracker's row model is the frame's.
    resolution: Field[CameraResolution] = Field(CameraResolution.P800, access=Field.INIT,
                                                description="Sensor mode, shared — the frame's shape")
    frame_height: Field[int] = Field(0, access=Field.INIT,
                                    description="Delivered frame height (px), shared; 0 = derived from the tilt")
    tilt: Field[float] = Field(0.0, access=Field.INIT, description="Camera up-tilt (°), shared")
    lens_fov: Field[float] = Field(0.0, access=Field.INIT, description="Lens field (°) across the sensor width, shared; 0 = fov")
    lens_centre_x: Field[float] = Field(0.0, access=Field.INIT, description="Optical centre offset (px), shared")
    lens_centre_y: Field[float] = Field(0.0, access=Field.INIT, description="Optical centre offset (px), shared")
    min_age: Field[int] = Field(5, min=0, max=9, step=1,
                                description="Minimum age in frames before a tracklet is considered.")
    min_height: Field[float] = Field(0.25, min=0.0, max=1.0, step=0.05,
                                     description="Minimum ROI height to accept a tracklet.")
    timeout: Field[float] = Field(2.0, min=1.0, max=5.0, step=0.1,
                                  description="Seconds a lost observation keeps anchoring — must outlast a seam crossing")
    emit_hold: Field[float] = Field(0.3, min=0.0, max=2.0, step=0.05,
                                    description="Seconds a world keeps being emitted after its last detection")
    seam: Group[SeamSettings] = Group(SeamSettings)
    parallax: Group[ParallaxSettings] = Group(ParallaxSettings)
