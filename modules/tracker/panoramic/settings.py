from modules.oak import CameraResolution
from modules.settings import BaseSettings, Field, Group


class SeamAngles(BaseSettings):
    """The frame's own azimuth spans, published by the tracker for whatever draws with them.

    Not settings: `fov` is the camera's field and `overlap` the azimuth two neighbours share, both
    fixed by the lens, the mount and the zone. Everything the fusion rules are tuned with is on
    `SeamSettings` in degrees or percent, so nothing here is a ratio of anything.

    `overlap` is in **world azimuth**, which is the number measurable against the panorama's degree
    grid and the one the overlap lines are drawn from. `angle_in_overlap` tests the *local*-angle
    equivalent, which `Geometry` derives alongside it and keeps to itself — the two differ because
    the local-to-azimuth map compresses.
    """
    fov: Field[float] = Field(0.0, access=Field.READ, description="Camera FOV (°)")
    overlap: Field[float] = Field(0.0, access=Field.READ,
                                  description="Azimuth (°) two neighbours share, at the zone's far edge")


class SeamSettings(BaseSettings):
    """When two cameras' views of a seam are one person, and where a person may be born.

    All in real units — degrees of world azimuth and a percentage of a measured height — rather
    than fractions of the overlap zone, so a number here means the same thing after a change of
    `fov`, mount or lens. Azimuth is the quantity the panorama check verifies; the distance
    estimate is not, so it is deliberately not a gate.
    """
    dead_zone: Field[float] = Field(5.0, min=0.0, max=20.0, step=0.5,
                                    description="No new person is born within this many ° of a camera's field edge")
    link_angle: Field[float] = Field(8.0, min=0.0, max=40.0, step=0.5,
                                     description="Two cameras' observations within this many ° of azimuth are one person")
    link_height: Field[float] = Field(15.0, min=0.0, max=100.0, step=1.0,
                                      description="Maximum % the two measured heights may differ when linking")
    hysteresis: Field[float] = Field(0.9, min=0.1, max=1.0, step=0.05,
                                     description="Lower values make active camera stickier.")
    angles: Group[SeamAngles] = Group(SeamAngles)


class RigSettings(BaseSettings):
    """The installation in metres: where the lenses are, and where people are tracked.

    **Two circles, so two prefixes.** ``camera_`` is the ring the lenses sit on; ``zone_`` is the
    floor people are tracked on. Everything is a **diameter**, as every figure in CALIBRATION.md
    is, so the group reads in one unit — radii exist only inside `Geometry`, which halves on the
    way in exactly as the render halves ``focus_diameter``.

    Nothing here is tuned. The camera pair is *measured with a tape*; the zone is *decided* and
    then taped on the floor. Three things depend on them:

    - **The parallax correction.** The cameras sit on a ring rather than at a shared optical
      centre, so the same person is seen at different world angles by neighbours — several degrees
      of disagreement at a seam. The distance to the person comes from where their feet meet the
      floor, which needs only the lens height, and that is enough to re-project every observation
      to the shared centre. At ``camera_diameter = 0`` the correction is disabled (identity).
    - **The overlap band** (`Geometry.angle_in_overlap`), derived at ``zone_max_diameter``: the
      widest band two cameras can share anywhere inside the zone, and so the most generous
      depth-free bound that never under-reports where people actually are.
    - **The parallax depth** (``parallax_diameter``, published below), derived at the zone's
      *harmonic* mean: the one depth every world azimuth is corrected at. Two depths from one zone
      on purpose — a flag must never under-report, so it takes the far edge; a correction wants its
      worst case smallest, so it takes the middle.
    - **The distance clamp**, from both diameters: a mangled bounding box can then only move the
      reported metres within the band people are in, never to a nonsensical depth.
    """
    camera_diameter: Field[float] = Field(0.0, min=0.0, max=2.0, step=0.01,
                                          description="Ø (m) of the ring the lenses sit on, measured. 0 disables the parallax correction")
    camera_height: Field[float] = Field(0.5, min=0.1, max=3.0, step=0.01, newline=True,
                                        description="Lens height above the floor (m), measured")
    zone_min_diameter: Field[float] = Field(3.0, min=0.5, max=20.0, step=0.1,
                                            description="Ø (m) of the tracked floor's near edge — the nearest distance claimed")
    zone_max_diameter: Field[float] = Field(7.0, min=1.0, max=30.0, step=0.1,
                                            description="Ø (m) of its far edge — sets the overlap band and the distance clamp")
    # Derived from the zone and published for the panorama, which places its marks on the same
    # cylinder the tracker corrects the azimuth at. The harmonic mean, because the correction is
    # linear in 1/d — see `Geometry._update_parallax_depth`.
    parallax_diameter: Field[float] = Field(4.2, access=Field.READ,
                                            description="Ø (m) the world azimuth is corrected at — derived, the zone's harmonic mean")
    # The delivered frame's row model, published by the tracker for whatever draws with its
    # numbers (the panorama). Rows are tangents of elevation: row = horizon_row - focal_rows * tan(e).
    vfov: Field[float] = Field(79.5, access=Field.READ, newline=True,
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
    # A property of the DETECTOR, not of the site, which is why it is here and not in `rig`: that
    # group is metres someone measured on the floor. Tuned by walking one person out until the
    # panorama's `H` stops drifting — the ROI is never rewritten, only the row derived from it
    # (`Geometry._foot_px`), so `min_height` and the crop extractor still see the detector's box.
    foot_offset: Field[float] = Field(0.0, min=0.0, max=0.2, step=0.005,
                                      description="How far below the feet the detector's box bottom sits (frame heights)")
    # Not under `seam`: this is the same camera re-finding a person it dropped, anywhere in its
    # field, and has nothing to do with two cameras meeting.
    reacquire_angle: Field[float] = Field(5.0, min=0.0, max=20.0, step=0.5,
                                          description="How far (°) a re-acquired person may be from the lost one and still be them")
    # The two windows a lost person lives in, adjacent because each is only clear beside the
    # other. `emit_timeout` is the shorter on purpose: a person stops driving the show well
    # before the tracker forgets them, which is what keeps a seam crossing linkable after the
    # near camera has given up. Inside it, a lost person still gets a pose from their last box —
    # which is how a dropped detection of a frame or two costs nothing.
    lost_timeout: Field[float] = Field(2.0, min=1.0, max=5.0, step=0.1,
                                       description="Seconds a lost person is remembered, for seam links and re-acquisition")
    emit_timeout: Field[float] = Field(0.3, min=0.0, max=2.0, step=0.05,
                                       description="Seconds a lost person is still emitted, so a pose survives a detection glitch")
    seam: Group[SeamSettings] = Group(SeamSettings)
    rig: Group[RigSettings] = Group(RigSettings)
