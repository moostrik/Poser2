from modules.oak import CameraResolution
from modules.settings import BaseSettings, Field, Group, Widget


class SeamSettings(BaseSettings):
    """When two cameras' views of a seam are one person, and where a person may be born.

    All in real units — degrees of world azimuth and a fraction of a measured height — rather
    than fractions of the overlap zone, so a number here means the same thing after a change of
    `fov`, mount or lens. Azimuth is the quantity the panorama check verifies; the distance
    estimate is not, so it is deliberately not a gate.

    **Only tunables live here.** The overlap two neighbours share is *derived* from the ring and
    the zone's far edge, so it is published on `RigSettings` beside the things it comes from,
    rather than in a sub-group here where it read as though it were one of these four.
    """
    dead_zone: Field[float] = Field(5.0, min=0.0, max=20.0, step=0.5,
                                    description="No new person is born within this many ° of a camera's field edge")
    link_angle: Field[float] = Field(8.0, min=0.0, max=40.0, step=0.5,
                                     description="Two cameras' observations within this many ° of azimuth are one person")
    # Normalised 0..1, like `height_filter`: a fraction of the larger of the two measured heights.
    link_height: Field[float] = Field(0.15, min=0.0, max=1.0, step=0.01,
                                      description="Maximum fraction the two measured heights may differ when linking")
    hysteresis: Field[float] = Field(0.9, min=0.1, max=1.0, step=0.05,
                                     description="Lower values make active camera stickier.")


class RigSettings(BaseSettings):
    """The installation in metres: where the lenses are, and where people are tracked.

    **Two circles, so two prefixes.** ``camera_`` is the ring the lenses sit on; ``zone_`` is the
    floor people are tracked on.

    **Everything is a RADIUS, measured from the fixture's axis**, because that is where the
    installation is built and taped from: the light fixture stands at the centre, so the origin is
    a physical object you can hook a tape to rather than a point to infer. Every triangle in
    `Geometry`, `panorama_map` and `panoramicstitch.frag` already takes a radius, and so does the
    panorama's ``R`` label — so a number here, a number in the footer, a number on a person's label
    and a tape on the floor are now **one number**, with nothing halved anywhere between them.
    (``camera_height`` is the exception that proves the rule: a height is not a radius.)

    Nothing here is tuned. The camera ring is *measured*; the zone is *decided* and then taped on
    the floor. Three things depend on them:

    - **The parallax correction.** The cameras sit on a ring rather than at a shared optical
      centre, so the same person is seen at different world angles by neighbours — several degrees
      of disagreement at a seam. The distance to the person comes from where their feet meet the
      floor, which needs only the lens height, and that is enough to re-project every observation
      to the shared centre. At ``camera_radius = 0`` the correction is disabled (identity).
    - **The overlap band** (`Geometry.angle_in_overlap`), derived at ``zone_max_radius``: the
      widest band two cameras can share anywhere inside the zone, and so the most generous
      depth-free bound that never under-reports where people actually are.
    - **The parallax depth** (``parallax_radius``, published below), derived at the zone's
      *harmonic* mean: the one depth every world azimuth is corrected at. Two depths from one zone
      on purpose — a flag must never under-report, so it takes the far edge; a correction wants its
      worst case smallest, so it takes the middle.
    - **The far edge**, ``zone_max_radius``, while the tracker's ``zone_filter`` is on: past it a
      person is not seen — not born, and dropped after the tracker's timeouts if they walk out
      (`Geometry.beyond_zone`). The panorama draws them as a grey line tagged ``past R…``. The near
      edge filters nothing, because close to the fixture the feet are often below the frame.
    """
    camera_radius: Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01,
                                        description="Distance (m) of each lens from the fixture axis. 0 disables the parallax correction")
    camera_height: Field[float] = Field(0.5, min=0.1, max=3.0, step=0.01,
                                        description="Lens height above the floor (m), measured")
    zone_min_radius: Field[float] = Field(1.5, min=0.25, max=10.0, step=0.05, newline=True,
                                          description="Radius (m) of the tracked floor's near edge — the nearest distance claimed")
    zone_max_radius: Field[float] = Field(3.5, min=0.5, max=15.0, step=0.05,
                                          description="Radius (m) of its far edge — sets the overlap band, and where zone_filter ignores people")
    # Derived from the zone and published for the panorama, which places its marks on the same
    # cylinder the tracker corrects the azimuth at. The harmonic mean, because the correction is
    # linear in 1/d — see `Geometry._update_parallax_depth`.
    parallax_radius: Field[float] = Field(2.1, access=Field.READ,
                                          description="Radius (m) the world azimuth is corrected at — derived, the zone's harmonic mean")
    # Beside the two it comes from — the ring and `zone_max_radius` — rather than under `seam`,
    # where it read as a fifth tunable. In WORLD AZIMUTH, which is what the panorama's degree grid
    # measures and what its overlap lines are drawn from; `angle_in_overlap` tests the *local*
    # equivalent, which `Geometry` derives alongside and keeps to itself (the map compresses, so
    # the two differ). `fov` is the setting above, not republished here.
    overlap: Field[float] = Field(0.0, access=Field.READ,
                                  description="Azimuth (°) two neighbours share, at the zone's far edge")
    # The delivered frame's shape, published by the tracker for whatever draws with its numbers (the
    # panorama). Horizontal first, as a frame is quoted. `hfov` and `tilt` mirror init fields the
    # panel hides by default, so the frame can be read here in one place.
    #
    # The two angles ARE the row model: rows are tangents of elevation, and `panorama_map.row_model`
    # rebuilds (horizon_row, focal_rows) from them exactly, so nothing in row space is published.
    # `angle_bottom` follows the tilt 1:1 (the frame is pinned at the sensor's lowest reach);
    # `angle_top` only partly (+1° to +1.5° per +3°), since a fixed number of tangent rows spend
    # themselves at the top. It is the frame's top ROW, which may sit above what the sensor fills.
    hfov: Field[float] = Field(127.0, access=Field.READ, newline=True,
                               description="Azimuth span (°) of the delivered frame, left edge to right")
    vfov: Field[float] = Field(79.5, access=Field.READ,
                              description="Angle span (°) of the delivered frame, bottom row to top row")
    tilt: Field[float] = Field(0.0, access=Field.READ,
                               description="Camera up-tilt (°), as the frame was built with")
    angle_bottom: Field[float] = Field(-39.7, access=Field.READ,
                                       description="Angle (°) from eye level of the frame's bottom row; negative is below")
    angle_top: Field[float] = Field(39.7, access=Field.READ,
                                    description="Angle (°) from eye level of the frame's top row")
    # What that frame allows, as radii from the fixture — the numbers the tilt is chosen by, live
    # for the running configuration (CALIBRATION.md, *Tilt — derived from the build*, has them as
    # design tables). Per column, against what the sensor actually fills rather than the frame's
    # rows, so the black arch counts. Feet once: the bottom row is covered at every bearing.
    # Hands twice, because the sensor's top edge falls toward the seams where people cross.
    feet_from: Field[float] = Field(0.0, access=Field.READ, newline=True,
                                    description="Nearest radius (m) with the feet in frame")
    hands_from: Field[float] = Field(0.0, access=Field.READ,
                                     description="Nearest radius (m) with 2.2 m raised hands in frame, on a camera's axis")
    hands_seam: Field[float] = Field(0.0, access=Field.READ,
                                     description="Nearest radius (m) with 2.2 m raised hands in frame, on a seam")


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
    # The intake's filters, together. A detection one of them drops is not counted, and the panorama
    # draws it as a grey line tagged with the filter (`young`, `small`, `past R…`).
    age_filter: Field[int] = Field(5, min=0, max=9, step=1,
                                   description="Minimum age in frames before a tracklet is considered.")
    height_filter: Field[float] = Field(0.25, min=0.0, max=1.0, step=0.05,
                                        description="Minimum ROI height to accept a tracklet.")
    # A switch, not a radius: whether the zone's far edge (`rig.zone_max_radius`) acts at all. Off
    # means off — nothing is filtered there and nothing on the strip mentions it; the other filters
    # still run.
    zone_filter: Field[bool] = Field(True, widget=Widget.switch,
                                     description="Ignore people past rig.zone_max_radius (dropped after the lost timeouts)")
    # A property of the DETECTOR, not of the site, which is why it is here and not in `rig`: that
    # group is metres someone measured on the floor. Tuned by walking one person out until the
    # panorama's `H` stops drifting — the ROI is never rewritten, only the row derived from it
    # (`Geometry._foot_px`), so `height_filter` and the crop extractor still see the detector's box.
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
