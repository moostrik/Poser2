# Standard library imports
from enum import IntEnum, auto

# Local application imports
from modules.settings import BaseSettings, Field, Widget


class PanoramaBlend(IntEnum):
    """How the two cameras' pixels combine where their fields overlap.

    The value **is** the stitch shader's `blendMode` uniform, so these must stay in step with the
    `#define`s at the top of `panoramicstitch.frag`, the way `MAX_CAMS` already does. `MAX` stays
    first and default: it is the plain picture, and the procedure in CALIBRATION.md is written
    around reading it.

    The first two merge the two views; the rest compare them, and outside an overlap — where there
    is only one view to compare — they fall back to the plain picture, except `DIFFERENCE`, which
    goes black so only the overlaps light up.
    """
    MAX        = 0        # the brighter of the two — a ghost reads as a doubled bright edge
    AVERAGE    = auto()   # both at half weight — a ghost reads as a soft double image
    MIN        = auto()   # the darker of the two — reads on bright content, where MAX saturates
    DIFFERENCE = auto()   # |a - b|: tune for black. Spoiled by the cameras' independent exposure
    SPLIT      = auto()   # one camera to red, its neighbour to green — says WHICH camera is left
    STRIPE     = auto()   # alternating columns — a straight edge zigzags; blind to exposure


class Part(IntEnum):
    """The pieces of the display, one member per renderer in this folder, named the same.

    `PanoramaLayerSettings.parts` is a checklist over these, so each piece can be taken off the
    strip without disturbing the others.
    """
    image        = 0      # StitchRenderer — the four camera frames
    seams        = auto()  # SeamRenderer — the seam rules that live in image space
    grid         = auto()  # GridRenderer — every reference mark in the strip's own two axes
    observations = auto()  # ObservationRenderer — a line per observation inside its tolerance
    labels       = auto()  # LabelRenderer — the per-person text


class PanoramaLayerSettings(BaseSettings):
    """The stitched 360-degree calibration view and the tracker data drawn over it.

    One display, one settings group: the image says whether the camera constants (`fov`, `tilt`)
    are right, the marks over it say whether the distance model is. They share a vertical scale,
    so a person's pixels and a person's numbers are compared in place.
    """
    parts: Field[list[Part]] = Field([Part.image, Part.grid, Part.observations, Part.labels],
                                     description="Which pieces of the display to draw")
    blend: Field[PanoramaBlend] = Field(PanoramaBlend.MAX,
                                        description="How overlapping cameras combine")
    focus_diameter: Field[float] = Field(4.5, min=1.0, max=12.0, step=0.5,
                                         description="Play-zone diameter (m) the image is stitched for — exact there, ghosts elsewhere")
    grid_degrees: Field[float] = Field(10.0, min=1.0, max=90.0, step=1.0,
                                       description="Grid spacing (°), the same on both axes")
    tilt: Field[float] = Field(0.0, access=Field.INIT,
                               description="Camera up-tilt (°), shared — shown in the footer; the rows come from the tracker")
    show_all_observations: Field[bool] = Field(True, widget=Widget.switch,
                                              description="Draw every camera's own opinion, not just the one the tracker picked")


# Fixed meanings, so not in ColorSettings: those are per-player track colours a user may tune, and
# a reader has to be able to tell a seam from an axis from the horizon.
GRID_COLOR:    tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.18)
HORIZON_COLOR: tuple[float, float, float, float] = (0.3, 1.0, 0.3, 1.0)    # its own hue, opaque
HORIZON_PX:    float = 1.0                        # the colour sets it apart, not the width
SEAM_COLOR:    tuple[float, float, float, float] = (1.0, 0.35, 0.0, 0.75)  # sector boundary
AXIS_COLOR:    tuple[float, float, float, float] = (0.0, 0.7, 1.0, 0.6)    # camera optical axis
# The two seam zones get their own hues, because they say opposite things and used to share one.
# Yellow is permissive (a second camera sees here), red prohibitive (no new person is born here).
# Their SHAPES and their HOMES differ too, and that is the point: yellow is a pair of lines in
# `GridRenderer`, because the overlap is a statement about azimuth and the degree grid measures it;
# red is a fill in `SeamRenderer`, because the dead zone is a band of one camera's image
# columns and only the picture can be read against it.
OVERLAP_COLOR:  tuple[float, float, float, float] = (1.0, 0.85, 0.0, 0.65)  # two cameras see it
DEAD_ZONE_COLOR: tuple[float, float, float, float] = (1.0, 0.15, 0.1, 0.10)  # no births here
# The tracked floor, as one band between the two diameters. The overlap's yellow, on purpose:
# together they say where the tracker works — the overlap bounds it in azimuth, the zone in
# distance — and shape tells them apart, the overlap a pair of verticals and the zone a horizontal
# field. Yellow also keeps both clear of the camera axes, which are the blue ones.
#
# A FIELD RATHER THAN TWO LINES, and the clipping is why it reads better: the zone reaches below
# what the strip can show at some presets, and a fill running off the bottom edge says "continues
# past here" by itself, where a line pinned to the boundary would have claimed a row that is not
# its own.
ZONE_COLOR:     tuple[float, float, float, float] = (1.0, 0.85, 0.0, 0.10)
LABEL_FG:      tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
LABEL_BG:      tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.6)
