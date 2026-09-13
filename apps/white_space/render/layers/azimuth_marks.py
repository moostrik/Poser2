"""Where the light strip overlay puts each person — the pure geometry, no GL, so it is unit-testable.

Two azimuths per person, as strip positions in [0, 1): the **eye** azimuth the light is placed at
(LERP frames, after ``EyeAzimuthExtractor``) and the tracker's **bbox-centre** azimuth it was
shifted from (PREDICT frames, upstream of the extractor). The gap between them is the correction.
"""

import math
from dataclasses import dataclass

from modules.pose.frame import Frame
from modules.pose.features import Azimuth
from modules.tracker import Tracklet

from apps.white_space.light.layers import angle_to_strip_position


@dataclass(frozen=True)
class AzimuthMark:
    track_id: int
    eye_x: float    # strip position [0, 1) of the eye azimuth; NaN when there is none
    bbox_x: float   # strip position [0, 1) of the bbox-centre azimuth; NaN when there is none


def _strip_x(frame: Frame | None) -> float:
    if frame is None:
        return math.nan
    return angle_to_strip_position(frame[Azimuth].value)


def build_azimuth_marks(eye_frames: dict[int, Frame], bbox_frames: dict[int, Frame],
                        tracklets: dict[int, Tracklet]) -> list[AzimuthMark]:
    """One mark per actively tracked person present in either frame set, ordered by track id.

    Only active tracklets, as the light layers count people; a person with neither azimuth is left out.
    """
    marks: list[AzimuthMark] = []
    for track_id in sorted(eye_frames.keys() | bbox_frames.keys()):
        tracklet: Tracklet | None = tracklets.get(track_id)
        if tracklet is None or not tracklet.is_active:
            continue
        eye_x: float = _strip_x(eye_frames.get(track_id))
        bbox_x: float = _strip_x(bbox_frames.get(track_id))
        if math.isnan(eye_x) and math.isnan(bbox_x):
            continue
        marks.append(AzimuthMark(track_id, eye_x, bbox_x))
    return marks


def signed_strip_gap(from_x: float, to_x: float) -> float:
    """Shortest signed distance from `from_x` to `to_x` on the strip, in [-0.5, 0.5).

    Positive is rightward. Taken across the 0/1 join when that is shorter, so a person standing on
    the join is not connected by a bar spanning the whole strip.
    """
    return (to_x - from_x + 0.5) % 1.0 - 0.5
