"""Where the projection-row overlay puts each person — the pure geometry, no GL, so it is unit-testable.

Two azimuths per person, as normalized azimuths in [0, 1): the **eye** azimuth the light is placed at
(LERP frames, after ``EyeAzimuthExtractor``) and the tracker's **bbox-centre** azimuth it was
shifted from (PREDICT frames, upstream of the extractor). The gap between them is the correction.
The row spans one turn, so a normalized azimuth is also the row's x.
"""

import math
from dataclasses import dataclass

from modules.pose.frame import Frame
from modules.pose.features import Azimuth

from apps.white_space.light.layers import normalize_azimuth


@dataclass(frozen=True)
class AzimuthMark:
    track_id: int
    eye_x: float    # normalized azimuth of the eye; NaN when there is none
    bbox_x: float   # normalized azimuth of the bbox centre; NaN when there is none


def _normalized_azimuth(frame: Frame | None) -> float:
    if frame is None:
        return math.nan
    return normalize_azimuth(frame[Azimuth].value)


def build_azimuth_marks(eye_frames: dict[int, Frame], bbox_frames: dict[int, Frame]) -> list[AzimuthMark]:
    """One mark per person with a pose in either frame set, ordered by track id — the same people
    the light layers draw. A person with neither azimuth is left out.
    """
    marks: list[AzimuthMark] = []
    for track_id in sorted(eye_frames.keys() | bbox_frames.keys()):
        eye_x: float = _normalized_azimuth(eye_frames.get(track_id))
        bbox_x: float = _normalized_azimuth(bbox_frames.get(track_id))
        if math.isnan(eye_x) and math.isnan(bbox_x):
            continue
        marks.append(AzimuthMark(track_id, eye_x, bbox_x))
    return marks


def signed_azimuth_gap(from_x: float, to_x: float) -> float:
    """Shortest signed distance from `from_x` to `to_x` in normalized azimuth, in [-0.5, 0.5).

    Positive is rightward. Taken across the 0/1 join when that is shorter, so a person standing on
    the join is not connected by a bar spanning the whole row.
    """
    return (to_x - from_x + 0.5) % 1.0 - 0.5
