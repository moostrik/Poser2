import math
from typing import Callable

from ..Nodes import FilterNode
from ...features import Azimuth, BBox, Points2D, PointLandmark
from ...frame import Frame, replace

_EYES: list[PointLandmark] = [PointLandmark.left_eye, PointLandmark.right_eye]


class EyeAzimuthExtractor(FilterNode):
    """Moves ``Azimuth`` from the bbox centre to the eyes.

    ``column_to_azimuth(cam_id, x)`` maps a normalised image column of a camera to a world
    azimuth in radians; it is the camera geometry, injected so this module stays camera-agnostic.
    ``BBox`` must be the crop rect the keypoints are normalised in, so a keypoint's image column
    is ``bbox.x + point.x * bbox.width`` and the box centre's is ``bbox.x + bbox.width / 2``.

    The azimuth is shifted by the difference between the two columns' azimuths rather than
    replaced by the eyes' own, so whatever the producer did to ``Azimuth`` (fusing two cameras'
    views at a seam) is kept. The eye column is the midpoint of both eyes, or the one valid eye.
    Leaves the frame unchanged when the azimuth, the box or both eyes are missing. Score kept.

    Must run exactly once per frame on an un-anchored ``Azimuth``: a second pass shifts again.
    """

    def __init__(self, column_to_azimuth: Callable[[int, float], float]) -> None:
        self._column_to_azimuth = column_to_azimuth

    def process(self, pose: Frame) -> Frame:
        azimuth = pose[Azimuth]
        if math.isnan(azimuth.value):
            return pose

        rect = pose[BBox].to_rect()
        if math.isnan(rect.x) or math.isnan(rect.width):
            return pose

        points = pose[Points2D]
        eye_xs: list[float] = [float(points.values[eye][0]) for eye in _EYES if points.get_valid(eye)]
        if not eye_xs:
            return pose
        eye_x: float = sum(eye_xs) / len(eye_xs)

        eye_column: float = rect.x + eye_x * rect.width
        centre_column: float = rect.x + 0.5 * rect.width
        delta: float = (self._column_to_azimuth(pose.cam_id, eye_column)
                        - self._column_to_azimuth(pose.cam_id, centre_column))

        # SingleAngle.from_value wraps to [-π, π), which also folds a delta taken across 0/2π.
        return replace(pose, {Azimuth: Azimuth.from_value(azimuth.value + delta, azimuth.score)})
