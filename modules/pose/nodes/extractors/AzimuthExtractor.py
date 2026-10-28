import math
from typing import Callable

from ..Nodes import FilterNode
from ...features import Azimuth, BBox, BBoxAzimuth, Points2D, PointLandmark
from ...frame import Frame, replace

_EYES: list[PointLandmark] = [PointLandmark.left_eye, PointLandmark.right_eye]


class AzimuthExtractor(FilterNode):
    """Derives ``Azimuth``, a person's azimuth at their eyes, from ``BBoxAzimuth``.

    ``column_to_azimuth(cam_id, x)`` maps a normalised image column of a camera to a world
    azimuth in radians; it is the camera geometry, injected so this module stays camera-agnostic.
    ``BBox`` must be the crop rect the keypoints are normalised in, so a keypoint's image column
    is ``bbox.x + point.x * bbox.width`` and the box centre's is ``bbox.x + bbox.width / 2``.

    ``BBoxAzimuth`` is shifted by the difference between the two columns' azimuths rather than
    replaced by the eyes' own, so whatever the producer did to it (fusing two cameras' views at a
    seam) is kept. The eye column is the midpoint of both eyes, or the one valid eye. Without a box
    or both eyes, ``Azimuth`` is ``BBoxAzimuth`` unshifted; without a ``BBoxAzimuth`` the frame is
    left unchanged. Score is ``BBoxAzimuth``'s.

    Run it where ``Points2D`` and ``BBox`` are still the pair inference produced: keypoints smoothed
    across frames no longer follow the crop they are normalised in, and the box's jitter returns.
    """

    def __init__(self, column_to_azimuth: Callable[[int, float], float]) -> None:
        self._column_to_azimuth = column_to_azimuth

    def process(self, pose: Frame) -> Frame:
        bbox_azimuth = pose[BBoxAzimuth]
        if math.isnan(bbox_azimuth.value):
            return pose

        delta: float = self._eye_delta(pose)
        # SingleAngle.from_value wraps to [-π, π), which also folds a delta taken across 0/2π.
        return replace(pose, {Azimuth: Azimuth.from_value(bbox_azimuth.value + delta, bbox_azimuth.score)})

    def _eye_delta(self, pose: Frame) -> float:
        """Azimuth of the eye column less that of the box centre; 0.0 without a box or eyes."""
        rect = pose[BBox].to_rect()
        if math.isnan(rect.x) or math.isnan(rect.width):
            return 0.0

        points = pose[Points2D]
        eye_xs: list[float] = [float(points.values[eye][0]) for eye in _EYES if points.get_valid(eye)]
        if not eye_xs:
            return 0.0
        eye_x: float = sum(eye_xs) / len(eye_xs)

        eye_column: float = rect.x + eye_x * rect.width
        centre_column: float = rect.x + 0.5 * rect.width
        return (self._column_to_azimuth(pose.cam_id, eye_column)
                - self._column_to_azimuth(pose.cam_id, centre_column))
