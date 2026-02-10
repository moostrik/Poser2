from modules.pose.features.Angles import AngleLandmark
from modules.pose.features.base.NormalizedSingleValue import NormalizedSingleValue


# Joint-specific normalization weights (higher = more sensitive)
ANGLE_MOTION_NORMALISATION: dict[AngleLandmark, float] = {
    AngleLandmark.left_shoulder:   1.0,
    AngleLandmark.right_shoulder:  1.0,
    AngleLandmark.left_elbow:      1.0,
    AngleLandmark.right_elbow:     1.0,
    AngleLandmark.left_hip:        2.0,
    AngleLandmark.right_hip:       2.0,
    AngleLandmark.left_knee:       2.0,
    AngleLandmark.right_knee:      2.0,
    AngleLandmark.head:            3.0,
}


class AngleMotion(NormalizedSingleValue):
    """Single normalized motion value representing overall body movement intensity.

    Value range: [0.0, 1.0] where 0 = no motion, 1 = maximum motion.
    Computed as the max of weighted per-joint angular velocities.
    """
    pass