from enum import IntEnum

# JOINTS
class PoseJoint(IntEnum):
    nose =          0
    left_eye =      1
    right_eye =     2
    left_ear =      3
    right_ear =     4
    left_shoulder = 5
    right_shoulder= 6
    left_elbow =    7
    right_elbow =   8
    left_wrist =    9
    right_wrist =   10
    left_hip =      11
    right_hip =     12
    left_knee =     13
    right_knee =    14
    left_ankle =    15
    right_ankle =   16
POSE_JOINT_NAMES: list[str] = [e.name for e in PoseJoint]
POSE_NUM_JOINTS:  int = len(PoseJoint)

# COLORS
POSE_COLOR_ALPHA_BASE:      float = 0.2
POSE_COLOR_CENTER:          tuple[float, float, float] = (0.8, 0.8, 0.8) # Light Gray
POSE_COLOR_LEFT:            tuple[float, float, float] = (1.0, 0.5, 0.0) # Orange
POSE_COLOR_LEFT_POSITIVE:   tuple[float, float, float] = (1.0, 1.0, 0.0) # Yellow
POSE_COLOR_LEFT_NEGATIVE:   tuple[float, float, float] = (1.0, 0.0, 0.0) # Red
POSE_COLOR_RIGHT:           tuple[float, float, float] = (0.0, 1.0, 1.0) # Cyan
POSE_COLOR_RIGHT_POSITIVE:  tuple[float, float, float] = (0.0, 0.5, 1.0) # Light Blue
POSE_COLOR_RIGHT_NEGATIVE:  tuple[float, float, float] = (0.0, 1.0, 0.5) # Light Green


# Define color for each joint
POSE_JOINT_COLORS: dict[PoseJoint, tuple[float, float, float]] = {
    # Central point
    PoseJoint.nose: POSE_COLOR_CENTER,

    # Left side points - orange
    PoseJoint.left_eye:         POSE_COLOR_LEFT,
    PoseJoint.left_ear:         POSE_COLOR_LEFT,
    PoseJoint.left_shoulder:    POSE_COLOR_LEFT,
    PoseJoint.left_elbow:       POSE_COLOR_LEFT,
    PoseJoint.left_wrist:       POSE_COLOR_LEFT,
    PoseJoint.left_hip:         POSE_COLOR_LEFT,
    PoseJoint.left_knee:        POSE_COLOR_LEFT,
    PoseJoint.left_ankle:       POSE_COLOR_LEFT,

    # Right side points - cyan
    PoseJoint.right_eye:        POSE_COLOR_RIGHT,
    PoseJoint.right_ear:        POSE_COLOR_RIGHT,
    PoseJoint.right_shoulder:   POSE_COLOR_RIGHT,
    PoseJoint.right_elbow:      POSE_COLOR_RIGHT,
    PoseJoint.right_wrist:      POSE_COLOR_RIGHT,
    PoseJoint.right_hip:        POSE_COLOR_RIGHT,
    PoseJoint.right_knee:       POSE_COLOR_RIGHT,
    PoseJoint.right_ankle:      POSE_COLOR_RIGHT
}

