import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from modules.pose.features.PosePoints import PosePointData
from modules.pose.features.PoseAngles import PoseAngleData, AngleJoint, ANGLE_JOINT_KEYPOINTS
from modules.pose.PoseTypes import PoseJoint, POSE_JOINT_COLORS
from modules.pose.PoseTypes import POSE_COLOR_ALPHA_BASE, POSE_COLOR_LEFT_POSITIVE, POSE_COLOR_LEFT_NEGATIVE, POSE_COLOR_RIGHT_POSITIVE, POSE_COLOR_RIGHT_NEGATIVE

# DEFINITIONS
POSE_VERTEX_LIST: list[list[PoseJoint]] = [
    [PoseJoint.nose,            PoseJoint.left_eye],
    [PoseJoint.nose,            PoseJoint.right_eye],
    [PoseJoint.left_eye,        PoseJoint.left_ear],
    [PoseJoint.right_eye,       PoseJoint.right_ear],
    [PoseJoint.left_shoulder,   PoseJoint.right_shoulder],
    [PoseJoint.left_shoulder,   PoseJoint.left_elbow],
    [PoseJoint.right_shoulder,  PoseJoint.right_elbow],
    [PoseJoint.left_elbow,      PoseJoint.left_wrist],
    [PoseJoint.right_elbow,     PoseJoint.right_wrist],
    [PoseJoint.left_shoulder,   PoseJoint.left_hip],
    [PoseJoint.right_shoulder,  PoseJoint.right_hip],
    [PoseJoint.left_hip,        PoseJoint.left_knee],
    [PoseJoint.right_hip,       PoseJoint.right_knee],
    [PoseJoint.left_knee,       PoseJoint.left_ankle],
    [PoseJoint.right_knee,      PoseJoint.right_ankle]
]
POSE_VERTEX_ARRAY: np.ndarray = np.array([kp.value for pose in POSE_VERTEX_LIST for kp in pose], dtype=np.int32)
POSE_VERTEX_INDICES: np.ndarray = np.arange(len(POSE_VERTEX_ARRAY), dtype=np.int32)

POSE_JOINT_TO_ANGLE_IDX: dict[PoseJoint, int] = {}
for angle_joint, keypoints in ANGLE_JOINT_KEYPOINTS.items():
    if len(keypoints) == 3:
        # For triplets, map the middle point (vertex) to this angle
        POSE_JOINT_TO_ANGLE_IDX[keypoints[1]] = angle_joint.value
    elif angle_joint == AngleJoint.head:
        # For head, map nose and eyes to head angle
        POSE_JOINT_TO_ANGLE_IDX[PoseJoint.nose] = angle_joint.value

# CLASSES
@dataclass(frozen=True)
class PoseVertexData:
    vertices: np.ndarray
    colors: np.ndarray

class PoseVertices:
    @staticmethod
    def compute_vertices(point_data: Optional[PosePointData]) -> Optional[PoseVertexData]:
        if point_data is None:
            return None

        vertices: np.ndarray = np.zeros((len(POSE_VERTEX_ARRAY), 2), dtype=np.float32)
        colors: np.ndarray = np.zeros((len(POSE_VERTEX_ARRAY), 4), dtype=np.float32)

        for i, (p1, p2) in enumerate(POSE_VERTEX_LIST):
            for j, joint in enumerate((p1, p2)):
                idx: int = i * 2 + j
                vertices[idx] = point_data.points[joint]
                colors[idx] = [*POSE_JOINT_COLORS[joint], (point_data.scores[joint] + POSE_COLOR_ALPHA_BASE) / (1.0 + POSE_COLOR_ALPHA_BASE)]

        vertex_data: PoseVertexData = PoseVertexData(vertices, colors)
        return vertex_data

    @staticmethod
    def compute_angled_vertices(point_data: Optional[PosePointData], angle_data: PoseAngleData) -> Optional[PoseVertexData]:

        if point_data is None:
            return None

        vertex_data: Optional[PoseVertexData] = PoseVertices.compute_vertices(point_data)
        if vertex_data is None:
            return None

        if not angle_data.has_data:
            return vertex_data

        colors: np.ndarray = vertex_data.colors

        for i, (p1, p2) in enumerate(POSE_VERTEX_LIST):
            for joint_pos, joint in enumerate((p1, p2)):
                idx: int | None = POSE_JOINT_TO_ANGLE_IDX.get(joint)
                if idx is not None:
                    angle: float = angle_data.angles[idx]
                    if not np.isnan(angle):
                        if joint.value % 2 == 1:  # Left side
                            colors[i*2 + joint_pos][0:3] = POSE_COLOR_LEFT_POSITIVE if angle >= 0 else POSE_COLOR_LEFT_NEGATIVE
                        else:  # Right side
                            colors[i*2 + joint_pos][0:3] = POSE_COLOR_RIGHT_POSITIVE if angle >= 0 else POSE_COLOR_RIGHT_NEGATIVE

        return PoseVertexData(vertex_data.vertices, colors)