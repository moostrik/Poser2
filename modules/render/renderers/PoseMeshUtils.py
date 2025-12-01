# Standard library imports
from dataclasses import dataclass
from typing import Optional

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import Angles, AngleLandmark
from modules.pose.features.Points2D import Points2D, PointLandmark
from modules.pose.nodes._utils.AngleUtils import ANGLE_KEYPOINTS
from modules.utils.PointsAndRects import Rect

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
POSE_JOINT_COLORS: dict[PointLandmark, tuple[float, float, float]] = {
    # Central point
    PointLandmark.nose: POSE_COLOR_CENTER,

    # Left side points - orange
    PointLandmark.left_eye:         POSE_COLOR_LEFT,
    PointLandmark.left_ear:         POSE_COLOR_LEFT,
    PointLandmark.left_shoulder:    POSE_COLOR_LEFT,
    PointLandmark.left_elbow:       POSE_COLOR_LEFT,
    PointLandmark.left_wrist:       POSE_COLOR_LEFT,
    PointLandmark.left_hip:         POSE_COLOR_LEFT,
    PointLandmark.left_knee:        POSE_COLOR_LEFT,
    PointLandmark.left_ankle:       POSE_COLOR_LEFT,

    # Right side points - cyan
    PointLandmark.right_eye:        POSE_COLOR_RIGHT,
    PointLandmark.right_ear:        POSE_COLOR_RIGHT,
    PointLandmark.right_shoulder:   POSE_COLOR_RIGHT,
    PointLandmark.right_elbow:      POSE_COLOR_RIGHT,
    PointLandmark.right_wrist:      POSE_COLOR_RIGHT,
    PointLandmark.right_hip:        POSE_COLOR_RIGHT,
    PointLandmark.right_knee:       POSE_COLOR_RIGHT,
    PointLandmark.right_ankle:      POSE_COLOR_RIGHT
}


# DEFINITIONS
POSE_VERTEX_LIST: list[list[PointLandmark]] = [
    [PointLandmark.nose,            PointLandmark.left_eye],
    [PointLandmark.nose,            PointLandmark.right_eye],
    [PointLandmark.left_eye,        PointLandmark.left_ear],
    [PointLandmark.right_eye,       PointLandmark.right_ear],
    [PointLandmark.left_shoulder,   PointLandmark.right_shoulder],
    [PointLandmark.left_shoulder,   PointLandmark.left_elbow],
    [PointLandmark.right_shoulder,  PointLandmark.right_elbow],
    [PointLandmark.left_elbow,      PointLandmark.left_wrist],
    [PointLandmark.right_elbow,     PointLandmark.right_wrist],
    [PointLandmark.left_shoulder,   PointLandmark.left_hip],
    [PointLandmark.right_shoulder,  PointLandmark.right_hip],
    [PointLandmark.left_hip,        PointLandmark.right_hip],
    [PointLandmark.left_hip,        PointLandmark.left_knee],
    [PointLandmark.right_hip,       PointLandmark.right_knee],
    [PointLandmark.left_knee,       PointLandmark.left_ankle],
    [PointLandmark.right_knee,      PointLandmark.right_ankle]
]
POSE_VERTEX_ARRAY: np.ndarray = np.array([kp.value for pose in POSE_VERTEX_LIST for kp in pose], dtype=np.int32)
POSE_VERTEX_INDICES: np.ndarray = np.arange(len(POSE_VERTEX_ARRAY), dtype=np.int32)

POSE_JOINT_TO_ANGLE_IDX: dict[PointLandmark, int] = {}
for angle_joint, keypoints in ANGLE_KEYPOINTS.items():
    if len(keypoints) == 3:
        # For triplets, map the middle point (vertex) to this angle
        POSE_JOINT_TO_ANGLE_IDX[keypoints[1]] = angle_joint.value
    elif angle_joint == AngleLandmark.head:
        # For head, map nose and eyes to head angle
        POSE_JOINT_TO_ANGLE_IDX[PointLandmark.nose] = angle_joint.value

# CLASSES
@dataclass(frozen=True)
class PoseVertexData:
    vertices: np.ndarray
    colors: np.ndarray

class PoseMeshUtils:

    @staticmethod
    def compute_vertices_and_colors(points: Points2D, rect: Rect | None = None, color: tuple[float, float, float, float] | None = None) -> PoseVertexData:
        """
        Compute vertex positions and colors from pose points.

        Args:
            points: 2D pose keypoints
            rect: Optional rectangle for coordinate transformation
            color: Optional uniform color (overrides per-joint colors)

        Returns:
            PoseVertexData with vertices and colors. Invalid lines (either endpoint invalid) have alpha=0.
        """
        n_vertices = len(POSE_VERTEX_ARRAY)
        vertices = np.zeros((n_vertices, 2), dtype=np.float32)
        colors = np.zeros((n_vertices, 4), dtype=np.float32)

        for i, (p1, p2) in enumerate(POSE_VERTEX_LIST):
            base_idx = i * 2

            # Check if BOTH points are valid
            p1_valid = points.get_valid(p1)
            p2_valid = points.get_valid(p2)
            line_valid = p1_valid and p2_valid

            if line_valid:
                # Get coordinates for both points
                x1, y1 = points.get(p1)
                x2, y2 = points.get(p2)
                score1 = points.get_score(p1)
                score2 = points.get_score(p2)

                # Apply rectangle transformation
                if rect is not None:
                    x1 = x1 * rect.width + rect.x
                    y1 = y1 * rect.height + rect.y
                    x2 = x2 * rect.width + rect.x
                    y2 = y2 * rect.height + rect.y

                # Set vertices
                vertices[base_idx] = [x1, y1]
                vertices[base_idx + 1] = [x2, y2]

                # Set colors for both vertices
                for j, (joint, score) in enumerate([(p1, score1), (p2, score2)]):
                    idx = base_idx + j
                    alpha: float = (score + POSE_COLOR_ALPHA_BASE) / (1.0 + POSE_COLOR_ALPHA_BASE)

                    if color is not None:
                        colors[idx] = [*color[0:3], alpha * color[3]]
                    else:
                        rgb = POSE_JOINT_COLORS[joint]
                        colors[idx] = [*rgb, alpha]
            else:
                # Invalid line: both vertices at origin, fully transparent
                vertices[base_idx] = [0.0, 0.0]
                vertices[base_idx + 1] = [0.0, 0.0]
                colors[base_idx] = [0.0, 0.0, 0.0, 0.0]
                colors[base_idx + 1] = [0.0, 0.0, 0.0, 0.0]

        return PoseVertexData(vertices, colors)




    @staticmethod
    def compute_angled_vertices(points: Points2D, angles: Angles) -> PoseVertexData:


        vertex_data: Optional[PoseVertexData] = PoseMeshUtils.compute_vertices_and_colors(points)
        if vertex_data is None:
            return None

        if angles.valid_count == 0:
            return vertex_data

        colors: np.ndarray = vertex_data.colors

        for i, (p1, p2) in enumerate(POSE_VERTEX_LIST):
            for joint_pos, joint in enumerate((p1, p2)):
                idx: int | None = POSE_JOINT_TO_ANGLE_IDX.get(joint)
                if idx is not None:
                    angle: float = angles.values[idx]
                    if not np.isnan(angle):
                        if joint.value % 2 == 1:  # Left side
                            colors[i*2 + joint_pos][0:3] = POSE_COLOR_LEFT_POSITIVE if angle >= 0 else POSE_COLOR_LEFT_NEGATIVE
                        else:  # Right side
                            colors[i*2 + joint_pos][0:3] = POSE_COLOR_RIGHT_POSITIVE if angle >= 0 else POSE_COLOR_RIGHT_NEGATIVE

        return PoseVertexData(vertex_data.vertices, colors)