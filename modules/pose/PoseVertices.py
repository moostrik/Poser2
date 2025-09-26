import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from modules.pose.PosePoints import PosePointData
from modules.pose.PoseAngles import PoseAngleData
from modules.pose.PoseTypes import (
    PoseVertexArray, PoseVertexList, PoseJointColors,
    BASE_ALPHA, LEFT_POSITIVE, LEFT_NEGATIVE, RIGHT_POSITIVE, RIGHT_NEGATIVE,
    PoseAngleJointIdx
)

# VERTICES
@dataclass(frozen=True)
class PoseVertexData:
    vertices: np.ndarray
    colors: np.ndarray

class PoseVertices:
    @staticmethod
    def compute_vertices(point_data: Optional[PosePointData]) -> Optional[PoseVertexData]:
        if point_data is None:
            return None

        vertices: np.ndarray = np.zeros((len(PoseVertexArray), 2), dtype=np.float32)
        colors: np.ndarray = np.zeros((len(PoseVertexArray), 4), dtype=np.float32)

        for i, (p1, p2) in enumerate(PoseVertexList):
            for j, joint in enumerate((p1, p2)):
                idx: int = i * 2 + j
                vertices[idx] = point_data.points[joint]
                colors[idx] = [*PoseJointColors[joint], (point_data.scores[joint] + BASE_ALPHA) / (1.0 + BASE_ALPHA)]

        vertex_data: PoseVertexData = PoseVertexData(vertices, colors)
        return vertex_data

    @staticmethod
    def compute_angled_vertices(point_data: Optional[PosePointData], angle_data: Optional[PoseAngleData]) -> Optional[PoseVertexData]:

        if point_data is None:
            return None

        vertex_data: Optional[PoseVertexData] = PoseVertices.compute_vertices(point_data)
        if vertex_data is None:
            return None

        if angle_data is None:
            return vertex_data

        colors: np.ndarray = vertex_data.colors.copy()

        for i, (p1, p2) in enumerate(PoseVertexList):
            for joint_pos, joint in enumerate((p1, p2)):
                idx: int | None = PoseAngleJointIdx.get(joint)
                if idx is not None:
                    angle: float = angle_data.angles[idx]
                    if not np.isnan(angle):
                        if joint.value % 2 == 1:  # Left side
                            colors[i*2 + joint_pos][0:3] = LEFT_POSITIVE if angle >= 0 else LEFT_NEGATIVE
                        else:  # Right side
                            colors[i*2 + joint_pos][0:3] = RIGHT_POSITIVE if angle >= 0 else RIGHT_NEGATIVE

        return PoseVertexData(vertex_data.vertices, colors)