from dataclasses import dataclass, field
from enum import Enum, IntEnum
import numpy as np
from typing import TypedDict
from typing import Optional, Callable
from modules.tracker.Tracklet import Tracklet
from modules.utils.PointsAndRects import Rect


# MODEL
POSEMODELWIDTH = 192
POSEMODELHEIGHT = 256

class PoseModelType(Enum):
    NONE =   0
    LARGE =  1
    MEDIUM = 2
    SMALL =  3
    TINY =   4
PoseModelTypeNames: list[str] = [e.name for e in PoseModelType]

PoseModelFileNames: list[tuple[str, str]] = [
    ('none', ''),
    ('rtmpose-l_8xb256-420e_aic-coco-256x192.py', 'rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'),
    ('rtmpose-m_8xb256-420e_aic-coco-256x192.py', 'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'),
    ('rtmpose-s_8xb256-420e_aic-coco-256x192.py', 'rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'),
    ('rtmpose-t_8xb256-420e_aic-coco-256x192.py', 'rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth')
]

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
PoseJointNames: list[str] = [e.name for e in PoseJoint]

# VERTICES
PoseVertexList: list[list[PoseJoint]] = [
    [PoseJoint.nose, PoseJoint.left_eye],
    [PoseJoint.nose, PoseJoint.right_eye],
    [PoseJoint.left_eye, PoseJoint.left_ear],
    [PoseJoint.right_eye, PoseJoint.right_ear],
    [PoseJoint.left_shoulder, PoseJoint.right_shoulder],
    [PoseJoint.left_shoulder, PoseJoint.left_elbow],
    [PoseJoint.right_shoulder, PoseJoint.right_elbow],
    [PoseJoint.left_elbow, PoseJoint.left_wrist],
    [PoseJoint.right_elbow, PoseJoint.right_wrist],
    [PoseJoint.left_shoulder, PoseJoint.left_hip],
    [PoseJoint.right_shoulder, PoseJoint.right_hip],
    [PoseJoint.left_hip, PoseJoint.left_knee],
    [PoseJoint.right_hip, PoseJoint.right_knee],
    [PoseJoint.left_knee, PoseJoint.left_ankle],
    [PoseJoint.right_knee, PoseJoint.right_ankle]
]
PoseVertexArray: np.ndarray = np.array([kp.value for pose in PoseVertexList for kp in pose], dtype=np.int32)
PoseVertexIndices: np.ndarray = np.arange(len(PoseVertexArray), dtype=np.int32)
PoseVertexColors: list[tuple[float, float, float]] = [
    (1.0, 0.7, 0.4),   # nose-left_eye
    (0.4, 0.7, 1.0),   # nose-right_eye
    (1.0, 0.8, 0.6),   # left_eye-left_ear
    (0.6, 0.8, 1.0),   # right_eye-right_ear
    (1.0, 1.0, 0.0),   # left_shoulder-right_shoulder
    (1.0, 0.7, 0.4),   # left_shoulder-left_elbow
    (0.4, 0.7, 1.0),   # right_shoulder-right_elbow
    (1.0, 0.8, 0.6),   # left_elbow-left_wrist
    (0.6, 0.8, 1.0),   # right_elbow-right_wrist
    (1.0, 0.6, 0.2),   # left_shoulder-left_hip
    (0.2, 0.6, 1.0),   # right_shoulder-right_hip
    (1.0, 0.7, 0.4),   # left_hip-left_knee
    (0.4, 0.7, 1.0),   # right_hip-right_knee
    (1.0, 0.8, 0.6),   # left_knee-left_ankle
    (0.6, 0.8, 1.0),   # right_knee-right_ankle
]

# POINT DATA
@dataclass (frozen=True)
class PosePointData():
    points: np.ndarray      # shape (17, 2)
    scores: np.ndarray      # shape (17, 1)

    _vertices: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _vertex_colors: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @property
    def vertices(self) -> np.ndarray:
        if self._vertices is not None:
            return self._vertices

        vertices: np.ndarray = np.zeros((len(PoseVertexArray), 2), dtype=np.float32)
        for i in range(len(PoseVertexArray)):
            vertices[i] = self.points[PoseVertexArray[i]]

        object.__setattr__(self, '_vertices', vertices)
        return vertices

    @property
    def vertex_colors(self) -> np.ndarray:
        if self._vertex_colors is not None:
            return self._vertex_colors

        vertex_colors: np.ndarray = np.zeros((len(PoseVertexArray), 4), dtype=np.float32)
        scores: np.ndarray = self.scores
        for i in range(len(PoseVertexList)):
            kp1: int = PoseVertexList[i][0].value
            kp2: int = PoseVertexList[i][1].value
            s1: float = scores[kp1]
            s2: float = scores[kp2]
            score: float = min(s1, s2)
            alpha: float = score
            C: tuple[float, float, float] = PoseVertexColors[i]
            vertex_colors[i*2] = [C[0], C[1], C[2], alpha]
            vertex_colors[i*2+1] = [C[0], C[1], C[2], alpha]

        object.__setattr__(self, '_vertex_colors', vertex_colors)
        return vertex_colors

# ANGLE DATA
PoseAngleJointTriplets: dict[PoseJoint, tuple[PoseJoint, PoseJoint, PoseJoint]] = {
    # PoseJoint.nose:           ( PoseJoint.left_eye,       PoseJoint.nose,           PoseJoint.right_eye   ),0
    # PoseJoint.left_eye:       ( PoseJoint.left_ear,       PoseJoint.left_eye,       PoseJoint.right_eye   ),
    # PoseJoint.right_eye:      ( PoseJoint.left_eye,       PoseJoint.right_eye,      PoseJoint.right_ear   ),
    PoseJoint.left_shoulder:  ( PoseJoint.left_hip,       PoseJoint.left_shoulder,  PoseJoint.left_elbow  ),
    PoseJoint.right_shoulder: ( PoseJoint.right_hip,      PoseJoint.right_shoulder, PoseJoint.right_elbow ),
    PoseJoint.left_elbow:     ( PoseJoint.left_shoulder,  PoseJoint.left_elbow,     PoseJoint.left_wrist  ),
    PoseJoint.right_elbow:    ( PoseJoint.right_shoulder, PoseJoint.right_elbow,    PoseJoint.right_wrist ),
    # PoseJoint.left_hip:       ( PoseJoint.left_shoulder,  PoseJoint.left_hip,       PoseJoint.left_knee   ),
    # PoseJoint.right_hip:      ( PoseJoint.right_shoulder, PoseJoint.right_hip,      PoseJoint.right_knee  ),
    # PoseJoint.left_knee:      ( PoseJoint.left_hip,       PoseJoint.left_knee,      PoseJoint.left_ankle  ),
    # PoseJoint.right_knee:     ( PoseJoint.right_hip,      PoseJoint.right_knee,     PoseJoint.right_ankle ),
}
PoseAngleJointLookup: dict[PoseJoint, int] = {joint: i for i, joint in enumerate(PoseAngleJointTriplets.keys())}
PoseAngleJointNames: list[str] = [e.name for e in PoseAngleJointTriplets.keys()]
NUM_POSE_ANGLES: int = len(PoseAngleJointTriplets)

@dataclass (frozen=True)
class PoseAngleData():
    angles: np.ndarray = np.full(NUM_POSE_ANGLES, np.nan, dtype=np.float32) # The computed joint angles (in radians [-Pi...Pi], or np.nan if invalid)
    scores: np.ndarray = np.zeros(NUM_POSE_ANGLES, dtype=np.float32)        # The minimum confidence score among the three PoseJoints

# THE POSE
@dataclass (frozen=True)
class Pose:
    tracklet: Tracklet = field(repr=False)

    crop_rect: Optional[Rect] = field(default = None)
    crop_image: Optional[np.ndarray] = field(default = None, repr=False)

    point_data: Optional[PosePointData] = field(default=None, repr=False)
    angle_data: Optional[PoseAngleData] = field(default=None)

    def __getattribute__(self, name):
        # Try to get attribute from Pose first
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # If not found, delegate to tracklet
            tracklet = super().__getattribute__('tracklet')
            if hasattr(tracklet, name):
                return getattr(tracklet, name)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getattr__(self, name):
        """
        Delegate attribute access to tracklet if attribute isn't found in Pose
        For backward compatibility, use pose.tracklet.id, etc. instead
        """
        if hasattr(self.tracklet, name):
            return getattr(self.tracklet, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @property
    def absolute_points(self) -> Optional[np.ndarray]:
        """
        Get PoseJoints in the original rectangle coordinates.
        Returns a tuple of (PoseJoints, scores) or None if not available.
        """
        if self.point_data is None or self.crop_rect is None:
            return None

        PoseJoints: np.ndarray = self.point_data.points  # Normalized coordinates within the model
        rect: Rect = self.crop_rect

        # Convert from normalized coordinates to actual pixel coordinates in the crop rect
        real_PoseJoints: np.ndarray = np.zeros_like(PoseJoints)
        real_PoseJoints[:, 0] = PoseJoints[:, 0] * rect.width + rect.x  # x coordinates
        real_PoseJoints[:, 1] = PoseJoints[:, 1] * rect.height + rect.y  # y coordinates

        return real_PoseJoints

    def get_approximate_person_length(self, threshold: float = 0.3) -> float | None:
        """
        Estimate the person's length by summing the lengths of both arms and both legs,
        only if all PoseJoints for a limb are above the confidence threshold.
        """

        if self.point_data is None:
            return None

        PoseJoints: np.ndarray = self.point_data.points
        scores: np.ndarray = self.point_data.scores
        height: float = self.crop_rect.height if self.crop_rect is not None else 1.0

        # Define the PoseJoint triplets for each limb
        limbs = [
            # Arms: shoulder -> elbow -> wrist
            (PoseJoint.left_shoulder, PoseJoint.left_elbow, PoseJoint.left_wrist),
            (PoseJoint.right_shoulder, PoseJoint.right_elbow, PoseJoint.right_wrist),
            # Legs: hip -> knee -> ankle
            (PoseJoint.left_hip, PoseJoint.left_knee, PoseJoint.left_ankle),
            (PoseJoint.right_hip, PoseJoint.right_knee, PoseJoint.right_ankle),
        ]

        limb_lengths: list[float] = []
        for kp1, kp2, kp3 in limbs:
            # Check if all PoseJoints for this limb are above threshold
            if (scores[kp1] > threshold and scores[kp2] > threshold and scores[kp3] > threshold):
                # Calculate limb length as sum of two segments
                seg1: float = float(np.linalg.norm(PoseJoints[kp1] - PoseJoints[kp2]))
                seg2: float = float(np.linalg.norm(PoseJoints[kp2] - PoseJoints[kp3]))
                limb_lengths.append(seg1 + seg2)

        return max(limb_lengths) * 2.5 * height if limb_lengths else None

PoseDict = dict[PoseJoint, Pose]
PoseCallback = Callable[[Pose], None]