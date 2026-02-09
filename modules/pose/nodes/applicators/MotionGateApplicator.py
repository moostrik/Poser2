# Standard library imports
from dataclasses import replace
from threading import Lock

import numpy as np

from modules.pose.features.MotionGate import MotionGate, configure_motion_gate
from modules.pose.features.base.NormalizedScalarFeature import AggregationMethod
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Frame import Frame, FrameDict


class MotionGateApplicatorConfig(NodeConfigBase):
    """Configuration for MotionGateApplicator."""

    def __init__(self, max_poses: int = 4) -> None:
        super().__init__()
        self.max_poses: int = max_poses


class MotionGateApplicator(FilterNode):
    """Filter that computes and applies motion gate for pose pairs.

    Motion gate is defined as: gate[A, B] = motion_A * motion_B
    where motion is the max aggregated AngleMotion value for each pose.

    Usage:
        1. Call submit(frames) with all current frames
        2. Call process(pose) for each pose to apply computed gate

    Thread-safe: Uses lock to protect stored motion gate dict.
    """

    def __init__(self, config: MotionGateApplicatorConfig | None = None) -> None:
        self._config = config if config is not None else MotionGateApplicatorConfig()
        configure_motion_gate(self._config.max_poses)
        self._motion_gate_dict: dict[int, MotionGate] = {}
        self._lock: Lock = Lock()

    def submit(self, frames: FrameDict) -> None:
        """Compute motion gates from all current frames.

        Args:
            frames: Dictionary mapping track_id -> Frame
        """
        max_poses = self._config.max_poses

        # Step 1: Extract motion values for each pose
        motion_values: dict[int, float] = {}
        for track_id, frame in frames.items():
            if frame.angle_motion is not None:
                motion = frame.angle_motion.aggregate(AggregationMethod.MAX)
                motion_values[track_id] = motion if not np.isnan(motion) else 0.0
            else:
                motion_values[track_id] = 0.0

        # Step 2: Build MotionGate for each pose
        motion_gate_dict: dict[int, MotionGate] = {}

        for track_id_i, motion_i in motion_values.items():
            values = np.zeros(max_poses, dtype=np.float32)
            scores = np.zeros(max_poses, dtype=np.float32)

            # Store self-motion at own index
            if track_id_i < max_poses:
                values[track_id_i] = motion_i
                scores[track_id_i] = 1.0

            # Store pairwise gates at other indices
            for track_id_j, motion_j in motion_values.items():
                if track_id_i == track_id_j:
                    continue  # Skip self-comparison for gate

                if track_id_j < max_poses:
                    # Gate = product of motions
                    gate = motion_i * motion_j
                    values[track_id_j] = gate
                    scores[track_id_j] = 1.0  # Full confidence when both present

            motion_gate_dict[track_id_i] = MotionGate(values=values, scores=scores)

        with self._lock:
            self._motion_gate_dict = motion_gate_dict

    def process(self, pose: Frame) -> Frame:
        """Apply pre-computed motion gate to this pose.

        Args:
            pose: Frame to enrich with motion gate data

        Returns:
            Frame with updated motion_gate field
        """
        with self._lock:
            motion_gate: MotionGate | None = self._motion_gate_dict.get(pose.track_id)

        if motion_gate is not None:
            pose = replace(pose, motion_gate=motion_gate)

        return pose
