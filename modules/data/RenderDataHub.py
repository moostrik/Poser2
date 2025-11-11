"""
Provides smoothed pose data for rendering at output framerate.

Manages pose tracking components that smooth noisy input and interpolate between
samples, enabling higher framerate output than input (e.g., 24 FPS input → 60 FPS output).

Counterpart to CaptureDataHub:
- CaptureDataHub: Stores raw pose data at capture rate (input FPS)
- RenderDataHub: Provides smoothed pose data at render rate (output FPS)

Key capabilities:
1. Coordinates viewport, angle, and correlation tracking
2. Provides thread-safe access to smoothed pose data
3. Enables higher framerate output through interpolation
4. Manages pose correlation analysis between tracklets

Thread Safety:
--------------
Designed for multi-threaded operation:
- Input thread: Calls add_poses() at ~30 FPS
- Render thread: Calls update() and get_pose() at 60+ FPS

Completely lock-free using fixed-size arrays:
- Array sizes never change (no structural modifications)
- Element assignment/reads are atomic (single reference operations)
- Interpolators have internal locks for their own state
- Pose cache is separate - written once per update(), read many times per frame
"""
import time as time

# Local application imports
from modules.pose import features
from modules.pose.interpolators import PoseChaseInterpolator, ChaseInterpolatorConfig
from modules.pose.Pose import Pose, PoseDict

from modules.Settings import Settings
from modules.utils.HotReloadMethods import HotReloadMethods


class RenderDataHub:
    def __init__(self, settings: Settings) -> None:
        self._num_players: int = settings.num_players

        self.interpolator_config: ChaseInterpolatorConfig = ChaseInterpolatorConfig(
            input_frequency=settings.camera_fps,
            responsiveness=0.2,
            friction=0.03
        )

        # Lazy init - create on first add_poses()
        self._interpolators: list[PoseChaseInterpolator | None] = [None] * self._num_players
        self._cached_poses: list[Pose | None] = [None] * self._num_players

        # self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def update(self) -> None:
        """Update all active interpolators and cache results. Completely lock-free."""
        current_time: float = time.time()
        for i, interpolator in enumerate(self._interpolators):
            if interpolator is not None:
                # sets cached pose to None if not ready yet
                self._cached_poses[i] = interpolator.update(current_time)

    def reset(self) -> None:
        """Reset all interpolators and clear cache. Completely lock-free."""
        for i, interpolator in enumerate(self._interpolators):
            if interpolator is not None:
                interpolator.reset()

            self._cached_poses[i] = None

    def add_poses(self, poses: PoseDict) -> None:
        """Add new pose data points for processing. Completely lock-free."""
        for id, pose in poses.items():
            if id >= self._num_players:
                continue

            if pose.lost:
                self._interpolators[id] = None
                self._cached_poses[id] = None
            else:
                interpolator = self._interpolators[id]
                if interpolator is None:
                    # Create and initialize together
                    interpolator = PoseChaseInterpolator(self.interpolator_config)
                    self._interpolators[id] = interpolator
                interpolator.process(pose)

    def is_active(self, pose_id: int) -> bool:
        """Check if player has cached pose available."""
        if pose_id >= self._num_players:
            return False
        return self._cached_poses[pose_id] is not None

    def get_pose(self, pose_id: int) -> Pose:
        """Get smoothed pose from cache.

        Raises KeyError if no pose available. Use is_active() to check first.
        """
        if pose_id >= self._num_players:
            raise KeyError(f"Pose ID {pose_id} out of range")

        pose: Pose | None = self._cached_poses[pose_id]
        if pose is None:
            raise KeyError(f"No pose found for tracklet ID {pose_id}")

        return pose

   #  BODY JOINT ANGLES
    def get_angles(self, tracklet_id: int) -> features.AngleFeature:
        """Get angles for the specified tracklet ID"""
        return self.get_pose(tracklet_id).angles

    def get_delta(self, tracklet_id: int) -> features.AngleFeature:
        """Get delta for the specified tracklet ID"""
        return self.get_pose(tracklet_id).deltas

    def get_symmetries(self, tracklet_id: int) -> features.SymmetryFeature:
        return self.get_pose(tracklet_id).symmetry

    # TIME
    def get_motion_time(self, tracklet_id: int) -> float:
        """Get motion time for the specified tracklet ID."""
        return self.get_pose(tracklet_id).motion_time

    def get_age(self, tracklet_id: int) -> float:
        """Get the age in seconds since the tracklet was first detected."""
        return self.get_pose(tracklet_id).age